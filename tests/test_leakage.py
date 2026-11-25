"""CRITICAL Stage 3 tests: strict end-time leakage equivalence, forbidden column checks, and horizon config overrides."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

from config.settings import get_settings
from src.feature_engineering import (
    assert_no_forbidden_features,
    build_forecast_dataset_from_frames,
    compute_deployable_feature_names,
)


def _make_multiday_synthetic_flows() -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(123)
    rows: list[dict] = []
    # Segment 0: 70 minutes starting 12:00, with an attack starting at minute 42
    # Segment 1 (after 90-min break > 60): 60 minutes starting 14:40, with an attack at minute 40
    seg_bases = [
        (datetime(2017, 7, 4, 12, 0, tzinfo=timezone.utc), 70, 42),
        (datetime(2017, 7, 4, 14, 40, tzinfo=timezone.utc), 60, 40),
    ]
    for base, n_mins, atk_min in seg_bases:
        for m in range(n_mins):
            # Include both fast flows (end in minute m) and slow flows (spanning 90s -> end in minute m+1)
            for k in range(8):
                dur_us = 2_000_000 if k < 6 else 95_000_000
                st = base + timedelta(minutes=m, seconds=k * 5)
                en = st + timedelta(microseconds=dur_us)
                is_atk = 1 if (m >= atk_min and m < atk_min + 4) else 0
                rows.append(
                    {
                        "day": "tuesday",
                        "start_ts": st,
                        "end_ts": en,
                        "Src IP": f"192.168.10.{1 + ((m + k) % 15)}",
                        "Dst IP": f"10.0.0.{1 + (k % 3)}",
                        "Dst Port": int([80, 443, 22, 21, 53][k % 5]),
                        "Flow Duration": dur_us,
                        "Total Fwd Packet": int(rng.integers(1, 20)),
                        "Total Bwd packets": int(rng.integers(0, 15)),
                        "Total Length of Fwd Packet": int(rng.integers(40, 1500)),
                        "Total Length of Bwd Packet": int(rng.integers(0, 2000)),
                        "SYN Flag Count": int(k % 2),
                        "Packet Length Mean": float(rng.uniform(40.0, 500.0)),
                        "is_attack": is_atk,
                    }
                )
    flows_df = pl.DataFrame(rows).with_columns(
        [
            pl.col("start_ts").cast(pl.Datetime("us", "UTC")),
            pl.col("end_ts").cast(pl.Datetime("us", "UTC")),
            pl.col("is_attack").cast(pl.Int8),
        ]
    )
    anom_df = flows_df.select(["start_ts", "end_ts"]).with_columns(
        [
            pl.Series("anomaly_score", rng.uniform(0.3, 0.7, size=flows_df.height), dtype=pl.Float32),
            pl.Series("is_anomalous", (rng.uniform(0, 1, size=flows_df.height) > 0.9).astype(np.int8)),
        ]
    )
    return flows_df, anom_df


def test_no_forbidden_columns_in_deployable_features() -> None:
    import pytest

    cfg = get_settings(horizons=[5, 15])
    deploy_cols = compute_deployable_feature_names(cfg.feature_windows)
    assert_no_forbidden_features(deploy_cols)
    assert len(deploy_cols) == 17 * (1 + len(cfg.feature_windows) * 6)
    for forbidden in ("y_5", "oracle_active_frac_5", "is_attack", "label", "hour_of_day", "day"):
        with pytest.raises(AssertionError):
            assert_no_forbidden_features([forbidden])


def test_leakage_equivalence_under_end_time_truncation() -> None:
    """For sampled minutes t, recomputing features from flows truncated at end_ts < t + 60s MUST match exactly."""
    cfg = get_settings(horizons=[5], feature_windows=[5, 15, 30], k_min=5, quiet_gap_min=5)
    flows_df, anom_df = _make_multiday_synthetic_flows()
    full_df = build_forecast_dataset_from_frames(flows_df, anom_df, cfg)
    deploy_cols = compute_deployable_feature_names(cfg.feature_windows)

    eligible = full_df.filter(pl.col("is_eligible") == 1)
    assert eligible.height > 15

    # Sample 8 decision minutes spread across both segments
    sample_indices = np.linspace(0, eligible.height - 1, 8, dtype=int)
    sampled_minutes = [eligible["minute_ts"][int(i)] for i in sample_indices]

    for t in sampled_minutes:
        cutoff = t + timedelta(seconds=cfg.bin_seconds)
        mask = flows_df["end_ts"] < pl.lit(cutoff)
        trunc_flows = flows_df.filter(mask)
        trunc_anom = anom_df.filter(mask)

        recomputed_df = build_forecast_dataset_from_frames(trunc_flows, trunc_anom, cfg)
        row_full = full_df.filter(pl.col("minute_ts") == pl.lit(t)).select(deploy_cols).to_numpy()
        row_trunc = recomputed_df.filter(pl.col("minute_ts") == pl.lit(t)).select(deploy_cols).to_numpy()

        assert row_full.shape == (1, len(deploy_cols))
        assert row_trunc.shape == (1, len(deploy_cols))
        assert np.allclose(row_full, row_trunc, rtol=1e-7, atol=1e-7), f"Leakage detected at minute {t}"


def test_config_override_horizons_changes_targets() -> None:
    flows_df, anom_df = _make_multiday_synthetic_flows()
    cfg_5 = get_settings(horizons=[5], k_min=5, quiet_gap_min=5)
    cfg_5_15 = get_settings(horizons=[5, 15], k_min=5, quiet_gap_min=5)

    df_5 = build_forecast_dataset_from_frames(flows_df, anom_df, cfg_5)
    df_5_15 = build_forecast_dataset_from_frames(flows_df, anom_df, cfg_5_15)

    assert "y_5" in df_5.columns and "y_15" not in df_5.columns
    assert "y_5" in df_5_15.columns and "y_15" in df_5_15.columns
    assert int(df_5_15["y_15"].sum()) > int(df_5_15["y_5"].sum())
