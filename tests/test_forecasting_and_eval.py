"""Tests for Stage 3 chronological splits, 1-to-1 alert-episode matching, and forecasting + eval pipeline."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

from config.settings import get_settings
from src.evaluation import compute_alert_episode_metrics, evaluate_model_predictions, run_evaluation
from src.feature_engineering import build_forecast_dataset_from_frames
from src.forecasting import find_chronological_onset_split, run_forecasting


def test_hand_computed_alert_episode_metrics_one_to_one() -> None:
    """Exact hand-computed verification of 1-to-1 matching, lead time, and false alerts/hr.

    Tests explicitly that TWO separate alert episodes before ONE onset yield
    1 hit episode + 1 false episode (multiple episodes cannot claim credit for 1 onset).
    """
    base = datetime(2017, 7, 7, 16, 0)
    ts = np.array([base + timedelta(minutes=i) for i in range(120)], dtype="datetime64[us]")
    seg = np.zeros(120, dtype=np.int32)
    onsets = np.zeros(120, dtype=np.int8)
    alerts = np.zeros(120, dtype=np.int8)

    # Onset 1 at minute 20:
    # Episode 1 = [15] (lead 5), gap to 19 is 4 > 2, Episode 2 = [19] (also within H=5 of onset 20!)
    # Under 1-to-1 matching: Episode 1 matches Onset 1 (lead=5), Episode 2 MUST be counted as false!
    onsets[20] = 1
    alerts[15] = 1
    alerts[19] = 1

    # False alerts at 50 and 52 (gap 2 <= 2 -> Episode 3 [50,52])
    alerts[50] = 1
    alerts[52] = 1

    # Separate false alert at 55 (gap from 52 is 3 > 2 -> Episode 4 [55])
    alerts[55] = 1

    # Onset 2 at minute 90; alert at 90 itself is NOT a < onset (Episode 5 [90] is false!)
    onsets[90] = 1
    alerts[90] = 1

    res = compute_alert_episode_metrics(
        minute_ts=ts,
        segment_id=seg,
        alerts=alerts,
        onset_flags=onsets,
        horizon_min=5,
        merge_gap_min=2,
        observed_minutes=120,
    )
    assert res["total_onsets"] == 2
    assert res["warned_onsets"] == 1
    assert res["hit_episodes"] == 1
    assert res["total_episodes"] == 5
    assert res["false_episodes"] == 4
    assert res["event_onset_recall"] == 0.5
    assert res["lead_times_min"] == [5.0]
    assert res["false_alerts_per_hour"] == 2.0


def test_eligible_observation_duration_used_for_false_alert_rate() -> None:
    """Verify false_alerts_per_hour divides by eligible minutes, not wall-clock minutes."""
    base = datetime(2017, 7, 7, 16, 0, tzinfo=timezone.utc)
    # 90 wall-clock minutes, only 60 eligible (1.0 eligible hour)
    ts = [base + timedelta(minutes=i) for i in range(90)]
    elig = [1] * 60 + [0] * 30
    onsets = [0] * 90
    y5 = [0] * 90
    probs_elig = np.zeros(60, dtype=np.float64)
    probs_elig[10] = 0.9  # 1 false alert episode

    tl = pl.DataFrame(
        {
            "minute_ts": pl.Series("minute_ts", ts).cast(pl.Datetime("us", "UTC")),
            "segment_id": pl.Series("segment_id", [0] * 90, dtype=pl.Int32),
            "is_eligible": pl.Series("is_eligible", elig, dtype=pl.Int8),
            "onset": pl.Series("onset", onsets, dtype=pl.Int8),
            "y_5": pl.Series("y_5", y5, dtype=pl.Int8),
        }
    )
    m = evaluate_model_predictions(tl, probs_elig, threshold=0.5, horizon_min=5, merge_gap_min=2)
    assert m["n_eligible_rows"] == 60
    assert m["n_minutes_wall_clock"] == 90
    assert abs(m["false_alerts_per_hour"] - 1.0) < 1e-6


def test_chronological_split_and_pipeline_smoke(tmp_path: Path) -> None:
    """Verify no overlap, required gaps, and full Stage 3 + Stage 4 execution on synthetic multi-onset timeline."""
    cfg = get_settings(
        project_root=tmp_path,
        horizons=[5],
        feature_windows=[5, 10],
        k_min=5,
        quiet_gap_min=5,
        min_train_onsets=2,
        min_val_onsets=2,
        min_test_onsets=1,
        bootstrap_reps=20,
        bootstrap_block_min=20,
        seeds=[0, 1],
    )
    cfg.ensure_dirs()

    rng = np.random.default_rng(7)
    base = datetime(2017, 7, 5, 12, 0, tzinfo=timezone.utc)
    onset_mins = {30, 65, 110, 145, 195, 225}
    rows = []
    for m in range(240):
        precursor = any(0 < (o - m) <= 5 for o in onset_mins)
        n_flows = 12 if m in onset_mins else (9 if precursor else 4)
        for k in range(n_flows):
            st = base + timedelta(minutes=m, seconds=k * 3)
            en = st + timedelta(seconds=1)
            rows.append(
                {
                    "day": "wednesday",
                    "start_ts": st,
                    "end_ts": en,
                    "Src IP": f"10.0.0.{1 + (k % 5)}",
                    "Dst IP": "192.168.10.50",
                    "Dst Port": 80 if not precursor else 22 + k,
                    "Flow Duration": 1_000_000,
                    "Total Fwd Packet": 10 if precursor else 2,
                    "Total Bwd packets": 2,
                    "Total Length of Fwd Packet": 800 if precursor else 100,
                    "Total Length of Bwd Packet": 100,
                    "SYN Flag Count": 1 if precursor else 0,
                    "Packet Length Mean": 150.0 if precursor else 50.0,
                    "is_attack": 1 if m in onset_mins else 0,
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
            pl.Series("anomaly_score", rng.uniform(0.3, 0.5, size=flows_df.height), dtype=pl.Float32),
            pl.Series("is_anomalous", np.zeros(flows_df.height, dtype=np.int8)),
        ]
    )
    fdf = build_forecast_dataset_from_frames(flows_df, anom_df, cfg)
    fdf.write_parquet(cfg.data_processed_dir / "forecast_features.parquet")

    split_res = find_chronological_onset_split(
        fdf,
        horizon_min=5,
        gap_minutes=cfg.split_gap_min,
        min_train_onsets=2,
        min_val_onsets=2,
        min_test_onsets=1,
    )
    info = split_res["split_info"]
    assert info["gap1_minutes"] >= cfg.split_gap_min
    assert info["gap2_minutes"] >= cfg.split_gap_min
    assert info["train_end"] < info["val_start"] < info["val_end"] < info["test_start"]

    run_forecasting(cfg)
    ev = run_evaluation(cfg)
    assert (cfg.results_dir / "verdict.json").exists()
    assert (cfg.results_dir / "forecast_metrics.csv").exists()
    assert (cfg.results_dir / "bootstrap_ci.json").exists()
    assert ev["verdict"]["verdict"] in {
        "OUTPERFORMS",
        "NOT DISTINGUISHABLE",
        "UNDERPERFORMS",
        "INCONCLUSIVE: insufficient events",
    }
