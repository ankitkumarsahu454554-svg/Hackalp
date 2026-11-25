"""Stage 3A: Strictly causal, end-time-available minute feature engineering and onset target construction."""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import polars as pl

from config.settings import Settings, get_settings
from src.data_loader import get_processed_parquet_paths

logger = logging.getLogger(__name__)

BASE_MINUTE_FEATURES = [
    "flow_count",
    "packets_per_sec",
    "bytes_per_sec",
    "mean_flow_duration_us",
    "median_flow_duration_us",
    "unique_src_ips",
    "unique_dst_ips",
    "unique_dst_ports",
    "dst_port_entropy",
    "src_ip_entropy",
    "syn_flag_rate",
    "mean_packet_length",
    "short_flow_fraction",
    "unseen_src_60m_fraction",
    "anom_mean",
    "anom_p95",
    "anom_frac",
]

ANOMALY_BASE_FEATURES = ["anom_mean", "anom_p95", "anom_frac"]

FORBIDDEN_EXACT_OR_SUBSTRINGS = (
    "label",
    "is_attack",
    "attack_active",
    "attack_flow_starts",
    "onset",
    "oracle_",
    "hour",
    "dayofweek",
    "day_of_week",
    "weekday",
    "minute_ts",
    "timestamp",
    "is_eligible",
)


def _shannon_entropy_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts[counts > 0] / float(total)
    return float(-np.sum(probs * np.log2(probs)))


def _rolling_ols_slope(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute causal right-aligned linear slope over the last ``window`` points."""
    n = len(arr)
    out = np.zeros(n, dtype=np.float64)
    if window <= 1 or n == 0:
        return out
    x = np.arange(window, dtype=np.float64)
    x_mean = x.mean()
    denom = np.sum((x - x_mean) ** 2)
    x_centered = x - x_mean
    for idx in range(n):
        start = max(0, idx - window + 1)
        seg = arr[start : idx + 1]
        m = len(seg)
        if m <= 1:
            out[idx] = 0.0
        elif m == window:
            out[idx] = float(np.dot(x_centered, seg - seg.mean()) / denom)
        else:
            xm = np.arange(m, dtype=np.float64)
            xm -= xm.mean()
            d = np.sum(xm ** 2)
            out[idx] = float(np.dot(xm, seg - seg.mean()) / d) if d > 0 else 0.0
    return out


def _rolling_stats_for_series(values: np.ndarray, windows: list[int], prefix: str) -> dict[str, np.ndarray]:
    """Compute strictly causal rolling mean, std, max, slope, burstiness, and ratio_to_mean."""
    n = len(values)
    out: dict[str, np.ndarray] = {}
    csum = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    csum2 = np.concatenate(([0.0], np.cumsum(values * values, dtype=np.float64)))

    for w in windows:
        means = np.zeros(n, dtype=np.float64)
        stds = np.zeros(n, dtype=np.float64)
        maxs = np.zeros(n, dtype=np.float64)
        for idx in range(n):
            s = max(0, idx - w + 1)
            cnt = idx - s + 1
            mean_val = (csum[idx + 1] - csum[s]) / cnt
            var_val = max(0.0, (csum2[idx + 1] - csum2[s]) / cnt - mean_val * mean_val)
            means[idx] = mean_val
            stds[idx] = np.sqrt(var_val)
            maxs[idx] = float(np.max(values[s : idx + 1]))

        slopes = _rolling_ols_slope(values, w)
        burstiness = stds / (np.abs(means) + 1e-6)
        ratio = values / (means + 1e-6)

        out[f"{prefix}_roll{w}_mean"] = means
        out[f"{prefix}_roll{w}_std"] = stds
        out[f"{prefix}_roll{w}_max"] = maxs
        out[f"{prefix}_roll{w}_slope"] = slopes
        out[f"{prefix}_roll{w}_burst"] = burstiness
        out[f"{prefix}_roll{w}_ratio"] = ratio
    return out


def compute_deployable_feature_names(feature_windows: list[int]) -> list[str]:
    """Return exact ordered list of deployable past-only feature column names."""
    cols = list(BASE_MINUTE_FEATURES)
    for base in BASE_MINUTE_FEATURES:
        for w in feature_windows:
            for stat in ("mean", "std", "max", "slope", "burst", "ratio"):
                cols.append(f"{base}_roll{w}_{stat}")
    return cols


def compute_anomaly_only_feature_names(feature_windows: list[int]) -> list[str]:
    """Return subset of deployable features using only Isolation Forest anomaly signals (for B2)."""
    cols = list(ANOMALY_BASE_FEATURES)
    for base in ANOMALY_BASE_FEATURES:
        for w in feature_windows:
            for stat in ("mean", "std", "max", "slope", "burst", "ratio"):
                cols.append(f"{base}_roll{w}_{stat}")
    return cols


def assert_no_forbidden_features(feature_names: list[str]) -> None:
    """Guardrail asserting zero label, timestamp, clock, oracle, or future columns enter deployable features."""
    for name in feature_names:
        lower = name.lower()
        if lower.startswith("y_") or lower == "day":
            raise AssertionError(f"Forbidden feature name detected in deployable features: {name}")
        for bad in FORBIDDEN_EXACT_OR_SUBSTRINGS:
            if bad in lower:
                raise AssertionError(f"Forbidden feature name detected in deployable features: {name}")


def _aggregate_minute_flows_df(
    flows_df: pl.DataFrame,
    anom_df: Optional[pl.DataFrame],
    bin_seconds: int = 60,
    segment_break_min: int = 60,
    unseen_lookback_min: int = 60,
) -> pl.DataFrame:
    """Aggregate flow records into causal per-minute features keyed by flow END time.

    Each flow contributes ONLY to ``avail_minute = floor(end_ts, bin_seconds)``.
    Empty minutes within a contiguous segment are filled with zeros.
    """
    if flows_df.height == 0:
        return pl.DataFrame()

    bin_us = int(bin_seconds) * 1_000_000
    # Select standardized column aliases safely (supports tiny synthetic test DataFrames)
    cols = set(flows_df.columns)
    fwd_pkt_col = "Total Fwd Packet" if "Total Fwd Packet" in cols else None
    bwd_pkt_col = "Total Bwd packets" if "Total Bwd packets" in cols else None
    fwd_len_col = "Total Length of Fwd Packet" if "Total Length of Fwd Packet" in cols else None
    bwd_len_col = "Total Length of Bwd Packet" if "Total Length of Bwd Packet" in cols else None
    syn_col = "SYN Flag Count" if "SYN Flag Count" in cols else None
    pkt_len_col = "Packet Length Mean" if "Packet Length Mean" in cols else None
    dur_col = "Flow Duration" if "Flow Duration" in cols else None

    end_us = flows_df["end_ts"].cast(pl.Datetime("us", "UTC")).to_numpy().astype("datetime64[us]").astype(np.int64)
    avail_us = (end_us // bin_us) * bin_us

    # Attach anomaly scores aligned by row order or timestamp
    if anom_df is not None and anom_df.height == flows_df.height:
        anom_scores = anom_df["anomaly_score"].to_numpy().astype(np.float64)
        anom_flags = anom_df["is_anomalous"].to_numpy().astype(np.float64)
    elif anom_df is not None and anom_df.height > 0:
        joined = flows_df.select(["start_ts", "end_ts"]).with_row_index("_idx").join(
            anom_df.select(["start_ts", "end_ts", "anomaly_score", "is_anomalous"]).unique(
                subset=["start_ts", "end_ts"], keep="first"
            ),
            on=["start_ts", "end_ts"],
            how="left",
        ).sort("_idx")
        anom_scores = joined["anomaly_score"].fill_null(0.0).to_numpy().astype(np.float64)
        anom_flags = joined["is_anomalous"].fill_null(0).to_numpy().astype(np.float64)
    else:
        anom_scores = np.zeros(flows_df.height, dtype=np.float64)
        anom_flags = np.zeros(flows_df.height, dtype=np.float64)

    # Extract numpy vectors for single-pass grouping by avail_us
    src_ips = flows_df["Src IP"].to_list() if "Src IP" in cols else ["0.0.0.0"] * flows_df.height
    dst_ips = flows_df["Dst IP"].to_list() if "Dst IP" in cols else ["0.0.0.0"] * flows_df.height
    dst_ports = flows_df["Dst Port"].to_numpy() if "Dst Port" in cols else np.zeros(flows_df.height, dtype=np.int64)
    durations = (
        flows_df[dur_col].cast(pl.Float64).clip(lower_bound=0.0).to_numpy()
        if dur_col
        else np.zeros(flows_df.height, dtype=np.float64)
    )
    pkts = (
        (flows_df[fwd_pkt_col].cast(pl.Float64).fill_null(0.0) if fwd_pkt_col else pl.lit(0.0))
        + (flows_df[bwd_pkt_col].cast(pl.Float64).fill_null(0.0) if bwd_pkt_col else pl.lit(0.0))
    )
    pkt_arr = flows_df.select(pkts.alias("_p"))["_p"].to_numpy() if (fwd_pkt_col or bwd_pkt_col) else np.zeros(flows_df.height)

    byt = (
        (flows_df[fwd_len_col].cast(pl.Float64).fill_null(0.0) if fwd_len_col else pl.lit(0.0))
        + (flows_df[bwd_len_col].cast(pl.Float64).fill_null(0.0) if bwd_len_col else pl.lit(0.0))
    )
    byt_arr = flows_df.select(byt.alias("_b"))["_b"].to_numpy() if (fwd_len_col or bwd_len_col) else np.zeros(flows_df.height)

    syn_arr = (
        (flows_df[syn_col].cast(pl.Float64).fill_null(0.0) > 0).to_numpy().astype(np.float64)
        if syn_col
        else np.zeros(flows_df.height, dtype=np.float64)
    )
    plen_arr = (
        flows_df[pkt_len_col].cast(pl.Float64).fill_null(0.0).to_numpy()
        if pkt_len_col
        else np.zeros(flows_df.height, dtype=np.float64)
    )

    # Sort indices by avail_us for fast contiguous bin slicing
    order = np.argsort(avail_us, kind="mergesort")
    sorted_avail = avail_us[order]
    uniq_bins, start_pos, bin_counts = np.unique(sorted_avail, return_index=True, return_counts=True)

    # Determine contiguous capture segments causally from observed availability bins
    break_us = int(segment_break_min) * 60 * 1_000_000
    diffs = np.diff(uniq_bins)
    break_idx = np.where(diffs > break_us)[0]
    seg_starts = np.concatenate(([0], break_idx + 1))
    seg_ends = np.concatenate((break_idx, [len(uniq_bins) - 1]))

    bin_lookup = {int(b): (int(s), int(c)) for b, s, c in zip(uniq_bins, start_pos, bin_counts)}

    records: list[dict[str, Any]] = []
    lookback_us = int(unseen_lookback_min) * 60 * 1_000_000

    for seg_id, (si, ei) in enumerate(zip(seg_starts, seg_ends)):
        t0 = int(uniq_bins[si])
        t1 = int(uniq_bins[ei])
        grid = np.arange(t0, t1 + bin_us, bin_us, dtype=np.int64)
        last_seen_src: dict[str, int] = {}

        for min_in_seg, m_us in enumerate(grid):
            m_int = int(m_us)
            entry = bin_lookup.get(m_int)
            if entry is None:
                records.append(
                    {
                        "minute_ts": m_int,
                        "segment_id": seg_id,
                        "minute_in_segment": min_in_seg,
                        "flow_count": 0.0,
                        "packets_per_sec": 0.0,
                        "bytes_per_sec": 0.0,
                        "mean_flow_duration_us": 0.0,
                        "median_flow_duration_us": 0.0,
                        "unique_src_ips": 0.0,
                        "unique_dst_ips": 0.0,
                        "unique_dst_ports": 0.0,
                        "dst_port_entropy": 0.0,
                        "src_ip_entropy": 0.0,
                        "syn_flag_rate": 0.0,
                        "mean_packet_length": 0.0,
                        "short_flow_fraction": 0.0,
                        "unseen_src_60m_fraction": 0.0,
                        "anom_mean": 0.0,
                        "anom_p95": 0.0,
                        "anom_frac": 0.0,
                    }
                )
                continue

            pos, cnt = entry
            idxs = order[pos : pos + cnt]
            dur_m = durations[idxs]
            pkt_m = pkt_arr[idxs]
            byt_m = byt_arr[idxs]
            syn_m = syn_arr[idxs]
            plen_m = plen_arr[idxs]
            dport_m = dst_ports[idxs]
            anom_m = anom_scores[idxs]
            aflg_m = anom_flags[idxs]

            srcs_m = [src_ips[k] for k in idxs]
            dsts_m = [dst_ips[k] for k in idxs]

            _, dport_cnts = np.unique(dport_m, return_counts=True)
            _, src_cnts = np.unique(srcs_m, return_counts=True)
            n_uniq_dst = len(set(dsts_m))

            # Unseen source fraction relative to [m - 60 min, m) within current segment
            unseen_cnt = 0
            cutoff_us = m_int - lookback_us
            for s_ip in srcs_m:
                prev_seen = last_seen_src.get(s_ip)
                if prev_seen is None or prev_seen < cutoff_us:
                    unseen_cnt += 1
            for s_ip in set(srcs_m):
                last_seen_src[s_ip] = m_int

            records.append(
                {
                    "minute_ts": m_int,
                    "segment_id": seg_id,
                    "minute_in_segment": min_in_seg,
                    "flow_count": float(cnt),
                    "packets_per_sec": float(pkt_m.sum() / bin_seconds),
                    "bytes_per_sec": float(byt_m.sum() / bin_seconds),
                    "mean_flow_duration_us": float(np.mean(dur_m)),
                    "median_flow_duration_us": float(np.median(dur_m)),
                    "unique_src_ips": float(len(src_cnts)),
                    "unique_dst_ips": float(n_uniq_dst),
                    "unique_dst_ports": float(len(dport_cnts)),
                    "dst_port_entropy": _shannon_entropy_from_counts(dport_cnts),
                    "src_ip_entropy": _shannon_entropy_from_counts(src_cnts),
                    "syn_flag_rate": float(np.mean(syn_m)),
                    "mean_packet_length": float(np.mean(plen_m)),
                    "short_flow_fraction": float(np.mean(dur_m <= 100_000.0)),
                    "unseen_src_60m_fraction": float(unseen_cnt / cnt),
                    "anom_mean": float(np.mean(anom_m)),
                    "anom_p95": float(np.percentile(anom_m, 95.0)),
                    "anom_frac": float(np.mean(aflg_m)),
                }
            )

    df_out = pl.DataFrame(records).with_columns(
        pl.col("minute_ts").cast(pl.Datetime("us", "UTC"))
    )
    return df_out


def add_rolling_and_targets(
    minute_df: pl.DataFrame,
    attack_starts_us: np.ndarray,
    cfg: Settings,
) -> pl.DataFrame:
    """Compute segment-isolated causal rolling features, attack_active, onset, targets y_H, and B1 oracle features."""
    if minute_df.height == 0:
        return minute_df

    bin_us = int(cfg.bin_seconds) * 1_000_000
    atk_bins = (attack_starts_us // bin_us) * bin_us if len(attack_starts_us) else np.empty(0, dtype=np.int64)
    uniq_atk, atk_cnts = np.unique(atk_bins, return_counts=True)
    atk_map = dict(zip(uniq_atk.tolist(), atk_cnts.tolist()))

    seg_frames: list[pl.DataFrame] = []
    warmup = max(cfg.feature_windows)
    max_h = max(cfg.horizons)

    for seg_id in minute_df["segment_id"].unique(maintain_order=True).to_list():
        seg_df = minute_df.filter(pl.col("segment_id") == seg_id).sort("minute_ts")
        n = seg_df.height
        m_us = seg_df["minute_ts"].to_numpy().astype("datetime64[us]").astype(np.int64)

        # Ground-truth attack_active & onset from flow START times
        atk_flow_starts = np.array([atk_map.get(int(m), 0) for m in m_us], dtype=np.int32)
        atk_active = (atk_flow_starts >= cfg.k_min).astype(np.int8)
        onsets = np.zeros(n, dtype=np.int8)
        quiet_run = 0
        for idx in range(n):
            if atk_active[idx] == 1:
                if quiet_run >= cfg.quiet_gap_min:
                    onsets[idx] = 1
                quiet_run = 0
            else:
                quiet_run += 1

        new_cols: dict[str, Any] = {
            "attack_flow_starts": pl.Series("attack_flow_starts", atk_flow_starts, dtype=pl.Int32),
            "attack_active": pl.Series("attack_active", atk_active, dtype=pl.Int8),
            "onset": pl.Series("onset", onsets, dtype=pl.Int8),
        }

        # Causal rolling features strictly within this segment
        for base in BASE_MINUTE_FEATURES:
            vals = seg_df[base].to_numpy().astype(np.float64)
            rolled = _rolling_stats_for_series(vals, cfg.feature_windows, base)
            for k, arr in rolled.items():
                new_cols[k] = pl.Series(k, arr, dtype=pl.Float64)

        # Future targets y_H(t) = 1 if any onset occurs in t+1 .. t+H in the SAME segment
        for h in cfg.horizons:
            y_h = np.zeros(n, dtype=np.int8)
            for idx in range(n):
                end_look = min(n, idx + h + 1)
                if idx + 1 < end_look and np.any(onsets[idx + 1 : end_look] == 1):
                    y_h[idx] = 1
            new_cols[f"y_{h}"] = pl.Series(f"y_{h}", y_h, dtype=pl.Int8)

        # B1 ORACLE persistence features (past TRUE attack_active labels up to t — clearly labelled non-deployable)
        atk_float = atk_active.astype(np.float64)
        for w in cfg.feature_windows:
            frac = np.zeros(n, dtype=np.float64)
            for idx in range(n):
                s = max(0, idx - w + 1)
                frac[idx] = float(np.mean(atk_float[s : idx + 1]))
            new_cols[f"oracle_active_frac_{w}"] = pl.Series(f"oracle_active_frac_{w}", frac, dtype=pl.Float64)

        mins_since = np.full(n, 9999.0, dtype=np.float64)
        last_atk_idx: Optional[int] = None
        for idx in range(n):
            if atk_active[idx] == 1:
                last_atk_idx = idx
                mins_since[idx] = 0.0
            elif last_atk_idx is not None:
                mins_since[idx] = float(idx - last_atk_idx)
        new_cols["oracle_minutes_since_attack"] = pl.Series("oracle_minutes_since_attack", mins_since, dtype=pl.Float64)

        # Eligibility mask: exclude attack-active minutes, warm-up < max(FEATURE_WINDOWS), and last max_h censored tail
        eligible = np.ones(n, dtype=np.int8)
        eligible[atk_active == 1] = 0
        eligible[: min(warmup, n)] = 0
        if max_h > 0 and n > max_h:
            eligible[n - max_h :] = 0
        else:
            eligible[:] = 0
        new_cols["is_eligible"] = pl.Series("is_eligible", eligible, dtype=pl.Int8)

        seg_frames.append(seg_df.with_columns(list(new_cols.values())))

    full_df = pl.concat(seg_frames)
    deploy_cols = compute_deployable_feature_names(cfg.feature_windows)
    assert_no_forbidden_features(deploy_cols)
    return full_df


def build_forecast_dataset_from_frames(
    flows_df: pl.DataFrame,
    anom_df: Optional[pl.DataFrame],
    cfg: Settings,
) -> pl.DataFrame:
    """In-memory builder used directly by leakage-equivalence and unit tests."""
    minute_df = _aggregate_minute_flows_df(
        flows_df,
        anom_df,
        bin_seconds=cfg.bin_seconds,
        segment_break_min=cfg.segment_break_min,
    )
    atk_starts = (
        flows_df.filter(pl.col("is_attack") == 1)["start_ts"]
        .cast(pl.Datetime("us", "UTC"))
        .to_numpy()
        .astype("datetime64[us]")
        .astype(np.int64)
    )
    return add_rolling_and_targets(minute_df, atk_starts, cfg)


def build_forecast_dataset(cfg: Optional[Settings] = None) -> pl.DataFrame:
    """Stream processed daily Parquets and Stage 2 anomaly scores to build the full Stage 3 minute table."""
    settings = cfg or get_settings()
    settings.ensure_dirs()
    parquets = get_processed_parquet_paths(settings)

    needed_flow_cols = [
        "day",
        "start_ts",
        "end_ts",
        "Src IP",
        "Dst IP",
        "Dst Port",
        "Flow Duration",
        "Total Fwd Packet",
        "Total Bwd packets",
        "Total Length of Fwd Packet",
        "Total Length of Bwd Packet",
        "SYN Flag Count",
        "Packet Length Mean",
        "is_attack",
    ]

    minute_parts: list[pl.DataFrame] = []
    atk_starts_list: list[np.ndarray] = []

    seg_offset = 0
    for pq in parquets:
        day_df = pl.read_parquet(pq, columns=needed_flow_cols)
        day_name = str(day_df["day"][0]) if day_df.height else pq.stem
        anom_path = settings.data_processed_dir / f"anomaly_scores_{day_name}.parquet"
        anom_df = pl.read_parquet(anom_path) if anom_path.exists() else None

        logger.info("Building causal minute aggregates for %s (%d flows)...", day_name, day_df.height)
        m_df = _aggregate_minute_flows_df(
            day_df,
            anom_df,
            bin_seconds=settings.bin_seconds,
            segment_break_min=settings.segment_break_min,
        )
        if m_df.height > 0:
            m_df = m_df.with_columns((pl.col("segment_id") + seg_offset).alias("segment_id"))
            seg_offset = int(m_df["segment_id"].max()) + 1
            minute_parts.append(m_df)

        atk_arr = (
            day_df.filter(pl.col("is_attack") == 1)["start_ts"]
            .cast(pl.Datetime("us", "UTC"))
            .to_numpy()
            .astype("datetime64[us]")
            .astype(np.int64)
        )
        if len(atk_arr):
            atk_starts_list.append(atk_arr)

        del day_df, anom_df
        gc.collect()

    all_minutes = pl.concat(minute_parts).sort("minute_ts")
    all_atk_starts = np.concatenate(atk_starts_list) if atk_starts_list else np.empty(0, dtype=np.int64)
    forecast_df = add_rolling_and_targets(all_minutes, all_atk_starts, settings)

    out_path = settings.data_processed_dir / "forecast_features.parquet"
    forecast_df.write_parquet(out_path, compression="zstd")
    return forecast_df
