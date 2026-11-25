"""Stage 1B: Data audit, cleaning, Parquet export, and minute-grid onset counting."""
from __future__ import annotations

import csv
import gc
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import polars as pl

from config.settings import Settings, get_settings
from src.data_loader import (
    DAY_ORDER,
    EXCLUDED_METADATA_AND_LEAKAGE_COLS,
    NON_NEGATIVE_PREFIXES,
    SENTINEL_MINUS_ONE_COLS,
    get_feature_columns,
    get_processed_parquet_paths,
    get_raw_csv_paths,
    infer_day_name,
    normalize_label_text,
    parse_timestamp_column,
)

logger = logging.getLogger(__name__)

# Published UNB / CNS2022 CICIDS2017 attack schedule reference (UTC windows from CNS 2022 documentation)
PUBLISHED_SCHEDULE_UTC = {
    "monday": [],
    "tuesday": [
        ("FTP-Patator", "2017-07-04 12:17:52", "2017-07-04 13:20:16", "09:17-10:20 ADT"),
        ("SSH-Patator", "2017-07-04 17:09:01", "2017-07-04 19:04:19", "14:09-16:04 ADT"),
    ],
    "wednesday": [
        ("DoS Slowloris", "2017-07-05 12:47:00", "2017-07-05 13:10:00", "09:47-10:10 ADT"),
        ("DoS Slowhttptest", "2017-07-05 13:14:00", "2017-07-05 13:35:00", "10:14-10:35 ADT"),
        ("DoS Hulk", "2017-07-05 13:43:00", "2017-07-05 14:07:00", "10:43-11:07 ADT"),
        ("DoS GoldenEye", "2017-07-05 14:10:00", "2017-07-05 14:24:00", "11:10-11:24 ADT"),
        ("Heartbleed", "2017-07-05 18:12:00", "2017-07-05 18:32:00", "15:12-15:32 ADT"),
    ],
    "thursday": [
        ("Web Attack - Brute Force", "2017-07-06 12:15:00", "2017-07-06 13:00:00", "09:15-10:00 ADT"),
        ("Web Attack - XSS", "2017-07-06 13:15:00", "2017-07-06 13:35:00", "10:15-10:35 ADT"),
        ("Web Attack - SQL Injection", "2017-07-06 13:40:00", "2017-07-06 13:42:00", "10:40-10:42 ADT"),
        ("Infiltration / Portscan", "2017-07-06 17:19:00", "2017-07-06 18:45:00", "14:19-15:45 ADT"),
    ],
    "friday": [
        ("Botnet", "2017-07-07 13:02:00", "2017-07-07 14:02:00", "10:02-11:02 ADT"),
        ("Portscan", "2017-07-07 16:55:00", "2017-07-07 18:24:00", "13:55-15:24 ADT"),
        ("DDoS", "2017-07-07 18:56:00", "2017-07-07 19:16:00", "15:56-16:16 ADT"),
    ],
}


@dataclass
class DayAuditSummary:
    day: str
    csv_name: str
    encoding: str
    raw_rows: int
    duplicates_removed: int
    clean_rows: int
    ts_min: str
    ts_max: str
    timestamp_info: dict[str, Any]
    duration_unit_verified: bool
    duration_rate_ratio_median: Optional[float]
    negative_duration_rows: int
    zero_duration_rows: int
    max_duration_us: int
    attempted_relabelled_rows: int
    dash_normalized_rows: int
    raw_label_counts: dict[str, int]
    clean_label_counts: dict[str, int]
    attack_windows: list[dict[str, Any]]
    nan_null_columns: dict[str, int]
    pos_inf_columns: dict[str, int]
    neg_inf_columns: dict[str, int]
    negative_value_columns: dict[str, int]
    parquet_path: str


def detect_csv_encoding(csv_path: Path) -> str:
    """Detect whether a CSV decodes cleanly as utf-8-sig / utf-8 or needs latin-1."""
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with csv_path.open("r", encoding=enc) as fh:
                for _ in range(5000):
                    line = fh.readline()
                    if not line:
                        break
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def build_minute_attack_grid(
    attack_start_ts: np.ndarray,
    all_start_ts: np.ndarray,
    all_end_ts: np.ndarray,
    bin_seconds: int = 60,
    k_min: int = 5,
    quiet_gap_min: int = 5,
    segment_break_min: int = 60,
) -> pl.DataFrame:
    """Build complete contiguous minute grid per capture segment and detect attack onsets.

    - A minute ``m`` is ``attack_active`` if ``attack_flow_starts >= k_min``.
    - A new segment starts whenever consecutive observed flow activity has a gap > ``segment_break_min``.
    - ``onset = 1`` at minute ``m`` iff ``attack_active == 1`` and the preceding
      ``quiet_gap_min`` minutes in the SAME segment all had ``attack_active == 0``.
      (The very first minutes of a segment cannot have ``quiet_gap_min`` prior quiet minutes,
      unless enough quiet minutes precede them.)
    """
    if len(all_start_ts) == 0:
        return pl.DataFrame(
            schema={
                "minute_ts": pl.Datetime("us", "UTC"),
                "segment_id": pl.Int32,
                "minute_in_segment": pl.Int32,
                "attack_flow_starts": pl.Int32,
                "attack_active": pl.Int8,
                "onset": pl.Int8,
            }
        )

    bin_us = int(bin_seconds) * 1_000_000
    # Convert datetime64[us] arrays to int64 microseconds since epoch
    starts_us = all_start_ts.astype("datetime64[us]").astype(np.int64)
    ends_us = all_end_ts.astype("datetime64[us]").astype(np.int64)
    atk_us = attack_start_ts.astype("datetime64[us]").astype(np.int64) if len(attack_start_ts) else np.empty(0, dtype=np.int64)

    flow_bins = (starts_us // bin_us) * bin_us
    end_bins = (ends_us // bin_us) * bin_us
    observed_bins = np.unique(np.concatenate([flow_bins, end_bins]))
    observed_bins.sort()

    atk_bins = (atk_us // bin_us) * bin_us
    uniq_atk_bins, atk_counts = np.unique(atk_bins, return_counts=True)
    atk_count_map = dict(zip(uniq_atk_bins.tolist(), atk_counts.tolist()))

    # Segment boundaries when gap between consecutive active flow bins exceeds segment_break_min
    break_us = int(segment_break_min) * 60 * 1_000_000
    diffs = np.diff(observed_bins)
    break_indices = np.where(diffs > break_us)[0]
    seg_starts = np.concatenate(([0], break_indices + 1))
    seg_ends = np.concatenate((break_indices, [len(observed_bins) - 1]))

    minute_ts_list: list[int] = []
    seg_id_list: list[int] = []
    min_in_seg_list: list[int] = []
    atk_starts_list: list[int] = []
    atk_active_list: list[int] = []
    onset_list: list[int] = []

    for seg_idx, (s_i, e_i) in enumerate(zip(seg_starts, seg_ends)):
        t_start = int(observed_bins[s_i])
        t_end = int(observed_bins[e_i])
        grid = np.arange(t_start, t_end + bin_us, bin_us, dtype=np.int64)
        n = len(grid)
        counts = np.array([atk_count_map.get(int(m), 0) for m in grid], dtype=np.int32)
        active = (counts >= k_min).astype(np.int8)
        onsets = np.zeros(n, dtype=np.int8)

        quiet_run = 0
        for idx in range(n):
            if active[idx] == 1:
                if quiet_run >= quiet_gap_min:
                    onsets[idx] = 1
                quiet_run = 0
            else:
                quiet_run += 1

        minute_ts_list.extend(grid.tolist())
        seg_id_list.extend([seg_idx] * n)
        min_in_seg_list.extend(list(range(n)))
        atk_starts_list.extend(counts.tolist())
        atk_active_list.extend(active.tolist())
        onset_list.extend(onsets.tolist())

    return pl.DataFrame(
        {
            "minute_ts": pl.Series("minute_ts", minute_ts_list, dtype=pl.Int64).cast(pl.Datetime("us", "UTC")),
            "segment_id": pl.Series("segment_id", seg_id_list, dtype=pl.Int32),
            "minute_in_segment": pl.Series("minute_in_segment", min_in_seg_list, dtype=pl.Int32),
            "attack_flow_starts": pl.Series("attack_flow_starts", atk_starts_list, dtype=pl.Int32),
            "attack_active": pl.Series("attack_active", atk_active_list, dtype=pl.Int8),
            "onset": pl.Series("onset", onset_list, dtype=pl.Int8),
        }
    )


def _audit_single_csv(csv_path: Path, out_parquet: Path, cfg: Settings) -> DayAuditSummary:
    """Read one daily CSV, audit anomalies, clean/deduplicate, sort chronologically, and save Parquet."""
    day = infer_day_name(csv_path)
    encoding = detect_csv_encoding(csv_path)
    logger.info("Auditing %s (day=%s, encoding=%s)...", csv_path.name, day, encoding)

    # Read header to normalize whitespace in column names
    with csv_path.open("r", encoding=encoding, newline="") as fh:
        raw_cols = next(csv.reader(fh))
    clean_cols = [c.strip() for c in raw_cols]

    polars_enc = "utf8" if encoding.startswith("utf-8") else "latin1"
    df = pl.read_csv(
        csv_path,
        encoding=polars_enc,
        has_header=True,
        new_columns=clean_cols,
        infer_schema_length=5000,
        null_values=["", "NA", "NaN", "nan", "null", "NULL"],
    )
    raw_rows = df.height

    # Strip string columns of accidental whitespace
    str_cols = [name for name, dtype in df.schema.items() if dtype == pl.String]
    if str_cols:
        df = df.with_columns([pl.col(c).str.strip_chars().alias(c) for c in str_cols])

    # Timestamp parsing (supports 24h UTC and legacy 12h AM/PM disambiguation)
    raw_ts_list = df["Timestamp"].to_list()
    normalized_ts, ts_info = parse_timestamp_column(raw_ts_list)
    df = df.with_columns(
        pl.Series("_norm_ts", normalized_ts, dtype=pl.String)
        .str.strptime(pl.Datetime("us"), format="%Y-%m-%d %H:%M:%S%.f", strict=False)
        .dt.replace_time_zone("UTC")
        .alias("start_ts")
    )

    # Duration audit
    dur_col = "Flow Duration"
    neg_dur_rows = int(df.filter(pl.col(dur_col) < 0).height)
    zero_dur_rows = int(df.filter(pl.col(dur_col) == 0).height)
    max_dur_us = int(df[dur_col].max() or 0)

    # Empirical unit verification: compare (Total Fwd + Total Bwd) / (Flow Duration / 1e6) vs Flow Packets/s
    ratio_df = df.filter((pl.col(dur_col) > 0) & (pl.col("Flow Packets/s") > 0)).select(
        (
            ((pl.col("Total Fwd Packet") + pl.col("Total Bwd packets")) / (pl.col(dur_col) / 1_000_000.0))
            / pl.col("Flow Packets/s")
        ).alias("ratio")
    )
    ratio_med = float(ratio_df["ratio"].median()) if ratio_df.height > 0 else None
    unit_ok = ratio_med is not None and 0.98 <= ratio_med <= 1.02

    # Compute end_ts = start_ts + max(Flow Duration, 0) microseconds
    df = df.with_columns(
        [
            pl.lit(day).alias("day"),
            (pl.col("start_ts") + pl.duration(microseconds=pl.col(dur_col).clip(lower_bound=0))).alias("end_ts"),
        ]
    )

    # Label normalization and Attempted relabelling per CNS 2022 documentation
    raw_labels = df["Label"].to_list()
    norm_labels = [normalize_label_text(x) for x in raw_labels]
    dash_normalized_rows = sum(1 for r, n in zip(raw_labels, norm_labels) if (r or "").strip() != n)

    raw_label_counts = {
        str(r["Label"]): int(r["len"])
        for r in df.group_by("Label").len().sort("len", descending=True).to_dicts()
    }

    has_attempted_col = "Attempted Category" in df.columns
    df = df.with_columns(
        [
            pl.Series("raw_label", raw_labels, dtype=pl.String),
            pl.Series("_norm_label", norm_labels, dtype=pl.String),
        ]
    )
    if has_attempted_col:
        attempted_mask = (pl.col("Attempted Category") != -1) | pl.col("_norm_label").str.ends_with("- Attempted")
    else:
        attempted_mask = pl.col("_norm_label").str.ends_with("- Attempted")

    attempted_relabelled_rows = int(df.filter(attempted_mask).height)
    df = df.with_columns(
        pl.when(attempted_mask).then(pl.lit("BENIGN")).otherwise(pl.col("_norm_label")).alias("label")
    ).drop("_norm_label")
    df = df.with_columns((pl.col("label") != "BENIGN").cast(pl.Int8).alias("is_attack"))

    # Audit NaN/Null, +/-inf, and unexpected negative values BEFORE any cleaning
    nan_null_cols: dict[str, int] = {}
    pos_inf_cols: dict[str, int] = {}
    neg_inf_cols: dict[str, int] = {}
    neg_val_cols: dict[str, int] = {}
    for col, dtype in df.schema.items():
        if col in EXCLUDED_METADATA_AND_LEAKAGE_COLS:
            continue
        null_cnt = int(df[col].null_count())
        if dtype in (pl.Float32, pl.Float64):
            nan_cnt = int(df.select(pl.col(col).is_nan().sum()).item() or 0)
            pinf_cnt = int(df.select((pl.col(col) == float("inf")).sum()).item() or 0)
            ninf_cnt = int(df.select((pl.col(col) == float("-inf")).sum()).item() or 0)
            if null_cnt + nan_cnt > 0:
                nan_null_cols[col] = null_cnt + nan_cnt
            if pinf_cnt > 0:
                pos_inf_cols[col] = pinf_cnt
            if ninf_cnt > 0:
                neg_inf_cols[col] = ninf_cnt
        elif null_cnt > 0:
            nan_null_cols[col] = null_cnt

        if dtype.is_numeric() and col not in SENTINEL_MINUS_ONE_COLS:
            is_nonneg_col = col.startswith(NON_NEGATIVE_PREFIXES) or "Duration" in col or "Length" in col or "Count" in col
            if is_nonneg_col:
                neg_cnt = int(df.filter(pl.col(col) < 0).height)
                if neg_cnt > 0:
                    neg_val_cols[col] = neg_cnt

    # Sort chronologically by start_ts, then deduplicate exact flows excluding sequential 'id'
    df = df.sort(["start_ts", "end_ts", "Flow ID"])
    dedup_subset = [c for c in clean_cols if c != "id"]
    df = df.unique(subset=dedup_subset, keep="first", maintain_order=True)
    clean_rows = df.height
    duplicates_removed = raw_rows - clean_rows

    clean_label_counts = {
        str(r["label"]): int(r["len"])
        for r in df.group_by("label").len().sort("len", descending=True).to_dicts()
    }

    # Attack windows per clean attack class
    attack_windows: list[dict[str, Any]] = []
    atk_summary = (
        df.filter(pl.col("is_attack") == 1)
        .group_by("label")
        .agg(
            [
                pl.len().alias("flows"),
                pl.col("start_ts").min().alias("first_start_utc"),
                pl.col("start_ts").max().alias("last_start_utc"),
                pl.col("end_ts").max().alias("last_end_utc"),
            ]
        )
        .sort("first_start_utc")
    )
    for row in atk_summary.to_dicts():
        attack_windows.append(
            {
                "label": row["label"],
                "flows": int(row["flows"]),
                "first_start_utc": str(row["first_start_utc"]),
                "last_start_utc": str(row["last_start_utc"]),
                "last_end_utc": str(row["last_end_utc"]),
            }
        )

    ts_min = str(df["start_ts"].min())
    ts_max = str(df["start_ts"].max())

    # Write cleaned daily Parquet
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_parquet, compression="zstd")

    summary = DayAuditSummary(
        day=day,
        csv_name=csv_path.name,
        encoding=encoding,
        raw_rows=raw_rows,
        duplicates_removed=duplicates_removed,
        clean_rows=clean_rows,
        ts_min=ts_min,
        ts_max=ts_max,
        timestamp_info=ts_info,
        duration_unit_verified=unit_ok,
        duration_rate_ratio_median=ratio_med,
        negative_duration_rows=neg_dur_rows,
        zero_duration_rows=zero_dur_rows,
        max_duration_us=max_dur_us,
        attempted_relabelled_rows=attempted_relabelled_rows,
        dash_normalized_rows=dash_normalized_rows,
        raw_label_counts=raw_label_counts,
        clean_label_counts=clean_label_counts,
        attack_windows=attack_windows,
        nan_null_columns=nan_null_cols,
        pos_inf_columns=pos_inf_cols,
        neg_inf_columns=neg_inf_cols,
        negative_value_columns=neg_val_cols,
        parquet_path=str(out_parquet),
    )

    del df
    gc.collect()
    return summary


def compute_onset_summary(cfg: Settings) -> pl.DataFrame:
    """Compute Stage 3 attack-onset counts across all processed days using a global timeline."""
    parquets = get_processed_parquet_paths(cfg)
    lf = pl.scan_parquet([str(p) for p in parquets]).select(["day", "start_ts", "end_ts", "is_attack"])
    df = lf.collect()

    all_starts = df["start_ts"].to_numpy()
    all_ends = df["end_ts"].to_numpy()
    atk_starts = df.filter(pl.col("is_attack") == 1)["start_ts"].to_numpy()

    grid = build_minute_attack_grid(
        attack_start_ts=atk_starts,
        all_start_ts=all_starts,
        all_end_ts=all_ends,
        bin_seconds=cfg.bin_seconds,
        k_min=cfg.k_min,
        quiet_gap_min=cfg.quiet_gap_min,
        segment_break_min=cfg.segment_break_min,
    )
    # Assign UTC date / weekday day label
    day_lookup = {
        "2017-07-03": "monday",
        "2017-07-04": "tuesday",
        "2017-07-05": "wednesday",
        "2017-07-06": "thursday",
        "2017-07-07": "friday",
    }
    grid = grid.with_columns(pl.col("minute_ts").dt.strftime("%Y-%m-%d").alias("utc_date"))
    grid = grid.with_columns(
        pl.col("utc_date").replace_strict(day_lookup, default=pl.col("utc_date")).alias("day")
    )

    per_day = (
        grid.group_by(["day", "utc_date"])
        .agg(
            [
                pl.len().alias("total_minutes"),
                pl.col("segment_id").n_unique().alias("segments"),
                pl.col("attack_active").sum().alias("attack_active_minutes"),
                pl.col("onset").sum().alias("onset_events"),
                pl.col("attack_flow_starts").sum().alias("attack_flow_starts"),
            ]
        )
    )
    order_map = {d: i for i, d in enumerate(DAY_ORDER)}
    rows = sorted(per_day.to_dicts(), key=lambda r: (order_map.get(r["day"], 99), r["utc_date"]))
    out_df = pl.DataFrame(rows)
    out_df.write_csv(cfg.results_dir / "onset_counts.csv")
    grid.write_parquet(cfg.data_processed_dir / "minute_attack_grid.parquet", compression="zstd")
    return out_df


def _write_audit_markdown(
    summaries: list[DayAuditSummary],
    onset_df: pl.DataFrame,
    feature_cols_with_port: list[str],
    feature_cols_no_port: list[str],
    cfg: Settings,
) -> Path:
    """Write comprehensive ``results/data_audit.md`` strictly from observed audit summaries."""
    manifest = {}
    if cfg.manifest_path.exists():
        manifest = json.loads(cfg.manifest_path.read_text(encoding="utf-8"))

    total_raw = sum(s.raw_rows for s in summaries)
    total_clean = sum(s.clean_rows for s in summaries)
    total_dups = sum(s.duplicates_removed for s in summaries)
    total_attempted = sum(s.attempted_relabelled_rows for s in summaries)
    total_onsets = int(onset_df["onset_events"].sum()) if onset_df.height > 0 else 0

    lines: list[str] = [
        "# Stage 1B Data Audit Report",
        "",
        "## Dataset Provenance & Version",
        f"- **Dataset Version**: {manifest.get('dataset_version', 'unknown')}",
        f"- **Source Name**: {manifest.get('source_name', 'unknown')}",
        f"- **Source URL**: {manifest.get('source_url', 'unknown')}",
        f"- **Synthetic Mode**: {manifest.get('synthetic_mode', False)}",
        f"- **Total Raw Flows**: {total_raw:,} | **Duplicates Removed**: {total_dups:,} | **Clean Flows**: {total_clean:,}",
        f"- **Attempted Flows Relabelled to BENIGN**: {total_attempted:,} (per CNS 2022 author guidance: `Attempted Category != -1` or `*- Attempted` -> `BENIGN`, never kept as separate attack classes)",
        "",
        "## Feature Column Governance & Leakage Prevention",
        f"- **Excluded Columns ({len(EXCLUDED_METADATA_AND_LEAKAGE_COLS)})**: `{sorted(EXCLUDED_METADATA_AND_LEAKAGE_COLS)}`",
        f"- **Classifier Features (with Dst Port)**: {len(feature_cols_with_port)} columns",
        f"- **Classifier Features (without Dst Port, ablation)**: {len(feature_cols_no_port)} columns (`Dst Port` excluded)",
        "",
        "## Attack-Onset Counts Per Day (Stage 3 Definition)",
        f"- Parameters: `BIN_SECONDS={cfg.bin_seconds}`, `K_MIN={cfg.k_min}`, `QUIET_GAP_MIN={cfg.quiet_gap_min}`, `SEGMENT_BREAK_MIN={cfg.segment_break_min}`",
        f"- **Total Onset Events Across Dataset**: **{total_onsets}**",
        "",
        "| Day | UTC Date | Observed Minutes | Segments | Attack-Active Minutes | Onset Events | Attack Flow Starts |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in onset_df.to_dicts():
        lines.append(
            f"| {r['day']} | {r['utc_date']} | {r['total_minutes']:,} | {r['segments']} | "
            f"{r['attack_active_minutes']:,} | {r['onset_events']} | {r['attack_flow_starts']:,} |"
        )

    lines.extend(["", "## Per-Day File Audit & Quality Diagnostics", ""])
    for s in summaries:
        lines.extend(
            [
                f"### {s.day.title()} (`{s.csv_name}`)",
                f"- **Encoding**: `{s.encoding}` | **Raw rows**: {s.raw_rows:,} | **Duplicates removed**: {s.duplicates_removed:,} | **Clean rows**: {s.clean_rows:,}",
                f"- **UTC Range**: `{s.ts_min}` -> `{s.ts_max}`",
                f"- **Timestamp Format & Timezone**: {s.timestamp_info.get('format_detected')} — {s.timestamp_info.get('timezone_assumption')}",
                f"- **AM/PM Inferred Rows**: {s.timestamp_info.get('ampm_inferred_rows', 0)} (monotonic disagreements: {s.timestamp_info.get('ampm_monotonic_disagreements', 0)})",
                f"- **Flow Duration Unit**: microseconds verified={s.duration_unit_verified} (median `(pkts/dur_s) / Flow Packets/s` ratio = {s.duration_rate_ratio_median:.6f})"
                if s.duration_rate_ratio_median is not None
                else f"- **Flow Duration Unit**: microseconds verified={s.duration_unit_verified}",
                f"- **Flow Duration Anomalies**: negative={s.negative_duration_rows}, zero={s.zero_duration_rows}, max={s.max_duration_us:,} us",
                f"- **Dash-Normalized Label Rows**: {s.dash_normalized_rows} | **Attempted -> BENIGN**: {s.attempted_relabelled_rows:,}",
                f"- **NaN / Null Columns**: {s.nan_null_columns if s.nan_null_columns else 'None'}",
                f"- **`+inf` Columns**: {s.pos_inf_columns if s.pos_inf_columns else 'None'}",
                f"- **`-inf` Columns**: {s.neg_inf_columns if s.neg_inf_columns else 'None'}",
                f"- **Unexpected Negative-Value Columns**: {s.negative_value_columns if s.negative_value_columns else 'None'}",
                "",
                "**Clean Label Distribution:**",
                "",
                "| Label | Count |",
                "| --- | ---: |",
            ]
        )
        for lbl, cnt in s.clean_label_counts.items():
            lines.append(f"| `{lbl}` | {cnt:,} |")

        lines.extend(["", "**Observed Attack Windows vs Published Schedule:**", ""])
        if s.attack_windows:
            lines.append("| Clean Attack Label | Flows | Observed First Start (UTC) | Observed Last Start (UTC) |")
            lines.append("| --- | ---: | --- | --- |")
            for aw in s.attack_windows:
                lines.append(
                    f"| `{aw['label']}` | {aw['flows']:,} | `{aw['first_start_utc']}` | `{aw['last_start_utc']}` |"
                )
        else:
            lines.append("- Benign-only baseline day (no attack labels present).")

        sched = PUBLISHED_SCHEDULE_UTC.get(s.day, [])
        if sched:
            lines.append("")
            lines.append("Published reference windows:")
            for name, st, en, adt in sched:
                lines.append(f"  - `{name}`: `{st}` -> `{en}` UTC ({adt})")
        lines.append("")

    lines.extend(
        [
            "## Label-Quality & Methodological Limitations Recorded",
            "- CICIDS2017 is a scripted 5-day lab capture; attack windows occur in concentrated morning/afternoon blocks.",
            "- CNS 2022 fixed CICFlowMeter TCP termination (`FIN`/`RST`), UTC timestamps, and `Attempted` labels, but `Flow Bytes/s` and `Flow Packets/s` can still exhibit `NaN`/`inf` if `Flow Duration == 0` and `-1` sentinels exist in `FWD/Bwd Init Win Bytes` and `ICMP Code/Type`.",
            "- Because onset events are sparse, forecasting statistical power depends directly on the observed onset counts reported above.",
        ]
    )

    out_path = cfg.results_dir / "data_audit.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def run_audit_and_prepare(cfg: Optional[Settings] = None) -> dict[str, Any]:
    """Run Stage 1B audit and export cleaned chronological daily Parquet files."""
    settings = cfg or get_settings()
    settings.ensure_dirs()
    csv_paths = get_raw_csv_paths(settings)

    summaries: list[DayAuditSummary] = []
    for cp in csv_paths:
        day = infer_day_name(cp)
        out_pq = settings.data_processed_dir / f"flows_{day}.parquet"
        summaries.append(_audit_single_csv(cp, out_pq, settings))

    # Inspect schema from first processed Parquet
    sample_schema = pl.scan_parquet(summaries[0].parquet_path).collect_schema().names()
    fcols_with = get_feature_columns(sample_schema, include_dst_port=True)
    fcols_no = get_feature_columns(sample_schema, include_dst_port=False)

    onset_df = compute_onset_summary(settings)
    audit_md_path = _write_audit_markdown(summaries, onset_df, fcols_with, fcols_no, settings)

    # Save machine-readable audit JSON for downstream reporting & dashboard
    audit_json = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_raw_rows": sum(s.raw_rows for s in summaries),
        "total_clean_rows": sum(s.clean_rows for s in summaries),
        "total_duplicates_removed": sum(s.duplicates_removed for s in summaries),
        "total_attempted_relabelled": sum(s.attempted_relabelled_rows for s in summaries),
        "total_onset_events": int(onset_df["onset_events"].sum()) if onset_df.height > 0 else 0,
        "feature_columns_with_dst_port": fcols_with,
        "feature_columns_without_dst_port": fcols_no,
        "onset_by_day": onset_df.to_dicts(),
        "days": [
            {
                "day": s.day,
                "csv_name": s.csv_name,
                "encoding": s.encoding,
                "raw_rows": s.raw_rows,
                "duplicates_removed": s.duplicates_removed,
                "clean_rows": s.clean_rows,
                "ts_min": s.ts_min,
                "ts_max": s.ts_max,
                "negative_duration_rows": s.negative_duration_rows,
                "zero_duration_rows": s.zero_duration_rows,
                "attempted_relabelled_rows": s.attempted_relabelled_rows,
                "clean_label_counts": s.clean_label_counts,
                "attack_windows": s.attack_windows,
                "nan_null_columns": s.nan_null_columns,
                "pos_inf_columns": s.pos_inf_columns,
                "neg_inf_columns": s.neg_inf_columns,
                "negative_value_columns": s.negative_value_columns,
            }
            for s in summaries
        ],
    }
    (settings.results_dir / "data_audit.json").write_text(json.dumps(audit_json, indent=2), encoding="utf-8")

    return {
        "audit_md": str(audit_md_path),
        "onset_counts_csv": str(settings.results_dir / "onset_counts.csv"),
        "total_clean_rows": audit_json["total_clean_rows"],
        "total_onset_events": audit_json["total_onset_events"],
        "onset_by_day": audit_json["onset_by_day"],
    }
