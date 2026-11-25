"""Unit tests for Stage 1B preprocessing, timestamp parsing, label normalization, and onset logic."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config.settings import get_settings
from src.data_loader import get_feature_columns, normalize_label_text, parse_timestamp_column
from src.preprocessing import build_minute_attack_grid, run_audit_and_prepare


def test_label_dash_normalization() -> None:
    assert normalize_label_text("Web Attack \ufffd XSS") == "Web Attack - XSS"
    assert normalize_label_text("Web Attack  Brute Force") == "Web Attack - Brute Force"
    assert normalize_label_text("FTP\u2013Patator") == "FTP-Patator"
    assert normalize_label_text("benign") == "BENIGN"


def test_timestamp_12h_ampm_disambiguation() -> None:
    raw = [
        "04/07/2017 9:15:00",
        "04/07/2017 12:05:00",
        "04/07/2017 1:30:00",
        "04/07/2017 4:45:00",
    ]
    parsed, info = parse_timestamp_column(raw)
    assert parsed == [
        "2017-07-04 09:15:00.000000",
        "2017-07-04 12:05:00.000000",
        "2017-07-04 13:30:00.000000",
        "2017-07-04 16:45:00.000000",
    ]
    assert info["ampm_inferred_rows"] == 2
    assert info["ampm_monotonic_disagreements"] == 0


def test_feature_column_exclusion() -> None:
    cols = [
        "id",
        "Flow ID",
        "Src IP",
        "Src Port",
        "Dst IP",
        "Dst Port",
        "Timestamp",
        "Flow Duration",
        "Total Fwd Packet",
        "Label",
        "Attempted Category",
    ]
    with_port = get_feature_columns(cols, include_dst_port=True)
    no_port = get_feature_columns(cols, include_dst_port=False)
    assert with_port == ["Dst Port", "Flow Duration", "Total Fwd Packet"]
    assert no_port == ["Flow Duration", "Total Fwd Packet"]


def test_onset_grid_k_min_quiet_gap_and_segments() -> None:
    # Build 20 minutes in segment 0, then a 70-minute break (> 60), then 10 minutes in segment 1
    base = np.datetime64("2017-07-04T12:00:00", "us")
    all_mins = list(range(0, 20)) + list(range(90, 100))
    all_starts = np.array([base + np.timedelta64(m, "m") for m in all_mins], dtype="datetime64[us]")
    all_ends = all_starts + np.timedelta64(1, "s")

    # Minute 6 has 5 attacks (after 6 quiet mins 0..5 >= quiet_gap_min=5 -> ONSET!)
    # Minute 7 has 4 attacks (< k_min=5 -> not active)
    # Minute 10 has 5 attacks (only 3 quiet mins 7,8,9 < 5 -> NOT onset)
    # Minute 17 has 5 attacks (6 quiet mins 11..16 >= 5 -> ONSET!)
    # Minute 92 (segment 1) has 5 attacks (only 2 quiet mins 90,91 in segment 1 -> NOT onset!)
    atk_list = (
        [6] * 5
        + [7] * 4
        + [10] * 5
        + [17] * 5
        + [92] * 5
    )
    atk_starts = np.array([base + np.timedelta64(m, "m") for m in atk_list], dtype="datetime64[us]")

    grid = build_minute_attack_grid(
        attack_start_ts=atk_starts,
        all_start_ts=all_starts,
        all_end_ts=all_ends,
        bin_seconds=60,
        k_min=5,
        quiet_gap_min=5,
        segment_break_min=60,
    )
    assert grid["segment_id"].n_unique() == 2
    onsets = grid.filter(pl_col := __import__("polars").col("onset") == 1)
    assert onsets.height == 2
    onset_mins = onsets["minute_ts"].to_list()
    assert onset_mins == [
        datetime(2017, 7, 4, 12, 6, tzinfo=timezone.utc),
        datetime(2017, 7, 4, 12, 17, tzinfo=timezone.utc),
    ]


def test_audit_end_to_end_on_tiny_csv(tmp_path: Path) -> None:
    cfg = get_settings(project_root=tmp_path, k_min=2, quiet_gap_min=2)
    cfg.ensure_dirs()
    csv_path = cfg.data_raw_dir / "tuesday.csv"
    rows = [
        "id,Flow ID,Src IP,Src Port,Dst IP,Dst Port,Protocol,Timestamp,Flow Duration,Total Fwd Packet,Total Bwd packets,Flow Packets/s,FWD Init Win Bytes,Label,Attempted Category"
    ]
    # 3 quiet minutes (12:00..12:02), then 2 attack rows at 12:03, plus 1 Attempted row & 1 exact dup (different id)
    rows.append("1,f1,1.1.1.1,1000,2.2.2.2,80,6,2017-07-04 12:00:00.000000,1000000,1,1,2.0,-1,BENIGN,-1")
    rows.append("2,f2,1.1.1.1,1001,2.2.2.2,80,6,2017-07-04 12:01:00.000000,1000000,1,1,2.0,-1,BENIGN,-1")
    rows.append("3,f3,1.1.1.1,1002,2.2.2.2,80,6,2017-07-04 12:02:00.000000,1000000,1,1,2.0,-1,FTP-Patator - Attempted,0")
    rows.append("4,f4,1.1.1.2,1003,2.2.2.2,21,6,2017-07-04 12:03:05.000000,1000000,2,2,4.0,1024,FTP-Patator,-1")
    rows.append("5,f5,1.1.1.2,1004,2.2.2.2,21,6,2017-07-04 12:03:10.000000,1000000,2,2,4.0,1024,FTP-Patator,-1")
    # Exact duplicate of row 5 except id=6
    rows.append("6,f5,1.1.1.2,1004,2.2.2.2,21,6,2017-07-04 12:03:10.000000,1000000,2,2,4.0,1024,FTP-Patator,-1")
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    summary = run_audit_and_prepare(cfg)
    assert summary["total_clean_rows"] == 5
    assert summary["total_onset_events"] == 1
    assert (cfg.results_dir / "data_audit.md").exists()
    assert (cfg.results_dir / "onset_counts.csv").exists()
    assert (cfg.data_processed_dir / "flows_tuesday.parquet").exists()
