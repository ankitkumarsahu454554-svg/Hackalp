"""Dedicated temporal split tests: global classification chronological embargoes & forecast onset-split invariants."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from src.classification import compute_global_chronological_split_bounds
from src.forecasting import find_chronological_onset_split


def test_global_classification_split_strictly_chronological() -> None:
    base = datetime(2017, 7, 3, 12, 0)
    # Multi-day spans across 5 days
    ts_list = []
    for d in range(5):
        day_base = base + timedelta(days=d)
        ts_list.extend([day_base + timedelta(minutes=m * 5) for m in range(96)])
    arr = np.array(ts_list, dtype="datetime64[us]")
    bounds = compute_global_chronological_split_bounds(arr, embargo_minutes=30)

    assert bounds["t_min"] <= bounds["train_end"]
    assert (bounds["val_start"] - bounds["train_end"]).total_seconds() >= 1800
    assert bounds["val_start"] <= bounds["val_end"]
    assert (bounds["test_start"] - bounds["val_end"]).total_seconds() >= 1800
    assert bounds["test_start"] <= bounds["t_max"]


def test_forecast_split_no_overlap_and_required_gaps() -> None:
    base = datetime(2017, 7, 4, 12, 0, tzinfo=timezone.utc)
    n = 400
    mins = [base + timedelta(minutes=i) for i in range(n)]
    onsets = np.zeros(n, dtype=np.int8)
    y5 = np.zeros(n, dtype=np.int8)
    elig = np.ones(n, dtype=np.int8)
    elig[:30] = 0
    for o in (50, 100, 160, 220, 290, 350):
        onsets[o] = 1
        y5[o - 5 : o] = 1
        elig[o] = 0

    df = pl.DataFrame(
        {
            "minute_ts": pl.Series("minute_ts", mins).cast(pl.Datetime("us", "UTC")),
            "segment_id": pl.Series("segment_id", [0] * n, dtype=pl.Int32),
            "is_eligible": pl.Series("is_eligible", elig, dtype=pl.Int8),
            "onset": pl.Series("onset", onsets, dtype=pl.Int8),
            "y_5": pl.Series("y_5", y5, dtype=pl.Int8),
        }
    )

    res = find_chronological_onset_split(df, horizon_min=5, gap_minutes=35, min_train_onsets=2, min_val_onsets=2, min_test_onsets=2)
    info = res["split_info"]
    assert info["gap1_minutes"] >= 35
    assert info["gap2_minutes"] >= 35
    labels = res["split_labels"]
    tr_ts = np.array(mins)[labels == "train"]
    va_ts = np.array(mins)[labels == "validation"]
    te_ts = np.array(mins)[labels == "test"]
    assert tr_ts.max() < va_ts.min()
    assert va_ts.max() < te_ts.min()
