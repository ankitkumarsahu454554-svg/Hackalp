"""Unit tests for Stage 2 globally chronological classifier splits and strict out-of-time anomaly detection."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

from config.settings import get_settings
from src.anomaly_detection import run_anomaly_detection
from src.classification import compute_global_chronological_split_bounds, run_classification


def _write_synthetic_day_parquet(path: Path, day: str, base_dt: datetime, n_rows: int, mode: str) -> None:
    rng = np.random.default_rng(42 if day == "monday" else 99)
    starts = [base_dt + timedelta(seconds=i * 30) for i in range(n_rows)]
    ends = [s + timedelta(seconds=1) for s in starts]
    f1 = rng.normal(10.0, 1.0, size=n_rows)
    f2 = rng.normal(50.0, 5.0, size=n_rows)
    dst_port = np.full(n_rows, 443, dtype=np.int64)
    labels = ["BENIGN"] * n_rows
    is_atk = np.zeros(n_rows, dtype=np.int8)

    if mode == "train_attacks":
        for idx in range(int(n_rows * 0.2), int(n_rows * 0.4)):
            labels[idx] = "FTP-Patator"
            is_atk[idx] = 1
            f1[idx] += 10.0
            f2[idx] += 20.0
            dst_port[idx] = 21
        for idx in range(int(n_rows * 0.5), int(n_rows * 0.7)):
            labels[idx] = "DoS Hulk"
            is_atk[idx] = 1
            f1[idx] += 15.0
            f2[idx] += 40.0
            dst_port[idx] = 80
    elif mode == "late_attacks":
        for idx in range(int(n_rows * 0.6), int(n_rows * 0.85)):
            labels[idx] = "Portscan"
            is_atk[idx] = 1
            f1[idx] += 25.0
            f2[idx] -= 30.0
            dst_port[idx] = 22

    df = pl.DataFrame(
        {
            "day": [day] * n_rows,
            "start_ts": pl.Series("start_ts", starts).cast(pl.Datetime("us", "UTC")),
            "end_ts": pl.Series("end_ts", ends).cast(pl.Datetime("us", "UTC")),
            "Flow ID": [f"{day}-f-{i}" for i in range(n_rows)],
            "Src IP": ["10.0.0.1"] * n_rows,
            "Src Port": [12345] * n_rows,
            "Dst IP": ["10.0.0.2"] * n_rows,
            "Dst Port": dst_port,
            "Flow Duration": [1_000_000] * n_rows,
            "f1": f1,
            "f2": f2,
            "raw_label": labels,
            "label": labels,
            "is_attack": is_atk,
        }
    )
    df.write_parquet(path)


def test_global_chronological_split_bounds_have_embargo() -> None:
    base = datetime(2017, 7, 3, 12, 0)
    ts = np.array([base + timedelta(minutes=i * 5) for i in range(200)], dtype="datetime64[us]")
    b = compute_global_chronological_split_bounds(ts, embargo_minutes=30)
    assert (b["val_start"] - b["train_end"]).total_seconds() >= 1800
    assert (b["test_start"] - b["val_end"]).total_seconds() >= 1800
    assert b["train_end"] < b["val_start"] <= b["val_end"] < b["test_start"]


def test_stage2_global_classification_and_strict_oot_anomaly(tmp_path: Path) -> None:
    cfg = get_settings(project_root=tmp_path, seeds=[0])
    cfg.ensure_dirs()
    _write_synthetic_day_parquet(
        cfg.data_processed_dir / "flows_monday.parquet",
        "monday",
        datetime(2017, 7, 3, 12, 0, tzinfo=timezone.utc),
        240,
        mode="benign",
    )
    _write_synthetic_day_parquet(
        cfg.data_processed_dir / "flows_tuesday.parquet",
        "tuesday",
        datetime(2017, 7, 4, 12, 0, tzinfo=timezone.utc),
        240,
        mode="train_attacks",
    )
    _write_synthetic_day_parquet(
        cfg.data_processed_dir / "flows_wednesday.parquet",
        "wednesday",
        datetime(2017, 7, 5, 12, 0, tzinfo=timezone.utc),
        240,
        mode="late_attacks",
    )

    cls_res = run_classification(cfg, embargo_minutes=15)
    gb = cls_res["global_split_bounds"]
    assert gb["train_end"] < gb["val_start"] < gb["val_end"] < gb["test_start"]
    assert "Portscan" in cls_res["with_dst_port"]["models"]["xgboost"]["test"]["unseen_in_training_classes"]
    assert (cfg.results_dir / "classification_metrics.json").exists()

    anom_res = run_anomaly_detection(cfg)
    assert anom_res["fit_period"] == "flows_monday.parquet"
    assert anom_res["discarded_from_out_of_time_eval_rows"] == 240
    # Ensure Monday benign rows are NOT included in out-of-time per-family total (Tue+Wed BENIGN < 480)
    oot_benign = [r for r in anom_res["per_family_out_of_time_only"] if r["label"] == "BENIGN"][0]
    assert oot_benign["total_out_of_time_flows"] < 480
    assert anom_res["out_of_time_roc_auc"] > 0.75
