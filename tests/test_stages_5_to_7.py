"""Tests for Stage 5 explainability, Stage 6 alert manager, and Stage 7 causal replay engine."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from config.settings import get_settings
from src.alert_manager import AlertManager
from src.explainability import explain_detection_local, explain_forecast_local
from src.feature_engineering import compute_deployable_feature_names
from src.replay_engine import HistoricalReplayEngine


def _build_replay_test_fixture(tmp_path: Path):
    cfg = get_settings(project_root=tmp_path, feature_windows=[5, 10], horizons=[5], k_min=5)
    fc_cols = compute_deployable_feature_names(cfg.feature_windows)
    det_cols = ["Dst Port", "Flow Duration", "Total Fwd Packet", "Total Bwd packets", "Total Length of Fwd Packet", "Total Length of Bwd Packet", "SYN Flag Count", "Packet Length Mean"]

    rng = np.random.default_rng(11)
    base = datetime(2017, 7, 7, 15, 0, tzinfo=timezone.utc)
    rows = []
    for m in range(30):
        for k in range(6):
            st = base + timedelta(minutes=m, seconds=k * 5)
            en = st + timedelta(seconds=1)
            is_atk = 1 if m >= 20 else 0
            rows.append(
                {
                    "start_ts": st,
                    "end_ts": en,
                    "Src IP": f"10.0.0.{1+k}",
                    "Dst IP": "192.168.10.50",
                    "Dst Port": 80,
                    "Flow Duration": 1_000_000,
                    "Total Fwd Packet": 20 if is_atk else 3,
                    "Total Bwd packets": 2,
                    "Total Length of Fwd Packet": 1500 if is_atk else 120,
                    "Total Length of Bwd Packet": 100,
                    "SYN Flag Count": 1 if is_atk else 0,
                    "Packet Length Mean": 200.0 if is_atk else 60.0,
                    "label": "DDoS" if is_atk else "BENIGN",
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

    # Fit tiny detector, anomaly, forecaster bundles
    X_det = flows_df.select(det_cols).to_numpy().astype(np.float32)
    le = LabelEncoder()
    y_det = le.fit_transform(flows_df["label"].to_numpy())
    imp = SimpleImputer(strategy="median").fit(X_det)
    det_xgb = XGBClassifier(n_estimators=10, max_depth=2, random_state=0).fit(imp.transform(X_det), y_det)
    det_bundle = {"pipeline": Pipeline([("imputer", imp), ("model", det_xgb)]), "label_encoder": le, "feature_columns": det_cols}

    iso_imp = SimpleImputer(strategy="median").fit(X_det)
    iso = IsolationForest(n_estimators=10, random_state=0).fit(iso_imp.transform(X_det))
    anom_bundle = {"imputer": iso_imp, "model": iso, "feature_columns": det_cols, "threshold": 0.55}

    X_fc = rng.normal(0.0, 1.0, size=(60, len(fc_cols)))
    y_fc = np.array([0] * 45 + [1] * 15)
    fc_xgb = XGBClassifier(n_estimators=10, max_depth=2, random_state=0).fit(X_fc, y_fc)
    fc_bundle = {
        "primary_model": fc_xgb,
        "seed_models": [fc_xgb],
        "calibrator": None,
        "feature_columns": fc_cols,
        "threshold": 0.25,
        "risk_tiers": {"medium": 0.25, "high": 0.55, "critical": 0.80},
        "horizon_min": 5,
        "X_train_background": X_fc[:20],
    }
    return cfg, flows_df, det_bundle, anom_bundle, fc_bundle, base


def test_explainability_and_alert_manager_separation(tmp_path: Path) -> None:
    cfg, _, det_bundle, _, fc_bundle, _ = _build_replay_test_fixture(tmp_path)
    reasons_fc, method_fc = explain_forecast_local(fc_bundle, fc_bundle["X_train_background"][:1], top_k=5)
    assert len(reasons_fc) == 5
    assert method_fc in {"shap_tree_explainer", "fallback_importance_weighted_zscore"}

    x_det_one = np.array([[80, 1_000_000, 20, 2, 1500, 100, 1, 200.0]], dtype=np.float32)
    reasons_det, _ = explain_detection_local(det_bundle, x_det_one, "DDoS", top_k=5)
    assert len(reasons_det) == 5

    am = AlertManager(
        risk_tiers={"medium": 0.25, "high": 0.55, "critical": 0.80},
        anom_p95=0.55,
        anom_p99=0.70,
        status_store_path=tmp_path / "statuses.json",
    )
    w = am.add_forecast_warning("2017-07-07 15:10:00 UTC", 0.60, 5, reasons_fc, anomaly_score=0.50)
    d = am.add_detection_alert("2017-07-07 15:20:00 UTC", "DDoS", 0.95, 0.75, reasons_det)
    assert w is not None and w.kind == "FORECAST WARNING" and w.severity == "High"
    assert d.kind == "DETECTION" and d.severity == "Critical"
    am.update_status(w.alert_id, "Acknowledged")
    assert am.alerts[0].analyst_status == "Acknowledged"
    assert (tmp_path / "statuses.json").exists()


def test_replay_never_reads_future_and_hides_future_labels(tmp_path: Path) -> None:
    cfg, flows_df, det_bundle, anom_bundle, fc_bundle, base = _build_replay_test_fixture(tmp_path)
    test_mins = [base + timedelta(minutes=m) for m in range(25)]

    am1 = AlertManager(fc_bundle["risk_tiers"], 0.55, 0.70)
    eng1 = HistoricalReplayEngine(flows_df, test_mins, det_bundle, anom_bundle, fc_bundle, am1, cfg)

    # Poison all rows after minute 10 (end_ts >= base + 11m) in a second copy
    cutoff = base + timedelta(minutes=11)
    poisoned_df = flows_df.with_columns(
        [
            pl.when(pl.col("end_ts") >= pl.lit(cutoff)).then(9_999_999).otherwise(pl.col("Total Fwd Packet")).alias("Total Fwd Packet"),
            pl.when(pl.col("start_ts") >= pl.lit(cutoff)).then(1).otherwise(pl.col("is_attack")).alias("is_attack"),
        ]
    )
    am2 = AlertManager(fc_bundle["risk_tiers"], 0.55, 0.70)
    eng2 = HistoricalReplayEngine(poisoned_df, test_mins, det_bundle, anom_bundle, fc_bundle, am2, cfg)

    # Step both through minute 10 (11 steps)
    for step_idx in range(11):
        s1 = eng1.step()
        s2 = eng2.step()
        assert s1 is not None and s2 is not None
        assert s1["forecast_risk"] == s2["forecast_risk"]
        assert s1["malicious_detected"] == s2["malicious_detected"]
        assert s1["anom_mean"] == s2["anom_mean"]
        # Visible ground truth only includes minutes <= current cursor
        vis = eng1.visible_ground_truth_up_to_cursor()
        assert len(vis) == step_idx + 1
        assert max(v["minute_ts"] for v in vis) == test_mins[step_idx]

    # Reset works cleanly
    eng1.reset()
    assert eng1.cursor == 0 and eng1.total_flows_processed == 0 and len(eng1.history) == 0
