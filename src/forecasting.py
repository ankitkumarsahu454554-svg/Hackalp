"""Stage 3B: Chronological onset-aware forecasting models, Platt calibration, and validation thresholding."""
from __future__ import annotations

import json
import logging
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config.settings import Settings, get_settings
from src.evaluation import evaluate_model_predictions
from src.feature_engineering import (
    assert_no_forbidden_features,
    build_forecast_dataset,
    compute_anomaly_only_feature_names,
    compute_deployable_feature_names,
)

logger = logging.getLogger(__name__)


def find_chronological_onset_split(
    df: pl.DataFrame,
    horizon_min: int,
    gap_minutes: int,
    min_train_onsets: int,
    min_val_onsets: int,
    min_test_onsets: int,
) -> dict[str, Any]:
    """Choose contiguous chronological train / gap / val / gap / test boundaries.

    Prefers meeting minimum onset counts across all 3 blocks while preserving
    maximum onsets (not dropped in gaps) and staying close to 60% / 15% / 25%.
    """
    df_sorted = df.sort("minute_ts")
    ts = df_sorted["minute_ts"].to_numpy()
    ts_m = ts.astype("datetime64[m]").astype(np.int64)
    elig = df_sorted["is_eligible"].to_numpy()
    y = df_sorted[f"y_{horizon_min}"].to_numpy()
    onsets = df_sorted["onset"].to_numpy()

    # An onset at index o is "predictable" if at least one eligible minute in [o-H, o-1] exists in same segment
    seg = df_sorted["segment_id"].to_numpy()
    predictable_onset_idx = []
    for oi in np.where(onsets == 1)[0]:
        if any(elig[k] == 1 and y[k] == 1 and seg[k] == seg[oi] for k in range(max(0, oi - horizon_min), oi)):
            predictable_onset_idx.append(int(oi))

    n = len(ts_m)
    if n < 3 * gap_minutes + 10 or len(predictable_onset_idx) < 3:
        # Simple fallback proportional split with required gaps
        t_end_idx = max(0, int(n * 0.60) - gap_minutes)
        v_start_idx = min(n - 1, int(np.searchsorted(ts_m, ts_m[t_end_idx] + gap_minutes)))
        v_end_idx = max(v_start_idx, int(n * 0.78) - gap_minutes)
        te_start_idx = min(n - 1, int(np.searchsorted(ts_m, ts_m[v_end_idx] + gap_minutes)))
        return _build_split_dict(df_sorted, ts, ts_m, elig, y, onsets, t_end_idx, v_start_idx, v_end_idx, te_start_idx, gap_minutes)

    # Candidate cutpoints: right after eligible quiet minutes or 1 min after an onset finishes?
    # Better: evaluate candidate indices right before each onset's lead window or right after each onset
    cand_indices = sorted(
        set(
            [int(max(0, oi - horizon_min - 2)) for oi in predictable_onset_idx]
            + [int(min(n - 1, oi + 5)) for oi in predictable_onset_idx]
            + list(range(int(n * 0.15), int(n * 0.90), max(1, n // 40)))
        )
    )

    total_pred_onsets = len(predictable_onset_idx)
    best_score: Optional[tuple] = None
    best_bounds: Optional[tuple[int, int, int, int]] = None

    for t_end_idx in cand_indices:
        v_start_idx = int(np.searchsorted(ts_m, ts_m[t_end_idx] + gap_minutes))
        if v_start_idx >= n - 5:
            continue
        # Count train onsets whose lead window lies inside [0..t_end_idx]
        tr_onsets = sum(1 for oi in predictable_onset_idx if oi <= t_end_idx)
        for v_end_idx in cand_indices:
            if v_end_idx <= v_start_idx + 2:
                continue
            te_start_idx = int(np.searchsorted(ts_m, ts_m[v_end_idx] + gap_minutes))
            if te_start_idx >= n - 2:
                continue
            va_onsets = sum(1 for oi in predictable_onset_idx if v_start_idx <= oi <= v_end_idx)
            te_onsets = sum(1 for oi in predictable_onset_idx if oi >= te_start_idx)

            # Ensure no boundary splits an onset's H-minute positive window across a gap!
            split_inside_y_tr = int(y[t_end_idx] == 1 and elig[t_end_idx] == 1)
            split_inside_y_va = int(y[v_end_idx] == 1 and elig[v_end_idx] == 1)

            meets_min = int(
                tr_onsets >= min_train_onsets and va_onsets >= min_val_onsets and te_onsets >= min_test_onsets
            )
            retained = tr_onsets + va_onsets + te_onsets
            if retained == 0:
                continue
            p_tr, p_va, p_te = tr_onsets / total_pred_onsets, va_onsets / total_pred_onsets, te_onsets / total_pred_onsets
            dist_target = abs(p_tr - 0.60) + abs(p_va - 0.15) + abs(p_te - 0.25)
            # Maximize: meets_min, clean boundary (not cutting positive lead window), retained onsets, then closeness to 60/15/25
            score = (meets_min, -(split_inside_y_tr + split_inside_y_va), retained, -dist_target)
            if best_score is None or score > best_score:
                best_score = score
                best_bounds = (t_end_idx, v_start_idx, v_end_idx, te_start_idx)

    if best_bounds is None:
        t_end_idx = int(n * 0.55)
        v_start_idx = min(n - 1, int(np.searchsorted(ts_m, ts_m[t_end_idx] + gap_minutes)))
        v_end_idx = min(n - 1, int(n * 0.75))
        te_start_idx = min(n - 1, int(np.searchsorted(ts_m, ts_m[v_end_idx] + gap_minutes)))
    else:
        t_end_idx, v_start_idx, v_end_idx, te_start_idx = best_bounds

    return _build_split_dict(df_sorted, ts, ts_m, elig, y, onsets, t_end_idx, v_start_idx, v_end_idx, te_start_idx, gap_minutes)


def _build_split_dict(
    df_sorted: pl.DataFrame,
    ts: np.ndarray,
    ts_m: np.ndarray,
    elig: np.ndarray,
    y: np.ndarray,
    onsets: np.ndarray,
    t_end_idx: int,
    v_start_idx: int,
    v_end_idx: int,
    te_start_idx: int,
    gap_minutes: int,
) -> dict[str, Any]:
    n = len(ts_m)
    labels = np.full(n, "gap", dtype=object)
    labels[: t_end_idx + 1] = "train"
    labels[v_start_idx : v_end_idx + 1] = "validation"
    labels[te_start_idx:] = "test"

    assert int(ts_m[v_start_idx] - ts_m[t_end_idx]) >= gap_minutes, "Train->Val gap violated!"
    assert int(ts_m[te_start_idx] - ts_m[v_end_idx]) >= gap_minutes, "Val->Test gap violated!"

    def _cnt(name: str) -> tuple[int, int, int]:
        m = labels == name
        on_cnt = int(np.sum(onsets[m] == 1))
        pos_cnt = int(np.sum((elig[m] == 1) & (y[m] == 1)))
        el_cnt = int(np.sum(elig[m] == 1))
        return on_cnt, pos_cnt, el_cnt

    tr_on, tr_pos, tr_el = _cnt("train")
    va_on, va_pos, va_el = _cnt("validation")
    te_on, te_pos, te_el = _cnt("test")

    return {
        "split_labels": labels,
        "split_info": {
            "train_start": str(ts[0]),
            "train_end": str(ts[t_end_idx]),
            "gap1_minutes": int(ts_m[v_start_idx] - ts_m[t_end_idx]),
            "val_start": str(ts[v_start_idx]),
            "val_end": str(ts[v_end_idx]),
            "gap2_minutes": int(ts_m[te_start_idx] - ts_m[v_end_idx]),
            "test_start": str(ts[te_start_idx]),
            "test_end": str(ts[-1]),
            "required_gap_minutes": int(gap_minutes),
        },
        "split_onsets": {
            "train_onsets": tr_on,
            "val_onsets": va_on,
            "test_onsets": te_on,
            "train_positive_rows": tr_pos,
            "val_positive_rows": va_pos,
            "test_positive_rows": te_pos,
            "train_eligible_rows": tr_el,
            "val_eligible_rows": va_el,
            "test_eligible_rows": te_el,
        },
    }


def _fit_platt_calibrator(raw_val_probs: np.ndarray, y_val: np.ndarray) -> tuple[Optional[LogisticRegression], dict[str, Any]]:
    """Fit Platt sigmoid calibration strictly on validation predictions."""
    n_pos = int(np.sum(y_val == 1))
    n_neg = int(np.sum(y_val == 0))
    if n_pos < 5 or n_neg < 5:
        return None, {"calibrated": False, "reason": f"Too few validation positives/negatives (pos={n_pos}, neg={n_neg})."}

    clipped = np.clip(raw_val_probs, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
    cal.fit(logits, y_val)
    slope = float(cal.coef_[0][0])
    if slope <= 0.0:
        return None, {
            "calibrated": False,
            "reason": f"Skipped Platt calibration because validation slope was non-positive ({slope:.4f}).",
        }
    return cal, {"calibrated": True, "slope": slope, "intercept": float(cal.intercept_[0])}


def _apply_platt(raw_probs: np.ndarray, cal: Optional[LogisticRegression]) -> np.ndarray:
    if cal is None:
        return np.clip(raw_probs, 0.0, 1.0)
    clipped = np.clip(raw_probs, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    return cal.predict_proba(logits)[:, 1]


def _select_validation_threshold(
    val_timeline_df: pl.DataFrame,
    val_probs_elig: np.ndarray,
    horizon_min: int,
    merge_gap_min: int,
    false_alert_budget_per_hour: float,
    max_alert_fraction: float = 0.35,
) -> float:
    """Choose a non-degenerate threshold strictly on validation to satisfy FALSE_ALERT_BUDGET_PER_HOUR.

    Rejects degenerate always-on thresholds (alert fraction > ``max_alert_fraction``)
    that artificially merge hours of continuous alerts into a single hit episode.
    Maximizes minute-level F1 (tie-breaking on event onset recall and precision)
    within the false-alert budget.
    """
    unique_probs = np.unique(np.round(val_probs_elig, 6))
    grid = np.unique(np.concatenate([unique_probs, np.linspace(0.01, 0.99, 99)]))
    best_feasible: Optional[tuple[float, float, float, float]] = None
    best_fallback: Optional[tuple[float, float, float]] = None
    n_elig = max(len(val_probs_elig), 1)

    for tau in grid:
        alert_frac = float(np.mean(val_probs_elig >= tau))
        if alert_frac <= 0.0:
            continue
        m = evaluate_model_predictions(
            val_timeline_df,
            val_probs_elig,
            threshold=float(tau),
            horizon_min=horizon_min,
            merge_gap_min=merge_gap_min,
        )
        fa_hr = float(m["false_alerts_per_hour"])
        ev_rec = float(m["event_onset_recall"] or 0.0)
        prec = float(m["precision"])
        rec = float(m["recall"])
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0

        if fa_hr <= false_alert_budget_per_hour and alert_frac <= max_alert_fraction:
            key = (f1, ev_rec, prec, -float(tau))
            if best_feasible is None or key > best_feasible:
                best_feasible = key
                best_tau_feas = float(tau)
        if alert_frac <= max_alert_fraction:
            fb_key = (-fa_hr, f1, ev_rec)
            if best_fallback is None or fb_key > best_fallback:
                best_fallback = fb_key
                best_tau_fb = float(tau)

    if best_feasible is not None:
        return best_tau_feas
    if best_fallback is not None:
        return best_tau_fb
    return float(np.percentile(val_probs_elig, 95.0)) if n_elig > 0 else 0.5


def _train_single_horizon(df_all: pl.DataFrame, horizon_min: int, cfg: Settings, is_primary: bool) -> dict[str, Any]:
    """Train B0-B3 and multi-seed XGBoost forecaster for one horizon H."""
    deploy_cols = compute_deployable_feature_names(cfg.feature_windows)
    anom_cols = compute_anomaly_only_feature_names(cfg.feature_windows)
    oracle_cols = [f"oracle_active_frac_{w}" for w in cfg.feature_windows] + ["oracle_minutes_since_attack"]
    assert_no_forbidden_features(deploy_cols)

    split_res = find_chronological_onset_split(
        df_all,
        horizon_min=horizon_min,
        gap_minutes=cfg.split_gap_min,
        min_train_onsets=cfg.min_train_onsets,
        min_val_onsets=cfg.min_val_onsets,
        min_test_onsets=cfg.min_test_onsets,
    )
    df_split = df_all.sort("minute_ts").with_columns(
        pl.Series("split", split_res["split_labels"].tolist(), dtype=pl.String)
    )

    tr_elig = df_split.filter((pl.col("split") == "train") & (pl.col("is_eligible") == 1))
    va_elig = df_split.filter((pl.col("split") == "validation") & (pl.col("is_eligible") == 1))
    te_elig = df_split.filter((pl.col("split") == "test") & (pl.col("is_eligible") == 1))
    va_timeline = df_split.filter(pl.col("split") == "validation").sort("minute_ts")

    y_tr = tr_elig[f"y_{horizon_min}"].to_numpy().astype(int)
    y_va = va_elig[f"y_{horizon_min}"].to_numpy().astype(int)

    X_tr_all = np.nan_to_num(tr_elig.select(deploy_cols).to_numpy().astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    X_va_all = np.nan_to_num(va_elig.select(deploy_cols).to_numpy().astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    X_te_all = np.nan_to_num(te_elig.select(deploy_cols).to_numpy().astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    X_tr_anom = np.nan_to_num(tr_elig.select(anom_cols).to_numpy().astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    X_va_anom = np.nan_to_num(va_elig.select(anom_cols).to_numpy().astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    X_te_anom = np.nan_to_num(te_elig.select(anom_cols).to_numpy().astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    X_tr_orc = np.nan_to_num(tr_elig.select(oracle_cols).to_numpy().astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    X_va_orc = np.nan_to_num(va_elig.select(oracle_cols).to_numpy().astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    X_te_orc = np.nan_to_num(te_elig.select(oracle_cols).to_numpy().astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    has_two_classes = len(np.unique(y_tr)) > 1

    # B0 Base rate
    base_rate = float(np.mean(y_tr)) if len(y_tr) else 0.0
    raw_preds: dict[str, dict[str, np.ndarray]] = {
        "B0_base_rate": {
            "val": np.full(len(y_va), base_rate, dtype=np.float64),
            "test": np.full(te_elig.height, base_rate, dtype=np.float64),
        }
    }

    # B1 ORACLE reference (not deployable)
    if has_two_classes:
        b1_pipe = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
                ("clf", LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=cfg.seeds[0])),
            ]
        )
        b1_pipe.fit(X_tr_orc, y_tr)
        raw_preds["B1_oracle_persistence"] = {
            "val": b1_pipe.predict_proba(X_va_orc)[:, 1],
            "test": b1_pipe.predict_proba(X_te_orc)[:, 1],
        }
    else:
        b1_pipe = None
        raw_preds["B1_oracle_persistence"] = {"val": raw_preds["B0_base_rate"]["val"], "test": raw_preds["B0_base_rate"]["test"]}

    # B2 Anomaly persistence (deployable)
    if has_two_classes:
        b2_pipe = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
                ("clf", LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=cfg.seeds[0])),
            ]
        )
        b2_pipe.fit(X_tr_anom, y_tr)
        raw_preds["B2_anomaly_persistence"] = {
            "val": b2_pipe.predict_proba(X_va_anom)[:, 1],
            "test": b2_pipe.predict_proba(X_te_anom)[:, 1],
        }
    else:
        b2_pipe = None
        raw_preds["B2_anomaly_persistence"] = {"val": raw_preds["B0_base_rate"]["val"], "test": raw_preds["B0_base_rate"]["test"]}

    # B3 Logistic Regression (deployable, small validation-selected C)
    best_b3_pipe = None
    if has_two_classes:
        best_c_ap = -1.0
        for c_val in (0.01, 0.1, 1.0):
            cand = Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                    ("clf", LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, random_state=cfg.seeds[0])),
                ]
            )
            cand.fit(X_tr_all, y_tr)
            vp = cand.predict_proba(X_va_all)[:, 1]
            ap = float(average_precision_score(y_va, vp)) if len(np.unique(y_va)) > 1 else 0.0
            if ap > best_c_ap:
                best_c_ap = ap
                best_b3_pipe = cand
        raw_preds["B3_logistic_regression"] = {
            "val": best_b3_pipe.predict_proba(X_va_all)[:, 1],
            "test": best_b3_pipe.predict_proba(X_te_all)[:, 1],
        }
    else:
        raw_preds["B3_logistic_regression"] = {"val": raw_preds["B0_base_rate"]["val"], "test": raw_preds["B0_base_rate"]["test"]}

    # M XGBoost forecaster (hyperparameter selection on validation ONLY, then all SEEDS)
    n_pos_tr = int(np.sum(y_tr == 1))
    n_neg_tr = int(np.sum(y_tr == 0))
    spw = float(n_neg_tr / max(n_pos_tr, 1))

    best_params = {"max_depth": 2, "learning_rate": 0.05, "n_estimators": 100, "min_child_weight": 3}
    if has_two_classes:
        best_val_ap = -1.0
        param_grid = [
            {"max_depth": d, "learning_rate": lr, "n_estimators": ne, "min_child_weight": mcw}
            for d in (2, 3)
            for lr in (0.03, 0.08)
            for ne in (80, 150)
            for mcw in (3, 5)
        ]
        for params in param_grid:
            clf = XGBClassifier(
                **params,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_lambda=2.0,
                scale_pos_weight=spw,
                tree_method="hist",
                n_jobs=2,
                random_state=cfg.seeds[0],
                eval_metric="logloss",
            )
            clf.fit(X_tr_all, y_tr)
            vp = clf.predict_proba(X_va_all)[:, 1]
            ap = float(average_precision_score(y_va, vp)) if len(np.unique(y_va)) > 1 else 0.0
            if ap > best_val_ap:
                best_val_ap = ap
                best_params = params

    seed_models: list[XGBClassifier] = []
    seed_val_probs: list[np.ndarray] = []
    seed_test_probs: list[np.ndarray] = []
    seed_metrics_list: list[dict[str, Any]] = []
    y_te = te_elig[f"y_{horizon_min}"].to_numpy().astype(int)
    te_timeline = df_split.filter(pl.col("split") == "test").sort("minute_ts")

    for s in cfg.seeds:
        if has_two_classes:
            clf_s = XGBClassifier(
                **best_params,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_lambda=2.0,
                scale_pos_weight=spw,
                tree_method="hist",
                n_jobs=2,
                random_state=s,
                eval_metric="logloss",
            )
            clf_s.fit(X_tr_all, y_tr)
            vp_s = clf_s.predict_proba(X_va_all)[:, 1]
            tp_s = clf_s.predict_proba(X_te_all)[:, 1]
        else:
            clf_s = None
            vp_s = raw_preds["B0_base_rate"]["val"]
            tp_s = raw_preds["B0_base_rate"]["test"]
        seed_models.append(clf_s)
        seed_val_probs.append(vp_s)
        seed_test_probs.append(tp_s)

        # Evaluate each seed through the exact same validation Platt calibration & threshold selection
        cal_s, cal_s_meta = _fit_platt_calibrator(vp_s, y_va)
        vp_s_cal = _apply_platt(vp_s, cal_s)
        tp_s_cal = _apply_platt(tp_s, cal_s)
        tau_s = _select_validation_threshold(
            va_timeline,
            vp_s_cal,
            horizon_min=horizon_min,
            merge_gap_min=cfg.alert_merge_gap_min,
            false_alert_budget_per_hour=cfg.false_alert_budget_per_hour,
        )
        m_s = evaluate_model_predictions(
            te_timeline,
            tp_s_cal,
            threshold=tau_s,
            horizon_min=horizon_min,
            merge_gap_min=cfg.alert_merge_gap_min,
        )
        seed_metrics_list.append(
            {
                "seed": s,
                "calibrated": cal_s_meta["calibrated"],
                "frozen_threshold": float(tau_s),
                "val_pr_auc_cal": float(average_precision_score(y_va, vp_s_cal)) if len(np.unique(y_va)) > 1 else None,
                "test_pr_auc_raw": float(average_precision_score(y_te, tp_s)) if len(np.unique(y_te)) > 1 else None,
                "test_pr_auc_cal": m_s["pr_auc"],
                "test_brier_cal": m_s["brier_score"],
                "test_event_onset_recall": m_s["event_onset_recall"],
                "test_false_alerts_per_hour": m_s["false_alerts_per_hour"],
            }
        )

    raw_preds["xgboost_forecaster"] = {
        "val": np.mean(seed_val_probs, axis=0),
        "test": np.mean(seed_test_probs, axis=0),
    }

    # Platt calibration (validation only) and frozen thresholds
    cal_preds: dict[str, dict[str, np.ndarray]] = {}
    calibration_info: dict[str, Any] = {}
    calibrators: dict[str, Optional[LogisticRegression]] = {}
    frozen_thresholds: dict[str, float] = {}

    for m_name, pr in raw_preds.items():
        if m_name == "B0_base_rate":
            cal_preds[m_name] = pr
            calibrators[m_name] = None
            calibration_info[m_name] = {"calibrated": False, "reason": "Constant base-rate baseline."}
            frozen_thresholds[m_name] = 0.5 if base_rate < 0.5 else base_rate
            continue

        cal, cal_meta = _fit_platt_calibrator(pr["val"], y_va)
        calibrators[m_name] = cal
        calibration_info[m_name] = cal_meta
        val_cal = _apply_platt(pr["val"], cal)
        test_cal = _apply_platt(pr["test"], cal)
        cal_preds[m_name] = {"val": val_cal, "test": test_cal}
        frozen_thresholds[m_name] = _select_validation_threshold(
            va_timeline,
            val_cal,
            horizon_min=horizon_min,
            merge_gap_min=cfg.alert_merge_gap_min,
            false_alert_budget_per_hour=cfg.false_alert_budget_per_hour,
        )

    # Assemble full prediction DataFrame across validation & test eligible rows
    elig_mask = (df_split["is_eligible"] == 1).to_numpy()
    split_arr = df_split["split"].to_numpy()
    val_elig_mask = (split_arr == "validation") & elig_mask
    test_elig_mask = (split_arr == "test") & elig_mask

    pred_cols: dict[str, np.ndarray] = {}
    for m_name, pr in cal_preds.items():
        arr = np.full(df_split.height, np.nan, dtype=np.float64)
        arr[val_elig_mask] = pr["val"]
        arr[test_elig_mask] = pr["test"]
        if m_name == "B0_base_rate":
            arr[(split_arr == "train") & elig_mask] = base_rate
        pred_cols[m_name] = arr

    out_preds_df = df_split.select(
        ["minute_ts", "segment_id", "minute_in_segment", "attack_active", "onset", f"y_{horizon_min}", "is_eligible", "split"]
    ).with_columns([pl.Series(k, v, dtype=pl.Float64) for k, v in pred_cols.items()])

    # Validation risk tiers for Alert Manager (derived strictly from validation calibrated XGB distribution)
    xgb_val_cal = cal_preds["xgboost_forecaster"]["val"]
    xgb_tau = frozen_thresholds["xgboost_forecaster"]
    above_tau = xgb_val_cal[xgb_val_cal >= xgb_tau]
    high_tier = float(np.percentile(above_tau, 50.0)) if len(above_tau) >= 2 else min(0.95, max(xgb_tau * 1.25, xgb_tau + 0.05))
    crit_tier = float(np.percentile(above_tau, 85.0)) if len(above_tau) >= 2 else min(0.99, max(high_tier * 1.15, high_tier + 0.05))

    # Secondary rolling-origin walk-forward folds if sufficient events exist
    wf_results = _run_walk_forward_secondary(df_all, deploy_cols, horizon_min, best_params, cfg)

    seed_test_prs_cal = [m["test_pr_auc_cal"] for m in seed_metrics_list if m.get("test_pr_auc_cal") is not None]
    seed_test_prs_raw = [m["test_pr_auc_raw"] for m in seed_metrics_list if m.get("test_pr_auc_raw") is not None]
    xgb_seed_summary = {
        "best_params": best_params,
        "seeds": cfg.seeds,
        "per_seed": seed_metrics_list,
        "single_seed_test_pr_auc_mean": float(np.mean(seed_test_prs_cal)) if seed_test_prs_cal else None,
        "single_seed_test_pr_auc_sd": float(np.std(seed_test_prs_cal, ddof=1)) if len(seed_test_prs_cal) > 1 else 0.0,
        "single_seed_raw_pr_auc_mean": float(np.mean(seed_test_prs_raw)) if seed_test_prs_raw else None,
        "single_seed_raw_pr_auc_sd": float(np.std(seed_test_prs_raw, ddof=1)) if len(seed_test_prs_raw) > 1 else 0.0,
        "ensemble_5seed_test_pr_auc": float(average_precision_score(y_te, cal_preds["xgboost_forecaster"]["test"])) if len(np.unique(y_te)) > 1 else None,
    }

    # Save models & schemas
    suffix = "" if is_primary else f"_h{horizon_min}"
    joblib.dump(
        {
            "primary_model": seed_models[0],
            "seed_models": seed_models,
            "calibrator": calibrators["xgboost_forecaster"],
            "feature_columns": deploy_cols,
            "threshold": frozen_thresholds["xgboost_forecaster"],
            "risk_tiers": {"medium": xgb_tau, "high": high_tier, "critical": crit_tier},
            "horizon_min": horizon_min,
            "X_train_background": X_tr_all[: min(256, len(X_tr_all))],
        },
        cfg.models_dir / f"forecaster_xgb_h{horizon_min}.joblib",
    )
    joblib.dump({"b2_pipeline": b2_pipe, "b3_pipeline": best_b3_pipe, "b1_oracle_pipeline": b1_pipe}, cfg.models_dir / f"forecaster_baselines_h{horizon_min}.joblib")

    out_preds_df.write_parquet(cfg.results_dir / f"forecast_predictions{suffix}.parquet", compression="zstd")

    summary = {
        "horizon_min": horizon_min,
        "split_info": split_res["split_info"],
        "split_onsets": split_res["split_onsets"],
        "calibration_info": calibration_info,
        "frozen_thresholds": frozen_thresholds,
        "validation_risk_tiers": {"medium": xgb_tau, "high": high_tier, "critical": crit_tier},
        "xgboost_seed_metrics": xgb_seed_summary,
        "walk_forward_secondary": wf_results,
    }
    (cfg.results_dir / f"forecast_training_summary{suffix}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if is_primary:
        (cfg.results_dir / "forecast_schema.json").write_text(
            json.dumps(
                {
                    "deployable_feature_count": len(deploy_cols),
                    "deployable_features": deploy_cols,
                    "anomaly_only_features": anom_cols,
                    "oracle_only_features_not_deployable": oracle_cols,
                    "horizons": cfg.horizons,
                    "feature_windows": cfg.feature_windows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        onsets_table = df_split.filter(pl.col("onset") == 1).select(
            ["minute_ts", "segment_id", "minute_in_segment", "attack_flow_starts", "split"]
        )
        onsets_table.write_csv(cfg.results_dir / "forecast_onsets.csv")

    return summary


def _run_walk_forward_secondary(
    df_all: pl.DataFrame,
    deploy_cols: list[str],
    horizon_min: int,
    best_params: dict[str, Any],
    cfg: Settings,
) -> list[dict[str, Any]]:
    """Run expanding-origin chronological walk-forward folds as a secondary sensitivity check."""
    df_s = df_all.sort("minute_ts")
    segs = sorted(df_s["segment_id"].unique().to_list())
    if len(segs) < 3:
        return []

    folds: list[dict[str, Any]] = []
    # Expand across days/segments starting from at least 2 segments (where attacks exist)
    for split_seg in segs[2:]:
        tr_df = df_s.filter((pl.col("segment_id") < split_seg) & (pl.col("is_eligible") == 1))
        te_df = df_s.filter((pl.col("segment_id") == split_seg) & (pl.col("is_eligible") == 1))
        y_tr = tr_df[f"y_{horizon_min}"].to_numpy().astype(int)
        y_te = te_df[f"y_{horizon_min}"].to_numpy().astype(int)
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
        X_tr = np.nan_to_num(tr_df.select(deploy_cols).to_numpy().astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        X_te = np.nan_to_num(te_df.select(deploy_cols).to_numpy().astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        spw = float(np.sum(y_tr == 0) / max(np.sum(y_tr == 1), 1))
        clf = XGBClassifier(
            **best_params,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_lambda=2.0,
            scale_pos_weight=spw,
            tree_method="hist",
            n_jobs=2,
            random_state=cfg.seeds[0],
            eval_metric="logloss",
        )
        clf.fit(X_tr, y_tr)
        p_te = clf.predict_proba(X_te)[:, 1]
        lr = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("clf", LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=cfg.seeds[0]))])
        lr.fit(X_tr, y_tr)
        p_lr = lr.predict_proba(X_te)[:, 1]
        folds.append(
            {
                "test_segment_id": int(split_seg),
                "train_positives": int(np.sum(y_tr)),
                "test_positives": int(np.sum(y_te)),
                "test_prevalence": float(np.mean(y_te)),
                "xgb_pr_auc": float(average_precision_score(y_te, p_te)),
                "b3_logreg_pr_auc": float(average_precision_score(y_te, p_lr)),
            }
        )
    return folds


def run_forecasting(cfg: Optional[Settings] = None) -> dict[str, Any]:
    """Build causal minute features if needed and train/calibrate Stage 3 forecasting models."""
    settings = cfg or get_settings()
    settings.ensure_dirs()
    feat_path = settings.data_processed_dir / "forecast_features.parquet"
    if feat_path.exists():
        df_all = pl.read_parquet(feat_path)
        missing_h = [h for h in settings.horizons if f"y_{h}" not in df_all.columns]
        if missing_h:
            df_all = build_forecast_dataset(settings)
    else:
        df_all = build_forecast_dataset(settings)

    summaries = {}
    for idx, h in enumerate(settings.horizons):
        summaries[str(h)] = _train_single_horizon(df_all, h, settings, is_primary=(idx == 0))
    return summaries


def run_k_min_sensitivity(
    cfg: Optional[Settings] = None,
    k_values: tuple[int, ...] = (1, 3, 5),
) -> list[dict[str, Any]]:
    """Run K_MIN sensitivity analysis (1, 3, 5) without altering primary K_MIN=5 artifacts."""
    import tempfile
    from src.evaluation import compute_block_bootstrap, determine_verdict
    from src.feature_engineering import add_rolling_and_targets, BASE_MINUTE_FEATURES

    base_cfg = cfg or get_settings()
    feat_path = base_cfg.data_processed_dir / "forecast_features.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing {feat_path}; run forecasting first.")

    df_existing = pl.read_parquet(feat_path)
    primary_h = int(base_cfg.horizons[0])

    # Reconstruct base minute dataframe + attack start timestamps from processed parquets
    from src.data_loader import get_processed_parquet_paths

    parquets = get_processed_parquet_paths(base_cfg)
    atk_starts_list = []
    for pq in parquets:
        arr = (
            pl.scan_parquet(str(pq))
            .filter(pl.col("is_attack") == 1)
            .select("start_ts")
            .collect()["start_ts"]
            .cast(pl.Datetime("us", "UTC"))
            .to_numpy()
            .astype("datetime64[us]")
            .astype(np.int64)
        )
        if len(arr):
            atk_starts_list.append(arr)
    all_atk_starts = np.concatenate(atk_starts_list) if atk_starts_list else np.empty(0, dtype=np.int64)
    base_minute_df = df_existing.select(["minute_ts", "segment_id", "minute_in_segment"] + BASE_MINUTE_FEATURES)

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as td:
        tmp_root = Path(td)
        for k_val in k_values:
            sub_root = tmp_root / f"k_{k_val}"
            sub_cfg = get_settings(
                project_root=sub_root,
                k_min=int(k_val),
                horizons=list(base_cfg.horizons),
                feature_windows=list(base_cfg.feature_windows),
                quiet_gap_min=base_cfg.quiet_gap_min,
                segment_break_min=base_cfg.segment_break_min,
                alert_merge_gap_min=base_cfg.alert_merge_gap_min,
                false_alert_budget_per_hour=base_cfg.false_alert_budget_per_hour,
                min_train_onsets=base_cfg.min_train_onsets,
                min_val_onsets=base_cfg.min_val_onsets,
                min_test_onsets=base_cfg.min_test_onsets,
                bootstrap_reps=base_cfg.bootstrap_reps,
                bootstrap_block_min=base_cfg.bootstrap_block_min,
                seeds=list(base_cfg.seeds),
            )
            sub_cfg.ensure_dirs()
            df_k = add_rolling_and_targets(base_minute_df, all_atk_starts, sub_cfg)
            summary_k = _train_single_horizon(df_k, primary_h, sub_cfg, is_primary=True)

            preds_k = pl.read_parquet(sub_cfg.results_dir / "forecast_predictions.parquet")
            test_tl = preds_k.filter(pl.col("split") == "test").sort("minute_ts")
            test_el = test_tl.filter(pl.col("is_eligible") == 1)
            thresholds_k = summary_k["frozen_thresholds"]

            model_names = [
                "B0_base_rate",
                "B1_oracle_persistence",
                "B2_anomaly_persistence",
                "B3_logistic_regression",
                "xgboost_forecaster",
            ]
            t_metrics = {}
            m_probs = {}
            for mn in model_names:
                probs = test_el[mn].to_numpy().astype(np.float64)
                m_probs[mn] = probs
                t_metrics[mn] = evaluate_model_predictions(
                    test_tl, probs, float(thresholds_k[mn]), primary_h, sub_cfg.alert_merge_gap_min
                )

            boot_k = compute_block_bootstrap(
                test_tl,
                m_probs,
                thresholds_k,
                horizon_min=primary_h,
                merge_gap_min=sub_cfg.alert_merge_gap_min,
                block_minutes=sub_cfg.bootstrap_block_min,
                reps=sub_cfg.bootstrap_reps,
                seed=sub_cfg.seeds[0],
            )
            verdict_k = determine_verdict(summary_k["split_onsets"], sub_cfg, boot_k, t_metrics)
            paired = boot_k.get("paired_differences", {})
            d_b2 = paired.get("xgboost_forecaster_minus_B2_anomaly_persistence", {}).get("pr_auc", {})
            d_b3 = paired.get("xgboost_forecaster_minus_B3_logistic_regression", {}).get("pr_auc", {})

            xm = t_metrics["xgboost_forecaster"]
            rows.append(
                {
                    "k_min": int(k_val),
                    "total_onsets": int(df_k["onset"].sum()),
                    "train_onsets": summary_k["split_onsets"]["train_onsets"],
                    "val_onsets": summary_k["split_onsets"]["val_onsets"],
                    "test_onsets": summary_k["split_onsets"]["test_onsets"],
                    "test_eligible_rows": xm["n_eligible_rows"],
                    "test_prevalence": xm["prevalence"],
                    "b0_pr_auc": t_metrics["B0_base_rate"]["pr_auc"],
                    "b2_pr_auc": t_metrics["B2_anomaly_persistence"]["pr_auc"],
                    "b3_pr_auc": t_metrics["B3_logistic_regression"]["pr_auc"],
                    "xgb_ensemble_pr_auc": xm["pr_auc"],
                    "xgb_single_seed_pr_auc_mean": summary_k["xgboost_seed_metrics"]["single_seed_test_pr_auc_mean"],
                    "xgb_single_seed_pr_auc_sd": summary_k["xgboost_seed_metrics"]["single_seed_test_pr_auc_sd"],
                    "xgb_brier_score": xm["brier_score"],
                    "xgb_event_onset_recall": xm["event_onset_recall"],
                    "xgb_mean_lead_time_min": xm["mean_lead_time_min"],
                    "xgb_false_alerts_per_hour": xm["false_alerts_per_hour"],
                    "diff_vs_b2_ci_low": d_b2.get("ci_low"),
                    "diff_vs_b2_ci_high": d_b2.get("ci_high"),
                    "diff_vs_b3_ci_low": d_b3.get("ci_low"),
                    "diff_vs_b3_ci_high": d_b3.get("ci_high"),
                    "verdict": verdict_k["verdict"],
                }
            )

    pl.DataFrame(rows).write_csv(base_cfg.results_dir / "k_min_sensitivity.csv")
    (base_cfg.results_dir / "k_min_sensitivity.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows
