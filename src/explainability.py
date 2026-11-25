"""Stage 5: Explainability (SHAP TreeExplainer with labelled permutation/gain fallback) for Detection and Forecasting."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import polars as pl
import shap
from sklearn.inspection import permutation_importance

from config.settings import Settings, get_settings
from src.classification import sanitize_feature_matrix
from src.data_loader import get_processed_parquet_paths
from src.feature_engineering import compute_deployable_feature_names

logger = logging.getLogger(__name__)


def _format_plain_reason(feature: str, value: float, shap_val: float, context: str) -> dict[str, Any]:
    """Format one top feature explanation into structured dict + plain-language sentence."""
    if context == "forecast":
        direction = "raises risk" if shap_val >= 0 else "lowers risk"
    else:
        direction = "pushes toward attack" if shap_val >= 0 else "pushes toward benign"
    sentence = f"`{feature}` = {value:.4g} {direction} (SHAP {shap_val:+.4f})."
    return {
        "feature": feature,
        "value": float(value),
        "contribution": float(shap_val),
        "abs_contribution": float(abs(shap_val)),
        "direction": direction,
        "sentence": sentence,
    }


def _extract_shap_2d(shap_values: Any, X: np.ndarray, class_idx: Optional[int] = None) -> np.ndarray:
    """Normalize SHAP output across binary/multiclass/SHAP-version return shapes into (N, F)."""
    arr = np.asarray(shap_values)
    if isinstance(shap_values, list):
        chosen = np.asarray(shap_values[class_idx if class_idx is not None else -1])
        return chosen
    if arr.ndim == 3:
        idx = class_idx if class_idx is not None else (arr.shape[2] - 1)
        return arr[:, :, idx]
    if arr.ndim == 2:
        return arr
    return np.zeros_like(X, dtype=np.float64)


def explain_forecast_local(
    forecaster_bundle: dict[str, Any],
    X_row: np.ndarray,
    top_k: int = 5,
    explainer: Optional[Any] = None,
) -> tuple[list[dict[str, Any]], str]:
    """Return top-k SHAP reasons (or labelled fallback) for a single forecast row (1, F)."""
    feat_cols: list[str] = forecaster_bundle["feature_columns"]
    model = forecaster_bundle["primary_model"]
    row = np.nan_to_num(np.asarray(X_row, dtype=np.float64).reshape(1, -1), nan=0.0, posinf=0.0, neginf=0.0)

    try:
        exp = explainer if explainer is not None else shap.TreeExplainer(model)
        sv = exp.shap_values(row)
        sv_2d = _extract_shap_2d(sv, row)[0]
        method = "shap_tree_explainer"
    except Exception as exc:
        logger.warning("Forecast local SHAP failed (%s); using feature-importance deviation fallback.", exc)
        bg = forecaster_bundle.get("X_train_background")
        bg_mean = np.mean(bg, axis=0) if bg is not None and len(bg) else np.zeros(row.shape[1])
        bg_std = np.std(bg, axis=0) + 1e-6 if bg is not None and len(bg) else np.ones(row.shape[1])
        imp = getattr(model, "feature_importances_", np.ones(row.shape[1]) / row.shape[1])
        sv_2d = ((row[0] - bg_mean) / bg_std) * imp
        method = "fallback_importance_weighted_zscore"

    top_idx = np.argsort(np.abs(sv_2d))[::-1][:top_k]
    reasons = [_format_plain_reason(feat_cols[int(i)], float(row[0, int(i)]), float(sv_2d[int(i)]), "forecast") for i in top_idx]
    return reasons, method


def explain_detection_local(
    detector_bundle: dict[str, Any],
    X_row: np.ndarray,
    predicted_class: str,
    top_k: int = 5,
    explainer: Optional[Any] = None,
) -> tuple[list[dict[str, Any]], str]:
    """Return top-k SHAP reasons (or labelled fallback) for a single detection flow row (1, F)."""
    feat_cols: list[str] = detector_bundle["feature_columns"]
    pipe = detector_bundle["pipeline"]
    le = detector_bundle["label_encoder"]
    imputer = pipe.named_steps["imputer"]
    xgb = pipe.named_steps["model"]

    row_san = sanitize_feature_matrix(np.asarray(X_row).reshape(1, -1))
    row_imp = imputer.transform(row_san)

    classes = le.classes_.tolist()
    if predicted_class in classes:
        cls_idx = classes.index(predicted_class)
    elif "BENIGN" in classes:
        # Explain away from BENIGN
        cls_idx = classes.index("BENIGN")
    else:
        cls_idx = 0

    try:
        exp = explainer if explainer is not None else shap.TreeExplainer(xgb)
        sv = exp.shap_values(row_imp)
        sv_2d = _extract_shap_2d(sv, row_imp, class_idx=cls_idx)[0]
        if predicted_class == "BENIGN":
            sv_2d = -sv_2d
        method = "shap_tree_explainer"
    except Exception as exc:
        logger.warning("Detection local SHAP failed (%s); using gain-importance fallback.", exc)
        imp = getattr(xgb, "feature_importances_", np.ones(row_imp.shape[1]) / row_imp.shape[1])
        sv_2d = np.sign(row_imp[0]) * imp
        method = "fallback_gain_importance"

    top_idx = np.argsort(np.abs(sv_2d))[::-1][:top_k]
    reasons = [_format_plain_reason(feat_cols[int(i)], float(row_imp[0, int(i)]), float(sv_2d[int(i)]), "detection") for i in top_idx]
    return reasons, method


def run_global_explainability(cfg: Optional[Settings] = None) -> dict[str, Any]:
    """Compute global SHAP importances (or labelled permutation fallback) separately for Forecast & Detection."""
    settings = cfg or get_settings()
    settings.ensure_dirs()
    primary_h = int(settings.horizons[0])

    # 1. Forecaster Explainability
    fc_model_path = settings.models_dir / f"forecaster_xgb_h{primary_h}.joblib"
    feat_path = settings.data_processed_dir / "forecast_features.parquet"
    preds_path = settings.results_dir / "forecast_predictions.parquet"
    if not fc_model_path.exists() or not feat_path.exists() or not preds_path.exists():
        raise FileNotFoundError("Missing forecaster model or features; run `--stage forecast` first.")

    fc_bundle = joblib.load(fc_model_path)
    fc_cols: list[str] = fc_bundle["feature_columns"]
    fc_model = fc_bundle["primary_model"]

    feat_df = pl.read_parquet(feat_path)
    preds_df = pl.read_parquet(preds_path).select(["minute_ts", "split"])
    joined = feat_df.join(preds_df, on="minute_ts", how="inner").filter(
        (pl.col("is_eligible") == 1) & (pl.col("split").is_in(["validation", "test"]))
    )
    if joined.height == 0:
        joined = feat_df.filter(pl.col("is_eligible") == 1)

    explain_df = joined.head(settings.shap_max_explain_rows)
    X_fc = np.nan_to_num(explain_df.select(fc_cols).to_numpy().astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    y_fc = explain_df[f"y_{primary_h}"].to_numpy().astype(int)

    try:
        explainer = shap.TreeExplainer(fc_model)
        sv = explainer.shap_values(X_fc)
        sv_2d = _extract_shap_2d(sv, X_fc)
        mean_abs = np.mean(np.abs(sv_2d), axis=0)
        fc_method = "shap_tree_explainer"
    except Exception as exc:
        logger.warning("Global forecast SHAP failed (%s); falling back to permutation importance.", exc)
        perm = permutation_importance(fc_model, X_fc, y_fc, n_repeats=5, random_state=settings.seeds[0], n_jobs=2)
        mean_abs = np.maximum(perm.importances_mean, 0.0)
        fc_method = "permutation_importance_fallback"

    order_fc = np.argsort(mean_abs)[::-1]
    fc_top = [
        {"feature": fc_cols[int(i)], "importance": float(mean_abs[int(i)])}
        for i in order_fc[:25]
    ]
    fc_report = {
        "component": "FORECASTING",
        "horizon_min": primary_h,
        "method": fc_method,
        "explained_rows": int(len(X_fc)),
        "top_features": fc_top,
    }
    (settings.results_dir / "shap_forecaster_global.json").write_text(json.dumps(fc_report, indent=2), encoding="utf-8")

    # 2. Classifier (Detection) Explainability — strictly separate
    det_model_path = settings.models_dir / "detector_xgb.joblib"
    if not det_model_path.exists():
        raise FileNotFoundError("Missing detector_xgb.joblib; run `--stage classify` first.")
    det_bundle = joblib.load(det_model_path)
    det_cols: list[str] = det_bundle["feature_columns"]
    det_pipe = det_bundle["pipeline"]
    det_le = det_bundle["label_encoder"]
    imputer = det_pipe.named_steps["imputer"]
    det_xgb = det_pipe.named_steps["model"]

    parquets = get_processed_parquet_paths(settings)
    sample_parts = []
    per_day_cap = max(settings.shap_max_explain_rows // len(parquets), 100)
    for pq in parquets:
        d = pl.read_parquet(pq, columns=["label"] + det_cols)
        step = max(d.height // per_day_cap, 1)
        sample_parts.append(d.gather_every(step).head(per_day_cap))
    det_sample = pl.concat(sample_parts).head(settings.shap_max_explain_rows)
    X_det = imputer.transform(sanitize_feature_matrix(det_sample.select(det_cols).to_numpy()))

    try:
        explainer_det = shap.TreeExplainer(det_xgb)
        sv_det = explainer_det.shap_values(X_det)
        arr_det = np.asarray(sv_det)
        if isinstance(sv_det, list):
            # Average absolute SHAP across all attack classes (and benign)
            mean_abs_det = np.mean([np.mean(np.abs(np.asarray(s)), axis=0) for s in sv_det], axis=0)
        elif arr_det.ndim == 3:
            mean_abs_det = np.mean(np.abs(arr_det), axis=(0, 2))
        else:
            mean_abs_det = np.mean(np.abs(arr_det), axis=0)
        det_method = "shap_tree_explainer"
    except Exception as exc:
        logger.warning("Global detection SHAP failed (%s); falling back to permutation importance.", exc)
        known_mask = np.isin(det_sample["label"].to_numpy(), det_le.classes_)
        if np.sum(known_mask) >= 10:
            y_det = det_le.transform(det_sample["label"].to_numpy()[known_mask])
            perm_d = permutation_importance(
                det_xgb, X_det[known_mask], y_det, n_repeats=3, random_state=settings.seeds[0], n_jobs=2
            )
            mean_abs_det = np.maximum(perm_d.importances_mean, 0.0)
        else:
            mean_abs_det = getattr(det_xgb, "feature_importances_", np.zeros(len(det_cols)))
        det_method = "permutation_importance_fallback"

    order_det = np.argsort(mean_abs_det)[::-1]
    det_top = [
        {"feature": det_cols[int(i)], "importance": float(mean_abs_det[int(i)])}
        for i in order_det[:25]
    ]
    det_report = {
        "component": "DETECTION",
        "method": det_method,
        "explained_rows": int(len(X_det)),
        "top_features": det_top,
    }
    (settings.results_dir / "shap_detector_global.json").write_text(json.dumps(det_report, indent=2), encoding="utf-8")

    return {"forecast": fc_report, "detection": det_report}
