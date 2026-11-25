"""Stage 2A: Intrusion Detection Classifiers (Random Forest & XGBoost) with globally chronological splits."""
from __future__ import annotations

import gc
import json
import logging
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

import joblib
import matplotlib
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.settings import Settings, get_settings
from src.data_loader import get_feature_columns, get_processed_parquet_paths

logger = logging.getLogger(__name__)


def sanitize_feature_matrix(arr: np.ndarray) -> np.ndarray:
    """Convert +/-inf to NaN while preserving float32 dtype for CPU/RAM efficiency."""
    out = np.array(arr, dtype=np.float32, copy=True)
    out[~np.isfinite(out)] = np.nan
    return out


def compute_global_chronological_split_bounds(
    all_start_ts: np.ndarray,
    embargo_minutes: int = 30,
    train_frac: float = 0.60,
    val_frac: float = 0.15,
) -> dict[str, Any]:
    """Compute globally chronological split timestamps across all capture days with embargo gaps.

    Guarantees:
    - ``train_end < val_start <= val_end < test_start`` across the entire multi-day timeline.
    - ``val_start - train_end >= embargo_minutes`` and ``test_start - val_end >= embargo_minutes``.
    - No later-day records can enter training when earlier-day records are evaluated.
    """
    if len(all_start_ts) < 10:
        raise ValueError("Insufficient rows to build a global chronological split.")

    sorted_ts = np.sort(all_start_ts.astype("datetime64[us]"))
    n = len(sorted_ts)
    embargo_us = np.timedelta64(int(embargo_minutes) * 60_000_000, "us")

    # Initial target cutoffs by chronological flow count (~60% train / ~15% val / ~25% test)
    idx_train_target = min(max(int(n * train_frac) - 1, 1), n - 5)
    train_end = sorted_ts[idx_train_target]

    # Advance val_start beyond train_end + embargo_minutes
    val_start_target = train_end + embargo_us
    val_start_idx = int(np.searchsorted(sorted_ts, val_start_target, side="left"))
    if val_start_idx >= n - 3:
        # Fallback by wall-clock duration if short synthetic sequence has dense timestamps
        total_span = sorted_ts[-1] - sorted_ts[0]
        half_usable = max(total_span - 2 * embargo_us, np.timedelta64(60_000_000, "us"))
        train_end = sorted_ts[0] + (half_usable * int(train_frac * 100)) // 100
        val_start_idx = int(np.searchsorted(sorted_ts, train_end + embargo_us, side="left"))
    val_start = sorted_ts[min(val_start_idx, n - 1)]

    remaining = max(n - val_start_idx, 1)
    val_count = max(int(n * val_frac), 1)
    idx_val_end = min(val_start_idx + val_count - 1, n - 3)
    val_end = sorted_ts[max(idx_val_end, val_start_idx)]

    test_start_target = val_end + embargo_us
    test_start_idx = int(np.searchsorted(sorted_ts, test_start_target, side="left"))
    if test_start_idx >= n:
        test_start_idx = n - 1
    test_start = sorted_ts[test_start_idx]

    # Convert numpy datetime64[us] to tz-aware UTC python datetimes
    from datetime import timezone

    def _to_utc(dt64: np.datetime64):
        dt = dt64.astype("datetime64[us]").item()
        return dt.replace(tzinfo=timezone.utc)

    t_min_dt = _to_utc(sorted_ts[0])
    tr_end_dt = _to_utc(train_end)
    va_start_dt = _to_utc(val_start)
    va_end_dt = _to_utc(val_end)
    te_start_dt = _to_utc(test_start)
    t_max_dt = _to_utc(sorted_ts[-1])

    assert (va_start_dt - tr_end_dt).total_seconds() >= embargo_minutes * 60 - 1e-6, "Train->Val embargo violated"
    assert (te_start_dt - va_end_dt).total_seconds() >= embargo_minutes * 60 - 1e-6, "Val->Test embargo violated"

    return {
        "t_min": t_min_dt,
        "train_end": tr_end_dt,
        "val_start": va_start_dt,
        "val_end": va_end_dt,
        "test_start": te_start_dt,
        "t_max": t_max_dt,
        "embargo_minutes": embargo_minutes,
    }


def compute_intraday_split_bounds(
    t_min: Any,
    t_max: Any,
    gap_minutes: int = 5,
    train_frac: float = 0.60,
    val_frac: float = 0.15,
) -> dict[str, Any]:
    """Legacy helper retained for reference/tests only; primary evaluation uses global chronological splits."""
    total_sec = (t_max - t_min).total_seconds()
    gap = timedelta(minutes=gap_minutes)
    usable_sec = max(total_sec - 2.0 * gap.total_seconds(), 60.0)
    train_end = t_min + timedelta(seconds=usable_sec * train_frac)
    val_start = train_end + gap
    val_end = val_start + timedelta(seconds=usable_sec * val_frac)
    test_start = val_end + gap
    return {
        "t_min": t_min,
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
        "test_start": test_start,
        "t_max": t_max,
    }


def _time_respecting_subsample(df: pl.DataFrame, max_rows: int) -> pl.DataFrame:
    """Subsample rows chronologically while preserving rare attack classes completely."""
    if df.height <= max_rows:
        return df.sort("start_ts")

    counts = df.group_by("label").len().to_dicts()
    n_classes = max(len(counts), 1)
    min_per_class = max(min(2000, max_rows // n_classes), 1)

    rare_parts: list[pl.DataFrame] = []
    large_parts: list[pl.DataFrame] = []
    remaining_budget = max_rows
    for row in counts:
        lbl = row["label"]
        cnt = int(row["len"])
        sub = df.filter(pl.col("label") == lbl)
        if cnt <= min_per_class:
            rare_parts.append(sub)
            remaining_budget = max(remaining_budget - cnt, 0)
        else:
            large_parts.append(sub)

    parts = list(rare_parts)
    if large_parts and remaining_budget > 0:
        large_df = pl.concat(large_parts)
        frac = min(1.0, remaining_budget / max(large_df.height, 1))
        step = max(int(round(1.0 / max(frac, 1e-6))), 1)
        sampled_large = large_df.sort("start_ts").gather_every(step).head(remaining_budget)
        parts.append(sampled_large)

    combined = pl.concat(parts).sort("start_ts")
    return combined.head(max_rows)


def _load_global_split_frames(
    cfg: Settings,
    feature_cols: list[str],
    embargo_minutes: int = 30,
    holdout_family: Optional[str] = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Load strictly globally chronological train / embargo / val / embargo / test frames."""
    parquets = get_processed_parquet_paths(cfg)
    ts_parts = [pl.scan_parquet(str(pq)).select("start_ts").collect()["start_ts"].to_numpy() for pq in parquets]
    all_ts = np.concatenate(ts_parts)
    bounds = compute_global_chronological_split_bounds(all_ts, embargo_minutes=embargo_minutes)
    del ts_parts, all_ts
    gc.collect()

    needed_cols = ["day", "start_ts", "end_ts", "Flow ID", "label", "is_attack"] + feature_cols
    train_parts: list[pl.DataFrame] = []
    val_parts: list[pl.DataFrame] = []
    test_parts: list[pl.DataFrame] = []

    raw_counts = {"train_raw": 0, "embargo_1_raw": 0, "val_raw": 0, "embargo_2_raw": 0, "test_raw": 0}
    day_assignments: list[dict[str, Any]] = []

    tr_end = bounds["train_end"]
    va_start = bounds["val_start"]
    va_end = bounds["val_end"]
    te_start = bounds["test_start"]

    for pq in parquets:
        df_day = pl.read_parquet(pq, columns=needed_cols)
        # Ensure flows ending after a cutoff do not leak into training/val
        tr = df_day.filter((pl.col("start_ts") <= pl.lit(tr_end)) & (pl.col("end_ts") <= pl.lit(va_start)))
        emb1 = df_day.filter((pl.col("start_ts") > pl.lit(tr_end)) & (pl.col("start_ts") < pl.lit(va_start)))
        va = df_day.filter(
            (pl.col("start_ts") >= pl.lit(va_start))
            & (pl.col("start_ts") <= pl.lit(va_end))
            & (pl.col("end_ts") <= pl.lit(te_start))
        )
        emb2 = df_day.filter((pl.col("start_ts") > pl.lit(va_end)) & (pl.col("start_ts") < pl.lit(te_start)))
        te = df_day.filter(pl.col("start_ts") >= pl.lit(te_start))

        raw_counts["train_raw"] += tr.height
        raw_counts["embargo_1_raw"] += emb1.height
        raw_counts["val_raw"] += va.height
        raw_counts["embargo_2_raw"] += emb2.height
        raw_counts["test_raw"] += te.height

        day_assignments.append(
            {
                "day": str(df_day["day"][0]) if df_day.height else pq.stem,
                "train_rows": tr.height,
                "embargo1_rows": emb1.height,
                "val_rows": va.height,
                "embargo2_rows": emb2.height,
                "test_rows": te.height,
            }
        )

        if holdout_family is not None:
            tr = tr.filter(pl.col("label") != holdout_family)
            va = va.filter(pl.col("label") != holdout_family)

        if tr.height > 0:
            train_parts.append(tr)
        if va.height > 0:
            val_parts.append(va)
        if te.height > 0:
            test_parts.append(te)

        del df_day
        gc.collect()

    train_df = _time_respecting_subsample(pl.concat(train_parts), cfg.classifier_max_train_rows)
    val_df = _time_respecting_subsample(pl.concat(val_parts), cfg.classifier_max_eval_rows)
    test_df = _time_respecting_subsample(pl.concat(test_parts), cfg.classifier_max_eval_rows)

    # Strict non-overlap chronological verification
    assert train_df["start_ts"].max() < val_df["start_ts"].min(), "Chronological violation: train overlaps validation"
    assert val_df["start_ts"].max() < test_df["start_ts"].min(), "Chronological violation: validation overlaps test"

    split_meta = {
        "train_start": str(bounds["t_min"]),
        "train_end": str(tr_end),
        "val_start": str(va_start),
        "val_end": str(va_end),
        "test_start": str(te_start),
        "test_end": str(bounds["t_max"]),
        "embargo_minutes": embargo_minutes,
        "raw_partition_counts": raw_counts,
        "per_day_partition_counts": day_assignments,
    }
    return train_df, val_df, test_df, split_meta


def _evaluate_predictions(
    y_true_str: np.ndarray,
    y_pred_str: np.ndarray,
    attack_prob: np.ndarray,
    seen_classes: set[str],
) -> dict[str, Any]:
    """Compute separate binary attack detection and multiclass classification metrics."""
    all_labels = sorted(set(y_true_str.tolist()) | set(y_pred_str.tolist()))
    report = classification_report(
        y_true_str,
        y_pred_str,
        labels=all_labels,
        output_dict=True,
        zero_division=0,
    )
    unseen_in_training = sorted([c for c in set(y_true_str.tolist()) if c not in seen_classes])
    seen_present = sorted([c for c in all_labels if c in seen_classes])

    macro_f1_all = float(f1_score(y_true_str, y_pred_str, labels=all_labels, average="macro", zero_division=0))
    macro_f1_seen = (
        float(f1_score(y_true_str, y_pred_str, labels=seen_present, average="macro", zero_division=0))
        if seen_present
        else 0.0
    )

    y_bin_true = (y_true_str != "BENIGN").astype(int)
    y_bin_pred = (y_pred_str != "BENIGN").astype(int)
    bin_prec = float(precision_score(y_bin_true, y_bin_pred, zero_division=0))
    bin_rec = float(recall_score(y_bin_true, y_bin_pred, zero_division=0))
    bin_f1 = float(f1_score(y_bin_true, y_bin_pred, zero_division=0))
    bin_roc = float(roc_auc_score(y_bin_true, attack_prob)) if len(np.unique(y_bin_true)) > 1 else None
    bin_pr = float(average_precision_score(y_bin_true, attack_prob)) if len(np.unique(y_bin_true)) > 1 else None

    cm = confusion_matrix(y_true_str, y_pred_str, labels=all_labels)
    per_class_rows = []
    for lbl in all_labels:
        entry = report.get(lbl, {})
        per_class_rows.append(
            {
                "class": lbl,
                "seen_in_training": lbl in seen_classes,
                "precision": float(entry.get("precision", 0.0)),
                "recall": float(entry.get("recall", 0.0)),
                "f1": float(entry.get("f1-score", 0.0)),
                "support": int(entry.get("support", 0)),
            }
        )

    return {
        "binary_detection": {
            "attack_prevalence": float(np.mean(y_bin_true)),
            "binary_precision": bin_prec,
            "binary_recall": bin_rec,
            "binary_attack_f1": bin_f1,
            "binary_roc_auc": bin_roc,
            "binary_pr_auc": bin_pr,
            "caveat": "Binary attack detection (1 - P(BENIGN)) can flag unseen attacks resembling seen families, which does NOT imply correct multiclass family attribution.",
        },
        "multiclass_classification": {
            "macro_f1_all_classes": macro_f1_all,
            "macro_f1_seen_classes_only": macro_f1_seen,
            "weighted_f1": float(report.get("weighted avg", {}).get("f1-score", 0.0)),
            "accuracy": float(report.get("accuracy", 0.0)),
            "unseen_in_training_classes": unseen_in_training,
            "per_class": per_class_rows,
            "confusion_matrix": {"labels": all_labels, "matrix": cm.tolist()},
        },
        # Backward-compatible flat keys for downstream tools/tables
        "macro_f1_all_classes": macro_f1_all,
        "macro_f1_seen_classes_only": macro_f1_seen,
        "weighted_f1": float(report.get("weighted avg", {}).get("f1-score", 0.0)),
        "accuracy": float(report.get("accuracy", 0.0)),
        "binary_attack_f1": bin_f1,
        "binary_roc_auc": bin_roc,
        "binary_pr_auc": bin_pr,
        "unseen_in_training_classes": unseen_in_training,
        "per_class": per_class_rows,
        "confusion_matrix": {"labels": all_labels, "matrix": cm.tolist()},
    }


def _plot_confusion_matrix(labels: list[str], cm: np.ndarray, title: str, out_path: Path) -> None:
    """Save clean normalized confusion matrix PNG."""
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.55), max(6, len(labels) * 0.5)))
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm.astype(float), np.where(row_sums == 0, 1, row_sums))
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _train_and_eval_pair(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    feature_cols: list[str],
    seed: int,
    save_models_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Fit RF and XGBoost (hist, class-weighted) on train and evaluate on val and test."""
    X_train = sanitize_feature_matrix(train_df.select(feature_cols).to_numpy())
    X_val = sanitize_feature_matrix(val_df.select(feature_cols).to_numpy())
    X_test = sanitize_feature_matrix(test_df.select(feature_cols).to_numpy())

    y_train_str = train_df["label"].to_numpy()
    y_val_str = val_df["label"].to_numpy()
    y_test_str = test_df["label"].to_numpy()

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_str)
    seen_classes = set(le.classes_.tolist())
    benign_idx = int(np.where(le.classes_ == "BENIGN")[0][0]) if "BENIGN" in seen_classes else None

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=2,
        random_state=seed,
    )
    rf.fit(X_train_imp, y_train)

    xgb_objective = "multi:softprob" if len(le.classes_) > 2 else "binary:logistic"
    xgb_kwargs: dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "n_jobs": 2,
        "random_state": seed,
        "eval_metric": "mlogloss" if len(le.classes_) > 2 else "logloss",
    }
    if len(le.classes_) > 2:
        xgb_kwargs["num_class"] = len(le.classes_)
        xgb_kwargs["objective"] = xgb_objective
    xgb = XGBClassifier(**xgb_kwargs)
    xgb.fit(X_train_imp, y_train, sample_weight=sample_weights)

    results: dict[str, Any] = {
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "seen_training_classes": sorted(seen_classes),
        "train_rows": int(len(y_train)),
        "val_rows": int(len(y_val_str)),
        "test_rows": int(len(y_test_str)),
        "models": {},
    }

    for model_name, model in (("random_forest", rf), ("xgboost", xgb)):
        val_proba = model.predict_proba(X_val_imp)
        test_proba = model.predict_proba(X_test_imp)
        val_pred = le.inverse_transform(np.argmax(val_proba, axis=1))
        test_pred = le.inverse_transform(np.argmax(test_proba, axis=1))

        if benign_idx is not None:
            val_atk_p = 1.0 - val_proba[:, benign_idx]
            test_atk_p = 1.0 - test_proba[:, benign_idx]
        else:
            val_atk_p = np.max(val_proba, axis=1)
            test_atk_p = np.max(test_proba, axis=1)

        results["models"][model_name] = {
            "validation": _evaluate_predictions(y_val_str, val_pred, val_atk_p, seen_classes),
            "test": _evaluate_predictions(y_test_str, test_pred, test_atk_p, seen_classes),
        }

    if save_models_dir is not None:
        save_models_dir.mkdir(parents=True, exist_ok=True)
        bundle_rf = Pipeline([("imputer", imputer), ("model", rf)])
        bundle_xgb = Pipeline([("imputer", imputer), ("model", xgb)])
        joblib.dump({"pipeline": bundle_rf, "label_encoder": le, "feature_columns": feature_cols}, save_models_dir / "detector_rf.joblib")
        joblib.dump({"pipeline": bundle_xgb, "label_encoder": le, "feature_columns": feature_cols}, save_models_dir / "detector_xgb.joblib")

    return results


def _choose_holdout_family(train_df: pl.DataFrame) -> Optional[str]:
    """Select a well-populated attack family from training data for the hold-out family experiment."""
    counts_df = train_df.filter(pl.col("is_attack") == 1).group_by("label").len().sort("len", descending=True)
    if counts_df.height == 0:
        return None
    for row in counts_df.to_dicts():
        if row["label"] in ("DoS Hulk", "FTP-Patator", "DoS GoldenEye") and int(row["len"]) >= 50:
            return str(row["label"])
    return str(counts_df["label"][0])


def run_classification(cfg: Optional[Settings] = None, embargo_minutes: int = 30) -> dict[str, Any]:
    """Run Stage 2A classification under a strictly global chronological split with embargo."""
    settings = cfg or get_settings()
    settings.ensure_dirs()
    seed = settings.seeds[0]

    parquets = get_processed_parquet_paths(settings)
    schema_names = pl.scan_parquet(str(parquets[0])).collect_schema().names()
    cols_with_port = get_feature_columns(schema_names, include_dst_port=True)
    cols_no_port = get_feature_columns(schema_names, include_dst_port=False)

    logger.info("Loading globally chronological splits (%d features with Dst Port)...", len(cols_with_port))
    train_df, val_df, test_df, split_meta = _load_global_split_frames(
        settings, cols_with_port, embargo_minutes=embargo_minutes
    )

    main_res = _train_and_eval_pair(
        train_df,
        val_df,
        test_df,
        cols_with_port,
        seed=seed,
        save_models_dir=settings.models_dir,
    )

    per_class_rows = []
    for m_name, m_data in main_res["models"].items():
        cm_info = m_data["test"]["confusion_matrix"]
        _plot_confusion_matrix(
            cm_info["labels"],
            np.array(cm_info["matrix"]),
            f"{m_name.upper()} Global Chronological Test Confusion Matrix",
            settings.results_dir / f"classifier_cm_{m_name}.png",
        )
        for r in m_data["test"]["per_class"]:
            per_class_rows.append({"model": m_name, "variant": "with_dst_port", **r})

    logger.info("Running Destination Port ablation (%d features)...", len(cols_no_port))
    no_port_res = _train_and_eval_pair(
        train_df,
        val_df,
        test_df,
        cols_no_port,
        seed=seed,
        save_models_dir=None,
    )
    for m_name, m_data in no_port_res["models"].items():
        for r in m_data["test"]["per_class"]:
            per_class_rows.append({"model": m_name, "variant": "without_dst_port", **r})

    pl.DataFrame(per_class_rows).write_csv(settings.results_dir / "classification_per_class.csv")

    # Separate hold-one-attack-family-out experiment
    holdout_family = _choose_holdout_family(train_df)
    holdout_report: dict[str, Any] = {"held_out_family": holdout_family}
    if holdout_family is not None:
        logger.info("Running hold-one-family-out experiment holding out '%s'...", holdout_family)
        h_train, h_val, h_test, _ = _load_global_split_frames(
            settings, cols_with_port, embargo_minutes=embargo_minutes, holdout_family=holdout_family
        )
        n_h_classes = h_train["label"].n_unique()
        if n_h_classes >= 2:
            h_res = _train_and_eval_pair(h_train, h_val, h_test, cols_with_port, seed=seed, save_models_dir=None)
            fam_frames = []
            for pq in parquets:
                f_sub = pl.scan_parquet(str(pq)).filter(pl.col("label") == holdout_family).select(cols_with_port).collect()
                if f_sub.height > 0:
                    fam_frames.append(f_sub)
            if fam_frames:
                fam_df = pl.concat(fam_frames).head(50_000)
                X_fam = sanitize_feature_matrix(fam_df.to_numpy())
                le_h = LabelEncoder()
                y_htr = le_h.fit_transform(h_train["label"].to_numpy())
                imp_h = SimpleImputer(strategy="median")
                X_htr = imp_h.fit_transform(sanitize_feature_matrix(h_train.select(cols_with_port).to_numpy()))
                X_fam_imp = imp_h.transform(X_fam)
                ben_idx_h = int(np.where(le_h.classes_ == "BENIGN")[0][0])

                xgb_h_kwargs: dict[str, Any] = {
                    "n_estimators": 150,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "tree_method": "hist",
                    "n_jobs": 2,
                    "random_state": seed,
                }
                if len(le_h.classes_) > 2:
                    xgb_h_kwargs["num_class"] = len(le_h.classes_)
                    xgb_h_kwargs["objective"] = "multi:softprob"
                xgb_h = XGBClassifier(**xgb_h_kwargs)
                xgb_h.fit(X_htr, y_htr, sample_weight=compute_sample_weight("balanced", y_htr))
                fam_proba = xgb_h.predict_proba(X_fam_imp)
                fam_pred = le_h.inverse_transform(np.argmax(fam_proba, axis=1))
                holdout_report.update(
                    {
                        "held_out_family_eval_flows": int(fam_df.height),
                        "xgb_binary_detection_rate_on_unseen_family": float(np.mean(fam_pred != "BENIGN")),
                        "xgb_mean_attack_probability_on_unseen_family": float(np.mean(1.0 - fam_proba[:, ben_idx_h])),
                        "top_predicted_labels_for_unseen_family": pl.Series("pred", fam_pred).value_counts(sort=True).head(5).to_dicts(),
                        "test_split_summary": {
                            k: {
                                "macro_f1_all": v["test"]["macro_f1_all_classes"],
                                "binary_attack_f1": v["test"]["binary_attack_f1"],
                            }
                            for k, v in h_res["models"].items()
                        },
                    }
                )
        else:
            holdout_report["note"] = "Skipped hold-out classifier: fewer than 2 classes remained in training."

    summary = {
        "split_protocol": f"globally_chronological_60_15_25_with_{embargo_minutes}min_embargo",
        "global_split_bounds": split_meta,
        "unseen_family_limitation_note": (
            "Under a strictly global chronological split across CICIDS2017, attack families scheduled on later days "
            "(validation/test) do not appear in earlier-day training. Exact multiclass recall on unseen attack families "
            "is 0.0 by construction; binary attack ROC-AUC/PR-AUC measures only benign-vs-attack feature separability, "
            "not unseen-family classification."
        ),
        "with_dst_port": main_res,
        "without_dst_port_ablation": no_port_res,
        "holdout_family_experiment": holdout_report,
    }
    (settings.results_dir / "classification_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
