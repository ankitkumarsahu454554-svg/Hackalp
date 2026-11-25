"""Stage 2B: Out-of-time Isolation Forest Anomaly Detection (verified benign fit only; strict out-of-time eval)."""
from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score

from config.settings import Settings, get_settings
from src.classification import sanitize_feature_matrix
from src.data_loader import get_feature_columns, get_processed_parquet_paths

logger = logging.getLogger(__name__)


def run_anomaly_detection(cfg: Optional[Settings] = None, threshold_percentile: float = 95.0) -> dict[str, Any]:
    """Fit Isolation Forest strictly on verified BENIGN flows from earliest benign period and evaluate out-of-time."""
    settings = cfg or get_settings()
    settings.ensure_dirs()
    seed = settings.seeds[0]

    parquets = get_processed_parquet_paths(settings)
    schema_names = pl.scan_parquet(str(parquets[0])).collect_schema().names()
    feature_cols = get_feature_columns(schema_names, include_dst_port=True)
    has_raw_label = "raw_label" in schema_names

    # Explicitly verify each candidate period's actual labels (never assume Monday is benign without checking!)
    benign_pq: Optional[Path] = None
    verified_label_checks: list[dict[str, Any]] = []
    for pq in parquets:
        lf_chk = pl.scan_parquet(str(pq))
        atk_cnt = int(lf_chk.select((pl.col("is_attack") == 1).sum()).collect().item() or 0)
        non_benign_lbl = int(lf_chk.select((pl.col("label") != "BENIGN").sum()).collect().item() or 0)
        non_benign_raw = (
            int(lf_chk.select((pl.col("raw_label") != "BENIGN").sum()).collect().item() or 0)
            if has_raw_label
            else non_benign_lbl
        )
        verified_label_checks.append(
            {
                "file": pq.name,
                "attack_rows": atk_cnt,
                "non_benign_clean_rows": non_benign_lbl,
                "non_benign_raw_rows": non_benign_raw,
            }
        )
        if benign_pq is None and atk_cnt == 0 and non_benign_lbl == 0 and non_benign_raw == 0:
            benign_pq = pq

    if benign_pq is None:
        benign_pq = parquets[0]
        logger.warning(
            "No purely benign day verified; isolating earliest clean BENIGN rows from %s.", benign_pq.name
        )
        filt = (pl.col("is_attack") == 0) & (pl.col("label") == "BENIGN")
        if has_raw_label:
            filt = filt & (pl.col("raw_label") == "BENIGN")
        benign_df = pl.scan_parquet(str(benign_pq)).filter(filt).sort("start_ts").collect()
    else:
        logger.info("Verified earliest 100%% BENIGN period: %s", benign_pq.name)
        benign_df = pl.scan_parquet(str(benign_pq)).sort("start_ts").collect()

    # Double-check fit_df has zero attack or non-benign raw labels
    assert benign_df.filter((pl.col("is_attack") != 0) | (pl.col("label") != "BENIGN")).height == 0
    if has_raw_label:
        assert benign_df.filter(pl.col("raw_label") != "BENIGN").height == 0

    n_benign_total = benign_df.height
    fit_cutoff = max(int(n_benign_total * 0.80), 1)
    fit_window_df = benign_df.head(fit_cutoff)
    calib_df = benign_df.tail(max(n_benign_total - fit_cutoff, 1))
    calib_cutoff_ts = calib_df["start_ts"].min()

    if fit_window_df.height > settings.anomaly_max_train_rows:
        step = max(fit_window_df.height // settings.anomaly_max_train_rows, 1)
        fit_df = fit_window_df.gather_every(step).head(settings.anomaly_max_train_rows)
    else:
        fit_df = fit_window_df
    discarded_excess_fit_rows = int(fit_window_df.height - fit_df.height)

    X_fit = sanitize_feature_matrix(fit_df.select(feature_cols).to_numpy())
    imputer = SimpleImputer(strategy="median")
    X_fit_imp = imputer.fit_transform(X_fit)

    iso = IsolationForest(
        n_estimators=150,
        max_samples=min(256, len(X_fit_imp)),
        contamination="auto",
        n_jobs=2,
        random_state=seed,
    )
    iso.fit(X_fit_imp)

    # Threshold calibrated strictly on held-out benign calibration tail (no test labels used)
    X_calib_imp = imputer.transform(sanitize_feature_matrix(calib_df.select(feature_cols).to_numpy()))
    calib_scores = (-iso.score_samples(X_calib_imp)).astype(np.float32)
    threshold = float(np.percentile(calib_scores, threshold_percentile))
    threshold_p99 = float(np.percentile(calib_scores, 99.0))

    joblib.dump(
        {
            "imputer": imputer,
            "model": iso,
            "feature_columns": feature_cols,
            "threshold": threshold,
            "threshold_percentile": threshold_percentile,
            "fit_source": benign_pq.name,
            "fit_end_before_ts": str(calib_cutoff_ts),
        },
        settings.models_dir / "isolation_forest.joblib",
    )

    del benign_df, fit_window_df, fit_df, calib_df, X_fit, X_fit_imp, X_calib_imp
    gc.collect()

    # Score all days in batches; compute out-of-time deployment evaluation STRICTLY on later days only!
    later_scores: list[np.ndarray] = []
    later_is_attack: list[np.ndarray] = []
    per_day_stats: list[dict[str, Any]] = []
    oot_family_totals: dict[str, int] = {}
    oot_family_flagged: dict[str, int] = {}
    oot_family_score_sums: dict[str, float] = {}
    discarded_in_sample_eval_rows = 0

    batch_size = settings.anomaly_batch_size
    for pq in parquets:
        df_day = pl.read_parquet(pq, columns=["day", "start_ts", "end_ts", "label", "is_attack"] + feature_cols)
        day_name = str(df_day["day"][0]) if df_day.height else pq.stem
        n_rows = df_day.height
        day_scores = np.empty(n_rows, dtype=np.float32)

        for start in range(0, n_rows, batch_size):
            end = min(start + batch_size, n_rows)
            X_batch = sanitize_feature_matrix(df_day.slice(start, end - start).select(feature_cols).to_numpy())
            X_batch_imp = imputer.transform(X_batch)
            day_scores[start:end] = (-iso.score_samples(X_batch_imp)).astype(np.float32)

        day_flags = (day_scores >= threshold).astype(np.int8)
        day_is_atk = df_day["is_attack"].to_numpy()
        day_labels = df_day["label"].to_list()

        # Persist per-day scores (downstream Stage 3 uses Tue-Fri strictly out-of-time; Monday kept only as fit baseline day)
        score_df = pl.DataFrame(
            {
                "day": df_day["day"],
                "start_ts": df_day["start_ts"],
                "end_ts": df_day["end_ts"],
                "label": df_day["label"],
                "is_attack": df_day["is_attack"],
                "anomaly_score": pl.Series("anomaly_score", day_scores, dtype=pl.Float32),
                "is_anomalous": pl.Series("is_anomalous", day_flags, dtype=pl.Int8),
            }
        )
        score_df.write_parquet(settings.data_processed_dir / f"anomaly_scores_{day_name}.parquet", compression="zstd")

        is_fit_period = pq == benign_pq
        if is_fit_period:
            # Never mix fitting/calibration-period observations into deployment-time evaluation metrics
            discarded_in_sample_eval_rows += n_rows
        else:
            later_scores.append(day_scores)
            later_is_attack.append(day_is_atk)
            for lbl, sc, flg in zip(day_labels, day_scores.tolist(), day_flags.tolist()):
                oot_family_totals[lbl] = oot_family_totals.get(lbl, 0) + 1
                oot_family_flagged[lbl] = oot_family_flagged.get(lbl, 0) + int(flg)
                oot_family_score_sums[lbl] = oot_family_score_sums.get(lbl, 0.0) + float(sc)

        day_roc = float(roc_auc_score(day_is_atk, day_scores)) if (not is_fit_period and len(np.unique(day_is_atk)) > 1) else None
        day_pr = float(average_precision_score(day_is_atk, day_scores)) if (not is_fit_period and len(np.unique(day_is_atk)) > 1) else None
        per_day_stats.append(
            {
                "day": day_name,
                "is_fitting_or_calibration_period": is_fit_period,
                "included_in_out_of_time_eval": not is_fit_period,
                "flows": int(n_rows),
                "attack_flows": int(day_is_atk.sum()),
                "anomalous_rate": float(day_flags.mean()),
                "mean_anomaly_score": float(day_scores.mean()),
                "roc_auc": day_roc,
                "pr_auc": day_pr,
            }
        )
        del df_day, score_df
        gc.collect()

    all_later_scores = np.concatenate(later_scores) if later_scores else np.empty(0, dtype=np.float32)
    all_later_y = np.concatenate(later_is_attack) if later_is_attack else np.empty(0, dtype=np.int8)

    overall_roc = float(roc_auc_score(all_later_y, all_later_scores)) if len(np.unique(all_later_y)) > 1 else None
    overall_pr = float(average_precision_score(all_later_y, all_later_scores)) if len(np.unique(all_later_y)) > 1 else None

    per_family_rows = []
    for lbl in sorted(oot_family_totals.keys()):
        tot = oot_family_totals[lbl]
        flg = oot_family_flagged[lbl]
        per_family_rows.append(
            {
                "label": lbl,
                "total_out_of_time_flows": tot,
                "flagged_anomalous": flg,
                "detection_rate": float(flg / max(tot, 1)),
                "mean_anomaly_score": float(oot_family_score_sums[lbl] / max(tot, 1)),
            }
        )
    pl.DataFrame(per_family_rows).write_csv(settings.results_dir / "anomaly_per_family.csv")

    holdout_highlight: Optional[dict[str, Any]] = None
    cls_metrics_path = settings.results_dir / "classification_metrics.json"
    if cls_metrics_path.exists():
        cls_data = json.loads(cls_metrics_path.read_text(encoding="utf-8"))
        held = cls_data.get("holdout_family_experiment", {}).get("held_out_family")
        if held:
            match = [r for r in per_family_rows if r["label"] == held]
            if match:
                holdout_highlight = match[0]

    report = {
        "fit_period": benign_pq.name,
        "verified_label_checks": verified_label_checks,
        "fit_period_total_rows": int(n_benign_total),
        "fit_rows_used": int(min(fit_cutoff, settings.anomaly_max_train_rows)),
        "discarded_excess_fit_rows": discarded_excess_fit_rows,
        "calibration_rows_benign_tail": int(max(n_benign_total - fit_cutoff, 1)),
        "discarded_from_out_of_time_eval_rows": discarded_in_sample_eval_rows,
        "threshold_percentile": threshold_percentile,
        "threshold": threshold,
        "threshold_p99_reference": threshold_p99,
        "note": (
            "Anomaly scores measure deviation from verified BENIGN baseline; NOT a probability of attack. "
            "Fitting and calibration rows are strictly excluded from out-of-time deployment metrics."
        ),
        "out_of_time_later_days_flows": int(len(all_later_y)),
        "out_of_time_attack_prevalence": float(all_later_y.mean()) if len(all_later_y) else 0.0,
        "out_of_time_roc_auc": overall_roc,
        "out_of_time_pr_auc": overall_pr,
        "holdout_family_anomaly_performance": holdout_highlight,
        "per_day": per_day_stats,
        "per_family_out_of_time_only": per_family_rows,
    }
    (settings.results_dir / "anomaly_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
