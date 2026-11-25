"""Stage 4: Forecasting evaluation, 1-to-1 alert-episode matching, block bootstrap, paired CIs, and verdict."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib
import numpy as np
import polars as pl
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


def compute_alert_episode_metrics(
    minute_ts: np.ndarray,
    segment_id: np.ndarray,
    alerts: np.ndarray,
    onset_flags: np.ndarray,
    horizon_min: int,
    merge_gap_min: int,
    observed_minutes: Optional[int] = None,
) -> dict[str, Any]:
    """Evaluate alert episodes with strict ONE-TO-ONE chronological episode-to-onset matching.

    Definitions:
    - Consecutive alert minutes within the same segment whose timestamp gap
      ``<= merge_gap_min`` minutes are merged into one alert episode.
    - An episode ``E`` qualifies for onset ``o`` iff some alert minute ``a in E`` in
      the same segment satisfies ``0 < (o - a) <= horizon_min``.
    - One-to-one chronological matching: each onset matches at most the earliest
      unmatched qualifying episode, and each episode matches at most one onset.
      Multiple episodes cannot claim credit for a single onset.
    - Lead time for a matched onset ``o`` = ``o - min(qualifying a in matched E)``.
    - Unmatched episodes count as false alert episodes.
    - False alerts per hour = ``false_episodes / (observed_minutes / 60.0)`` where
      ``observed_minutes`` is the eligible decision duration.
    """
    n = len(minute_ts)
    obs_mins = int(observed_minutes) if observed_minutes is not None else n
    obs_hours = max(obs_mins / 60.0, 1e-6)

    if n == 0:
        return {
            "total_onsets": 0,
            "warned_onsets": 0,
            "warned_onset_indices": [],
            "event_onset_recall": None,
            "lead_times_min": [],
            "mean_lead_time_min": None,
            "median_lead_time_min": None,
            "total_episodes": 0,
            "hit_episodes": 0,
            "false_episodes": 0,
            "false_alerts_per_hour": 0.0,
        }

    ts_min = minute_ts.astype("datetime64[m]").astype(np.int64)
    alert_idx = np.where(alerts == 1)[0]
    onset_idx = np.where(onset_flags == 1)[0]

    # Merge consecutive alert minutes within same segment with timestamp diff <= merge_gap_min
    episodes: list[list[int]] = []
    for idx in alert_idx:
        if not episodes:
            episodes.append([int(idx)])
            continue
        prev_idx = episodes[-1][-1]
        same_seg = int(segment_id[idx]) == int(segment_id[prev_idx])
        gap_m = int(ts_min[idx] - ts_min[prev_idx])
        if same_seg and gap_m <= merge_gap_min:
            episodes[-1].append(int(idx))
        else:
            episodes.append([int(idx)])

    # Strict 1-to-1 greedy chronological matching (earliest unmatched qualifying episode per onset)
    matched_episodes: set[int] = set()
    warned_onset_indices: list[int] = []
    lead_times: list[float] = []

    for oi in onset_idx:
        o_seg = int(segment_id[oi])
        o_m = int(ts_min[oi])
        for ep_i, ep in enumerate(episodes):
            if ep_i in matched_episodes:
                continue
            qualifying_a = [
                int(ai)
                for ai in ep
                if int(segment_id[ai]) == o_seg and 0 < (o_m - int(ts_min[ai])) <= horizon_min
            ]
            if qualifying_a:
                matched_episodes.add(ep_i)
                warned_onset_indices.append(int(oi))
                first_a_m = min(int(ts_min[ai]) for ai in qualifying_a)
                lead_times.append(float(o_m - first_a_m))
                break

    total_onsets = int(len(onset_idx))
    warned_count = int(len(warned_onset_indices))
    hit_episodes = int(len(matched_episodes))
    false_episodes = int(len(episodes) - hit_episodes)
    event_recall = float(warned_count / total_onsets) if total_onsets > 0 else None

    return {
        "total_onsets": total_onsets,
        "warned_onsets": warned_count,
        "warned_onset_indices": warned_onset_indices,
        "event_onset_recall": event_recall,
        "lead_times_min": lead_times,
        "mean_lead_time_min": float(np.mean(lead_times)) if lead_times else None,
        "median_lead_time_min": float(np.median(lead_times)) if lead_times else None,
        "total_episodes": len(episodes),
        "hit_episodes": hit_episodes,
        "false_episodes": false_episodes,
        "false_alerts_per_hour": float(false_episodes / obs_hours),
    }


def evaluate_model_predictions(
    timeline_df: pl.DataFrame,
    probs_eligible: np.ndarray,
    threshold: float,
    horizon_min: int,
    merge_gap_min: int,
) -> dict[str, Any]:
    """Compute minute-level and 1-to-1 event-level metrics using eligible decision duration."""
    elig_mask = (timeline_df["is_eligible"] == 1).to_numpy()
    elig_df = timeline_df.filter(pl.col("is_eligible") == 1)
    y_true = elig_df[f"y_{horizon_min}"].to_numpy().astype(int)

    probs_clipped = np.clip(probs_eligible.astype(np.float64), 0.0, 1.0)
    preds = (probs_clipped >= threshold).astype(int)

    alerts_full = np.zeros(timeline_df.height, dtype=np.int8)
    alerts_full[elig_mask] = preds.astype(np.int8)

    # Use eligible observation minutes (where alerts can actually be emitted) for false alerts/hr
    eligible_minutes = int(elig_df.height)
    ep_metrics = compute_alert_episode_metrics(
        minute_ts=timeline_df["minute_ts"].to_numpy(),
        segment_id=timeline_df["segment_id"].to_numpy(),
        alerts=alerts_full,
        onset_flags=timeline_df["onset"].to_numpy(),
        horizon_min=horizon_min,
        merge_gap_min=merge_gap_min,
        observed_minutes=eligible_minutes,
    )

    prevalence = float(np.mean(y_true)) if len(y_true) else 0.0
    has_both = len(np.unique(y_true)) > 1
    pr_auc = float(average_precision_score(y_true, probs_clipped)) if has_both else None
    roc_auc = float(roc_auc_score(y_true, probs_clipped)) if has_both else None
    brier = float(brier_score_loss(y_true, probs_clipped)) if len(y_true) else None
    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))

    return {
        "n_minutes_wall_clock": int(timeline_df.height),
        "n_eligible_rows": eligible_minutes,
        "eligible_hours": float(eligible_minutes / 60.0),
        "n_positive_rows": int(np.sum(y_true)),
        "prevalence": prevalence,
        "threshold": float(threshold),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier_score": brier,
        "precision": prec,
        "recall": rec,
        **ep_metrics,
    }


def _block_indices_for_timeline(
    minute_ts: np.ndarray,
    block_minutes: int,
) -> list[np.ndarray]:
    """Partition timeline indices into contiguous ~block_minutes chronological blocks."""
    if len(minute_ts) == 0:
        return []
    ts_m = minute_ts.astype("datetime64[m]").astype(np.int64)
    block_ids = (ts_m - ts_m[0]) // max(int(block_minutes), 1)
    blocks: list[np.ndarray] = []
    for bid in np.unique(block_ids):
        idxs = np.where(block_ids == bid)[0]
        if len(idxs) > 0:
            blocks.append(idxs)
    return blocks


def compute_block_bootstrap(
    test_timeline_df: pl.DataFrame,
    model_probs_eligible: dict[str, np.ndarray],
    model_thresholds: dict[str, float],
    horizon_min: int,
    merge_gap_min: int,
    block_minutes: int = 30,
    reps: int = 500,
    seed: int = 0,
) -> dict[str, Any]:
    """Run block bootstrap (30-min blocks) using 1-to-1 matched onset flags and paired differences vs B2 & B3."""
    elig_mask = (test_timeline_df["is_eligible"] == 1).to_numpy()
    ts_all = test_timeline_df["minute_ts"].to_numpy()
    seg_all = test_timeline_df["segment_id"].to_numpy()
    onset_all = test_timeline_df["onset"].to_numpy()
    y_all = test_timeline_df[f"y_{horizon_min}"].to_numpy().astype(int)

    blocks = _block_indices_for_timeline(ts_all, block_minutes)
    n_blocks = len(blocks)
    if n_blocks == 0:
        return {"error": "Empty test timeline; bootstrap skipped."}

    rng = np.random.default_rng(seed)

    full_probs: dict[str, np.ndarray] = {}
    onset_warned_by_model: dict[str, np.ndarray] = {}

    for m_name, elig_p in model_probs_eligible.items():
        fp = np.full(len(ts_all), np.nan, dtype=np.float64)
        fp[elig_mask] = elig_p
        fa = np.zeros(len(ts_all), dtype=np.int8)
        fa[elig_mask] = (elig_p >= model_thresholds[m_name]).astype(np.int8)
        full_probs[m_name] = fp

        # Use the exact 1-to-1 matched warned onset indices
        ep_res = compute_alert_episode_metrics(
            minute_ts=ts_all,
            segment_id=seg_all,
            alerts=fa,
            onset_flags=onset_all,
            horizon_min=horizon_min,
            merge_gap_min=merge_gap_min,
            observed_minutes=int(np.sum(elig_mask)),
        )
        warned_arr = np.zeros(len(ts_all), dtype=np.int8)
        for oi in ep_res["warned_onset_indices"]:
            warned_arr[oi] = 1
        onset_warned_by_model[m_name] = warned_arr

    boot_stats: dict[str, dict[str, list[float]]] = {
        m: {"pr_auc": [], "brier_score": [], "event_onset_recall": []} for m in model_probs_eligible
    }

    for _ in range(reps):
        drawn_blocks = rng.integers(0, n_blocks, size=n_blocks)
        idx_cat = np.concatenate([blocks[b] for b in drawn_blocks])
        elig_sub = elig_mask[idx_cat]
        y_sub = y_all[idx_cat][elig_sub]
        has_both = len(np.unique(y_sub)) > 1
        onsets_sub = onset_all[idx_cat]
        n_onsets_sub = int(np.sum(onsets_sub))

        for m_name in model_probs_eligible:
            p_sub = full_probs[m_name][idx_cat][elig_sub]
            if has_both and len(y_sub) > 0:
                boot_stats[m_name]["pr_auc"].append(float(average_precision_score(y_sub, p_sub)))
            else:
                boot_stats[m_name]["pr_auc"].append(float("nan"))

            if len(y_sub) > 0:
                boot_stats[m_name]["brier_score"].append(float(brier_score_loss(y_sub, p_sub)))
            else:
                boot_stats[m_name]["brier_score"].append(float("nan"))

            if n_onsets_sub > 0:
                w_sub = int(np.sum(onset_warned_by_model[m_name][idx_cat]))
                boot_stats[m_name]["event_onset_recall"].append(float(w_sub / n_onsets_sub))
            else:
                boot_stats[m_name]["event_onset_recall"].append(float("nan"))

    def _summarize_ci(vals: list[float]) -> dict[str, Any]:
        arr = np.array(vals, dtype=np.float64)
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            return {"bootstrap_mean": None, "ci_low": None, "ci_high": None, "valid_reps": 0}
        return {
            "bootstrap_mean": float(np.mean(valid)),
            "ci_low": float(np.percentile(valid, 2.5)),
            "ci_high": float(np.percentile(valid, 97.5)),
            "valid_reps": int(len(valid)),
        }

    model_cis: dict[str, Any] = {}
    for m_name, metrics_dict in boot_stats.items():
        model_cis[m_name] = {k: _summarize_ci(v) for k, v in metrics_dict.items()}

    paired_diffs: dict[str, Any] = {}
    target_model = "xgboost_forecaster"
    if target_model in boot_stats:
        for base in ("B0_base_rate", "B2_anomaly_persistence", "B3_logistic_regression"):
            if base not in boot_stats:
                continue
            diff_entry: dict[str, Any] = {}
            for metric_name in ("pr_auc", "brier_score", "event_onset_recall"):
                a = np.array(boot_stats[target_model][metric_name], dtype=np.float64)
                b = np.array(boot_stats[base][metric_name], dtype=np.float64)
                mask = np.isfinite(a) & np.isfinite(b)
                if np.sum(mask) == 0:
                    diff_entry[metric_name] = {
                        "mean_diff": None,
                        "ci_low": None,
                        "ci_high": None,
                        "excludes_zero": False,
                        "valid_reps": 0,
                    }
                    continue
                d = a[mask] - b[mask]
                lo = float(np.percentile(d, 2.5))
                hi = float(np.percentile(d, 97.5))
                diff_entry[metric_name] = {
                    "mean_diff": float(np.mean(d)),
                    "ci_low": lo,
                    "ci_high": hi,
                    "excludes_zero": bool(lo > 0.0 or hi < 0.0),
                    "valid_reps": int(np.sum(mask)),
                }
            paired_diffs[f"{target_model}_minus_{base}"] = diff_entry

    return {
        "block_minutes": block_minutes,
        "n_blocks": n_blocks,
        "bootstrap_reps": reps,
        "model_cis": model_cis,
        "paired_differences": paired_diffs,
    }


def determine_verdict(
    split_counts: dict[str, int],
    cfg: Settings,
    bootstrap_res: dict[str, Any],
    test_metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compute automatic verdict: OUTPERFORMS / NOT DISTINGUISHABLE / UNDERPERFORMS / INCONCLUSIVE."""
    insufficient = (
        split_counts.get("train_onsets", 0) < cfg.min_train_onsets
        or split_counts.get("val_onsets", 0) < cfg.min_val_onsets
        or split_counts.get("test_onsets", 0) < cfg.min_test_onsets
    )
    if insufficient:
        verdict = "INCONCLUSIVE: insufficient events"
        rationale = (
            f"Onset counts (train={split_counts.get('train_onsets')}, "
            f"val={split_counts.get('val_onsets')}, test={split_counts.get('test_onsets')}) "
            f"do not meet required minimums ({cfg.min_train_onsets}/{cfg.min_val_onsets}/{cfg.min_test_onsets})."
        )
    else:
        paired = bootstrap_res.get("paired_differences", {})
        d_b2 = paired.get("xgboost_forecaster_minus_B2_anomaly_persistence", {}).get("pr_auc", {})
        d_b3 = paired.get("xgboost_forecaster_minus_B3_logistic_regression", {}).get("pr_auc", {})
        b2_lo, b2_hi = d_b2.get("ci_low"), d_b2.get("ci_high")
        b3_lo, b3_hi = d_b3.get("ci_low"), d_b3.get("ci_high")

        if b2_lo is not None and b3_lo is not None and b2_lo > 0.0 and b3_lo > 0.0:
            verdict = "OUTPERFORMS"
            rationale = (
                f"XGBoost forecaster PR-AUC 95% paired bootstrap CIs are strictly positive vs both "
                f"B2 ([{b2_lo:+.4f}, {b2_hi:+.4f}]) and B3 ([{b3_lo:+.4f}, {b3_hi:+.4f}])."
            )
        elif (b2_hi is not None and b2_hi < 0.0) or (b3_hi is not None and b3_hi < 0.0):
            verdict = "UNDERPERFORMS"
            rationale = (
                f"XGBoost forecaster PR-AUC 95% paired bootstrap CI falls strictly below zero vs at least one "
                f"deployable baseline (vs B2: [{b2_lo}, {b2_hi}], vs B3: [{b3_lo}, {b3_hi}])."
            )
        else:
            verdict = "NOT DISTINGUISHABLE"
            rationale = (
                f"XGBoost forecaster PR-AUC 95% paired bootstrap intervals overlap zero against deployable "
                f"baselines (vs B2: [{b2_lo}, {b2_hi}], vs B3: [{b3_lo}, {b3_hi}])."
            )

    oracle_ref = test_metrics.get("B1_oracle_persistence", {})
    return {
        "verdict": verdict,
        "rationale": rationale,
        "split_onsets": split_counts,
        "xgboost_5seed_ensemble_test_pr_auc": test_metrics.get("xgboost_forecaster", {}).get("pr_auc"),
        "b2_anomaly_test_pr_auc": test_metrics.get("B2_anomaly_persistence", {}).get("pr_auc"),
        "b3_logreg_test_pr_auc": test_metrics.get("B3_logistic_regression", {}).get("pr_auc"),
        "b0_baserate_test_pr_auc": test_metrics.get("B0_base_rate", {}).get("pr_auc"),
        "b1_oracle_reference_not_deployable": {
            "pr_auc": oracle_ref.get("pr_auc"),
            "brier_score": oracle_ref.get("brier_score"),
            "event_onset_recall": oracle_ref.get("event_onset_recall"),
            "note": "ORACLE REFERENCE — NOT DEPLOYABLE (uses past true labels).",
        },
    }


def _generate_evaluation_plots(
    preds_df: pl.DataFrame,
    test_metrics: dict[str, dict[str, Any]],
    calibration_info: dict[str, Any],
    horizon_min: int,
    out_dir: Path,
) -> None:
    """Save PR curves, calibration curves (calibrated models only), lead-time histogram, and comparison chart."""
    test_elig = preds_df.filter((pl.col("split") == "test") & (pl.col("is_eligible") == 1))
    y_true = test_elig[f"y_{horizon_min}"].to_numpy().astype(int)
    prevalence = float(np.mean(y_true)) if len(y_true) else 0.0

    model_cols = [
        ("B0_base_rate", "B0 Base Rate", "--"),
        ("B1_oracle_persistence", "B1 Oracle (Not Deployable)", ":"),
        ("B2_anomaly_persistence", "B2 Anomaly Persistence", "-."),
        ("B3_logistic_regression", "B3 Logistic Regression", "-"),
        ("xgboost_forecaster", "M XGBoost Forecaster (5-seed ensemble)", "-"),
    ]

    # 1. PR Curves
    fig, ax = plt.subplots(figsize=(8.2, 5.5))
    for col, label, ls in model_cols:
        if col not in test_elig.columns:
            continue
        scores = test_elig[col].to_numpy().astype(float)
        if len(np.unique(y_true)) > 1:
            p_arr, r_arr, _ = precision_recall_curve(y_true, scores)
            ap = average_precision_score(y_true, scores)
            cal_tag = "" if calibration_info.get(col, {}).get("calibrated", True) or col == "B0_base_rate" else " [Uncalibrated]"
            ax.plot(r_arr, p_arr, linestyle=ls, linewidth=2, label=f"{label}{cal_tag} (PR-AUC={ap:.3f})")
    ax.axhline(prevalence, color="gray", linestyle="--", linewidth=1.2, label=f"Prevalence baseline ({prevalence:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Test Precision-Recall Curves (Horizon H={horizon_min} min)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"pr_curves_h{horizon_min}.png", dpi=150)
    plt.close(fig)

    # 2. Calibration Curve (explicitly flag uncalibrated models)
    if np.sum(y_true) >= 5:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        for col, label, ls in model_cols[1:]:
            if col not in test_elig.columns:
                continue
            is_cal = bool(calibration_info.get(col, {}).get("calibrated", False))
            tag = "Platt-calibrated" if is_cal else "Uncalibrated score"
            scores = np.clip(test_elig[col].to_numpy().astype(float), 0.0, 1.0)
            prob_true, prob_pred = calibration_curve(y_true, scores, n_bins=5, strategy="quantile")
            ax.plot(prob_pred, prob_true, marker="o", linestyle=ls, label=f"{label} ({tag})")
        ax.set_xlabel("Mean Predicted Score / Probability")
        ax.set_ylabel("Observed Positive Fraction")
        ax.set_title(f"Test Reliability Curve (H={horizon_min} min)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"calibration_h{horizon_min}.png", dpi=150)
        plt.close(fig)

    # 3. Lead-time Histogram
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    any_leads = False
    for m_name, label, _ in model_cols[2:]:
        lts = test_metrics.get(m_name, {}).get("lead_times_min", [])
        if lts:
            any_leads = True
            ax.hist(lts, bins=np.arange(0.5, horizon_min + 1.5, 1.0), alpha=0.55, label=f"{label} (n={len(lts)})")
    ax.set_xlabel("Warning Lead Time Before Onset (minutes)")
    ax.set_ylabel("Warned Onsets Count")
    ax.set_title(f"Test Warning Lead Time Distribution (1-to-1 Matched, H={horizon_min} min)")
    if any_leads:
        ax.legend(loc="best", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No warned onsets at frozen threshold", ha="center", va="center", transform=ax.transAxes)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"lead_time_hist_h{horizon_min}.png", dpi=150)
    plt.close(fig)

    # 4. Comparison bar chart (PR-AUC and Event Recall)
    names = [m for m, _, _ in model_cols if m in test_metrics]
    pr_vals = [float(test_metrics[m].get("pr_auc") or 0.0) for m in names]
    ev_vals = [float(test_metrics[m].get("event_onset_recall") or 0.0) for m in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - 0.18, pr_vals, width=0.35, label="Point-Estimate Test PR-AUC")
    ax.bar(x + 0.18, ev_vals, width=0.35, label="Event Onset Recall (1-to-1)")
    ax.axhline(prevalence, color="red", linestyle="--", linewidth=1, label=f"Prevalence ({prevalence:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Forecasting Model Comparison on Test Split (H={horizon_min} min)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"forecast_comparison_h{horizon_min}.png", dpi=150)
    plt.close(fig)


def run_evaluation(cfg: Optional[Settings] = None) -> dict[str, Any]:
    """Run Stage 4 full statistical evaluation, block bootstrap, plots, and automated verdict."""
    settings = cfg or get_settings()
    settings.ensure_dirs()

    preds_path = settings.results_dir / "forecast_predictions.parquet"
    meta_path = settings.results_dir / "forecast_training_summary.json"
    if not preds_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Missing forecasting outputs in {settings.results_dir}. "
            "Run `python -m src.pipeline --stage forecast` first."
        )

    preds_df = pl.read_parquet(preds_path)
    train_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    primary_h = int(settings.horizons[0])

    test_timeline = preds_df.filter(pl.col("split") == "test").sort("minute_ts")
    test_elig = test_timeline.filter(pl.col("is_eligible") == 1)

    model_names = [
        "B0_base_rate",
        "B1_oracle_persistence",
        "B2_anomaly_persistence",
        "B3_logistic_regression",
        "xgboost_forecaster",
    ]
    thresholds: dict[str, float] = train_meta.get("frozen_thresholds", {})
    calibration_info: dict[str, Any] = train_meta.get("calibration_info", {})
    test_metrics: dict[str, dict[str, Any]] = {}
    model_probs_elig: dict[str, np.ndarray] = {}

    for m_name in model_names:
        if m_name not in test_elig.columns:
            continue
        probs = test_elig[m_name].to_numpy().astype(np.float64)
        tau = float(thresholds.get(m_name, 0.5))
        model_probs_elig[m_name] = probs
        m_eval = evaluate_model_predictions(
            test_timeline,
            probs,
            threshold=tau,
            horizon_min=primary_h,
            merge_gap_min=settings.alert_merge_gap_min,
        )
        cal_meta = calibration_info.get(m_name, {})
        m_eval["is_calibrated"] = bool(cal_meta.get("calibrated", False))
        m_eval["calibration_note"] = cal_meta.get("reason", "Platt sigmoid fitted on validation only.")
        test_metrics[m_name] = m_eval

    bootstrap_res = compute_block_bootstrap(
        test_timeline,
        model_probs_elig,
        thresholds,
        horizon_min=primary_h,
        merge_gap_min=settings.alert_merge_gap_min,
        block_minutes=settings.bootstrap_block_min,
        reps=settings.bootstrap_reps,
        seed=settings.seeds[0],
    )

    verdict = determine_verdict(
        split_counts=train_meta["split_onsets"],
        cfg=settings,
        bootstrap_res=bootstrap_res,
        test_metrics=test_metrics,
    )

    _generate_evaluation_plots(preds_df, test_metrics, calibration_info, primary_h, settings.results_dir)

    seed_info = train_meta.get("xgboost_seed_metrics", {})
    csv_rows = []
    for m_name, metrics in test_metrics.items():
        cis = bootstrap_res.get("model_cis", {}).get(m_name, {})
        csv_rows.append(
            {
                "model": m_name,
                "is_deployable": m_name != "B1_oracle_persistence",
                "is_calibrated": metrics["is_calibrated"],
                "estimator_type": "5-seed_ensemble" if m_name == "xgboost_forecaster" else "single_estimator",
                "horizon_min": primary_h,
                "threshold": metrics["threshold"],
                "eligible_minutes": metrics["n_eligible_rows"],
                "prevalence": metrics["prevalence"],
                "pr_auc": metrics["pr_auc"],
                "bootstrap_pr_auc_mean": cis.get("pr_auc", {}).get("bootstrap_mean"),
                "pr_auc_ci_low": cis.get("pr_auc", {}).get("ci_low"),
                "pr_auc_ci_high": cis.get("pr_auc", {}).get("ci_high"),
                "single_seed_pr_auc_mean": seed_info.get("single_seed_test_pr_auc_mean") if m_name == "xgboost_forecaster" else None,
                "single_seed_pr_auc_sd": seed_info.get("single_seed_test_pr_auc_sd") if m_name == "xgboost_forecaster" else None,
                "roc_auc": metrics["roc_auc"],
                "brier_score": metrics["brier_score"],
                "brier_ci_low": cis.get("brier_score", {}).get("ci_low"),
                "brier_ci_high": cis.get("brier_score", {}).get("ci_high"),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "event_onset_recall": metrics["event_onset_recall"],
                "event_recall_ci_low": cis.get("event_onset_recall", {}).get("ci_low"),
                "event_recall_ci_high": cis.get("event_onset_recall", {}).get("ci_high"),
                "warned_onsets": metrics["warned_onsets"],
                "total_onsets": metrics["total_onsets"],
                "mean_lead_time_min": metrics["mean_lead_time_min"],
                "median_lead_time_min": metrics["median_lead_time_min"],
                "hit_episodes": metrics["hit_episodes"],
                "false_episodes": metrics["false_episodes"],
                "false_alerts_per_hour": metrics["false_alerts_per_hour"],
            }
        )
    pl.DataFrame(csv_rows).write_csv(settings.results_dir / "forecast_metrics.csv")

    full_eval = {
        "primary_horizon_min": primary_h,
        "split_info": train_meta.get("split_info"),
        "split_onsets": train_meta.get("split_onsets"),
        "calibration_info": calibration_info,
        "xgboost_seed_metrics": seed_info,
        "walk_forward_secondary": train_meta.get("walk_forward_secondary"),
        "test_metrics": test_metrics,
        "bootstrap": bootstrap_res,
        "verdict": verdict,
    }
    (settings.results_dir / "forecast_metrics.json").write_text(json.dumps(full_eval, indent=2), encoding="utf-8")
    (settings.results_dir / "bootstrap_ci.json").write_text(json.dumps(bootstrap_res, indent=2), encoding="utf-8")
    (settings.results_dir / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    return full_eval
