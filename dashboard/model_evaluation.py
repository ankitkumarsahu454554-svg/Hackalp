"""Page 5: Scientific Model Evaluation, Verdict, Bootstrap CIs, K_MIN Sensitivity, Provenance, and Limitations."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config.settings import Settings
from dashboard.ui_utils import check_pipeline_ready, load_csv_safe, load_json_safe


def render(cfg: Settings) -> None:
    st.title("Model Evaluation & Scientific Audit")
    if not check_pipeline_ready(cfg):
        return

    verdict = load_json_safe(cfg.results_dir / "verdict.json") or {}
    fc_metrics = load_json_safe(cfg.results_dir / "forecast_metrics.json") or {}
    cls_metrics = load_json_safe(cfg.results_dir / "classification_metrics.json") or {}
    anom_metrics = load_json_safe(cfg.results_dir / "anomaly_metrics.json") or {}
    audit_json = load_json_safe(cfg.results_dir / "data_audit.json") or {}
    manifest = load_json_safe(cfg.manifest_path) or {}

    # 1. Automated Verdict verbatim
    v_label = verdict.get("verdict", "UNKNOWN")
    if v_label == "OUTPERFORMS":
        st.success(f"### Automated Verdict: {v_label}\n{verdict.get('rationale', '')}")
    elif v_label.startswith("INCONCLUSIVE"):
        st.warning(f"### Automated Verdict: {v_label}\n{verdict.get('rationale', '')}")
    else:
        st.info(f"### Automated Verdict: {v_label}\n{verdict.get('rationale', '')}")

    # 2. Required Limitations Banner / Panel (always visible)
    split_onsets = verdict.get("split_onsets", {})
    total_onsets = audit_json.get("total_onset_events", "N/A")
    test_onsets = split_onsets.get("test_onsets", "N/A")
    st.warning(
        "**Key Scientific Limitations**:\n"
        "- CICIDS2017 is a controlled, scripted historical benchmark.\n"
        f"- Only **{total_onsets}** dataset-defined attack onsets were identified (`K_MIN={cfg.k_min}`, `QUIET_GAP_MIN={cfg.quiet_gap_min}`).\n"
        f"- Only **{test_onsets}** onsets were present in the reported test partition.\n"
        "- Results do not establish generalization to real production networks.\n"
        f"- The current forecasting verdict is **`{v_label}`**."
    )

    # 3. Forecasting Comparison Table (B0-B3 vs XGBoost)
    st.subheader(f"Attack-Onset Forecasting (H = {fc_metrics.get('primary_horizon_min', cfg.horizons[0])} min)")
    st.caption(
        "Event matching is strict 1-to-1 chronological (multiple alert episodes cannot claim credit for a single onset). "
        "False alerts/hour are computed over eligible decision minutes only."
    )
    fc_df = load_csv_safe(cfg.results_dir / "forecast_metrics.csv")
    if fc_df is not None:
        st.dataframe(fc_df.to_pandas(), width="stretch", hide_index=True)

    seed_info = fc_metrics.get("xgboost_seed_metrics", {})
    if seed_info:
        st.caption(
            f"XGBoost statistical breakdown across seeds `{seed_info.get('seeds')}`: "
            f"**5-seed calibrated ensemble Test PR-AUC** = `{seed_info.get('ensemble_5seed_test_pr_auc', 0):.4f}` | "
            f"**Single-seed calibrated Test PR-AUC (mean ± SD)** = "
            f"`{seed_info.get('single_seed_test_pr_auc_mean', 0):.4f} ± {seed_info.get('single_seed_test_pr_auc_sd', 0):.4f}`."
        )

    # Paired Bootstrap Table
    paired = fc_metrics.get("bootstrap", {}).get("paired_differences", {})
    if paired:
        st.subheader("Paired Block Bootstrap Differences (XGBoost 5-Seed Ensemble minus Baseline)")
        p_rows = []
        for comp, d in paired.items():
            for metric_name, vals in d.items():
                p_rows.append(
                    {
                        "Comparison": comp,
                        "Metric": metric_name,
                        "Bootstrap Mean Diff": vals.get("mean_diff"),
                        "95% CI Low": vals.get("ci_low"),
                        "95% CI High": vals.get("ci_high"),
                        "Excludes Zero": vals.get("excludes_zero"),
                    }
                )
        st.dataframe(pd.DataFrame(p_rows), width="stretch", hide_index=True)

    # Plots
    plot_cols = st.columns(2)
    h = fc_metrics.get("primary_horizon_min", cfg.horizons[0])
    for idx, (fname, cap) in enumerate(
        [
            (f"pr_curves_h{h}.png", "Precision-Recall Curves vs Prevalence Baseline"),
            (f"calibration_h{h}.png", "Calibration Reliability Curve"),
            (f"lead_time_hist_h{h}.png", "Warning Lead-Time Histogram (1-to-1 Matched)"),
            (f"forecast_comparison_h{h}.png", "PR-AUC & 1-to-1 Event Onset Recall Comparison"),
        ]
    ):
        p = cfg.results_dir / fname
        if p.exists():
            plot_cols[idx % 2].image(str(p), caption=cap, width="stretch")

    # 4. K_MIN Sensitivity Analysis
    k_df = load_csv_safe(cfg.results_dir / "k_min_sensitivity.csv")
    if k_df is not None:
        st.subheader("Sensitivity Analysis: K_MIN ∈ {1, 3, 5}")
        st.dataframe(k_df.to_pandas(), width="stretch", hide_index=True)

    # 5. Onset Counts
    st.subheader("Attack-Onset Counts Per Day")
    onset_df = load_csv_safe(cfg.results_dir / "onset_counts.csv")
    if onset_df is not None:
        st.dataframe(onset_df.to_pandas(), width="stretch", hide_index=True)

    # 6. Detection (Global Chronological Split)
    st.subheader(f"Module A: Intrusion Detection ({cls_metrics.get('split_protocol', 'global chronological')})")
    if cls_metrics.get("unseen_family_limitation_note"):
        st.caption(cls_metrics["unseen_family_limitation_note"])

    with_p = cls_metrics.get("with_dst_port", {}).get("models", {})
    no_p = cls_metrics.get("without_dst_port_ablation", {}).get("models", {})
    cls_rows = []
    for variant, md in (("With Dst Port (83)", with_p), ("Without Dst Port (82)", no_p)):
        for mname, d in md.items():
            te = d.get("test", {})
            bin_d = te.get("binary_detection", {})
            multi_d = te.get("multiclass_classification", {})
            cls_rows.append(
                {
                    "Variant": variant,
                    "Model": mname,
                    "Binary Attack F1": bin_d.get("binary_attack_f1", te.get("binary_attack_f1")),
                    "Binary ROC-AUC": bin_d.get("binary_roc_auc", te.get("binary_roc_auc")),
                    "Binary PR-AUC": bin_d.get("binary_pr_auc", te.get("binary_pr_auc")),
                    "Multiclass Macro-F1 (All)": multi_d.get("macro_f1_all_classes", te.get("macro_f1_all_classes")),
                    "Multiclass Weighted-F1": multi_d.get("weighted_f1", te.get("weighted_f1")),
                    "Unseen Classes in Test": ", ".join(
                        multi_d.get("unseen_in_training_classes", te.get("unseen_in_training_classes", []))
                    ),
                }
            )
    if cls_rows:
        st.dataframe(pd.DataFrame(cls_rows), width="stretch", hide_index=True)

    cm_cols = st.columns(2)
    for idx, mname in enumerate(("random_forest", "xgboost")):
        cm_p = cfg.results_dir / f"classifier_cm_{mname}.png"
        if cm_p.exists():
            cm_cols[idx].image(str(cm_p), caption=f"{mname.upper()} Global Chronological Test Confusion Matrix", width="stretch")

    exp_cls = load_json_safe(cfg.results_dir / "exploratory_intraday_classification.json")
    if exp_cls:
        with st.expander("Exploratory Legacy Comparison: Intra-Day 60/15/25 Split (Not Globally Chronological)"):
            st.warning(exp_cls.get("methodological_warning", "Exploratory reference only."))
            st.json(
                {
                    m: v.get("test", {}).get("binary_pr_auc")
                    for m, v in exp_cls.get("with_dst_port", {}).get("models", {}).items()
                }
            )

    # 7. Anomaly Detection (Strict Out-of-Time)
    st.subheader("Module B: Out-of-Time Isolation Forest Anomaly Detection")
    st.json(
        {
            "fit_period": anom_metrics.get("fit_period"),
            "fit_rows_used": anom_metrics.get("fit_rows_used"),
            "calibration_rows_benign_tail": anom_metrics.get("calibration_rows_benign_tail"),
            "discarded_from_out_of_time_eval_rows": anom_metrics.get("discarded_from_out_of_time_eval_rows"),
            "threshold_percentile": anom_metrics.get("threshold_percentile"),
            "threshold": anom_metrics.get("threshold"),
            "out_of_time_later_days_flows": anom_metrics.get("out_of_time_later_days_flows"),
            "out_of_time_roc_auc": anom_metrics.get("out_of_time_roc_auc"),
            "out_of_time_pr_auc": anom_metrics.get("out_of_time_pr_auc"),
            "holdout_family_anomaly_performance": anom_metrics.get("holdout_family_anomaly_performance"),
        }
    )
    anom_fam = load_csv_safe(cfg.results_dir / "anomaly_per_family.csv")
    if anom_fam is not None:
        st.dataframe(anom_fam.to_pandas(), width="stretch", hide_index=True)

    with st.expander("Data Provenance & Manifest"):
        st.json(manifest)
