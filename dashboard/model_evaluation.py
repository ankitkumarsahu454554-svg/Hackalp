"""Page 5: Scientific Model Evaluation, Verdict, Bootstrap CIs, K_MIN Sensitivity, and Provenance."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.ui import (
    COLOR_CRITICAL,
    COLOR_CYAN,
    COLOR_WARNING,
    render_metric_grid,
    render_section_header,
    style_plotly_fig,
)
from config.settings import Settings
from dashboard.ui_utils import (
    check_pipeline_ready,
    get_runtime_status_context,
    load_csv_safe,
    load_json_safe,
    render_shell_header,
)


def render(cfg: Settings) -> None:
    ctx = get_runtime_status_context(cfg)
    render_shell_header(
        cfg,
        title="Model Evaluation & Scientific Audit",
        description="Complete reproducible benchmark across Forecasting (H=5m), Intrusion Detection (Module A), Out-of-Time Anomaly Scoring (Module B), uncertainty intervals, and provenance.",
        ctx=ctx,
    )
    if not check_pipeline_ready(cfg):
        return

    verdict = load_json_safe(cfg.results_dir / "verdict.json") or {}
    fc_metrics = load_json_safe(cfg.results_dir / "forecast_metrics.json") or {}
    cls_metrics = load_json_safe(cfg.results_dir / "classification_metrics.json") or {}
    anom_metrics = load_json_safe(cfg.results_dir / "anomaly_metrics.json") or {}
    audit_json = load_json_safe(cfg.results_dir / "data_audit.json") or {}
    manifest = load_json_safe(cfg.manifest_path) or {}

    # 1. Automated Scientific Verdict Verbatim
    v_label = verdict.get("verdict", "UNKNOWN")
    v_cls = "outperforms" if v_label == "OUTPERFORMS" else ""
    st.markdown(
        f'<div class="cs-verdict-box {v_cls}">'
        f'<div style="font-size:0.74rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#F59E0B;">Automated Statistical Verdict</div>'
        f'<div style="font-size:1.35rem;font-weight:700;color:#F8FAFC;margin:0.15rem 0 0.35rem 0;">{v_label}</div>'
        f'<div style="font-size:0.86rem;color:#CBD5E1;line-height:1.45;">{verdict.get("rationale", "")}</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # 2. Required Scientific Limitations Panel (Always Visible)
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

    # Top Forecasting Summary Cards
    fc_df = load_csv_safe(cfg.results_dir / "forecast_metrics.csv")
    seed_info = fc_metrics.get("xgboost_seed_metrics", {})
    xgb_row = None
    if fc_df is not None:
        fc_pd = fc_df.to_pandas()
        match = fc_pd[fc_pd["model"] == "xgboost_forecaster"]
        if not match.empty:
            xgb_row = match.iloc[0]

    render_metric_grid(
        [
            {
                "label": "XGB 5-Seed Test PR-AUC",
                "value": f"{verdict.get('xgboost_5seed_ensemble_test_pr_auc', 0):.4f}",
                "sub": f"Test base rate: {verdict.get('b0_baserate_test_pr_auc', 0):.4f} ({split_onsets.get('test_positive_rows', 0)}/{split_onsets.get('test_eligible_rows', 0)})",
                "badge": "Ensemble",
                "tone": "cyan",
            },
            {
                "label": "Single-Seed Mean ± SD",
                "value": f"{seed_info.get('single_seed_test_pr_auc_mean', 0):.4f} ± {seed_info.get('single_seed_test_pr_auc_sd', 0):.4f}",
                "sub": f"Seeds {seed_info.get('seeds', [0,1,2,3,4])}, individually calibrated",
                "badge": "5 Seeds",
                "tone": "violet",
            },
            {
                "label": "Bootstrap PR-AUC [95% CI]",
                "value": f"{float(xgb_row['bootstrap_pr_auc_mean']):.4f}" if xgb_row is not None else "—",
                "sub": f"95% CI [{float(xgb_row['pr_auc_ci_low']):.3f}, {float(xgb_row['pr_auc_ci_high']):.3f}] (30m blocks)" if xgb_row is not None else "",
                "badge": "500 Reps",
                "tone": "warning",
            },
            {
                "label": "1-to-1 Event Recall",
                "value": f"{float(xgb_row['event_onset_recall']):.3f} ({int(xgb_row['warned_onsets'])}/{int(xgb_row['total_onsets'])})" if xgb_row is not None else "—",
                "sub": f"Mean lead: {float(xgb_row['mean_lead_time_min']):.2f} min" if xgb_row is not None else "",
                "badge": "Strict 1:1",
                "tone": "healthy",
            },
            {
                "label": "False Alerts / Eligible Hr",
                "value": f"{float(xgb_row['false_alerts_per_hour']):.3f}" if xgb_row is not None else "—",
                "sub": f"{int(xgb_row['false_episodes'])} false episodes across {int(xgb_row['eligible_minutes'])} elig. min" if xgb_row is not None else "",
                "badge": "Budget",
                "tone": "warning",
            },
        ]
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Forecasting & Baselines (H=5m)",
            "Intrusion Detection (Module A)",
            "Out-of-Time Anomaly (Module B)",
            "Data Provenance & Audit",
        ]
    )

    with tab1:
        render_section_header(
            f"Model Comparison (H = {fc_metrics.get('primary_horizon_min', cfg.horizons[0])} min)",
            "Strictly chronological test metrics (validation calibration and thresholds frozen). B1_oracle_persistence is NOT deployable.",
        )
        if fc_df is not None:
            st.dataframe(fc_df.to_pandas(), width="stretch", hide_index=True)

        with st.expander("Metric Definitions & Evaluation Protocol"):
            st.markdown(
                "- **PR-AUC**: Area under the Precision–Recall curve; baseline equals test positive prevalence (`0.0955`).\n"
                "- **ROC-AUC**: Area under the Receiver Operating Characteristic curve across all threshold settings.\n"
                "- **Brier Score**: Mean squared error of predicted onset probabilities (lower is better calibrated).\n"
                "- **1-to-1 Event Recall**: Fraction of dataset-defined attack onsets (`6` test onsets) matched greedily 1-to-1 by a prior merged alert episode within `(t, t+5]`.\n"
                "- **False Alerts / Eligible Hr**: Unmatched alert episodes divided by eligible decision hours (`314 / 60 = 5.233 hr`).\n"
                "- **Lead Time (min)**: Minutes between the start of a matched alert episode and its corresponding attack onset."
            )

        paired = fc_metrics.get("bootstrap", {}).get("paired_differences", {})
        if paired:
            render_section_header("Paired Block Bootstrap Differences", "30-minute block resampling (500 reps) comparing XGBoost 5-Seed Ensemble minus baselines.")
            p_rows = []
            for comp, d in paired.items():
                for metric_name, vals in d.items():
                    p_rows.append(
                        {
                            "Comparison": comp,
                            "Metric": metric_name,
                            "Bootstrap Mean Diff": round(float(vals.get("mean_diff", 0)), 5),
                            "95% CI Low": round(float(vals.get("ci_low", 0)), 5),
                            "95% CI High": round(float(vals.get("ci_high", 0)), 5),
                            "Excludes Zero": bool(vals.get("excludes_zero", False)),
                        }
                    )
            st.dataframe(pd.DataFrame(p_rows), width="stretch", hide_index=True)

        render_section_header("Diagnostic Curves & Lead-Time Distributions", "Precision-Recall, Calibration Reliability, Lead-Time Histogram, and Model Comparison plots.")
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

        k_df = load_csv_safe(cfg.results_dir / "k_min_sensitivity.csv")
        if k_df is not None:
            render_section_header("Sensitivity Analysis: K_MIN in {1, 3, 5}", "Checks verdict stability across onset flow-count thresholds.")
            st.dataframe(k_df.to_pandas(), width="stretch", hide_index=True)

        onset_df = load_csv_safe(cfg.results_dir / "onset_counts.csv")
        if onset_df is not None:
            render_section_header("Attack-Onset Counts Per Day", "Observed capture minutes, attack-active minutes, and onset events across CICIDS2017_improved.")
            st.dataframe(onset_df.to_pandas(), width="stretch", hide_index=True)

    with tab2:
        render_section_header(
            f"Intrusion Detection ({cls_metrics.get('split_protocol', 'global chronological')})",
            cls_metrics.get("unseen_family_limitation_note", ""),
        )
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
                        "Binary Attack F1": round(float(bin_d.get("binary_attack_f1", 0)), 4),
                        "Binary ROC-AUC": round(float(bin_d.get("binary_roc_auc", 0)), 4),
                        "Binary PR-AUC": round(float(bin_d.get("binary_pr_auc", 0)), 4),
                        "Multiclass Macro-F1 (All)": round(float(multi_d.get("macro_f1_all_classes", 0)), 4),
                        "Multiclass Weighted-F1": round(float(multi_d.get("weighted_f1", 0)), 4),
                        "Unseen Test Classes": ", ".join(multi_d.get("unseen_in_training_classes", [])),
                    }
                )
        if cls_rows:
            st.dataframe(pd.DataFrame(cls_rows), width="stretch", hide_index=True)

        hold = cls_metrics.get("holdout_family_experiment", {})
        if hold:
            render_metric_grid(
                [
                    {"label": "Hold-One-Family-Out Target", "value": str(hold.get("held_out_family", "-")), "sub": f"{hold.get('held_out_family_eval_flows', 0):,} unseen flows", "badge": "Zero-Day Sim", "tone": "violet"},
                    {"label": "XGB Unseen Detection Rate", "value": f"{float(hold.get('xgb_binary_detection_rate_on_unseen_family', 0))*100:.2f}%", "sub": f"Mean P(Attack): {float(hold.get('xgb_mean_attack_probability_on_unseen_family', 0)):.4f}", "badge": "Supervised", "tone": "warning"},
                    {"label": "IF Unseen Detection Rate", "value": f"{float(anom_metrics.get('holdout_family_anomaly_performance', {}).get('detection_rate', 0))*100:.2f}%", "sub": "Out-of-time Isolation Forest on DoS Hulk", "badge": "Unsupervised", "tone": "healthy"},
                ]
            )

        cm_cols = st.columns(2)
        for idx, mname in enumerate(("random_forest", "xgboost")):
            cm_p = cfg.results_dir / f"classifier_cm_{mname}.png"
            if cm_p.exists():
                cm_cols[idx].image(str(cm_p), caption=f"{mname.upper()} Global Chronological Test Confusion Matrix", width="stretch")

        per_cls_df = load_csv_safe(cfg.results_dir / "classification_per_class.csv")
        if per_cls_df is not None:
            with st.expander("Per-Class Precision / Recall / F1 Breakdown"):
                st.dataframe(per_cls_df.to_pandas(), width="stretch", hide_index=True)

        exp_cls = load_json_safe(cfg.results_dir / "exploratory_intraday_classification.json")
        if exp_cls:
            with st.expander("Exploratory Legacy Reference: Intra-Day 60/15/25 Split (Archived)"):
                st.warning(exp_cls.get("methodological_warning", "Exploratory reference only."))
                exp_rows = []
                for m, v in exp_cls.get("with_dst_port", {}).get("models", {}).items():
                    te = v.get("test", {})
                    exp_rows.append(
                        {
                            "Model": m,
                            "Binary PR-AUC": te.get("binary_pr_auc"),
                            "Binary ROC-AUC": te.get("binary_roc_auc"),
                            "Binary Attack F1": te.get("binary_attack_f1"),
                            "Weighted F1": te.get("weighted_f1"),
                        }
                    )
                if exp_rows:
                    st.dataframe(pd.DataFrame(exp_rows), width="stretch", hide_index=True)

    with tab3:
        render_section_header("Strict Out-of-Time Isolation Forest", anom_metrics.get("note", "Deviation score from verified benign Monday baseline."))
        render_metric_grid(
            [
                {"label": "Monday Benign Fit", "value": f"{int(anom_metrics.get('fit_rows_used', 0)):,}", "sub": f"{int(anom_metrics.get('discarded_from_out_of_time_eval_rows', 0)):,} Monday rows excluded from OOT", "badge": "Verified Benign", "tone": "healthy"},
                {"label": "95th Pct Threshold", "value": f"{float(anom_metrics.get('threshold', 0)):.4f}", "sub": f"Calibration tail: {int(anom_metrics.get('calibration_rows_benign_tail', 0)):,} benign flows", "badge": "p95", "tone": "cyan"},
                {"label": "OOT ROC-AUC (Tue–Fri)", "value": f"{float(anom_metrics.get('out_of_time_roc_auc', 0)):.4f}", "sub": f"{int(anom_metrics.get('out_of_time_later_days_flows', 0)):,} deployment flows", "badge": "OOT", "tone": "violet"},
                {"label": "OOT PR-AUC (Tue–Fri)", "value": f"{float(anom_metrics.get('out_of_time_pr_auc', 0)):.4f}", "sub": f"Attack prevalence: {float(anom_metrics.get('out_of_time_attack_prevalence', 0)):.4f}", "badge": "OOT", "tone": "warning"},
            ]
        )

        oot_days = [d for d in anom_metrics.get("per_day", []) if d.get("included_in_out_of_time_eval")]
        if oot_days:
            day_names = [d["day"].capitalize() for d in oot_days]
            fig_day = go.Figure()
            fig_day.add_trace(go.Bar(x=day_names, y=[d["roc_auc"] for d in oot_days], name="ROC-AUC", marker_color=COLOR_CYAN))
            fig_day.add_trace(go.Bar(x=day_names, y=[d["pr_auc"] for d in oot_days], name="PR-AUC", marker_color=COLOR_WARNING))
            style_plotly_fig(fig_day, title="Out-of-Time Anomaly Performance by Capture Day", xaxis_title="Day", yaxis_title="Score", height=320)
            fig_day.update_layout(barmode="group")
            st.plotly_chart(fig_day, width="stretch")

        anom_fam = load_csv_safe(cfg.results_dir / "anomaly_per_family.csv")
        if anom_fam is not None:
            st.dataframe(anom_fam.to_pandas(), width="stretch", hide_index=True)

    with tab4:
        render_section_header("Dataset Provenance & Integrity Verification", "Cryptographic SHA-256 verification and audit totals.")
        render_metric_grid(
            [
                {"label": "Dataset Version", "value": str(manifest.get("dataset_version", "-")), "sub": str(manifest.get("source_name", "")), "badge": str(manifest.get("verification_status", "-")), "tone": "healthy"},
                {"label": "Clean Audited Flows", "value": f"{int(audit_json.get('total_clean_rows', 0)):,}", "sub": f"Duplicates removed: {int(audit_json.get('total_duplicates_removed', 0)):,}", "badge": "Clean", "tone": "cyan"},
                {"label": "Attempted Relabelled", "value": f"{int(audit_json.get('total_attempted_relabelled', 0)):,}", "sub": "Normalized to BENIGN per CNS 2022", "badge": "Hygiene", "tone": "violet"},
                {"label": "Synthetic Mode", "value": str(manifest.get("synthetic_mode", False)), "sub": f"Retrieved UTC: {str(manifest.get('retrieved_utc', '-'))[:19]}", "badge": "Provenance", "tone": "healthy" if not manifest.get("synthetic_mode") else "critical"},
            ]
        )
        files_list = manifest.get("files", [])
        if files_list:
            st.dataframe(pd.DataFrame(files_list), width="stretch", hide_index=True)
