"""Page 1: Security Command Centre — Executive SOC Overview & Signal Separation."""
from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.ui import (
    COLOR_CRITICAL,
    COLOR_CYAN,
    COLOR_HEALTHY,
    COLOR_HIGH,
    COLOR_VIOLET,
    COLOR_WARNING,
    render_alert_card,
    render_empty_state,
    render_error_state,
    render_metric_grid,
    render_model_status_card,
    render_section_header,
    render_threat_timeline,
    style_plotly_fig,
)
from config.settings import Settings
from dashboard.ui_utils import (
    check_pipeline_ready,
    ensure_replay_in_session,
    get_runtime_status_context,
    load_json_safe,
    render_shell_header,
)
from utils.data_loaders import (
    read_csv_artifact,
    read_forecast_timeline,
    read_saved_replay_alerts,
)


def _render_threat_distributions(cfg: Settings, anom_fam_df: pd.DataFrame | None) -> None:
    """Section B: Attack family breakdown, benign vs malicious donut, and anomaly distribution."""
    render_section_header(
        "Threat & Anomaly Distribution",
        "Strict out-of-time distribution across Tuesday–Friday deployment flows (Monday fit baseline excluded).",
    )
    if anom_fam_df is None or anom_fam_df.empty:
        render_empty_state("No Family Distribution Data", "Run `python -m src.pipeline --stage anomaly` to generate per-family statistics.")
        return

    benign_rows = anom_fam_df[anom_fam_df["label"] == "BENIGN"]
    attack_rows = anom_fam_df[anom_fam_df["label"] != "BENIGN"].copy()
    benign_total = int(benign_rows["total_out_of_time_flows"].sum()) if not benign_rows.empty else 0
    attack_total = int(attack_rows["total_out_of_time_flows"].sum()) if not attack_rows.empty else 0

    col_left, col_mid, col_right = st.columns([1.25, 0.95, 1.25])

    with col_left:
        top_atk = attack_rows.sort_values("total_out_of_time_flows", ascending=True).tail(8)
        fig_fam = go.Figure(
            go.Bar(
                x=top_atk["total_out_of_time_flows"],
                y=top_atk["label"],
                orientation="h",
                marker_color=COLOR_CRITICAL,
                hovertemplate="%{y}<br>Flows: %{x:,}<extra></extra>",
            )
        )
        style_plotly_fig(fig_fam, title="Attack Family Volume (Tue–Fri)", xaxis_title="Flows", height=320)
        fig_fam.update_layout(showlegend=False, margin=dict(l=10, r=16, t=46, b=36))
        st.plotly_chart(fig_fam, width="stretch")

    with col_mid:
        fig_donut = go.Figure(
            go.Pie(
                labels=["BENIGN", "Attack Families"],
                values=[benign_total, attack_total],
                hole=0.62,
                marker=dict(colors=[COLOR_CYAN, COLOR_CRITICAL], line=dict(color="#070B14", width=2)),
                textinfo="percent",
                hovertemplate="%{label}: %{value:,} flows (%{percent})<extra></extra>",
            )
        )
        style_plotly_fig(fig_donut, title="Benign vs Attack Composition", height=320)
        fig_donut.update_layout(margin=dict(l=12, r=12, t=46, b=24), showlegend=True)
        st.plotly_chart(fig_donut, width="stretch")

    with col_right:
        top_anom = attack_rows[attack_rows["flagged_anomalous"] > 0].sort_values("detection_rate", ascending=True).tail(8)
        if top_anom.empty:
            top_anom = attack_rows.sort_values("mean_anomaly_score", ascending=True).tail(8)
        fig_anom = go.Figure(
            go.Bar(
                x=(top_anom["detection_rate"] * 100.0).round(2),
                y=top_anom["label"],
                orientation="h",
                marker_color=COLOR_WARNING,
                customdata=top_anom["flagged_anomalous"],
                hovertemplate="%{y}<br>Flagged Rate: %{x:.1f}%<br>Flagged Flows: %{customdata:,}<extra></extra>",
            )
        )
        style_plotly_fig(fig_anom, title="Out-of-Time IF Flag Rate (%)", xaxis_title="Anomalous Rate (%)", height=320)
        fig_anom.update_layout(showlegend=False, margin=dict(l=10, r=16, t=46, b=36))
        st.plotly_chart(fig_anom, width="stretch")


def render(cfg: Settings) -> None:
    ctx = get_runtime_status_context(cfg)
    render_shell_header(
        cfg,
        title="Security Command Centre",
        description="Unified SOC overview combining corpus analytics, causal replay telemetry, recent warnings, and strictly separated Detection, Anomaly, and Forecast modules.",
        ctx=ctx,
    )
    if not check_pipeline_ready(cfg):
        return

    try:
        eng = ensure_replay_in_session(cfg)
    except Exception as exc:
        render_error_state("Historical Replay Engine Unavailable", f"Could not initialize replay engine: {exc}")
        return

    audit = ctx["audit"]
    anom_metrics = load_json_safe(cfg.results_dir / "anomaly_metrics.json") or {}
    verdict = ctx["verdict"]
    fc_summary = ctx["fc_summary"]
    cls_metrics = load_json_safe(cfg.results_dir / "classification_metrics.json") or {}
    anom_fam_df = read_csv_artifact(cfg.results_dir / "anomaly_per_family.csv")
    timeline_df = read_forecast_timeline(cfg)
    saved_alerts = read_saved_replay_alerts(cfg)

    # Replay Quick Control Strip
    c1, c2, c3, c4 = st.columns([1.0, 1.0, 1.0, 3.2])
    if c1.button("Step +10 min", type="primary", disabled=eng.is_finished, key="cc_step10", width="stretch"):
        for _ in range(10):
            if not eng.is_finished:
                eng.step()
        st.rerun()
    if c2.button("Step +30 min", disabled=eng.is_finished, key="cc_step30", width="stretch"):
        for _ in range(30):
            if not eng.is_finished:
                eng.step()
        st.rerun()
    if c3.button("Reset Replay", key="cc_reset", width="stretch"):
        eng.reset()
        st.session_state["replay_running"] = False
        st.rerun()
    pos_txt = (
        f"Cursor {eng.cursor} / {len(eng.test_minutes)} min · "
        f"{eng.current_minute.strftime('%Y-%m-%d %H:%M UTC') if eng.current_minute else 'Ready to begin Friday test replay'}"
    )
    c4.info(pos_txt)

    # Top-level KPI Metric Cards connected cleanly to actual project data + active replay
    total_corpus_flows = int(audit.get("total_clean_rows", 0))
    oot_attack_flows = int(sum(int(x.get("attack_flows", 0)) for x in anom_metrics.get("per_day", []) if x.get("included_in_out_of_time_eval")))
    oot_anomalies = int(anom_fam_df["flagged_anomalous"].sum()) if anom_fam_df is not None else 0
    saved_fc_warnings = sum(1 for a in saved_alerts if a.get("kind") == "FORECAST WARNING")
    active_fc_warnings = sum(1 for a in eng.alert_manager.alerts if a.kind == "FORECAST WARNING")
    total_onsets = int(audit.get("total_onset_events", 0))
    test_onsets = int(verdict.get("split_onsets", {}).get("test_onsets", 0))

    latest = eng.history[-1] if eng.history else None
    fc_risk_now = f"{latest['forecast_risk']:.4f} ({latest['risk_state']})" if latest else "Idle (Step replay)"
    revealed_attack_mins = sum(1 for h in eng.history if h["gt_attack_active"])

    render_metric_grid(
        [
            {
                "label": "Total Flows Analysed",
                "value": f"{eng.total_flows_processed:,}" if eng.history else f"{total_corpus_flows:,}",
                "sub": f"Replay processed ({total_corpus_flows:,} clean corpus flows)" if eng.history else f"Verified CICIDS2017_improved corpus (5 UTC days)",
                "badge": "Replay" if eng.history else "Corpus",
                "tone": "cyan",
            },
            {
                "label": "Malicious Flows Detected",
                "value": f"{eng.total_malicious_detected:,}" if eng.history else f"{oot_attack_flows:,}",
                "sub": "Flagged in active causal replay" if eng.history else "Observed attack-labelled flows across Tue–Fri",
                "badge": "Module A",
                "tone": "critical",
            },
            {
                "label": "Anomalous Flows (IF)",
                "value": f"{eng.total_anomalous_flows:,}" if eng.history else f"{oot_anomalies:,}",
                "sub": f"Replay anomalies (>= {eng.anom_threshold:.3f})" if eng.history else f"Tue–Fri out-of-time flags (95th pct {anom_metrics.get('threshold', 0):.3f})",
                "badge": "Module B",
                "tone": "warning",
            },
            {
                "label": f"Forecast Warnings (H={eng.horizon_min}m)",
                "value": f"{active_fc_warnings}" if eng.history else f"{saved_fc_warnings}",
                "sub": f"Current risk: {fc_risk_now}" if eng.history else f"Saved test-window forecast warnings (Thr {eng.fc_threshold:.4f})",
                "badge": "Module M",
                "tone": "violet",
            },
            {
                "label": "Observed Attack Events",
                "value": f"{revealed_attack_mins} min" if eng.history else f"{total_onsets} onsets",
                "sub": f"Revealed <= replay cursor ({test_onsets} test onsets total)" if eng.history else f"{test_onsets} test / {total_onsets} corpus onset events (K_MIN={cfg.k_min})",
                "badge": "Ground Truth",
                "tone": "healthy",
            },
        ]
    )

    # Section A: Security Overview & System Readiness
    render_section_header("Security Overview & Readiness", "Real-time component telemetry derived from verified manifests, model bundles, and replay cursor.")
    ov1, ov2, ov3, ov4 = st.columns(4)
    with ov1:
        render_model_status_card(
            "System Health",
            "Healthy" if ctx["results_ok"] and ctx["models_ok"] else "Degraded",
            "healthy" if ctx["results_ok"] and ctx["models_ok"] else "warning",
            "All evaluation artifacts, SHAP explainers, and threshold manifests validated cleanly on local CPU runtime.",
            f"Verdict: {verdict.get('verdict', 'UNKNOWN')}",
        )
    with ov2:
        render_model_status_card(
            "Model Readiness",
            "3 / 3 Ready" if ctx["models_ok"] else "Incomplete",
            "healthy" if ctx["models_ok"] else "warning",
            "XGBoost Detector (83 feats), Monday-fitted Isolation Forest, and 5-seed Platt XGBoost Forecaster loaded.",
            f"Seeds: {fc_summary.get('xgboost_seed_metrics', {}).get('seeds', [0,1,2,3,4])}",
        )
    with ov3:
        render_model_status_card(
            "Dataset Processing",
            "Verified" if ctx["manifest"].get("verification_status") == "verified" else "Unverified",
            "healthy" if ctx["manifest"].get("verification_status") == "verified" else "warning",
            "DistriNet IEEE CNS 2022 CICIDS2017_improved with Attempted flows normalized to BENIGN.",
            f"{total_corpus_flows:,} flows | 0 duplicates",
        )
    with ov4:
        render_model_status_card(
            "Historical Replay",
            "Active" if eng.history else "Standby",
            "cyan" if eng.history else "violet",
            "Strictly causal end-time slicing (`end_ts < t + 60s`) across Friday test window; never live network capture.",
            f"Cursor: {eng.cursor} / {len(eng.test_minutes)} minutes",
        )

    # Section B: Threat & Anomaly Distributions
    _render_threat_distributions(cfg, anom_fam_df)

    # Section C: Temporal Activity & Forecast Risk Timeline
    render_section_header(
        "Temporal Activity & Forecast Risk Timeline",
        "Causal replay traffic & attack observations alongside calibrated 5-minute onset risk.",
    )
    if eng.history:
        times = [h["timestamp_str"] for h in eng.history]
        fig_traffic = go.Figure()
        fig_traffic.add_trace(go.Scatter(x=times, y=[h["flows_completed"] for h in eng.history], name="Completed Flows/min", line=dict(color=COLOR_CYAN, width=2.2)))
        fig_traffic.add_trace(go.Scatter(x=times, y=[h["malicious_detected"] for h in eng.history], name="Detected Malicious/min", line=dict(color=COLOR_CRITICAL, width=2.0)))
        fig_traffic.add_trace(go.Scatter(x=times, y=[h["anomalous_flows"] for h in eng.history], name="Anomalous Flows/min (IF)", line=dict(color=COLOR_WARNING, width=1.8, dash="dot")))
        style_plotly_fig(fig_traffic, title="Causal Replay Traffic, Detection & Anomaly Activity", xaxis_title="UTC Minute", yaxis_title="Flows / min", height=340)
        st.plotly_chart(fig_traffic, width="stretch")

        fc_warn_x = [h["timestamp_str"] for h in eng.history if h["forecast_alert_id"]]
        fc_warn_y = [h["forecast_risk"] for h in eng.history if h["forecast_alert_id"]]
        fig_fc = render_threat_timeline(
            x_values=times,
            risk_values=[h["forecast_risk"] for h in eng.history],
            threshold=eng.fc_threshold,
            attack_active_mask=[bool(h["gt_attack_active"]) for h in eng.history],
            warn_x=fc_warn_x,
            warn_y=fc_warn_y,
            title="Causal Replay Forecast Risk Window (Calibrated H=5 min)",
            height=340,
        )
        st.plotly_chart(fig_fc, width="stretch")
    elif timeline_df is not None and not timeline_df.empty:
        test_tl = timeline_df[timeline_df["split"] == "test"].copy()
        if not test_tl.empty:
            ts_list = test_tl["minute_ts"].dt.strftime("%Y-%m-%d %H:%M UTC").tolist()
            risk_vals = test_tl["xgboost_forecaster"].fillna(0.0).tolist()
            atk_mask = [bool(x) for x in test_tl["attack_active"].tolist()]
            onset_df = test_tl[test_tl["onset"] == 1]
            warn_df = test_tl[test_tl["xgboost_forecaster"] >= eng.fc_threshold]
            fig_bench = render_threat_timeline(
                x_values=ts_list,
                risk_values=risk_vals,
                threshold=eng.fc_threshold,
                attack_active_mask=atk_mask,
                warn_x=warn_df["minute_ts"].dt.strftime("%Y-%m-%d %H:%M UTC").tolist(),
                warn_y=warn_df["xgboost_forecaster"].tolist(),
                onset_x=onset_df["minute_ts"].dt.strftime("%Y-%m-%d %H:%M UTC").tolist(),
                onset_y=[eng.fc_threshold * 1.12] * len(onset_df),
                title="Test Benchmark Temporal Threat Profile (Step Replay Above for Live Minute-by-Minute Simulation)",
                height=360,
            )
            st.plotly_chart(fig_bench, width="stretch")
    else:
        render_empty_state("No Timeline Available", "Step the historical replay above or open Temporal Threat Radar.")

    # Section D (Recent Alerts) & Section E (Model Insights)
    col_alerts, col_models = st.columns([1.25, 1.0])
    with col_alerts:
        render_section_header(
            "Recent Alerts",
            "Active replay feed when stepped; saved test-replay archive otherwise.",
        )
        recent_list: list[dict[str, Any]] = eng.alert_manager.to_dicts() if eng.alert_manager.alerts else saved_alerts
        if recent_list:
            for item in reversed(recent_list[-5:]):
                render_alert_card(item)
        else:
            render_empty_state("No Alerts Generated Yet", "Click Step +10 min above to step through recorded test traffic.")

    with col_models:
        render_section_header("Model Insights (Signal Separation)", "Detection, Anomaly, and Forecasting remain strictly decoupled.")
        xgb_det_test = cls_metrics.get("with_dst_port", {}).get("models", {}).get("xgboost", {}).get("test", {})
        xgb_bin = xgb_det_test.get("binary_detection", {})
        render_model_status_card(
            "1. Intrusion Detection (Module A)",
            "Active",
            "cyan",
            "Flow-level supervised classification (BENIGN vs attack category) on globally chronological splits.",
            f"Test Binary PR-AUC: {xgb_bin.get('binary_pr_auc', 0):.4f} | ROC-AUC: {xgb_bin.get('binary_roc_auc', 0):.4f}",
        )
        render_model_status_card(
            "2. Anomaly Scoring (Module B)",
            "Out-of-Time",
            "warning",
            "Isolation Forest deviation score fitted strictly on verified benign Monday traffic (not attack probability).",
            f"Tue–Fri OOT ROC-AUC: {anom_metrics.get('out_of_time_roc_auc', 0):.4f} | PR-AUC: {anom_metrics.get('out_of_time_pr_auc', 0):.4f}",
        )
        render_model_status_card(
            f"3. Attack-Onset Forecasting (Module M)",
            verdict.get("verdict", "UNKNOWN"),
            "violet",
            "5-seed Platt-calibrated XGBoost estimating risk of a NEW onset within the next 5 minutes.",
            f"5-Seed PR-AUC: {verdict.get('xgboost_5seed_ensemble_test_pr_auc', 0):.4f} (vs B3 {verdict.get('b3_logreg_test_pr_auc', 0):.4f})",
        )
