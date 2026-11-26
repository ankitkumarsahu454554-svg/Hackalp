"""Page 3: Explainable AI — Separate Forecasting & Detection SHAP Workspaces."""
from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st

from components.ui import (
    COLOR_CRITICAL,
    COLOR_CYAN,
    COLOR_INFO,
    COLOR_WARNING,
    render_alert_card,
    render_empty_state,
    render_metric_grid,
    render_section_header,
    style_plotly_fig,
)
from config.settings import Settings
from dashboard.ui_utils import (
    check_pipeline_ready,
    get_runtime_status_context,
    load_json_safe,
    render_shell_header,
)
from utils.data_loaders import read_saved_replay_alerts


def _plot_global_bar(data: dict[str, Any], title: str, color: str) -> go.Figure:
    feats = data.get("top_features", [])[:15]
    names = [x["feature"] for x in reversed(feats)]
    vals = [x["importance"] for x in reversed(feats)]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker_color=color,
            hovertemplate="%{y}<br>Mean |SHAP|: %{x:.4f}<extra></extra>",
        )
    )
    style_plotly_fig(
        fig,
        title=f"{title} ({data.get('method', 'unknown')}, n={data.get('explained_rows', 0):,})",
        xaxis_title="Mean |SHAP| / Attribution Magnitude",
        height=440,
    )
    fig.update_layout(showlegend=False, margin=dict(l=16, r=20, t=50, b=40))
    return fig


def _render_local_alert_section(cfg: Settings, kind_filter: str, accent_color: str) -> None:
    render_section_header(
        f"Local Alert Explanation ({kind_filter})",
        "Inspect positive and negative SHAP contributions driving an individual prediction or warning.",
    )
    alerts: list[dict[str, Any]] = []
    source_note = "Active Replay Session"
    if "replay_engine" in st.session_state and st.session_state["replay_engine"].alert_manager.alerts:
        alerts = [a for a in st.session_state["replay_engine"].alert_manager.to_dicts() if a["kind"] == kind_filter]
    if not alerts:
        saved = read_saved_replay_alerts(cfg)
        alerts = [a for a in saved if a.get("kind") == kind_filter]
        source_note = "Saved Replay Archive (`results/replay_alerts.json`)"

    if not alerts:
        render_empty_state(
            f"No {kind_filter} Alerts Available",
            "Step the Historical Replay on Command Centre or Temporal Threat Radar to generate alerts for local SHAP inspection.",
        )
        return

    st.caption(f"Explanation Source: {source_note} ({len(alerts)} alerts available)")
    options = [f"{a['alert_id']} | {a['timestamp']} | {a['severity']} | {a.get('attack_category', '-')}" for a in alerts]
    chosen_idx = st.selectbox(
        f"Select {kind_filter} Alert",
        range(len(options)),
        format_func=lambda i: options[i],
        key=f"xai_{kind_filter}",
    )
    alert = alerts[chosen_idx]
    reasons = alert.get("top_5_reasons", [])

    left_col, right_col = st.columns([1.2, 1.0])
    with left_col:
        if reasons:
            names = [r["feature"] for r in reversed(reasons)]
            vals = [float(r["contribution"]) for r in reversed(reasons)]
            bar_colors = [accent_color if v >= 0 else COLOR_INFO for v in vals]
            fig = go.Figure(
                go.Bar(
                    x=vals,
                    y=names,
                    orientation="h",
                    marker_color=bar_colors,
                    hovertemplate="%{y}<br>SHAP Contribution: %{x:+.4f}<extra></extra>",
                )
            )
            style_plotly_fig(
                fig,
                title=f"Top-5 Feature Contributions ({alert['alert_id']})",
                xaxis_title="SHAP Contribution (+ pushes higher / - pushes lower)",
                height=320,
            )
            fig.add_vline(x=0.0, line_color="#64748B", line_width=1)
            fig.update_layout(showlegend=False, margin=dict(l=16, r=20, t=48, b=38))
            st.plotly_chart(fig, width="stretch")
        else:
            render_empty_state("No Local Feature Reasons", "Selected alert did not include local SHAP attributions.")

    with right_col:
        render_alert_card(alert)
        st.markdown("**Plain-English Feature Contribution Breakdown**")
        if reasons:
            for idx, r in enumerate(reasons, 1):
                sign_label = "Raises" if float(r["contribution"]) >= 0 else "Lowers"
                st.markdown(f"{idx}. **{sign_label}**: {r['sentence']}")
        else:
            st.caption("No detailed per-feature breakdown available.")


def render(cfg: Settings) -> None:
    ctx = get_runtime_status_context(cfg)
    render_shell_header(
        cfg,
        title="Explainable AI (SHAP)",
        description="Transparent model attributions for pre-onset forecasting (Module M) and flow-level intrusion detection (Module A), kept strictly separate.",
        ctx=ctx,
    )
    if not check_pipeline_ready(cfg):
        return

    shap_fc = load_json_safe(cfg.results_dir / "shap_forecaster_global.json")
    shap_det = load_json_safe(cfg.results_dir / "shap_detector_global.json")
    if not shap_fc or not shap_det:
        render_empty_state("SHAP Summaries Missing", "Run `python -m src.pipeline --stage explain` to compute global SHAP explanations.")
        return

    st.info(
        "**Model Explanation vs Causal Proof:** SHAP values quantify how input features contributed to the model's statistical output. "
        "Feature attribution explains model behaviour and does **not** establish causal proof of attacker intent or guaranteed future intrusion."
    )

    fc_top1 = (shap_fc.get("top_features") or [{}])[0]
    det_top1 = (shap_det.get("top_features") or [{}])[0]
    render_metric_grid(
        [
            {
                "label": "Forecaster Explainer",
                "value": str(shap_fc.get("method", "unknown")),
                "sub": f"Horizon H={shap_fc.get('horizon_min', cfg.horizons[0])} min",
                "badge": "Module M",
                "tone": "cyan",
            },
            {
                "label": "Top Forecast Driver",
                "value": str(fc_top1.get("feature", "-")),
                "sub": f"Mean |SHAP|: {float(fc_top1.get('importance', 0)):.4f} (n={shap_fc.get('explained_rows', 0):,})",
                "badge": "Global",
                "tone": "violet",
            },
            {
                "label": "Detector Explainer",
                "value": str(shap_det.get("method", "unknown")),
                "sub": "XGBoost flow classifier (83 features)",
                "badge": "Module A",
                "tone": "critical",
            },
            {
                "label": "Top Detection Driver",
                "value": str(det_top1.get("feature", "-")),
                "sub": f"Mean |SHAP|: {float(det_top1.get('importance', 0)):.4f} (n={shap_det.get('explained_rows', 0):,})",
                "badge": "Global",
                "tone": "warning",
            },
        ]
    )

    tab_fc, tab_det = st.tabs(["Forecasting Explainability (Module M)", "Detection Explainability (Module A)"])

    with tab_fc:
        render_section_header("Global Forecasting Feature Importance", "Explains pre-onset calibrated risk drivers using strictly causal rolling traffic and anomaly features.")
        st.plotly_chart(_plot_global_bar(shap_fc, "Global Forecaster Feature Importance", COLOR_CYAN), width="stretch")
        _render_local_alert_section(cfg, kind_filter="FORECAST WARNING", accent_color=COLOR_WARNING)

    with tab_det:
        render_section_header("Global Detection Feature Importance", "Explains flow-level malicious vs benign classification drivers independently from forecasting.")
        st.plotly_chart(_plot_global_bar(shap_det, "Global Detector Feature Importance", COLOR_CRITICAL), width="stretch")
        _render_local_alert_section(cfg, kind_filter="DETECTION", accent_color=COLOR_CRITICAL)
