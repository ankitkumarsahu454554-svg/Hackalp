"""Page 3: Explainable AI (Separate Forecast & Detection SHAP Explanations)."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from config.settings import Settings
from dashboard.ui_utils import check_pipeline_ready, ensure_replay_in_session, load_json_safe


def _plot_global_bar(data: dict, title: str, color: str) -> go.Figure:
    feats = data.get("top_features", [])[:15]
    names = [x["feature"] for x in reversed(feats)]
    vals = [x["importance"] for x in reversed(feats)]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=color))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#070B14",
        plot_bgcolor="#0F172A",
        title=f"{title} ({data.get('method', 'unknown')}, n={data.get('explained_rows', 0)})",
        xaxis_title="Mean |SHAP| / Importance",
        height=440,
    )
    return fig


def render(cfg: Settings) -> None:
    st.title("Explainable AI")
    if not check_pipeline_ready(cfg):
        return

    shap_fc = load_json_safe(cfg.results_dir / "shap_forecaster_global.json")
    shap_det = load_json_safe(cfg.results_dir / "shap_detector_global.json")
    if not shap_fc or not shap_det:
        st.warning("SHAP summaries not generated yet. Run `python -m src.pipeline --stage explain`.")
        return

    tab_fc, tab_det = st.tabs(["Forecasting Explainability (Module M)", "Detection Explainability (Module A)"])

    with tab_fc:
        st.caption(f"Method: `{shap_fc.get('method')}` — explains pre-onset forecast risk drivers.")
        st.plotly_chart(_plot_global_bar(shap_fc, "Global Forecaster Feature Importance", "#00D4FF"), width="stretch")
        _render_local_alert_section(cfg, kind_filter="FORECAST WARNING", color="#FF8C42")

    with tab_det:
        st.caption(f"Method: `{shap_det.get('method')}` — explains observed flow classification drivers (kept strictly separate from forecasting).")
        st.plotly_chart(_plot_global_bar(shap_det, "Global Detector Feature Importance", "#FF4D4F"), width="stretch")
        _render_local_alert_section(cfg, kind_filter="DETECTION", color="#FF4D4F")


def _render_local_alert_section(cfg: Settings, kind_filter: str, color: str) -> None:
    st.subheader(f"Per-Alert Local Explanation ({kind_filter})")
    alerts = []
    if "replay_engine" in st.session_state:
        alerts = [a for a in st.session_state["replay_engine"].alert_manager.to_dicts() if a["kind"] == kind_filter]
    if not alerts:
        saved = load_json_safe(cfg.results_dir / "replay_alerts.json")
        if isinstance(saved, list):
            alerts = [a for a in saved if a.get("kind") == kind_filter]

    if not alerts:
        st.info(f"No {kind_filter} alerts available yet. Step the Historical Replay or check saved alerts.")
        return

    options = [f"{a['alert_id']} | {a['timestamp']} | {a['severity']}" for a in alerts]
    chosen_idx = st.selectbox(f"Select {kind_filter} Alert", range(len(options)), format_func=lambda i: options[i], key=f"xai_{kind_filter}")
    alert = alerts[chosen_idx]
    reasons = alert.get("top_5_reasons", [])

    for r in reasons:
        st.markdown(f"- {r['sentence']}")

    if reasons:
        names = [r["feature"] for r in reversed(reasons)]
        vals = [r["contribution"] for r in reversed(reasons)]
        bar_colors = [color if v >= 0 else "#38BDF8" for v in vals]
        fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=bar_colors))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#070B14",
            plot_bgcolor="#0F172A",
            title=f"Top-5 Feature Contributions for {alert['alert_id']}",
            xaxis_title="SHAP Contribution (+ raises / - lowers)",
            height=300,
        )
        st.plotly_chart(fig, width="stretch")
