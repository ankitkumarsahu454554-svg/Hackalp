"""Page 1: Security Command Centre."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from config.settings import Settings
from dashboard.ui_utils import check_pipeline_ready, ensure_replay_in_session, load_json_safe


def render(cfg: Settings) -> None:
    st.title("Security Command Centre")
    if not check_pipeline_ready(cfg):
        return

    try:
        eng = ensure_replay_in_session(cfg)
    except Exception as exc:
        st.error(f"Could not initialize Historical Replay Engine: {exc}")
        return

    # Quick controls so user can step directly from Command Centre or Temporal Threat Radar
    c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
    if c1.button("Step +10 min", key="cc_step10"):
        for _ in range(10):
            if not eng.is_finished:
                eng.step()
    if c2.button("Step +30 min", key="cc_step30"):
        for _ in range(30):
            if not eng.is_finished:
                eng.step()
    if c3.button("Reset Replay", key="cc_reset"):
        eng.reset()
        st.session_state["replay_running"] = False
    c4.info(
        f"Replay position: **{eng.cursor} / {len(eng.test_minutes)} min** "
        f"({eng.current_minute.strftime('%Y-%m-%d %H:%M UTC') if eng.current_minute else 'Not started'})"
    )

    latest = eng.history[-1] if eng.history else None
    crit_count = sum(1 for a in eng.alert_manager.alerts if a.severity == "Critical")
    fc_risk = latest["forecast_risk"] if latest else 0.0
    risk_state = latest["risk_state"] if latest else "Idle"

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Flows Processed (Replay)", f"{eng.total_flows_processed:,}")
    m2.metric("Malicious Flows Detected", f"{eng.total_malicious_detected:,}")
    m3.metric("Anomalous Flows (IF)", f"{eng.total_anomalous_flows:,}")
    m4.metric(f"Forecast Risk (H={eng.horizon_min}m)", f"{fc_risk:.3f}" if latest else "—", delta=risk_state)
    m5.metric("Critical Alerts", f"{crit_count}")

    st.divider()
    if not eng.history:
        st.info("Press **Step +10 min** above or open **Temporal Threat Radar** to run the historical replay.")
        fc_meta = load_json_safe(cfg.results_dir / "forecast_training_summary.json") or {}
        st.caption(
            f"Test replay window: `{fc_meta.get('split_info', {}).get('test_start', '-')}` to "
            f"`{fc_meta.get('split_info', {}).get('test_end', '-')}`."
        )
        return

    times = [h["timestamp_str"] for h in eng.history]
    flows = [h["flows_completed"] for h in eng.history]
    mal = [h["malicious_detected"] for h in eng.history]
    anom = [h["anomalous_flows"] for h in eng.history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=flows, name="Total Completed Flows/min", line=dict(color="#00D4FF", width=2)))
    fig.add_trace(go.Scatter(x=times, y=mal, name="Detected Malicious Flows/min", line=dict(color="#FF4D4F", width=2)))
    fig.add_trace(go.Scatter(x=times, y=anom, name="Anomalous Flows/min (IF)", line=dict(color="#FF8C42", width=1.8, dash="dot")))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#070B14",
        plot_bgcolor="#0F172A",
        title="Historical Replay Traffic & Detection Timeline (Up to Current Replay Time)",
        xaxis_title="UTC Minute",
        yaxis_title="Flow Count",
        height=400,
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Separation of Signals")
    col_a, col_b, col_c = st.columns(3)
    col_a.markdown("**1. Detection (Module A)**\nObserved malicious flows classified per flow (`BENIGN` vs attack class).")
    col_b.markdown("**2. Anomaly Score (Module B)**\nIsolation Forest deviation from Monday benign baseline (not a probability).")
    col_c.markdown(f"**3. Forecast Risk (Module M)**\nCalibrated risk of a *new* onset in the next {eng.horizon_min} min (threshold `{eng.fc_threshold:.4f}`).")
