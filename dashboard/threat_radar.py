"""Page 2: Temporal Threat Radar with Causal Historical Replay Controls."""
from __future__ import annotations

import time

import plotly.graph_objects as go
import streamlit as st

from config.settings import Settings
from dashboard.ui_utils import check_pipeline_ready, ensure_replay_in_session


def _render_radar_body(cfg: Settings) -> None:
    eng = ensure_replay_in_session(cfg)

    # Advance engine if running
    if st.session_state.get("replay_running", False) and not eng.is_finished:
        steps_per_tick = max(1, int(st.session_state.get("replay_speed", cfg.replay_default_speed)))
        for _ in range(steps_per_tick):
            if eng.is_finished:
                st.session_state["replay_running"] = False
                break
            eng.step()
        if eng.is_finished:
            st.session_state["replay_running"] = False

    # Controls bar
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 2])
    if c1.button("Start", type="primary", disabled=eng.is_finished, key="tr_start"):
        st.session_state["replay_running"] = True
        st.rerun()
    if c2.button("Pause", disabled=not st.session_state.get("replay_running", False), key="tr_pause"):
        st.session_state["replay_running"] = False
        st.rerun()
    if c3.button("Step +1m", disabled=eng.is_finished, key="tr_step1"):
        st.session_state["replay_running"] = False
        eng.step()
        st.rerun()
    if c4.button("Reset", key="tr_reset"):
        st.session_state["replay_running"] = False
        eng.reset()
        st.rerun()
    speed = c5.slider(
        "Speed (min / tick)",
        min_value=1,
        max_value=30,
        value=int(st.session_state.get("replay_speed", cfg.replay_default_speed)),
        key="tr_speed_slider",
    )
    st.session_state["replay_speed"] = speed

    status_label = "RUNNING" if st.session_state.get("replay_running") else ("COMPLETED" if eng.is_finished else "PAUSED / READY")
    cur_ts = eng.current_minute.strftime("%Y-%m-%d %H:%M:%S UTC") if eng.current_minute else "Not started"

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Replay Status", status_label)
    s2.metric("Current Simulation Time", cur_ts)
    latest = eng.history[-1] if eng.history else None
    risk_val = latest["forecast_risk"] if latest else 0.0
    risk_tier = latest["risk_state"] if latest else "Low"
    s3.metric(f"Forecast Risk (H={eng.horizon_min}m)", f"{risk_val:.4f}" if latest else "—")
    s4.metric("Risk Tier", risk_tier)

    if not eng.history:
        st.info("Click **Start** or **Step +1m** to begin causal historical replay. Ground-truth attack windows remain hidden until their timestamps are reached.")
        return

    times = [h["timestamp_str"] for h in eng.history]
    risks = [h["forecast_risk"] for h in eng.history]
    gt_revealed = eng.visible_ground_truth_up_to_cursor()
    gt_y = [max(risks + [eng.fc_threshold * 1.4]) if g["gt_attack_active"] else 0.0 for g in gt_revealed]

    fc_warn_x = [h["timestamp_str"] for h in eng.history if h["forecast_alert_id"]]
    fc_warn_y = [h["forecast_risk"] for h in eng.history if h["forecast_alert_id"]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=gt_y,
            fill="tozeroy",
            mode="none",
            name="Revealed Actual Attack Active (<= current replay time)",
            fillcolor="rgba(255, 77, 79, 0.22)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=risks,
            name="Calibrated Onset Forecast Risk",
            line=dict(color="#00D4FF", width=2.4),
        )
    )
    if fc_warn_x:
        fig.add_trace(
            go.Scatter(
                x=fc_warn_x,
                y=fc_warn_y,
                mode="markers",
                name="Forecast Warnings Raised",
                marker=dict(color="#FF8C42", size=8, symbol="triangle-up"),
            )
        )
    fig.add_hline(
        y=eng.fc_threshold,
        line_dash="dash",
        line_color="#FF8C42",
        annotation_text=f"Validation Threshold ({eng.fc_threshold:.4f})",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#070B14",
        plot_bgcolor="#0F172A",
        title="Temporal Threat Radar — Onset Risk History vs Revealed Ground-Truth Attack Periods",
        xaxis_title="Simulation UTC Time",
        yaxis_title="Calibrated Onset Risk (Next 5 min)",
        height=430,
    )
    st.plotly_chart(fig, width="stretch")

    # Recent Forecast Warnings vs Revealed Ground Truth table
    fc_warnings = [a for a in eng.alert_manager.alerts if a.kind == "FORECAST WARNING"]
    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader(f"Forecast Warnings ({len(fc_warnings)})")
        if fc_warnings:
            for w in reversed(fc_warnings[-8:]):
                st.warning(f"**{w.timestamp}** | Risk `{w.forecast_risk:.4f}` | Severity `{w.severity}` — {w.advisory_note}")
        else:
            st.caption("No forecast warnings raised up to current replay timestamp.")
    with c_right:
        st.subheader("Revealed Attack Minutes (<= current time)")
        active_gt = [h for h in eng.history if h["gt_attack_active"]]
        if active_gt:
            for g in reversed(active_gt[-8:]):
                st.error(f"**{g['timestamp_str']}** | Attack starts: `{g['gt_attack_starts']}` | Classes: `{', '.join(g['gt_labels'])}`")
        else:
            st.caption("No attack-active minutes reached yet in current replay.")


def render(cfg: Settings) -> None:
    st.title("Temporal Threat Radar")
    if not check_pipeline_ready(cfg):
        return

    try:
        ensure_replay_in_session(cfg)
    except Exception as exc:
        st.error(f"Replay engine unavailable: {exc}")
        return

    frag_fn = getattr(st, "fragment", None)
    if frag_fn is not None and st.session_state.get("replay_running", False):
        @frag_fn(run_every=1.0)
        def _live_tick() -> None:
            _render_radar_body(cfg)

        _live_tick()
    else:
        _render_radar_body(cfg)
        if st.session_state.get("replay_running", False):
            time.sleep(1.0)
            st.rerun()
