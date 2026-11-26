"""Page 2: Temporal Threat Radar — Signature 5-Min Horizon Radar & Causal Historical Replay."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
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
    render_section_header,
    render_threat_timeline,
    style_plotly_fig,
)
from config.settings import Settings
from dashboard.ui_utils import (
    check_pipeline_ready,
    ensure_replay_in_session,
    get_runtime_status_context,
    render_shell_header,
)
from utils.data_loaders import read_csv_artifact


def _build_radar_scope_figure(eng: Any, risk_val: float, risk_tier: str) -> go.Figure:
    """Create a restrained, analytical polar radar plot of recent trajectory & 5-minute forecast window."""
    # Sample last 12 causal steps around 0..300 degrees; reserve 300..360 deg sector for H=+5m horizon risk
    hist_tail = eng.history[-12:] if eng.history else []
    max_r = max([float(h["forecast_risk"]) for h in hist_tail] + [float(eng.fc_threshold) * 1.4, 0.20])

    fig = go.Figure()

    # Validation threshold ring
    ring_theta = np.linspace(0, 360, 73)
    fig.add_trace(
        go.Scatterpolar(
            r=[float(eng.fc_threshold)] * len(ring_theta),
            theta=ring_theta,
            mode="lines",
            name=f"Validation Alert Ring ({eng.fc_threshold:.4f})",
            line=dict(color=COLOR_WARNING, width=1.6, dash="dash"),
            hoverinfo="skip",
        )
    )

    # 5-Minute Future Forecast Window sector highlight (300 -> 360 deg)
    sector_theta = [0, 300, 330, 360, 0]
    sector_r = [0, risk_val, risk_val, risk_val, 0]
    tier_color = {
        "Critical": "rgba(239, 68, 68, 0.30)",
        "High": "rgba(249, 115, 22, 0.26)",
        "Medium": "rgba(245, 158, 11, 0.22)",
    }.get(risk_tier, "rgba(0, 212, 255, 0.16)")
    line_color = {
        "Critical": COLOR_CRITICAL,
        "High": COLOR_HIGH,
        "Medium": COLOR_WARNING,
    }.get(risk_tier, COLOR_CYAN)

    fig.add_trace(
        go.Scatterpolar(
            r=sector_r,
            theta=sector_theta,
            fill="toself",
            fillcolor=tier_color,
            line=dict(color=line_color, width=2),
            name=f"Next +{eng.horizon_min}m Forecast Window ({risk_val:.4f})",
            hovertemplate=f"H=+{eng.horizon_min}m Risk: {risk_val:.4f} ({risk_tier})<extra></extra>",
        )
    )

    if hist_tail:
        thetas = np.linspace(30, 285, len(hist_tail)) if len(hist_tail) > 1 else [285]
        r_vals = [float(h["forecast_risk"]) for h in hist_tail]
        labels = [h["timestamp_str"][-12:-4] for h in hist_tail]
        fig.add_trace(
            go.Scatterpolar(
                r=r_vals,
                theta=thetas,
                mode="lines+markers",
                name="Recent Causal Trajectory (<= t)",
                line=dict(color=COLOR_CYAN, width=2.2),
                marker=dict(
                    size=[10 if h["gt_attack_active"] else 6 for h in hist_tail],
                    color=[COLOR_CRITICAL if h["gt_attack_active"] else COLOR_CYAN for h in hist_tail],
                ),
                text=labels,
                hovertemplate="%{text} UTC<br>Risk: %{r:.4f}<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        polar=dict(
            bgcolor="#0B1222",
            radialaxis=dict(
                visible=True,
                range=[0, max_r],
                gridcolor="rgba(148, 163, 184, 0.14)",
                tickfont=dict(size=10, color="#94A3B8"),
            ),
            angularaxis=dict(
                tickvals=[30, 120, 210, 285, 330],
                ticktext=["t-11m", "t-7m", "t-3m", "Cursor (t)", "+5m Window"],
                gridcolor="rgba(148, 163, 184, 0.14)",
                tickfont=dict(size=10, color="#CBD5E1"),
                direction="clockwise",
            ),
        ),
        margin=dict(l=38, r=38, t=42, b=36),
        height=355,
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5, font=dict(size=10.5)),
        title=dict(text="Temporal Threat Scope (Causal Sweep & +5m Horizon)", font=dict(size=13.5, color="#F1F5F9")),
    )
    return fig


def _render_radar_body(cfg: Settings, ctx: dict[str, Any]) -> None:
    eng = ensure_replay_in_session(cfg)

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
    c1, c2, c3, c4, c5, c6 = st.columns([0.9, 0.9, 0.95, 1.05, 0.9, 1.8])
    if c1.button("Start", type="primary", disabled=eng.is_finished, key="tr_start", width="stretch"):
        st.session_state["replay_running"] = True
        st.rerun()
    if c2.button("Pause", disabled=not st.session_state.get("replay_running", False), key="tr_pause", width="stretch"):
        st.session_state["replay_running"] = False
        st.rerun()
    if c3.button("Step +1m", disabled=eng.is_finished, key="tr_step1", width="stretch"):
        st.session_state["replay_running"] = False
        eng.step()
        st.rerun()
    if c4.button("Step +10m", disabled=eng.is_finished, key="tr_step10", width="stretch"):
        st.session_state["replay_running"] = False
        for _ in range(10):
            if not eng.is_finished:
                eng.step()
        st.rerun()
    if c5.button("Reset", key="tr_reset", width="stretch"):
        st.session_state["replay_running"] = False
        eng.reset()
        st.rerun()
    speed = c6.slider(
        "Speed (min / tick)",
        min_value=1,
        max_value=30,
        value=int(st.session_state.get("replay_speed", cfg.replay_default_speed)),
        key="tr_speed_slider",
    )
    st.session_state["replay_speed"] = speed

    status_label = "RUNNING" if st.session_state.get("replay_running") else ("COMPLETED" if eng.is_finished else "PAUSED / READY")
    cur_ts = eng.current_minute.strftime("%Y-%m-%d %H:%M UTC") if eng.current_minute else "Not started"
    latest = eng.history[-1] if eng.history else None
    risk_val = float(latest["forecast_risk"]) if latest else 0.0
    risk_tier = str(latest["risk_state"]) if latest else "Low"
    tier_tone = {"Critical": "critical", "High": "warning", "Medium": "warning"}.get(risk_tier, "healthy")

    render_metric_grid(
        [
            {
                "label": "Replay Status",
                "value": status_label,
                "sub": f"Position {eng.cursor} / {len(eng.test_minutes)} test minutes",
                "badge": "Causal" if not eng.history else "Active",
                "tone": "cyan",
            },
            {
                "label": "Current Replay Time",
                "value": cur_ts,
                "sub": "Future labels & flows strictly masked",
                "badge": "UTC",
                "tone": "violet",
            },
            {
                "label": f"Forecast Risk (Next +{eng.horizon_min}m)",
                "value": f"{risk_val:.4f}" if latest else "—",
                "sub": f"Estimated risk only (Thr {eng.fc_threshold:.4f})",
                "badge": "Estimate",
                "tone": tier_tone,
            },
            {
                "label": "Risk Tier Classification",
                "value": risk_tier,
                "sub": "Derived from frozen validation quantiles",
                "badge": risk_tier,
                "tone": tier_tone,
            },
        ]
    )

    # Radar Polar Scope + Forecast Context Panel
    radar_col, info_col = st.columns([1.05, 1.25])
    with radar_col:
        st.plotly_chart(_build_radar_scope_figure(eng, risk_val, risk_tier), width="stretch")

    with info_col:
        verdict = ctx["verdict"]
        fc_onsets_df = read_csv_artifact(cfg.results_dir / "forecast_onsets.csv")
        test_onsets_list = fc_onsets_df[fc_onsets_df["split"] == "test"] if fc_onsets_df is not None else None
        st.markdown(
            '<div class="cs-surface-card">'
            '<h4 style="margin:0 0 0.45rem 0;color:#F1F5F9;font-size:0.98rem;">How to Read the Temporal Threat Radar</h4>'
            '<div style="color:#94A3B8;font-size:0.83rem;line-height:1.5;">'
            '<strong>1. Historical Observed Attacks:</strong> Ground-truth attack-active intervals are revealed <em>only</em> after their UTC timestamp passes the replay cursor.<br>'
            f'<strong>2. Replay Position (t):</strong> Uses strictly terminated flows (<code>end_ts &lt; t + 60s</code>) to avoid future leakage.<br>'
            f'<strong>3. Next +{eng.horizon_min}-Minute Risk Window:</strong> Displays calibrated XGBoost onset probability against the frozen validation budget threshold (<code>{eng.fc_threshold:.4f}</code>). <strong>This is a risk estimate, never a guaranteed attack.</strong>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="cs-surface-card">'
            '<h4 style="margin:0 0 0.35rem 0;color:#F1F5F9;font-size:0.94rem;">Forecast Benchmark Context</h4>'
            f'<div style="color:#CBD5E1;font-size:0.82rem;line-height:1.5;">'
            f'Test Verdict: <strong>{verdict.get("verdict", "NOT DISTINGUISHABLE")}</strong> &middot; '
            f'5-Seed Ensemble PR-AUC: <code>{verdict.get("xgboost_5seed_ensemble_test_pr_auc", 0):.4f}</code> &middot; '
            f'Test Onsets: <code>{verdict.get("split_onsets", {}).get("test_onsets", 6)}</code>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if test_onsets_list is not None and not test_onsets_list.empty:
            st.caption("Recorded Friday Test-Partition Onset Schedule (revealed causally during replay):")
            st.dataframe(test_onsets_list[["minute_ts", "attack_flow_starts"]], width="stretch", hide_index=True, height=150)

    if not eng.history:
        render_empty_state(
            "Replay Ready at Cursor 0",
            "Click Start, Step +1m, or Step +10m above to advance through the recorded Friday test traffic causally.",
        )
        return

    times = [h["timestamp_str"] for h in eng.history]
    risks = [h["forecast_risk"] for h in eng.history]
    gt_revealed = eng.visible_ground_truth_up_to_cursor()
    atk_mask = [bool(g["gt_attack_active"]) for g in gt_revealed]
    fc_warn_x = [h["timestamp_str"] for h in eng.history if h["forecast_alert_id"]]
    fc_warn_y = [h["forecast_risk"] for h in eng.history if h["forecast_alert_id"]]

    render_section_header("Observed Attack History vs 5-Minute Forecast Horizon", "Shaded red zones indicate revealed past attack windows; amber markers indicate forecast warnings.")
    fig_tl = render_threat_timeline(
        x_values=times,
        risk_values=risks,
        threshold=eng.fc_threshold,
        attack_active_mask=atk_mask,
        warn_x=fc_warn_x,
        warn_y=fc_warn_y,
        replay_cursor_x=times[-1] if times else None,
        title=f"Temporal Threat Radar — Causal Risk Trajectory up to {times[-1]}",
        height=420,
    )
    st.plotly_chart(fig_tl, width="stretch")

    # Split Feed: Recent Forecast Warnings vs Revealed Attack Windows
    fc_warnings = [a for a in eng.alert_manager.alerts if a.kind == "FORECAST WARNING"]
    c_left, c_right = st.columns(2)
    with c_left:
        render_section_header(f"Forecast Warnings Raised ({len(fc_warnings)})", "Pre-onset risk estimates exceeding frozen threshold.")
        if fc_warnings:
            shown = [a for a in reversed(eng.alert_manager.to_dicts()) if a["kind"] == "FORECAST WARNING"][:6]
            for w in shown:
                render_alert_card(w)
        else:
            render_empty_state("No Forecast Warnings Raised", "No minutes up to the current replay cursor exceeded the validation threshold.")

    with c_right:
        active_gt = [h for h in eng.history if h["gt_attack_active"]]
        render_section_header(f"Revealed Attack Minutes ({len(active_gt)})", "Ground-truth starts observed strictly <= current replay time.")
        if active_gt:
            for g in reversed(active_gt[-6:]):
                render_alert_card(
                    {
                        "alert_id": f"GT-{g['timestamp_str']}",
                        "timestamp": g["timestamp_str"],
                        "kind": "REVEALED GROUND TRUTH",
                        "attack_category": ", ".join(g["gt_labels"]) or "Attack Active",
                        "severity": "Critical",
                        "analyst_status": f"{g['gt_attack_starts']} attack starts",
                        "top_5_reasons": [{"sentence": f"Observed {g['gt_attack_starts']} malicious flow starts during this minute."}],
                    }
                )
        else:
            render_empty_state("No Attack-Active Minutes Reached", "Attack-active minutes will appear here once their UTC timestamp is reached.")


def render(cfg: Settings) -> None:
    ctx = get_runtime_status_context(cfg)
    render_shell_header(
        cfg,
        title="Temporal Threat Radar",
        description="Real-time-paced causal historical replay contrasting past attack observations against calibrated 5-minute pre-onset risk estimates.",
        ctx=ctx,
    )
    if not check_pipeline_ready(cfg):
        return

    try:
        ensure_replay_in_session(cfg)
    except Exception as exc:
        render_error_state("Replay Engine Unavailable", str(exc))
        return

    frag_fn = getattr(st, "fragment", None)
    if frag_fn is not None and st.session_state.get("replay_running", False):
        @frag_fn(run_every=1.0)
        def _live_tick() -> None:
            _render_radar_body(cfg, get_runtime_status_context(cfg))

        _live_tick()
    else:
        _render_radar_body(cfg, ctx)
        if st.session_state.get("replay_running", False):
            time.sleep(1.0)
            st.rerun()
