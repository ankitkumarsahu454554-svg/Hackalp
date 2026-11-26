"""Reusable cybersecurity SaaS UI components for CYBERSENTINEL AI."""
from __future__ import annotations

import html
from typing import Any, Iterable, Optional

import plotly.graph_objects as go
import streamlit as st

COLOR_BG_CANVAS = "#070B14"
COLOR_BG_SURFACE = "#0F172A"
COLOR_CYAN = "#00D4FF"
COLOR_VIOLET = "#8B5CF6"
COLOR_CRITICAL = "#EF4444"
COLOR_HIGH = "#F97316"
COLOR_WARNING = "#F59E0B"
COLOR_HEALTHY = "#10B981"
COLOR_INFO = "#38BDF8"
COLOR_TEXT_PRIMARY = "#F1F5F9"
COLOR_TEXT_SECONDARY = "#94A3B8"


def _esc(val: Any) -> str:
    return html.escape("" if val is None else str(val), quote=True)


def render_status_indicator(label: str, tone: str = "healthy") -> str:
    """Return HTML for a compact status pill (healthy | warning | critical | cyan | violet | muted)."""
    valid_tones = {"healthy", "warning", "critical", "cyan", "violet", "muted"}
    safe_tone = tone if tone in valid_tones else "muted"
    return (
        f'<span class="cs-status-pill">'
        f'<span class="cs-status-dot cs-dot-{safe_tone}"></span>'
        f'{_esc(label)}'
        f'</span>'
    )


def render_severity_badge(severity: str) -> str:
    """Return HTML badge for Critical, High, Medium, Low, or Informational severities."""
    norm = (severity or "Informational").strip().lower()
    if norm not in {"critical", "high", "medium", "low", "informational"}:
        norm = "informational"
    return f'<span class="cs-badge cs-badge-{norm}">{_esc(severity or "Informational")}</span>'


def render_page_header(
    title: str,
    description: str,
    eyebrow: str = "CYBERSENTINEL AI // TEMPORAL THREAT RADAR",
    badges: Optional[list[tuple[str, str]]] = None,
) -> None:
    """Render the consistent top header bar with live system/dataset/model/replay status pills."""
    badge_html = "".join(render_status_indicator(lbl, tone) for lbl, tone in (badges or []))
    html_block = (
        '<div class="cs-page-header">'
        '<div class="cs-page-header-left">'
        f'<div class="cs-page-eyebrow">{_esc(eyebrow)}</div>'
        f'<h1 class="cs-page-title">{_esc(title)}</h1>'
        f'<div class="cs-page-desc">{_esc(description)}</div>'
        '</div>'
        f'<div class="cs-header-badges">{badge_html}</div>'
        '</div>'
    )
    st.markdown(html_block, unsafe_allow_html=True)


def render_section_header(title: str, subtitle: str = "") -> None:
    """Render a consistent horizontal section header with subtle divider."""
    sub_html = f'<div class="cs-section-sub">{_esc(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="cs-section-header"><div>'
        f'<h3 class="cs-section-title">{_esc(title)}</h3>'
        f'{sub_html}'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def render_metric_grid(cards: Iterable[dict[str, Any]]) -> None:
    """Render responsive wrap-ready metric cards (`label`, `value`, optional `sub`, `tone`)."""
    items: list[str] = []
    for c in cards:
        tone = c.get("tone")
        badge = render_status_indicator(c["badge"], tone) if c.get("badge") and tone else ""
        sub_txt = f'<div class="cs-metric-sub">{_esc(c["sub"])}</div>' if c.get("sub") else ""
        val_str = str(c.get("value", "—"))
        val_cls = "cs-metric-value compact" if len(val_str) > 13 else "cs-metric-value"
        items.append(
            '<div class="cs-metric-card">'
            '<div class="cs-metric-top">'
            f'<span class="cs-metric-label">{_esc(c.get("label", ""))}</span>'
            f'{badge}'
            '</div>'
            f'<div class="{val_cls}">{_esc(val_str)}</div>'
            f'{sub_txt}'
            '</div>'
        )
    st.markdown(f'<div class="cs-metric-grid">{"".join(items)}</div>', unsafe_allow_html=True)


def render_model_status_card(
    title: str,
    status_label: str,
    status_tone: str,
    description: str,
    key_stat: str,
) -> None:
    """Render a reusable Module A / B / M status card."""
    pill = render_status_indicator(status_label, status_tone)
    st.markdown(
        '<div class="cs-model-card">'
        '<div class="cs-model-card-header">'
        f'<span class="cs-model-card-title">{_esc(title)}</span>'
        f'{pill}'
        '</div>'
        f'<div class="cs-model-card-desc">{_esc(description)}</div>'
        f'<div class="cs-model-card-stat">{_esc(key_stat)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_alert_card(alert: dict[str, Any]) -> None:
    """Render one structured alert item with severity badge and top reason."""
    sev = str(alert.get("severity", "Informational"))
    sev_cls = f"sev-{sev.strip().lower()}" if sev.strip().lower() in {"critical", "high", "medium", "low"} else ""
    badge = render_severity_badge(sev)
    kind = str(alert.get("kind", "ALERT"))
    ts = str(alert.get("timestamp", "-"))
    cat = str(alert.get("attack_category", "-"))
    reasons = alert.get("top_5_reasons") or []
    top_reason = reasons[0].get("sentence", alert.get("advisory_note", "")) if reasons else alert.get("advisory_note", "")

    score_parts: list[str] = []
    if alert.get("forecast_risk") is not None:
        score_parts.append(f"Risk: {float(alert['forecast_risk']):.4f}")
    if alert.get("detection_score") is not None:
        score_parts.append(f"Detection P: {float(alert['detection_score']):.3f}")
    if alert.get("anomaly_score") is not None:
        score_parts.append(f"IF Score: {float(alert['anomaly_score']):.3f}")
    score_parts.append(f"Status: {alert.get('analyst_status', 'New')}")
    meta_str = f"{kind} | {ts} | {cat} | " + " | ".join(score_parts)

    st.markdown(
        f'<div class="cs-alert-card {sev_cls}">'
        '<div class="cs-alert-card-head">'
        f'<span class="cs-alert-id">{_esc(alert.get("alert_id", "ALERT"))}</span>'
        f'{badge}'
        '</div>'
        f'<div class="cs-alert-meta">{_esc(meta_str)}</div>'
        f'<div class="cs-alert-reason">{_esc(top_reason)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_chart_header(title: str, subtitle: str = "") -> None:
    """Consistent compact label immediately above a Plotly chart."""
    st.markdown(f"**{title}**" + (f"  \n<span style='color:#94A3B8;font-size:0.82rem;'>{_esc(subtitle)}</span>" if subtitle else ""), unsafe_allow_html=True)


def render_empty_state(title: str, message: str) -> None:
    """Render a clean bordered empty-state placeholder."""
    st.markdown(
        '<div class="cs-state-box">'
        f'<div class="cs-state-title">{_esc(title)}</div>'
        f'<div class="cs-state-body">{_esc(message)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_loading_state(title: str, message: str) -> None:
    """Render a clean loading-state notice."""
    st.markdown(
        '<div class="cs-state-box">'
        f'<div class="cs-state-title">{_esc(title)}</div>'
        f'<div class="cs-state-body">{_esc(message)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_error_state(title: str, message: str) -> None:
    """Render a clean actionable error-state banner."""
    st.markdown(
        '<div class="cs-state-box error">'
        f'<div class="cs-state-title">{_esc(title)}</div>'
        f'<div class="cs-state-body">{_esc(message)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def style_plotly_fig(
    fig: go.Figure,
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = "",
    height: int = 380,
) -> go.Figure:
    """Apply cohesive midnight/slate SaaS styling and responsive margins to any Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLOR_BG_SURFACE,
        plot_bgcolor="#0B1222",
        font=dict(family="Inter, Segoe UI, sans-serif", color=COLOR_TEXT_PRIMARY, size=12),
        title=dict(text=title, font=dict(size=14, color=COLOR_TEXT_PRIMARY), x=0.01, xanchor="left") if title else None,
        xaxis=dict(
            title=xaxis_title,
            gridcolor="rgba(148, 163, 184, 0.10)",
            zerolinecolor="rgba(148, 163, 184, 0.15)",
            tickfont=dict(color=COLOR_TEXT_SECONDARY),
        ),
        yaxis=dict(
            title=yaxis_title,
            gridcolor="rgba(148, 163, 184, 0.10)",
            zerolinecolor="rgba(148, 163, 184, 0.15)",
            tickfont=dict(color=COLOR_TEXT_SECONDARY),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=11, color=COLOR_TEXT_SECONDARY),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=46, r=20, t=54 if title else 28, b=42),
        height=height,
        hovermode="x unified",
    )
    return fig


def render_threat_timeline(
    x_values: list[Any],
    risk_values: list[float],
    threshold: float,
    attack_active_mask: Optional[list[bool]] = None,
    warn_x: Optional[list[Any]] = None,
    warn_y: Optional[list[float]] = None,
    onset_x: Optional[list[Any]] = None,
    onset_y: Optional[list[float]] = None,
    replay_cursor_x: Optional[Any] = None,
    title: str = "Temporal Threat Risk Timeline",
    height: int = 410,
) -> go.Figure:
    """Reusable temporal risk timeline with observed attack regions, forecast warnings, and replay marker."""
    fig = go.Figure()
    top_y = max([float(r) for r in risk_values if r is not None] + [float(threshold) * 1.35, 0.15])

    if attack_active_mask:
        gt_y = [top_y if flag else 0.0 for flag in attack_active_mask]
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=gt_y,
                fill="tozeroy",
                mode="none",
                name="Observed Attack Window",
                fillcolor="rgba(239, 68, 68, 0.18)",
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=risk_values,
            name="Calibrated 5-Min Onset Risk",
            line=dict(color=COLOR_CYAN, width=2.4),
            hovertemplate="%{x}<br>Estimated Risk: %{y:.4f}<extra></extra>",
        )
    )

    if warn_x and warn_y:
        fig.add_trace(
            go.Scatter(
                x=warn_x,
                y=warn_y,
                mode="markers",
                name="Forecast Warnings",
                marker=dict(color=COLOR_WARNING, size=9, symbol="triangle-up", line=dict(color="#070B14", width=1)),
                hovertemplate="%{x}<br>Warning Risk: %{y:.4f}<extra></extra>",
            )
        )

    if onset_x and onset_y:
        fig.add_trace(
            go.Scatter(
                x=onset_x,
                y=onset_y,
                mode="markers",
                name="Dataset Onset Marker",
                marker=dict(color=COLOR_CRITICAL, size=9, symbol="diamond"),
                hovertemplate="%{x}<br>Observed Attack Onset<extra></extra>",
            )
        )

    fig.add_hline(
        y=float(threshold),
        line_dash="dash",
        line_color=COLOR_WARNING,
        annotation_text=f"Validation Threshold ({threshold:.4f})",
        annotation_font_color=COLOR_WARNING,
    )

    if replay_cursor_x is not None:
        fig.add_vline(
            x=replay_cursor_x,
            line_dash="dot",
            line_color=COLOR_VIOLET,
        )

    return style_plotly_fig(
        fig,
        title=title,
        xaxis_title="UTC Timestamp",
        yaxis_title="Calibrated Onset Risk (Next 5 min)",
        height=height,
    )
