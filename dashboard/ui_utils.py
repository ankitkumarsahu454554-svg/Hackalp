"""Shared dashboard helpers for theme injection, sidebar provenance, headers, and replay state."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import polars as pl
import streamlit as st

from components.ui import render_page_header, render_status_indicator
from config.settings import Settings, get_settings
from src.data_loader import is_synthetic_mode
from utils.data_loaders import (
    load_text_cached,
    read_json_artifact,
    read_saved_replay_alerts,
)


def get_cfg() -> Settings:
    return get_settings()


def load_json_safe(path: Path) -> Optional[dict[str, Any]]:
    data = read_json_artifact(path)
    return data if isinstance(data, dict) else None


def load_json_list_safe(path: Path) -> list[dict[str, Any]]:
    data = read_json_artifact(path)
    return data if isinstance(data, list) else []


def load_csv_safe(path: Path) -> Optional[pl.DataFrame]:
    if not path.exists():
        return None
    return pl.read_csv(path)


def inject_theme_css(cfg: Settings) -> None:
    """Inject the unified CYBERSENTINEL AI CSS design system once per render."""
    css_path = cfg.project_root / "assets" / "styles" / "theme.css"
    css = load_text_cached(str(css_path))
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def get_brand_svg(cfg: Settings, variant: str = "sidebar") -> str:
    """Load crisp vector SVG brand logo markup."""
    fname = {
        "sidebar": "cybersentinel_logo_sidebar.svg",
        "primary": "cybersentinel_logo_primary.svg",
        "light": "cybersentinel_logo_light.svg",
        "icon": "cybersentinel_icon.svg",
    }.get(variant, "cybersentinel_logo_sidebar.svg")
    svg = load_text_cached(str(cfg.project_root / "assets" / "brand" / fname))
    return svg or ""


def get_runtime_status_context(cfg: Settings) -> dict[str, Any]:
    """Summarize verified dataset, model, and replay status for headers and sidebar."""
    manifest = load_json_safe(cfg.manifest_path) or {}
    verdict = load_json_safe(cfg.results_dir / "verdict.json") or {}
    fc_summary = load_json_safe(cfg.results_dir / "forecast_training_summary.json") or {}
    audit = load_json_safe(cfg.results_dir / "data_audit.json") or {}

    models_ok = all(
        (cfg.models_dir / m).exists()
        for m in ("detector_xgb.joblib", "isolation_forest.joblib", f"forecaster_xgb_h{cfg.horizons[0]}.joblib")
    )
    results_ok = all(
        (cfg.results_dir / r).exists()
        for r in ("verdict.json", "forecast_metrics.csv", "classification_metrics.json", "anomaly_metrics.json")
    )

    eng = st.session_state.get("replay_engine")
    if eng is not None and eng.history:
        replay_state = f"Replay: {eng.cursor}/{len(eng.test_minutes)} min"
        replay_tone = "cyan" if not eng.is_finished else "healthy"
        last_ts = eng.history[-1]["timestamp_str"]
    elif eng is not None:
        replay_state = f"Replay Ready (0/{len(eng.test_minutes)} min)"
        replay_tone = "violet"
        last_ts = fc_summary.get("split_info", {}).get("test_end", "Not stepped")
    else:
        replay_state = "Historical Replay Mode"
        replay_tone = "violet"
        last_ts = fc_summary.get("split_info", {}).get("test_end", "Offline Archive")

    return {
        "manifest": manifest,
        "verdict": verdict,
        "fc_summary": fc_summary,
        "audit": audit,
        "models_ok": models_ok,
        "results_ok": results_ok,
        "replay_state": replay_state,
        "replay_tone": replay_tone,
        "last_ts": str(last_ts),
        "synthetic": is_synthetic_mode(cfg),
        "saved_alerts_count": len(read_saved_replay_alerts(cfg)),
    }


def render_shell_header(cfg: Settings, title: str, description: str, ctx: Optional[dict[str, Any]] = None) -> None:
    """Render standardized SaaS page header, live status pills, and credibility notice."""
    info = ctx or get_runtime_status_context(cfg)
    manifest = info["manifest"]
    ds_ver = manifest.get("dataset_version", "Unverified")
    ds_ok = manifest.get("verification_status") == "verified"

    badges = [
        (f"Dataset: {ds_ver}", "healthy" if ds_ok else "warning"),
        ("Models: Ready" if info["models_ok"] else "Models: Artifacts Missing", "healthy" if info["models_ok"] else "warning"),
        (info["replay_state"], info["replay_tone"]),
        (f"Window End: {info['last_ts'][:19]}", "muted"),
    ]
    render_page_header(title=title, description=description, badges=badges)

    if info["synthetic"]:
        st.markdown(
            '<div class="cs-synth-banner">SYNTHETIC: NOT RESULTS &mdash; Running on synthetic smoke data only. Never report synthetic metrics as research findings.</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="cs-credibility-bar">'
        '<strong>Research Transparency Notice:</strong> CYBERSENTINEL AI is a research-oriented network threat analysis and temporal forecasting system. '
        'Results are based on the available dataset and historical replay (never live packet capture). '
        'Forecasts represent calibrated onset risk estimates, not guaranteed future attacks.'
        '</div>',
        unsafe_allow_html=True,
    )


def render_sidebar_brand_and_status(cfg: Settings, ctx: dict[str, Any]) -> None:
    """Render sidebar vector logo, dataset/model/replay status card, and version indicator."""
    import base64

    svg = get_brand_svg(cfg, "sidebar")
    if svg:
        b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
        st.markdown(
            f'<div class="cs-sidebar-brand">'
            f'<img src="data:image/svg+xml;base64,{b64}" alt="CYBERSENTINEL AI — Temporal Threat Radar" style="width:100%;height:auto;display:block;" />'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown("### CYBERSENTINEL AI\n`TEMPORAL THREAT RADAR`")


def render_sidebar_footer_status(cfg: Settings, ctx: dict[str, Any]) -> None:
    """Render system status metrics and version tag below navigation."""
    manifest = ctx["manifest"]
    ds_ver = manifest.get("dataset_version", "Not fetched")
    ver_stat = manifest.get("verification_status", "missing")
    model_stat = "Loaded / Verified" if ctx["models_ok"] else ("Metrics Only" if ctx["results_ok"] else "Not Built")
    replay_pill = render_status_indicator(ctx["replay_state"], ctx["replay_tone"])

    st.markdown(
        '<div class="cs-sidebar-panel">'
        '<div class="cs-nav-label" style="margin-top:0;">System Telemetry</div>'
        '<div class="cs-sidebar-kv">'
        '<span class="cs-sidebar-kv-label">Dataset</span>'
        f'<span class="cs-sidebar-kv-value">{ds_ver}</span>'
        '</div>'
        '<div class="cs-sidebar-kv">'
        '<span class="cs-sidebar-kv-label">Integrity</span>'
        f'<span class="cs-sidebar-kv-value">{ver_stat}</span>'
        '</div>'
        '<div class="cs-sidebar-kv">'
        '<span class="cs-sidebar-kv-label">Pipeline</span>'
        f'<span class="cs-sidebar-kv-value">{model_stat}</span>'
        '</div>'
        '<div style="margin-top:0.45rem;">'
        f'{replay_pill}'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    if ctx["synthetic"]:
        st.error("SYNTHETIC: NOT RESULTS")

    st.caption("CYBERSENTINEL AI · Temporal Threat Radar · v1.0.0")


def render_global_banners(cfg: Settings) -> dict[str, Any]:
    """Backward-compatible wrapper injecting CSS and returning manifest metadata."""
    inject_theme_css(cfg)
    ctx = get_runtime_status_context(cfg)
    render_shell_header(
        cfg,
        title="CYBERSENTINEL AI",
        description="Temporal Threat Radar — Research Network Threat Analysis & Attack-Onset Forecasting.",
        ctx=ctx,
    )
    return ctx["manifest"]


def check_pipeline_ready(cfg: Settings) -> bool:
    """Verify core evaluation results exist; display actionable message if missing."""
    required = [
        cfg.results_dir / "verdict.json",
        cfg.results_dir / "forecast_metrics.csv",
        cfg.results_dir / "classification_metrics.json",
        cfg.results_dir / "anomaly_metrics.json",
    ]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        st.warning(
            f"Pipeline outputs missing ({', '.join(missing)}). "
            "Run the pipeline first: `python -m src.pipeline --stage all`"
        )
        return False
    return True


def ensure_replay_in_session(cfg: Settings):
    """Lazily initialize the HistoricalReplayEngine in st.session_state."""
    if "replay_engine" not in st.session_state:
        from src.replay_engine import load_replay_engine

        st.session_state["replay_engine"] = load_replay_engine(cfg)
    if "replay_running" not in st.session_state:
        st.session_state["replay_running"] = False
    if "replay_speed" not in st.session_state:
        st.session_state["replay_speed"] = cfg.replay_default_speed
    return st.session_state["replay_engine"]
