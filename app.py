"""CYBERSENTINEL AI (Temporal Threat Radar) — Production Streamlit Application Shell."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

_FAVICON_PATH = Path(__file__).resolve().parent / "assets" / "brand" / "favicon.png"
_PAGE_ICON: str = str(_FAVICON_PATH) if _FAVICON_PATH.exists() else "CS"

st.set_page_config(
    page_title="CYBERSENTINEL AI — Temporal Threat Radar",
    page_icon=_PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

from dashboard import (
    alert_management,
    command_centre,
    explainable_ai,
    model_evaluation,
    threat_radar,
)
from dashboard.ui_utils import (
    get_cfg,
    get_runtime_status_context,
    inject_theme_css,
    render_sidebar_brand_and_status,
    render_sidebar_footer_status,
)

PAGES = {
    "1. Security Command Centre": command_centre.render,
    "2. Temporal Threat Radar": threat_radar.render,
    "3. Explainable AI": explainable_ai.render,
    "4. Alert Management": alert_management.render,
    "5. Model Evaluation": model_evaluation.render,
}

_NAV_LABELS = {
    "1. Security Command Centre": "Command Centre",
    "2. Temporal Threat Radar": "Temporal Threat Radar",
    "3. Explainable AI": "Explainable AI (SHAP)",
    "4. Alert Management": "Alert Management",
    "5. Model Evaluation": "Model Evaluation",
}


def main() -> None:
    cfg = get_cfg()
    inject_theme_css(cfg)
    ctx = get_runtime_status_context(cfg)

    with st.sidebar:
        render_sidebar_brand_and_status(cfg, ctx)
        st.markdown('<div class="cs-nav-label">Operations &amp; Research</div>', unsafe_allow_html=True)
        choice = st.radio(
            "Select View",
            list(PAGES.keys()),
            format_func=lambda k: _NAV_LABELS.get(k, k),
            label_visibility="collapsed",
        )
        render_sidebar_footer_status(cfg, ctx)

    PAGES[choice](cfg)


if __name__ == "__main__":
    main()
