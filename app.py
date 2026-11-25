"""CYBERSENTINEL AI (Temporal Threat Radar) — Main Streamlit Dashboard Entrypoint."""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="CYBERSENTINEL AI — Temporal Threat Radar",
    page_icon="🛡️",
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
from dashboard.ui_utils import get_cfg, render_global_banners

PAGES = {
    "1. Security Command Centre": command_centre.render,
    "2. Temporal Threat Radar": threat_radar.render,
    "3. Explainable AI": explainable_ai.render,
    "4. Alert Management": alert_management.render,
    "5. Model Evaluation": model_evaluation.render,
}


def main() -> None:
    cfg = get_cfg()
    render_global_banners(cfg)

    with st.sidebar:
        st.title("Navigation")
        choice = st.radio("Select View", list(PAGES.keys()), label_visibility="collapsed")
        st.divider()
        st.caption("CYBERSENTINEL AI — Temporal Threat Radar")

    PAGES[choice](cfg)


if __name__ == "__main__":
    main()
