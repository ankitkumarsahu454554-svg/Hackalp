"""Shared dashboard helpers for loading verified results, banners, and replay state."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import polars as pl
import streamlit as st

from config.settings import Settings, get_settings
from src.data_loader import is_synthetic_mode


def get_cfg() -> Settings:
    return get_settings()


def load_json_safe(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_list_safe(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv_safe(path: Path) -> Optional[pl.DataFrame]:
    if not path.exists():
        return None
    return pl.read_csv(path)


def render_global_banners(cfg: Settings) -> dict[str, Any]:
    """Render mandatory top disclaimer banner, synthetic warning banner if active, and sidebar provenance."""
    st.markdown(
        """
        <style>
        .cs-banner {
            background: linear-gradient(90deg, #0b1628 0%, #10243e 100%);
            border-left: 4px solid #00D4FF;
            padding: 10px 16px;
            border-radius: 6px;
            margin-bottom: 12px;
            font-size: 0.92rem;
            color: #E2E8F0;
        }
        .cs-synth-banner {
            background: #5c1018;
            border: 1px solid #FF4D4F;
            padding: 10px 16px;
            border-radius: 6px;
            margin-bottom: 12px;
            color: #FFE4E6;
            font-weight: 700;
        }
        </style>
        <div class="cs-banner">
            <strong>CYBERSENTINEL AI (Temporal Threat Radar)</strong> &mdash;
            Historical replay of recorded data. Forecasts are risk estimates, not confirmed attacks.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if is_synthetic_mode(cfg):
        st.markdown(
            '<div class="cs-synth-banner">SYNTHETIC: NOT RESULTS &mdash; Running on synthetic smoke data only.</div>',
            unsafe_allow_html=True,
        )

    manifest = load_json_safe(cfg.manifest_path) or {}
    with st.sidebar:
        st.markdown("### Dataset Provenance")
        st.caption(f"**Source**: {manifest.get('source_name', 'Not fetched yet')}")
        st.caption(f"**Version**: {manifest.get('dataset_version', 'N/A')}")
        st.caption(f"**Status**: `{manifest.get('verification_status', 'missing')}`")
        if manifest.get("synthetic_mode"):
            st.error("SYNTHETIC: NOT RESULTS")
    return manifest


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
