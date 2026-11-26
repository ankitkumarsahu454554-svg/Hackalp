"""Memory-safe, cached data loaders for the Streamlit dashboard."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import polars as pl
import streamlit as st

from config.settings import Settings


@st.cache_data(show_spinner=False)
def load_text_cached(path_str: str) -> Optional[str]:
    """Read a text or SVG asset once and cache it across UI reruns."""
    p = Path(path_str)
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def load_json_cached(path_str: str, mtime_ns: int = 0) -> Optional[Any]:
    """Load and parse JSON with cache key invalidated on file modification."""
    del mtime_ns
    p = Path(path_str)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_csv_cached(path_str: str, mtime_ns: int = 0) -> Optional[pd.DataFrame]:
    """Load small result CSV files into pandas with cache invalidation."""
    del mtime_ns
    p = Path(path_str)
    if not p.exists():
        return None
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def load_forecast_timeline_cached(path_str: str, mtime_ns: int = 0) -> Optional[pd.DataFrame]:
    """Load the compact minute-level forecast predictions timeline (approx 34 KB)."""
    del mtime_ns
    p = Path(path_str)
    if not p.exists():
        return None
    df = pl.read_parquet(p).sort("minute_ts").to_pandas()
    df["minute_ts"] = pd.to_datetime(df["minute_ts"], utc=True)
    return df


def _mtime_or_zero(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def read_json_artifact(path: Path) -> Optional[Any]:
    return load_json_cached(str(path), _mtime_or_zero(path))


def read_csv_artifact(path: Path) -> Optional[pd.DataFrame]:
    return load_csv_cached(str(path), _mtime_or_zero(path))


def read_forecast_timeline(cfg: Settings) -> Optional[pd.DataFrame]:
    p = cfg.results_dir / "forecast_predictions.parquet"
    return load_forecast_timeline_cached(str(p), _mtime_or_zero(p))


def read_saved_replay_alerts(cfg: Settings) -> list[dict[str, Any]]:
    data = read_json_artifact(cfg.results_dir / "replay_alerts.json")
    return data if isinstance(data, list) else []
