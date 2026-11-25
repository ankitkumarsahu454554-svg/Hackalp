"""Typed data-loading and schema helpers for CYBERSENTINEL AI."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional, Sequence

import polars as pl

from config.settings import Settings, get_settings

DAY_ORDER = ["monday", "tuesday", "wednesday", "thursday", "friday"]

EXCLUDED_METADATA_AND_LEAKAGE_COLS = {
    "id",
    "Flow ID",
    "Src IP",
    "Src Port",
    "Dst IP",
    "Timestamp",
    "Label",
    "Attempted Category",
    "raw_label",
    "label",
    "is_attack",
    "start_ts",
    "end_ts",
    "day",
}

SENTINEL_MINUS_ONE_COLS = {
    "FWD Init Win Bytes",
    "Bwd Init Win Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "ICMP Code",
    "ICMP Type",
    "Attempted Category",
}

NON_NEGATIVE_PREFIXES = (
    "Flow Duration",
    "Total ",
    "Fwd Packet Length",
    "Bwd Packet Length",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Packet Length",
    "FIN Flag",
    "SYN Flag",
    "RST Flag",
    "PSH Flag",
    "ACK Flag",
    "URG Flag",
    "CWR Flag",
    "ECE Flag",
    "Fwd PSH",
    "Bwd PSH",
    "Fwd URG",
    "Bwd URG",
    "Fwd RST",
    "Bwd RST",
    "Average Packet Size",
    "Subflow ",
    "Active ",
    "Idle ",
    "Fwd Seg Size",
    "Bwd Segment",
    "Fwd Segment",
    "Fwd Act Data Pkts",
)


def load_manifest_safe(cfg: Settings) -> dict[str, Any]:
    """Read ``DATA_MANIFEST.json`` if present; raise FileNotFoundError otherwise."""
    if not cfg.manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {cfg.manifest_path}. Run `python -m src.pipeline --stage fetch` first."
        )
    return json.loads(cfg.manifest_path.read_text(encoding="utf-8"))


def is_synthetic_mode(cfg: Optional[Settings] = None) -> bool:
    """Return True if synthetic_mode is flagged in config or DATA_MANIFEST.json."""
    settings = cfg or get_settings()
    if settings.synthetic_mode:
        return True
    if settings.manifest_path.exists():
        data = json.loads(settings.manifest_path.read_text(encoding="utf-8"))
        return bool(data.get("synthetic_mode", False))
    return False


def get_raw_csv_paths(cfg: Optional[Settings] = None) -> list[Path]:
    """Return chronologically ordered raw flow CSVs from manifest or ``data/raw``."""
    settings = cfg or get_settings()
    if not settings.data_raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {settings.data_raw_dir}. Run `python -m src.pipeline --stage fetch`."
        )
    csvs = list(settings.data_raw_dir.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No raw CSV files found in {settings.data_raw_dir}. "
            "Run `python -m scripts.fetch_data` or inspect DATA_NEEDED.md."
        )

    def _sort_key(p: Path) -> tuple[int, str]:
        stem = p.stem.lower()
        for idx, day in enumerate(DAY_ORDER):
            if day in stem:
                return (idx, p.name.lower())
        return (len(DAY_ORDER), p.name.lower())

    return sorted(csvs, key=_sort_key)


def get_processed_parquet_paths(cfg: Optional[Settings] = None) -> list[Path]:
    """Return ordered processed daily Parquet paths; raise actionable error if missing."""
    settings = cfg or get_settings()
    parquets = list(settings.data_processed_dir.glob("flows_*.parquet"))
    if not parquets:
        raise FileNotFoundError(
            f"No processed Parquet files found in {settings.data_processed_dir}. "
            "Run `python -m src.pipeline --stage audit` first."
        )

    def _sort_key(p: Path) -> tuple[int, str]:
        stem = p.stem.lower()
        for idx, day in enumerate(DAY_ORDER):
            if day in stem:
                return (idx, p.name.lower())
        return (len(DAY_ORDER), p.name.lower())

    return sorted(parquets, key=_sort_key)


def scan_processed_flows(cfg: Optional[Settings] = None, day: Optional[str] = None) -> pl.LazyFrame:
    """Return a lazy Polars scan over cleaned processed Parquet files."""
    paths = get_processed_parquet_paths(cfg)
    if day is not None:
        matched = [p for p in paths if day.lower() in p.stem.lower()]
        if not matched:
            raise FileNotFoundError(f"No processed Parquet found for day='{day}'. Available: {[p.name for p in paths]}")
        paths = matched
    return pl.scan_parquet([str(p) for p in paths])


def get_feature_columns(schema_names: Sequence[str], include_dst_port: bool = True) -> list[str]:
    """Return ML flow feature columns, strictly excluding IDs, IPs, timestamps, Src Port, and labels."""
    cols: list[str] = []
    for name in schema_names:
        if name in EXCLUDED_METADATA_AND_LEAKAGE_COLS:
            continue
        if not include_dst_port and name == "Dst Port":
            continue
        cols.append(name)
    return cols


def normalize_label_text(value: Optional[str]) -> str:
    """Normalize corrupted dashes and extra whitespace in CICIDS2017 label strings."""
    if value is None:
        return "UNKNOWN"
    text = str(value).strip()
    # Replace common corrupted / Unicode dashes (en-dash, em-dash, minus, replacement char)
    text = re.sub(r"[\u2010-\u2015\u2212\ufffd]+", "-", text)
    # Fix UNB double-space dash corruption such as "Web Attack  Brute Force"
    text = re.sub(r"\s{2,}", " - ", text)
    text = re.sub(r"\s*-\s*", " - ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Preserve standard compact labels like FTP-Patator / SSH-Patator
    text = text.replace("FTP - Patator", "FTP-Patator").replace("SSH - Patator", "SSH-Patator")
    if text.upper() == "BENIGN":
        return "BENIGN"
    return text


def infer_day_name(csv_path: Path) -> str:
    """Extract canonical weekday name or stem from a CSV filename."""
    stem = csv_path.stem.lower()
    for day in DAY_ORDER:
        if day in stem:
            return day
    return stem


def parse_timestamp_column(ts_values: list[str]) -> tuple[list[Optional[str]], dict[str, Any]]:
    """Disambiguate 12-hour clock timestamps without AM/PM when needed; pass through ISO 24h UTC.

    Rule for 12-hour timestamps (e.g. ``dd/mm/yyyy H:MM`` without AM/PM):
    Capture runs roughly 09:00-17:00 local time. Hours 01..07 are inferred PM (+12h),
    hours 08..11 are AM, and hour 12 is noon (PM). Monotonic transitions across 12->1
    are tracked; any row whose monotonic check disagrees with the window rule is counted.
    """
    sample = next((v.strip() for v in ts_values if v and str(v).strip()), "")
    iso_24h_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(\.\d+)?$")
    if iso_24h_pattern.match(sample):
        return ts_values, {
            "format_detected": "ISO-24h UTC with microseconds (YYYY-MM-DD HH:MM:SS.ffffff)",
            "timezone_assumption": "UTC (09:00-17:00 ADT corresponds to ~12:00-20:00 UTC in July)",
            "ampm_inferred_rows": 0,
            "ampm_monotonic_disagreements": 0,
        }

    from datetime import datetime, timedelta

    parsed_strs: list[Optional[str]] = []
    inferred_pm = 0
    disagreements = 0
    prev_dt: Optional[datetime] = None
    saw_afternoon = False
    formats = ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M")

    for raw in ts_values:
        txt = str(raw).strip() if raw is not None else ""
        if not txt:
            parsed_strs.append(None)
            continue
        if re.search(r"\b(AM|PM)\b", txt, re.I):
            for fmt in ("%d/%m/%Y %I:%M:%S %p", "%m/%d/%Y %I:%M:%S %p", "%d/%m/%Y %I:%M %p"):
                try:
                    dt = datetime.strptime(txt, fmt)
                    parsed_strs.append(dt.strftime("%Y-%m-%d %H:%M:%S.%f"))
                    prev_dt = dt
                    break
                except ValueError:
                    continue
            else:
                parsed_strs.append(None)
            continue

        dt_base: Optional[datetime] = None
        for fmt in formats:
            try:
                dt_base = datetime.strptime(txt, fmt)
                break
            except ValueError:
                continue
        if dt_base is None:
            parsed_strs.append(None)
            continue

        # Apply capture-window heuristic: 01:00..07:59 -> 13:00..19:59
        dt_adj = dt_base
        if 1 <= dt_base.hour <= 7:
            dt_adj = dt_base + timedelta(hours=12)
            inferred_pm += 1
            saw_afternoon = True
        elif dt_base.hour == 12:
            saw_afternoon = True
        elif saw_afternoon and prev_dt is not None and dt_adj < prev_dt - timedelta(minutes=30):
            # Monotonic check disagrees with simple window rule
            disagreements += 1
            if dt_adj.hour < 12:
                dt_adj = dt_adj + timedelta(hours=12)
                inferred_pm += 1

        prev_dt = dt_adj
        parsed_strs.append(dt_adj.strftime("%Y-%m-%d %H:%M:%S.%f"))

    return parsed_strs, {
        "format_detected": "12-hour or legacy slash-date format (disambiguated to 24h)",
        "timezone_assumption": "America/Moncton ADT (09:00-17:00 capture window heuristic)",
        "ampm_inferred_rows": inferred_pm,
        "ampm_monotonic_disagreements": disagreements,
    }
