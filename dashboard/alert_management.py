"""Page 4: Alert Management with Separate Detection & Forecast Alerts and Persistent Analyst Status."""
from __future__ import annotations

import json
import pandas as pd
import streamlit as st

from config.settings import Settings
from dashboard.ui_utils import check_pipeline_ready, ensure_replay_in_session, load_json_list_safe
from src.alert_manager import VALID_STATUSES


def render(cfg: Settings) -> None:
    st.title("Alert Management")
    if not check_pipeline_ready(cfg):
        return

    eng = ensure_replay_in_session(cfg)
    alerts = eng.alert_manager.to_dicts()
    source_note = "Active Replay Session Alerts"
    if not alerts:
        alerts = load_json_list_safe(cfg.results_dir / "replay_alerts.json")
        source_note = "Saved Replay Alerts (`results/replay_alerts.json`)"

    st.caption(f"Source: {source_note}. Every forecast warning is an early-warning estimate; analyst verifies before action.")
    if not alerts:
        st.info("No alerts yet. Advance the Historical Replay on **Temporal Threat Radar** to generate alerts.")
        return

    kind_filter = st.multiselect(
        "Filter Kind",
        ["FORECAST WARNING", "DETECTION"],
        default=["FORECAST WARNING", "DETECTION"],
    )
    sev_filter = st.multiselect(
        "Filter Severity",
        ["Critical", "High", "Medium", "Low"],
        default=["Critical", "High", "Medium", "Low"],
    )
    filtered = [a for a in alerts if a["kind"] in kind_filter and a["severity"] in sev_filter]

    rows = []
    status_overrides = {}
    status_file = cfg.results_dir / "alert_statuses.json"
    if status_file.exists():
        status_overrides = json.loads(status_file.read_text(encoding="utf-8"))

    for a in filtered:
        aid = a["alert_id"]
        cur_status = eng.alert_manager.statuses.get(aid, status_overrides.get(aid, a.get("analyst_status", "New")))
        reasons_txt = "; ".join(r["sentence"] for r in a.get("top_5_reasons", [])[:3])
        rows.append(
            {
                "Alert ID": aid,
                "Timestamp": a["timestamp"],
                "Kind": a["kind"],
                "Attack Category": a["attack_category"],
                "Detection Score": round(a["detection_score"], 4) if a["detection_score"] is not None else None,
                "Anomaly Score": round(a["anomaly_score"], 4) if a["anomaly_score"] is not None else None,
                "Forecast Risk": round(a["forecast_risk"], 4) if a["forecast_risk"] is not None else None,
                "Severity": a["severity"],
                "Top Reasons": reasons_txt,
                "Analyst Status": cur_status,
            }
        )

    df = pd.DataFrame(rows)
    edited = st.data_editor(
        df,
        width="stretch",
        hide_index=True,
        disabled=[c for c in df.columns if c != "Analyst Status"],
        column_config={
            "Analyst Status": st.column_config.SelectboxColumn(
                "Analyst Status",
                options=list(VALID_STATUSES),
                required=True,
            )
        },
        key="alerts_editor",
    )

    if st.button("Save Analyst Status Updates", type="primary"):
        for _, r in edited.iterrows():
            eng.alert_manager.update_status(str(r["Alert ID"]), str(r["Analyst Status"]))
        st.success(f"Saved {len(edited)} alert status entries to `results/alert_statuses.json`.")
