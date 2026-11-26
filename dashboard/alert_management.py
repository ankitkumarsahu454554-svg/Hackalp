"""Page 4: Alert Management — SOC Alert Triage, Search, Filtering, and Persistent Workflow."""
from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from components.ui import (
    render_alert_card,
    render_empty_state,
    render_metric_grid,
    render_section_header,
    render_severity_badge,
)
from config.settings import Settings
from dashboard.ui_utils import (
    check_pipeline_ready,
    ensure_replay_in_session,
    get_runtime_status_context,
    render_shell_header,
)
from src.alert_manager import VALID_STATUSES
from utils.data_loaders import read_saved_replay_alerts

_SEVERITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3, "Informational": 4}
_ALL_SEVERITIES = ["Critical", "High", "Medium", "Low", "Informational"]


def render(cfg: Settings) -> None:
    ctx = get_runtime_status_context(cfg)
    render_shell_header(
        cfg,
        title="Alert Management",
        description="SOC triage queue separating pre-onset Forecast Warnings from observed Detection alerts, with search, filtering, SHAP reasons, and persistent status workflow.",
        ctx=ctx,
    )
    if not check_pipeline_ready(cfg):
        return

    eng = ensure_replay_in_session(cfg)
    alerts = eng.alert_manager.to_dicts()
    source_note = "Active Replay Session Alerts"
    if not alerts:
        alerts = read_saved_replay_alerts(cfg)
        source_note = "Saved Replay Alerts (`results/replay_alerts.json`)"

    if not alerts:
        render_empty_state(
            "No Alerts in Queue",
            "Advance the Historical Replay on Command Centre or Temporal Threat Radar to populate Detection and Forecast Warning alerts.",
        )
        return

    status_overrides: dict[str, str] = {}
    status_file = cfg.results_dir / "alert_statuses.json"
    if status_file.exists():
        status_overrides = json.loads(status_file.read_text(encoding="utf-8"))

    def _cur_status(a: dict[str, Any]) -> str:
        aid = a["alert_id"]
        return str(eng.alert_manager.statuses.get(aid, status_overrides.get(aid, a.get("analyst_status", "New"))))

    # Summary KPI cards
    sev_counts = {s: sum(1 for a in alerts if a.get("severity") == s) for s in _ALL_SEVERITIES}
    new_count = sum(1 for a in alerts if _cur_status(a) == "New")
    render_metric_grid(
        [
            {"label": "Total Alerts", "value": f"{len(alerts):,}", "sub": source_note, "badge": "Queue", "tone": "cyan"},
            {"label": "Critical", "value": f"{sev_counts['Critical']:,}", "sub": "Immediate analyst triage", "badge": "Critical", "tone": "critical"},
            {"label": "High", "value": f"{sev_counts['High']:,}", "sub": "Elevated confidence / risk", "badge": "High", "tone": "warning"},
            {"label": "Medium", "value": f"{sev_counts['Medium']:,}", "sub": "Validation threshold tier", "badge": "Medium", "tone": "warning"},
            {"label": "Low / Info", "value": f"{sev_counts['Low'] + sev_counts['Informational']:,}", "sub": f"Unreviewed (New): {new_count:,}", "badge": "Triage", "tone": "healthy"},
        ]
    )

    # Search, Filter & Sort Toolbar
    render_section_header("Filter & Search Queue", "Every forecast warning is an early-warning estimate; analyst verifies before containment.")
    f1, f2, f3, f4, f5 = st.columns([1.4, 1.1, 1.1, 1.0, 1.0])
    search_q = f1.text_input("Search ID, Category, Timestamp, Reason", value="", placeholder="e.g. FCW-, Botnet, Dst Port", key="am_search")
    kind_filter = f2.multiselect("Kind", ["FORECAST WARNING", "DETECTION"], default=["FORECAST WARNING", "DETECTION"], key="am_kind")
    sev_filter = f3.multiselect("Severity", _ALL_SEVERITIES, default=_ALL_SEVERITIES, key="am_sev")
    status_filter = f4.multiselect("Analyst Status", list(VALID_STATUSES), default=list(VALID_STATUSES), key="am_status")
    sort_by = f5.selectbox("Sort Order", ["Newest First", "Oldest First", "Severity (High -> Low)"], key="am_sort")

    q_low = search_q.strip().lower()
    filtered: list[dict[str, Any]] = []
    for a in alerts:
        if a.get("kind") not in kind_filter:
            continue
        if a.get("severity") not in sev_filter:
            continue
        if _cur_status(a) not in status_filter:
            continue
        if q_low:
            hay = " ".join(
                [
                    str(a.get("alert_id", "")),
                    str(a.get("timestamp", "")),
                    str(a.get("attack_category", "")),
                    " ".join(r.get("sentence", "") for r in a.get("top_5_reasons", [])),
                ]
            ).lower()
            if q_low not in hay:
                continue
        filtered.append(a)

    if sort_by == "Newest First":
        filtered.sort(key=lambda x: str(x.get("timestamp", "")), reverse=True)
    elif sort_by == "Oldest First":
        filtered.sort(key=lambda x: str(x.get("timestamp", "")))
    else:
        filtered.sort(key=lambda x: (_SEVERITY_ORDER.get(str(x.get("severity")), 9), str(x.get("timestamp", ""))))

    if not filtered:
        render_empty_state("No Alerts Match Filters", "Adjust your search query or severity/status filters above.")
        return

    rows = []
    for a in filtered:
        aid = a["alert_id"]
        reasons_txt = "; ".join(r["sentence"] for r in a.get("top_5_reasons", [])[:2])
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
                "Analyst Status": _cur_status(a),
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
        height=320,
        key="alerts_editor",
    )

    if st.button("Save Batch Status Updates", type="primary", key="am_save_batch"):
        for _, r in edited.iterrows():
            eng.alert_manager.update_status(str(r["Alert ID"]), str(r["Analyst Status"]))
        st.success(f"Persisted {len(edited)} alert status entries to `results/alert_statuses.json`.")
        st.rerun()

    # Selected Alert Detail & Persistent Triage Actions
    render_section_header("Selected Alert Inspector & Persistent Actions", "Inspect SHAP explanations and apply persistent triage workflow transitions.")
    id_options = [f"{a['alert_id']} | {a['severity']} | {a['kind']} | {_cur_status(a)}" for a in filtered]
    sel_idx = st.selectbox("Inspect Alert", range(len(id_options)), format_func=lambda i: id_options[i], key="am_inspect")
    sel = filtered[sel_idx]
    sel_id = sel["alert_id"]

    d_left, d_right = st.columns([1.3, 1.0])
    with d_left:
        render_alert_card({**sel, "analyst_status": _cur_status(sel)})
        st.markdown("**Top SHAP Feature Explanations**")
        reasons = sel.get("top_5_reasons", [])
        if reasons:
            for idx, r in enumerate(reasons, 1):
                st.markdown(f"{idx}. {r['sentence']}")
        else:
            st.caption("No local SHAP features recorded.")

    with d_right:
        st.markdown(
            f"**Current Status:** `{_cur_status(sel)}` &nbsp; " + render_severity_badge(sel.get("severity", "Informational")),
            unsafe_allow_html=True,
        )
        st.caption(sel.get("advisory_note", ""))
        b1, b2 = st.columns(2)
        b3, b4 = st.columns(2)
        if b1.button("Acknowledge", key="am_ack", width="stretch"):
            eng.alert_manager.update_status(sel_id, "Acknowledged")
            st.success(f"{sel_id} marked Acknowledged.")
            st.rerun()
        if b2.button("Resolve", key="am_res", width="stretch"):
            eng.alert_manager.update_status(sel_id, "Resolved")
            st.success(f"{sel_id} marked Resolved.")
            st.rerun()
        if b3.button("Escalate", key="am_esc", width="stretch"):
            eng.alert_manager.update_status(sel_id, "Escalated")
            st.warning(f"{sel_id} marked Escalated.")
            st.rerun()
        if b4.button("Dismiss", key="am_dis", width="stretch"):
            eng.alert_manager.update_status(sel_id, "Dismissed")
            st.info(f"{sel_id} marked Dismissed.")
            st.rerun()

    # Alert History (Persisted Status Overrides)
    if status_overrides or eng.alert_manager.statuses:
        merged_statuses = {**status_overrides, **eng.alert_manager.statuses}
        reviewed_rows = [
            {"Alert ID": k, "Persisted Analyst Status": v}
            for k, v in merged_statuses.items()
            if v != "New"
        ]
        if reviewed_rows:
            with st.expander(f"Persisted Analyst Action History ({len(reviewed_rows)} reviewed alerts)"):
                st.dataframe(pd.DataFrame(reviewed_rows), width="stretch", hide_index=True)
