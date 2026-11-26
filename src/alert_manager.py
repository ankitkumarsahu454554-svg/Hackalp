"""Stage 6: Alert Manager with separate Detection and Forecast Warning streams and validation-derived tiers."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

VALID_STATUSES = ("New", "Acknowledged", "Escalated", "Resolved", "Dismissed")


@dataclass
class AlertRecord:
    alert_id: str
    timestamp: str
    kind: str  # "DETECTION" or "FORECAST WARNING"
    attack_category: str
    detection_score: Optional[float]
    anomaly_score: Optional[float]
    forecast_risk: Optional[float]
    horizon_min: Optional[int]
    severity: str
    advisory_note: str
    top_5_reasons: list[dict[str, Any]]
    analyst_status: str = "New"


def classify_forecast_severity(risk: float, risk_tiers: dict[str, float]) -> str:
    """Map calibrated forecast risk to Low/Medium/High/Critical using validation-derived tiers."""
    med = float(risk_tiers["medium"])
    high = float(risk_tiers["high"])
    crit = float(risk_tiers["critical"])
    if risk >= crit:
        return "Critical"
    if risk >= high:
        return "High"
    if risk >= med:
        return "Medium"
    return "Low"


def classify_detection_severity(
    detection_score: float,
    anomaly_score: float,
    anom_p95: float,
    anom_p99: float,
) -> str:
    """Map classifier attack probability + anomaly score to severity using out-of-time baseline quantiles."""
    if detection_score >= 0.90 and anomaly_score >= anom_p99:
        return "Critical"
    if detection_score >= 0.75 or anomaly_score >= anom_p99:
        return "High"
    if detection_score >= 0.50 or anomaly_score >= anom_p95:
        return "Medium"
    return "Low"


class AlertManager:
    """Manages separate DETECTION and FORECAST WARNING alerts with persistent analyst status."""

    def __init__(
        self,
        risk_tiers: dict[str, float],
        anom_p95: float,
        anom_p99: float,
        status_store_path: Optional[Path] = None,
    ) -> None:
        self.risk_tiers = risk_tiers
        self.anom_p95 = float(anom_p95)
        self.anom_p99 = float(anom_p99)
        self.status_store_path = status_store_path
        self.statuses: dict[str, str] = self._load_statuses()
        self.alerts: list[AlertRecord] = []
        self._seen_ids: set[str] = set()

    def _load_statuses(self) -> dict[str, str]:
        if self.status_store_path and self.status_store_path.exists():
            return json.loads(self.status_store_path.read_text(encoding="utf-8"))
        return {}

    def _save_statuses(self) -> None:
        if self.status_store_path:
            self.status_store_path.parent.mkdir(parents=True, exist_ok=True)
            self.status_store_path.write_text(json.dumps(self.statuses, indent=2), encoding="utf-8")

    def add_forecast_warning(
        self,
        timestamp: str,
        forecast_risk: float,
        horizon_min: int,
        top_5_reasons: list[dict[str, Any]],
        anomaly_score: Optional[float] = None,
    ) -> Optional[AlertRecord]:
        """Raise a FORECAST WARNING when calibrated risk meets the validation-derived Medium threshold."""
        severity = classify_forecast_severity(forecast_risk, self.risk_tiers)
        if severity == "Low":
            return None
        aid = f"FCW-{timestamp.replace(':', '').replace(' ', '_')}"
        if aid in self._seen_ids:
            return None
        self._seen_ids.add(aid)
        rec = AlertRecord(
            alert_id=aid,
            timestamp=timestamp,
            kind="FORECAST WARNING",
            attack_category="Pre-onset Risk (Not Confirmed Attack)",
            detection_score=None,
            anomaly_score=float(anomaly_score) if anomaly_score is not None else None,
            forecast_risk=float(forecast_risk),
            horizon_min=int(horizon_min),
            severity=severity,
            advisory_note=f"Early warning (H={horizon_min} min), analyst verifies before action.",
            top_5_reasons=top_5_reasons,
            analyst_status=self.statuses.get(aid, "New"),
        )
        self.alerts.append(rec)
        return rec

    def add_detection_alert(
        self,
        timestamp: str,
        attack_category: str,
        detection_score: float,
        anomaly_score: float,
        top_5_reasons: list[dict[str, Any]],
        suffix: str = "0",
    ) -> AlertRecord:
        """Raise a DETECTION alert for observed malicious or anomalous traffic."""
        severity = classify_detection_severity(detection_score, anomaly_score, self.anom_p95, self.anom_p99)
        aid = f"DET-{timestamp.replace(':', '').replace(' ', '_')}-{suffix}"
        if aid in self._seen_ids:
            aid = f"{aid}-{len(self.alerts)}"
        self._seen_ids.add(aid)
        rec = AlertRecord(
            alert_id=aid,
            timestamp=timestamp,
            kind="DETECTION",
            attack_category=attack_category,
            detection_score=float(detection_score),
            anomaly_score=float(anomaly_score),
            forecast_risk=None,
            horizon_min=None,
            severity=severity,
            advisory_note="Observed flow flagged by classifier/anomaly detector; analyst verifies before action.",
            top_5_reasons=top_5_reasons,
            analyst_status=self.statuses.get(aid, "New"),
        )
        self.alerts.append(rec)
        return rec

    def update_status(self, alert_id: str, new_status: str) -> None:
        """Update and persist analyst workflow status for an alert."""
        if new_status not in VALID_STATUSES:
            raise ValueError(f"Invalid analyst status '{new_status}'. Choose from {VALID_STATUSES}.")
        self.statuses[alert_id] = new_status
        for rec in self.alerts:
            if rec.alert_id == alert_id:
                rec.analyst_status = new_status
        self._save_statuses()

    def to_dicts(self) -> list[dict[str, Any]]:
        return [asdict(a) for a in self.alerts]

    def save_alerts(self, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.to_dicts(), indent=2), encoding="utf-8")


def build_alert_manager_from_results(cfg: Optional[Settings] = None) -> AlertManager:
    """Instantiate AlertManager using validation-derived thresholds stored in results/."""
    settings = cfg or get_settings()
    summary_path = settings.results_dir / "forecast_training_summary.json"
    anom_path = settings.results_dir / "anomaly_metrics.json"
    if not summary_path.exists() or not anom_path.exists():
        raise FileNotFoundError(
            "Missing forecast_training_summary.json or anomaly_metrics.json. Run pipeline stages first."
        )
    fc_meta = json.loads(summary_path.read_text(encoding="utf-8"))
    anom_meta = json.loads(anom_path.read_text(encoding="utf-8"))
    return AlertManager(
        risk_tiers=fc_meta["validation_risk_tiers"],
        anom_p95=float(anom_meta["threshold"]),
        anom_p99=float(anom_meta["threshold_p99_reference"]),
        status_store_path=settings.results_dir / "alert_statuses.json",
    )
