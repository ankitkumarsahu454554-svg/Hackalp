"""Stage 7: Strictly causal minute-by-minute Historical Replay Engine for test-period traffic."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import polars as pl

from config.settings import Settings, get_settings
from src.alert_manager import AlertManager, classify_forecast_severity
from src.classification import sanitize_feature_matrix
from src.explainability import explain_detection_local, explain_forecast_local
from src.feature_engineering import BASE_MINUTE_FEATURES, _rolling_stats_for_series, _shannon_entropy_from_counts

logger = logging.getLogger(__name__)


class HistoricalReplayEngine:
    """Replays test-period traffic chronologically one minute at a time without future leakage.

    At step minute ``t``:
    - Uses ONLY flows whose ``end_ts < t + 60s`` (completed at or before end of minute ``t``).
    - Computes causal minute features and rolling windows strictly over ``<= t``.
    - Runs Detection (XGBoost classifier), Anomaly (Isolation Forest), and Forecast (XGBoost + Platt).
    - Reveals ground-truth labels ONLY for minutes ``<= t``.
    """

    def __init__(
        self,
        flows_df: pl.DataFrame,
        test_minutes: list[datetime],
        detector_bundle: dict[str, Any],
        anomaly_bundle: dict[str, Any],
        forecaster_bundle: dict[str, Any],
        alert_manager: AlertManager,
        cfg: Optional[Settings] = None,
    ) -> None:
        self.cfg = cfg or get_settings()
        self.flows_df = flows_df.sort("end_ts")
        self._end_us = self.flows_df["end_ts"].cast(pl.Datetime("us", "UTC")).to_numpy().astype("datetime64[us]").astype(np.int64)
        self._gt_df = flows_df.select(["start_ts", "label", "is_attack"]).sort("start_ts")
        self._start_us = self._gt_df["start_ts"].cast(pl.Datetime("us", "UTC")).to_numpy().astype("datetime64[us]").astype(np.int64)
        self.test_minutes = sorted(test_minutes)
        self.detector_bundle = detector_bundle
        self.anomaly_bundle = anomaly_bundle
        self.forecaster_bundle = forecaster_bundle
        self.alert_manager = alert_manager

        self.det_cols: list[str] = detector_bundle["feature_columns"]
        self.det_pipe = detector_bundle["pipeline"]
        self.det_le = detector_bundle["label_encoder"]
        self.benign_idx = (
            int(np.where(self.det_le.classes_ == "BENIGN")[0][0])
            if "BENIGN" in self.det_le.classes_
            else None
        )
        self.anom_cols: list[str] = anomaly_bundle["feature_columns"]
        self.anom_imp = anomaly_bundle["imputer"]
        self.anom_model = anomaly_bundle["model"]
        self.anom_threshold = float(anomaly_bundle["threshold"])

        self.fc_cols: list[str] = forecaster_bundle["feature_columns"]
        self.fc_models = forecaster_bundle.get("seed_models") or [forecaster_bundle["primary_model"]]
        self.fc_calibrator = forecaster_bundle.get("calibrator")
        self.fc_threshold = float(forecaster_bundle["threshold"])
        self.fc_risk_tiers = forecaster_bundle["risk_tiers"]
        self.horizon_min = int(forecaster_bundle["horizon_min"])

        import shap
        try:
            self._fc_explainer = shap.TreeExplainer(forecaster_bundle["primary_model"])
        except Exception:
            self._fc_explainer = None
        try:
            self._det_explainer = shap.TreeExplainer(self.det_pipe.named_steps["model"])
        except Exception:
            self._det_explainer = None

        self.reset()

    def reset(self) -> None:
        """Reset replay simulation cursor and accumulated causal history."""
        self.cursor: int = 0
        self.running: bool = False
        self.total_flows_processed: int = 0
        self.total_malicious_detected: int = 0
        self.total_anomalous_flows: int = 0
        self.history: list[dict[str, Any]] = []
        self._base_series: dict[str, list[float]] = {k: [] for k in BASE_MINUTE_FEATURES}
        self._last_seen_src: dict[str, int] = {}
        self.alert_manager.alerts.clear()
        self.alert_manager._seen_ids.clear()

    @property
    def is_finished(self) -> bool:
        return self.cursor >= len(self.test_minutes)

    @property
    def current_minute(self) -> Optional[datetime]:
        if self.cursor == 0:
            return None
        return self.test_minutes[self.cursor - 1]

    def _compute_minute_base_features(self, bin_flows: pl.DataFrame, minute_dt: datetime) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Score flows completed in minute ``t`` and aggregate purely causal minute features."""
        n_flows = bin_flows.height
        if n_flows == 0:
            zeros = {k: 0.0 for k in BASE_MINUTE_FEATURES}
            return zeros, np.empty(0), np.empty(0), np.empty(0, dtype=str), np.empty(0)

        # Module B: Out-of-time Isolation Forest
        X_anom = self.anom_imp.transform(sanitize_feature_matrix(bin_flows.select(self.anom_cols).to_numpy()))
        anom_scores = (-self.anom_model.score_samples(X_anom)).astype(np.float64)
        anom_flags = (anom_scores >= self.anom_threshold).astype(np.int8)

        # Module A: Classifier Detection
        X_det = sanitize_feature_matrix(bin_flows.select(self.det_cols).to_numpy())
        det_proba = self.det_pipe.predict_proba(X_det)
        det_pred_idx = np.argmax(det_proba, axis=1)
        det_pred_labels = self.det_le.inverse_transform(det_pred_idx)
        det_atk_probs = (1.0 - det_proba[:, self.benign_idx]) if self.benign_idx is not None else np.max(det_proba, axis=1)

        # Causal traffic aggregates
        durations = bin_flows["Flow Duration"].cast(pl.Float64).clip(lower_bound=0.0).to_numpy()
        pkts = (bin_flows["Total Fwd Packet"].cast(pl.Float64).fill_null(0.0) + bin_flows["Total Bwd packets"].cast(pl.Float64).fill_null(0.0)).to_numpy()
        byts = (bin_flows["Total Length of Fwd Packet"].cast(pl.Float64).fill_null(0.0) + bin_flows["Total Length of Bwd Packet"].cast(pl.Float64).fill_null(0.0)).to_numpy()
        syns = (bin_flows["SYN Flag Count"].cast(pl.Float64).fill_null(0.0) > 0).to_numpy().astype(np.float64)
        plens = bin_flows["Packet Length Mean"].cast(pl.Float64).fill_null(0.0).to_numpy()
        dports = bin_flows["Dst Port"].to_numpy()
        srcs = bin_flows["Src IP"].to_list()
        dsts = bin_flows["Dst IP"].to_list()

        _, dp_cnts = np.unique(dports, return_counts=True)
        _, sr_cnts = np.unique(srcs, return_counts=True)

        m_us = int(np.datetime64(minute_dt.replace(tzinfo=None), "us").astype(np.int64))
        cutoff_us = m_us - 60 * 60 * 1_000_000
        unseen_cnt = sum(1 for s in srcs if (self._last_seen_src.get(s) is None or self._last_seen_src[s] < cutoff_us))
        for s in set(srcs):
            self._last_seen_src[s] = m_us

        base = {
            "flow_count": float(n_flows),
            "packets_per_sec": float(pkts.sum() / self.cfg.bin_seconds),
            "bytes_per_sec": float(byts.sum() / self.cfg.bin_seconds),
            "mean_flow_duration_us": float(np.mean(durations)),
            "median_flow_duration_us": float(np.median(durations)),
            "unique_src_ips": float(len(sr_cnts)),
            "unique_dst_ips": float(len(set(dsts))),
            "unique_dst_ports": float(len(dp_cnts)),
            "dst_port_entropy": _shannon_entropy_from_counts(dp_cnts),
            "src_ip_entropy": _shannon_entropy_from_counts(sr_cnts),
            "syn_flag_rate": float(np.mean(syns)),
            "mean_packet_length": float(np.mean(plens)),
            "short_flow_fraction": float(np.mean(durations <= 100_000.0)),
            "unseen_src_60m_fraction": float(unseen_cnt / n_flows),
            "anom_mean": float(np.mean(anom_scores)),
            "anom_p95": float(np.percentile(anom_scores, 95.0)),
            "anom_frac": float(np.mean(anom_flags)),
        }
        return base, anom_scores, anom_flags, det_pred_labels, det_atk_probs

    def step(self) -> Optional[dict[str, Any]]:
        """Advance historical replay by exactly 1 minute and return the step state."""
        if self.is_finished:
            self.running = False
            return None

        t = self.test_minutes[self.cursor]
        t_end = t + timedelta(seconds=self.cfg.bin_seconds)
        t_us = int(np.datetime64(t.replace(tzinfo=None), "us").astype(np.int64))
        t_end_us = int(np.datetime64(t_end.replace(tzinfo=None), "us").astype(np.int64))

        # STRICT CAUSAL SLICE: only flows whose END time is in [t, t_end)
        # Never touch any row with end_ts >= t_end!
        lo = int(np.searchsorted(self._end_us, t_us, side="left"))
        hi = int(np.searchsorted(self._end_us, t_end_us, side="left"))
        bin_flows = self.flows_df.slice(lo, max(hi - lo, 0))
        base, anom_scores, anom_flags, det_labels, det_probs = self._compute_minute_base_features(bin_flows, t)

        for k in BASE_MINUTE_FEATURES:
            self._base_series[k].append(base[k])

        # Build current minute's full causal rolling feature vector
        feat_dict: dict[str, float] = dict(base)
        for k in BASE_MINUTE_FEATURES:
            arr = np.array(self._base_series[k][-max(self.cfg.feature_windows) :], dtype=np.float64)
            rolled = _rolling_stats_for_series(arr, self.cfg.feature_windows, k)
            for rk, rarr in rolled.items():
                feat_dict[rk] = float(rarr[-1])

        X_fc_row = np.array([[feat_dict[c] for c in self.fc_cols]], dtype=np.float64)
        raw_probs = [float(m.predict_proba(X_fc_row)[0, 1]) for m in self.fc_models]
        raw_mean = float(np.mean(raw_probs))
        if self.fc_calibrator is not None:
            clipped = float(np.clip(raw_mean, 1e-6, 1.0 - 1e-6))
            logit = np.array([[np.log(clipped / (1.0 - clipped))]])
            cal_risk = float(self.fc_calibrator.predict_proba(logit)[0, 1])
        else:
            cal_risk = raw_mean

        risk_state = classify_forecast_severity(cal_risk, self.fc_risk_tiers)

        # Aggregate detection counts for minute t
        n_flows = bin_flows.height
        mal_mask = det_labels != "BENIGN"
        n_mal = int(np.sum(mal_mask)) if n_flows else 0
        n_anom = int(np.sum(anom_flags)) if n_flows else 0
        self.total_flows_processed += n_flows
        self.total_malicious_detected += n_mal
        self.total_anomalous_flows += n_anom

        ts_str = t.strftime("%Y-%m-%d %H:%M:%S UTC")

        # Raise FORECAST WARNING if above frozen threshold (after initial 30-min warm-up inside replay or using history)
        fc_alert = None
        if len(self._base_series["flow_count"]) >= max(self.cfg.feature_windows) and cal_risk >= self.fc_threshold:
            reasons_fc, _ = explain_forecast_local(self.forecaster_bundle, X_fc_row, top_k=5, explainer=self._fc_explainer)
            fc_alert = self.alert_manager.add_forecast_warning(
                timestamp=ts_str,
                forecast_risk=cal_risk,
                horizon_min=self.horizon_min,
                top_5_reasons=reasons_fc,
                anomaly_score=base["anom_mean"],
            )

        # Raise top DETECTION alert for minute t if malicious or strongly anomalous flow observed
        det_alert = None
        if n_flows > 0 and (n_mal > 0 or n_anom > 0):
            score_rank = det_probs + 0.25 * anom_flags
            top_i = int(np.argmax(score_rank))
            if det_probs[top_i] >= 0.50 or anom_flags[top_i] == 1:
                x_det_row = bin_flows.slice(top_i, 1).select(self.det_cols).to_numpy()
                pred_cls = str(det_labels[top_i]) if det_labels[top_i] != "BENIGN" else "ANOMALOUS_TRAFFIC"
                reasons_det, _ = explain_detection_local(self.detector_bundle, x_det_row, str(det_labels[top_i]), top_k=5, explainer=self._det_explainer)
                det_alert = self.alert_manager.add_detection_alert(
                    timestamp=ts_str,
                    attack_category=pred_cls,
                    detection_score=float(det_probs[top_i]),
                    anomaly_score=float(anom_scores[top_i]),
                    top_5_reasons=reasons_det,
                    suffix=str(top_i),
                )

        # Ground truth revealed ONLY for flows whose start_ts falls in minute t (<= current replay time!)
        # NEVER passed to any model above.
        gt_lo = int(np.searchsorted(self._start_us, t_us, side="left"))
        gt_hi = int(np.searchsorted(self._start_us, t_end_us, side="left"))
        started_in_t = self._gt_df.slice(gt_lo, max(gt_hi - gt_lo, 0))
        gt_attack_starts = int(started_in_t.filter(pl.col("is_attack") == 1).height)
        gt_attack_active = gt_attack_starts >= self.cfg.k_min
        gt_labels = sorted(set(started_in_t.filter(pl.col("is_attack") == 1)["label"].to_list()))

        step_record = {
            "minute_ts": t,
            "timestamp_str": ts_str,
            "flows_completed": n_flows,
            "malicious_detected": n_mal,
            "anomalous_flows": n_anom,
            "forecast_risk": cal_risk,
            "risk_state": risk_state,
            "anom_mean": base["anom_mean"],
            "bytes_per_sec": base["bytes_per_sec"],
            "gt_attack_starts": gt_attack_starts,
            "gt_attack_active": gt_attack_active,
            "gt_labels": gt_labels,
            "forecast_alert_id": fc_alert.alert_id if fc_alert else None,
            "detection_alert_id": det_alert.alert_id if det_alert else None,
        }
        self.history.append(step_record)
        self.cursor += 1
        return step_record

    def visible_ground_truth_up_to_cursor(self) -> list[dict[str, Any]]:
        """Return revealed ground truth strictly up to current replay timestamp (never future)."""
        return [
            {
                "minute_ts": h["minute_ts"],
                "gt_attack_active": h["gt_attack_active"],
                "gt_attack_starts": h["gt_attack_starts"],
                "gt_labels": h["gt_labels"],
            }
            for h in self.history
        ]


def load_replay_engine(cfg: Optional[Settings] = None) -> HistoricalReplayEngine:
    """Load test-period flows (with pre-test context on test day) and trained bundles for replay."""
    settings = cfg or get_settings()
    summary = json.loads((settings.results_dir / "forecast_training_summary.json").read_text(encoding="utf-8"))
    primary_h = int(settings.horizons[0])

    det_bundle = joblib.load(settings.models_dir / "detector_xgb.joblib")
    anom_bundle = joblib.load(settings.models_dir / "isolation_forest.joblib")
    fc_bundle = joblib.load(settings.models_dir / f"forecaster_xgb_h{primary_h}.joblib")

    from src.alert_manager import build_alert_manager_from_results

    alert_mgr = build_alert_manager_from_results(settings)
    test_start_str = summary["split_info"]["test_start"]
    test_end_str = summary["split_info"]["test_end"]
    t_start = datetime.fromisoformat(test_start_str.replace("Z", "+00:00"))
    if t_start.tzinfo is None:
        from datetime import timezone
        t_start = t_start.replace(tzinfo=timezone.utc)
    t_end = datetime.fromisoformat(test_end_str.replace("Z", "+00:00"))
    if t_end.tzinfo is None:
        from datetime import timezone
        t_end = t_end.replace(tzinfo=timezone.utc)

    # Load only flows covering [t_start - 60m, t_end + 60s] for lookback context + test replay
    needed_cols = sorted(
        set(
            det_bundle["feature_columns"]
            + anom_bundle["feature_columns"]
            + ["start_ts", "end_ts", "Src IP", "Dst IP", "Dst Port", "Flow Duration", "Total Fwd Packet", "Total Bwd packets", "Total Length of Fwd Packet", "Total Length of Bwd Packet", "SYN Flag Count", "Packet Length Mean", "label", "is_attack"]
        )
    )
    from src.data_loader import get_processed_parquet_paths

    parquets = get_processed_parquet_paths(settings)
    context_start = t_start - timedelta(minutes=max(settings.feature_windows) + 30)
    parts = []
    for pq in parquets:
        sub = pl.scan_parquet(str(pq)).filter(
            (pl.col("end_ts") >= pl.lit(context_start)) & (pl.col("start_ts") <= pl.lit(t_end + timedelta(seconds=60)))
        ).select(needed_cols).collect()
        if sub.height > 0:
            parts.append(sub)
    flows_df = pl.concat(parts).sort("end_ts")

    # Build contiguous minute list covering [context_start_test_warm_up .. t_end]
    # Note: warm up the causal rolling state from t_start - max(FEATURE_WINDOWS) to t_start - 1 without emitting test alerts before t_start,
    # OR start test minutes cleanly at t_start using 30-min prior context if available.
    test_minutes: list[datetime] = []
    cur = t_start
    while cur <= t_end:
        test_minutes.append(cur)
        cur += timedelta(minutes=1)

    engine = HistoricalReplayEngine(
        flows_df=flows_df,
        test_minutes=test_minutes,
        detector_bundle=det_bundle,
        anomaly_bundle=anom_bundle,
        forecaster_bundle=fc_bundle,
        alert_manager=alert_mgr,
        cfg=settings,
    )
    # Seed rolling buffers and 60-min source-lookback causally using context minutes before t_start
    warmup_m = max(max(settings.feature_windows), 60)
    for offset in range(warmup_m, 0, -1):
        wt = t_start - timedelta(minutes=offset)
        wt_end = wt + timedelta(seconds=settings.bin_seconds)
        w_flows = flows_df.filter((pl.col("end_ts") >= pl.lit(wt)) & (pl.col("end_ts") < pl.lit(wt_end)))
        base, _, _, _, _ = engine._compute_minute_base_features(w_flows, wt)
        for k in BASE_MINUTE_FEATURES:
            engine._base_series[k].append(base[k])

    return engine
