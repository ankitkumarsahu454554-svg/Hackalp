"""Dedicated Historical Replay Engine tests for end-time causality, delayed label reveal, and controls."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import polars as pl

from src.alert_manager import AlertManager
from src.replay_engine import HistoricalReplayEngine
from tests.test_stages_5_to_7 import _build_replay_test_fixture


def test_replay_causality_and_controls_step_pause_reset(tmp_path: Path) -> None:
    cfg, flows_df, det_bundle, anom_bundle, fc_bundle, base = _build_replay_test_fixture(tmp_path)
    test_mins = [base + timedelta(minutes=m) for m in range(25)]

    am = AlertManager(fc_bundle["risk_tiers"], 0.55, 0.70)
    eng = HistoricalReplayEngine(flows_df, test_mins, det_bundle, anom_bundle, fc_bundle, am, cfg)

    assert eng.cursor == 0
    assert not eng.is_finished
    assert eng.visible_ground_truth_up_to_cursor() == []

    # Step 15 minutes; ground truth reveals strictly up to minute 14 (no attack starts until min 20)
    for _ in range(15):
        eng.step()
    assert eng.cursor == 15
    vis = eng.visible_ground_truth_up_to_cursor()
    assert len(vis) == 15
    assert all(v["gt_attack_active"] is False for v in vis)

    # Step through minute 22 (after attack starts at min 20) -> ground truth reveals only then
    for _ in range(8):
        eng.step()
    vis2 = eng.visible_ground_truth_up_to_cursor()
    assert len(vis2) == 23
    assert any(v["gt_attack_active"] is True for v in vis2[20:])

    # Reset clears state and hides revealed labels again
    eng.reset()
    assert eng.cursor == 0
    assert eng.visible_ground_truth_up_to_cursor() == []
    assert eng.total_flows_processed == 0
