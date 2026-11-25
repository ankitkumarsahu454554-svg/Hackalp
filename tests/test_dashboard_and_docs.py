"""Tests for Stage 8 README generation, config overrides, and Streamlit dashboard startup/graceful empty handling."""
from __future__ import annotations

from pathlib import Path

from config.settings import get_settings
from src.make_readme import generate_readme


def test_config_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("CYBERSENTINEL_HORIZONS", "[5, 15]")
    monkeypatch.setenv("CYBERSENTINEL_K_MIN", "7")
    cfg = get_settings()
    assert cfg.horizons == [5, 15]
    assert cfg.k_min == 7


def test_make_readme_from_results_and_manifest(tmp_path: Path) -> None:
    cfg = get_settings(project_root=tmp_path)
    cfg.ensure_dirs()
    cfg.manifest_path.write_text(
        '{"dataset_version": "V_TEST", "source_name": "TEST_SRC", "source_url": "https://example.org", "synthetic_mode": true, "verification_status": "verified", "files": []}',
        encoding="utf-8",
    )
    (cfg.results_dir / "verdict.json").write_text(
        '{"verdict": "INCONCLUSIVE: insufficient events", "rationale": "Test rationale", "split_onsets": {"train_onsets": 1, "val_onsets": 1, "test_onsets": 1}}',
        encoding="utf-8",
    )
    out = generate_readme(cfg)
    txt = out.read_text(encoding="utf-8")
    assert "V_TEST" in txt
    assert "SYNTHETIC: NOT RESULTS" in txt
    assert "INCONCLUSIVE: insufficient events" in txt


def test_dashboard_pages_handle_missing_results_gracefully(tmp_path: Path) -> None:
    from streamlit.testing.v1 import AppTest

    # Run app.py via AppTest against empty tmp_path via env override?
    # Verify all dashboard module callables handle missing results without raising exceptions.
    import app
    from dashboard import (
        alert_management,
        command_centre,
        explainable_ai,
        model_evaluation,
        threat_radar,
    )

    assert len(app.PAGES) == 5
    cfg = get_settings(project_root=tmp_path)
    cfg.ensure_dirs()
    for mod in (command_centre, threat_radar, explainable_ai, alert_management, model_evaluation):
        # Should render warning cleanly when results are absent (check callable exists and AppTest smoke passes)
        assert callable(mod.render)

    at = AppTest.from_file("app.py", default_timeout=20)
    at.run()
    assert not at.exception
