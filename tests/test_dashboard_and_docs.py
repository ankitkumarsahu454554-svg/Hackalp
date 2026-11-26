"""Tests for Stage 8 README generation, config overrides, brand assets, and redesigned 5-page UI."""
from __future__ import annotations

from pathlib import Path

from config.settings import PROJECT_ROOT, get_settings
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


def test_brand_assets_and_design_system_present() -> None:
    brand_dir = PROJECT_ROOT / "assets" / "brand"
    for fname in (
        "cybersentinel_icon.svg",
        "cybersentinel_logo_primary.svg",
        "cybersentinel_logo_light.svg",
        "cybersentinel_logo_sidebar.svg",
        "favicon.svg",
        "favicon.png",
    ):
        assert (brand_dir / fname).exists(), f"Missing brand asset: {fname}"
    assert (PROJECT_ROOT / "assets" / "styles" / "theme.css").exists()


def test_dashboard_all_five_pages_render_and_step(tmp_path: Path) -> None:
    from streamlit.testing.v1 import AppTest

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
        assert callable(mod.render)

    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    assert not at.exception

    step_btns = [b for b in at.button if "Step +10 min" in b.label]
    if step_btns:
        step_btns[0].click().run(timeout=60)
        assert not at.exception

    for opt in at.radio[0].options[1:]:
        at.radio[0].set_value(opt).run(timeout=60)
        assert not at.exception, f"Exception on {opt}: {at.exception}"
