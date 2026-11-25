"""Tests for Stage 1A data acquisition safety, verification, idempotence, and offline fallback."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from config.settings import get_settings
from src.data_acquisition import acquire_data, compute_sha256, extract_zip_safe


def _write_sample_csv(path: Path) -> None:
    path.write_text(
        "Flow ID,Src IP,Dst IP,Timestamp,Flow Duration,Label\n"
        "10.0.0.1-10.0.0.2-80-12345-6,10.0.0.1,10.0.0.2,2017-07-03 12:00:00.000000,1000,BENIGN\n",
        encoding="utf-8",
    )


def test_offline_writes_data_needed_and_synthetic_flag(tmp_path: Path) -> None:
    cfg = get_settings(project_root=tmp_path)
    result = acquire_data(cfg, internet_checker=lambda _sess: False)
    assert result.success is False
    assert result.synthetic_mode is True
    assert (tmp_path / "DATA_NEEDED.md").exists()
    manifest = json.loads(cfg.manifest_path.read_text(encoding="utf-8"))
    assert manifest["synthetic_mode"] is True
    assert manifest["verification_status"] == "missing_real_data"
    assert (cfg.results_dir / "data_provenance.md").exists()


def test_local_existing_files_verified_and_skipped_without_overwrite(tmp_path: Path) -> None:
    cfg = get_settings(project_root=tmp_path)
    cfg.ensure_dirs()
    csv_path = cfg.data_raw_dir / "monday_sample.csv"
    _write_sample_csv(csv_path)
    original_bytes = csv_path.read_bytes()

    res1 = acquire_data(cfg, internet_checker=lambda _sess: False)
    assert res1.success is True
    assert res1.synthetic_mode is False
    assert cfg.manifest_path.exists()
    manifest = json.loads(cfg.manifest_path.read_text(encoding="utf-8"))
    assert manifest["files"][0]["sha256"] == compute_sha256(csv_path)

    # Re-run: skips cleanly and never overwrites raw file
    res2 = acquire_data(cfg, internet_checker=lambda _sess: False)
    assert res2.success is True
    assert csv_path.read_bytes() == original_bytes


def test_size_or_hash_mismatch_detected_without_overwrite(tmp_path: Path) -> None:
    cfg = get_settings(project_root=tmp_path)
    cfg.ensure_dirs()
    csv_path = cfg.data_raw_dir / "tuesday_sample.csv"
    _write_sample_csv(csv_path)
    acquire_data(cfg, internet_checker=lambda _sess: False)

    # Corrupt file externally to simulate tamper/mismatch against manifest
    csv_path.write_text("CORRUPTED_CONTENT\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="mismatch"):
        acquire_data(cfg, internet_checker=lambda _sess: False)
    assert csv_path.read_text(encoding="utf-8") == "CORRUPTED_CONTENT\n"


def test_zip_slip_blocked(tmp_path: Path) -> None:
    zpath = tmp_path / "evil.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("../../outside.csv", "a,b\n1,2\n")
    with pytest.raises(ValueError, match="Unsafe zip member"):
        extract_zip_safe(zpath, tmp_path / "extract", {})
