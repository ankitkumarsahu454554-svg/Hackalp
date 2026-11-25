"""Central typed configuration for CYBERSENTINEL AI (Temporal Threat Radar).

Every setting is overridable via dataclass instantiation, keyword arguments,
or ``CYBERSENTINEL_*`` environment variables.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BIN_SECONDS: int = 60
QUIET_GAP_MIN: int = 5
K_MIN: int = 5
HORIZONS: list[int] = [5]
FEATURE_WINDOWS: list[int] = [5, 15, 30]
SEGMENT_BREAK_MIN: int = 60
ALERT_MERGE_GAP_MIN: int = 2
FALSE_ALERT_BUDGET_PER_HOUR: float = 1.0
MIN_TRAIN_ONSETS: int = 5
MIN_VAL_ONSETS: int = 5
MIN_TEST_ONSETS: int = 3
BOOTSTRAP_REPS: int = 500
BOOTSTRAP_BLOCK_MIN: int = 30
SEEDS: list[int] = [0, 1, 2, 3, 4]
REPLAY_DEFAULT_SPEED: int = 10
MAX_DOWNLOAD_GB: float = 15.0
SECONDARY_DATASETS: list[str] = []


@dataclass
class Settings:
    """Runtime configuration container for all pipeline stages."""

    project_root: Path = PROJECT_ROOT
    bin_seconds: int = BIN_SECONDS
    quiet_gap_min: int = QUIET_GAP_MIN
    k_min: int = K_MIN
    horizons: list[int] = field(default_factory=lambda: list(HORIZONS))
    feature_windows: list[int] = field(default_factory=lambda: list(FEATURE_WINDOWS))
    segment_break_min: int = SEGMENT_BREAK_MIN
    alert_merge_gap_min: int = ALERT_MERGE_GAP_MIN
    false_alert_budget_per_hour: float = FALSE_ALERT_BUDGET_PER_HOUR
    min_train_onsets: int = MIN_TRAIN_ONSETS
    min_val_onsets: int = MIN_VAL_ONSETS
    min_test_onsets: int = MIN_TEST_ONSETS
    bootstrap_reps: int = BOOTSTRAP_REPS
    bootstrap_block_min: int = BOOTSTRAP_BLOCK_MIN
    seeds: list[int] = field(default_factory=lambda: list(SEEDS))
    replay_default_speed: int = REPLAY_DEFAULT_SPEED
    max_download_gb: float = MAX_DOWNLOAD_GB
    secondary_datasets: list[str] = field(default_factory=lambda: list(SECONDARY_DATASETS))

    # CPU / RAM safety caps (tuned for ~8 GB RAM laptop)
    classifier_max_train_rows: int = 200_000
    classifier_max_eval_rows: int = 150_000
    anomaly_max_train_rows: int = 150_000
    anomaly_batch_size: int = 100_000
    shap_max_explain_rows: int = 2000
    duckdb_memory_limit: str = "1500MB"
    duckdb_threads: int = 2

    # Replay / synthetic flags
    synthetic_mode: bool = False

    @property
    def data_raw_dir(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def data_processed_dir(self) -> Path:
        return self.project_root / "data" / "processed"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def results_dir(self) -> Path:
        return self.project_root / "results"

    @property
    def manifest_path(self) -> Path:
        return self.data_raw_dir / "DATA_MANIFEST.json"

    @property
    def split_gap_min(self) -> int:
        return max(self.feature_windows) + max(self.horizons)

    def ensure_dirs(self) -> None:
        """Create required project directories if missing."""
        for p in (self.data_raw_dir, self.data_processed_dir, self.models_dir, self.results_dir):
            p.mkdir(parents=True, exist_ok=True)


def _parse_env_value(raw: str, current: Any) -> Any:
    if isinstance(current, bool):
        return raw.strip().lower() in {"1", "true", "yes"}
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        stripped = raw.strip()
        if stripped.startswith("["):
            return json.loads(stripped)
        if not stripped:
            return []
        if current and isinstance(current[0], int):
            return [int(x.strip()) for x in stripped.split(",") if x.strip()]
        return [x.strip() for x in stripped.split(",") if x.strip()]
    if isinstance(current, Path):
        return Path(raw)
    return raw


def get_settings(**overrides: Any) -> Settings:
    """Return a ``Settings`` object with ``CYBERSENTINEL_*`` env and explicit overrides applied."""
    cfg = Settings()
    for field_name in cfg.__dataclass_fields__:
        env_key = f"CYBERSENTINEL_{field_name.upper()}"
        if env_key in os.environ:
            current = getattr(cfg, field_name)
            setattr(cfg, field_name, _parse_env_value(os.environ[env_key], current))
    for key, val in overrides.items():
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown setting override: {key}")
        setattr(cfg, key, val)
    return cfg
