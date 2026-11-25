"""Stage-by-stage CLI orchestrator for CYBERSENTINEL AI."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


def _run_fetch(cfg: Settings) -> None:
    from src.data_acquisition import acquire_data

    res = acquire_data(cfg)
    if not res.success:
        raise RuntimeError(
            f"Data acquisition failed; see {res.data_needed_path}. Errors: {res.errors}"
        )


def _run_audit(cfg: Settings) -> None:
    from src.preprocessing import run_audit_and_prepare

    run_audit_and_prepare(cfg)


def _run_classify(cfg: Settings) -> None:
    from src.classification import run_classification

    run_classification(cfg)


def _run_anomaly(cfg: Settings) -> None:
    from src.anomaly_detection import run_anomaly_detection

    run_anomaly_detection(cfg)


def _run_forecast(cfg: Settings) -> None:
    from src.forecasting import run_forecasting

    run_forecasting(cfg)


def _run_evaluate(cfg: Settings) -> None:
    from src.evaluation import run_evaluation
    from src.explainability import run_global_explainability
    from src.forecasting import run_k_min_sensitivity

    run_evaluation(cfg)
    run_k_min_sensitivity(cfg)
    run_global_explainability(cfg)


def _run_explain(cfg: Settings) -> None:
    from src.explainability import run_global_explainability

    run_global_explainability(cfg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CYBERSENTINEL AI Temporal Threat Radar pipeline CLI")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["fetch", "audit", "classify", "anomaly", "forecast", "evaluate", "explain", "all"],
        help="Pipeline stage to execute.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    cfg = get_settings()
    cfg.ensure_dirs()

    dispatch = {
        "fetch": _run_fetch,
        "audit": _run_audit,
        "classify": _run_classify,
        "anomaly": _run_anomaly,
        "forecast": _run_forecast,
        "evaluate": _run_evaluate,
        "explain": _run_explain,
    }
    if args.stage == "all":
        for name in ("fetch", "audit", "classify", "anomaly", "forecast", "evaluate"):
            logger.info("Running stage: %s", name)
            dispatch[name](cfg)
    else:
        dispatch[args.stage](cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
