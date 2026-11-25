"""CLI entrypoint for resumable, idempotent dataset acquisition."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from src.data_acquisition import acquire_data


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch and verify CYBERSENTINEL AI datasets.")
    parser.add_argument("--max-download-gb", type=float, default=None, help="Override MAX_DOWNLOAD_GB cap.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    overrides = {}
    if args.max_download_gb is not None:
        overrides["max_download_gb"] = args.max_download_gb
    cfg = get_settings(**overrides)
    result = acquire_data(cfg)

    print(
        json.dumps(
            {
                "success": result.success,
                "synthetic_mode": result.synthetic_mode,
                "source_name": result.source_name,
                "source_url": result.source_url,
                "verified_csv_count": len(result.verified_csvs),
                "manifest_path": str(result.manifest_path),
                "provenance_path": str(result.provenance_path),
                "data_needed_path": str(result.data_needed_path) if result.data_needed_path else None,
                "errors": result.errors,
            },
            indent=2,
        )
    )
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
