#!/usr/bin/env python3
"""Check chronological continuity and timestamp quality."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peg_bench.data.diagnose_time import diagnose_time_continuity


def configure_logging(log_path: str | Path) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Check date continuity and timestamp anomalies.")
    parser.add_argument("--manifest", type=str, default="outputs/manifests/data_manifest.csv")
    parser.add_argument("--output_dir", type=str, default="outputs/diagnosis")
    parser.add_argument("--expected_days", type=int, default=30)
    parser.add_argument("--tolerance_days", type=int, default=3)
    parser.add_argument("--log_file", type=str, default="outputs/logs/check_time_continuity.log")
    args = parser.parse_args()
    configure_logging(args.log_file)
    result = diagnose_time_continuity(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        expected_days=args.expected_days,
        tolerance_days=args.tolerance_days,
    )
    logging.info("Continuity report written to %s", result["continuity_path"])
    logging.info("Missing dates written to %s", result["missing_path"])
    logging.info("Duplicate dates written to %s", result["duplicate_path"])
    logging.info("Timestamp anomalies written to %s", result["anomaly_path"])


if __name__ == "__main__":
    main()

