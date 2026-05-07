#!/usr/bin/env python3
"""Build a standardized manifest from raw tensor files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peg_bench.data.build_manifest import build_data_manifest


def configure_logging(log_path: str | Path) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data manifest for PEG-Bench diagnosis.")
    parser.add_argument("--input_dir", type=str, default="data/raw")
    parser.add_argument("--output", type=str, default="outputs/manifests/data_manifest.csv")
    parser.add_argument("--diagnosis_output_dir", type=str, default="outputs/diagnosis")
    parser.add_argument("--inventory_path", type=str, default="")
    parser.add_argument("--log_file", type=str, default="outputs/logs/build_data_manifest.log")
    args = parser.parse_args()
    configure_logging(args.log_file)
    inventory_path = args.inventory_path or None
    result = build_data_manifest(
        input_dir=args.input_dir,
        manifest_output=args.output,
        diagnosis_output_dir=args.diagnosis_output_dir,
        inventory_path=inventory_path,
    )
    logging.info("Manifest written to %s", result["manifest_path"])
    logging.info("Tensor schema written to %s", result["schema_path"])
    logging.info("Key summary written to %s", result["key_summary_path"])


if __name__ == "__main__":
    main()

