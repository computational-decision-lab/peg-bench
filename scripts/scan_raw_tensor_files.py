#!/usr/bin/env python3
"""Scan raw tensor files and export a basic inventory."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peg_bench.data.scan_files import scan_tensor_files


def configure_logging(log_path: str | Path) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan raw tensor files for PEG-Bench diagnosis.")
    parser.add_argument("--input_dir", type=str, default="data/raw")
    parser.add_argument("--output_dir", type=str, default="outputs/diagnosis")
    parser.add_argument("--log_file", type=str, default="outputs/logs/scan_raw_tensor_files.log")
    args = parser.parse_args()
    configure_logging(args.log_file)
    logging.info("Scanning tensor files under %s", args.input_dir)
    result = scan_tensor_files(args.input_dir, args.output_dir)
    logging.info("Inventory written to %s", result["inventory_path"])
    logging.info("Unreadable file list written to %s", result["unreadable_path"])
    logging.info("Summary written to %s", result["summary_path"])


if __name__ == "__main__":
    main()

