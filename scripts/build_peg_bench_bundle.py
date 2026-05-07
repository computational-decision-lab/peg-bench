#!/usr/bin/env python3
"""Build a PEG-Bench-style preprocessing bundle from raw Tardis files."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json

from src.data.peg_bench_dataset import PEGBenchPreprocessConfig, build_peg_bench_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PEG-Bench preprocessing bundle from raw Tardis feeds.")
    parser.add_argument("--raw-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--train-days", type=int, default=None)
    parser.add_argument("--val-days", type=int, default=None)
    parser.add_argument("--test-days", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--interval-ms", type=int, default=1000)
    parser.add_argument("--volume-threshold", type=float, default=100_000.0)
    parser.add_argument("--max-days", type=int, default=None)
    args = parser.parse_args()

    bundle = build_peg_bench_bundle(
        PEGBenchPreprocessConfig(
            raw_root=args.raw_root,
            output_root=args.output_root,
            start_date=args.start_date,
            end_date=args.end_date,
            train_days=args.train_days,
            val_days=args.val_days,
            test_days=args.test_days,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            depth=args.depth,
            interval_ms=args.interval_ms,
            volume_threshold=args.volume_threshold,
            max_days=args.max_days,
        )
    )
    print(json.dumps(bundle["coverage"], indent=2))


if __name__ == "__main__":
    main()
