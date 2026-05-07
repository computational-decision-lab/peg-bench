#!/usr/bin/env python3
"""Build split-level PEG-Bench panels without starting model training."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json

from peg_bench.data.panel_builder import SPLITS, build_split_panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PEG-Bench split panels from a processed bundle.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--mode", type=str, default="time", choices=["time", "volume"])
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--split-plan", type=str, default=None)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for split in SPLITS:
        panel_path = output_root / f"{args.mode}_{split}.npz"
        summaries[split] = build_split_panel(
            manifest_path=args.manifest,
            split=split,
            mode=args.mode,
            output_path=panel_path,
            split_plan_path=args.split_plan,
        )
    summary_path = output_root / f"{args.mode}_panel_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
