#!/usr/bin/env python3
"""Build a window-level CSV index for one PEG-Bench split panel."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json

from peg_bench.data.window_index import build_panel_window_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a segmented window index from a PEG-Bench split panel.")
    parser.add_argument("--panel", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--target-key", type=str, default="return_20")
    args = parser.parse_args()

    summary = build_panel_window_index(
        panel_path=args.panel,
        output_path=args.output,
        seq_len=args.seq_len,
        stride=args.stride,
        target_key=args.target_key,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
