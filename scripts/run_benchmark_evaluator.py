#!/usr/bin/env python3
"""Run the benchmark-level evaluator over per-day method reports."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json

from src.eval.benchmark_evaluator import (
    BenchmarkEvaluatorConfig,
    evaluate_benchmark_methods,
    load_method_specs_from_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate daily reports into a PEG-Bench report card.")
    parser.add_argument("--spec", type=str, required=True, help="JSON file describing method names and report paths.")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path.")
    parser.add_argument("--markdown", type=str, default=None, help="Optional markdown report output path.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()

    method_specs, split_lookup, spec_payload = load_method_specs_from_json(args.spec)
    result = evaluate_benchmark_methods(
        method_specs,
        split_lookup=split_lookup,
        config=BenchmarkEvaluatorConfig(
            output_path=args.output,
            markdown_path=args.markdown,
            evaluation_split=args.split,
            top_k=args.top_k,
            bootstrap_samples=args.bootstrap_samples,
        ),
    )
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({"methods": list(result["methods"].keys()), "leaderboard": result["leaderboard"]}, indent=2))
    if spec_payload.get("bundle_manifest"):
        print(f"Used split index from {spec_payload['bundle_manifest']}")


if __name__ == "__main__":
    main()
