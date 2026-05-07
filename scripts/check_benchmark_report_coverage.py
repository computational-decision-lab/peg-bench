#!/usr/bin/env python3
"""Check expected-vs-observed day-level report coverage for a benchmark split."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json

from src.eval.benchmark_evaluator import load_method_specs_from_json
from src.eval.report_coverage import summarize_report_coverage, write_report_coverage_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Check benchmark day-level report coverage for one split.")
    parser.add_argument("--spec", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-md", type=str, required=True)
    args = parser.parse_args()

    method_specs, split_lookup, _ = load_method_specs_from_json(args.spec)
    summary = summarize_report_coverage(
        method_specs=method_specs,
        split_lookup=split_lookup,
        evaluation_split=args.split,
    )
    paths = write_report_coverage_outputs(
        summary=summary,
        csv_path=args.output_csv,
        json_path=args.output_json,
        markdown_path=args.output_md,
    )
    print(json.dumps({"paths": paths, "methods": summary["methods"]}, indent=2))


if __name__ == "__main__":
    main()
