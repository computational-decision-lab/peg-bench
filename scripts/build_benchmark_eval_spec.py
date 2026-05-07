#!/usr/bin/env python3
"""Build a benchmark-evaluator spec JSON from method report globs."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import glob
import json


def _parse_method_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"Invalid --method value {value!r}. Expected format name=glob_pattern.")
    name, pattern = value.split("=", 1)
    name = name.strip()
    pattern = pattern.strip()
    if not name or not pattern:
        raise ValueError(f"Invalid --method value {value!r}. Expected non-empty name and glob pattern.")
    return name, pattern


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a benchmark-evaluator spec JSON from method report globs.")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--split-plan", type=str, default=None)
    parser.add_argument("--bundle-manifest", type=str, default=None)
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help="Method definition in the form name=/path/to/reports/**/*.json . Can be passed multiple times.",
    )
    args = parser.parse_args()

    methods: list[dict[str, object]] = []
    for raw in args.method:
        name, pattern = _parse_method_arg(raw)
        reports = sorted(glob.glob(pattern, recursive=True))
        methods.append({"name": name, "reports": reports})

    payload: dict[str, object] = {"methods": methods}
    if args.split_plan:
        payload["split_plan_path"] = str(Path(args.split_plan).resolve())
    if args.bundle_manifest:
        payload["bundle_manifest"] = str(Path(args.bundle_manifest).resolve())

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
