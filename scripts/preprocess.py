#!/usr/bin/env python3
"""Run raw-data preprocessing from YAML config."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import yaml

from src.data.peg_bench_dataset import PEGBenchPreprocessConfig, build_peg_bench_bundle
from src.data.pipeline import DataPipelineConfig, run_data_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw L2 + trade data into KAIROS-X bars.")
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    args = parser.parse_args()
    payload = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    pipeline_type = payload.pop("pipeline_type", "single_day")
    if pipeline_type == "peg_bench":
        result = build_peg_bench_bundle(PEGBenchPreprocessConfig(**payload))
    else:
        result = run_data_pipeline(DataPipelineConfig(**payload))
    print(result)


if __name__ == "__main__":
    main()
