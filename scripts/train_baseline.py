#!/usr/bin/env python3
"""Train the Kairos baseline from YAML config."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import yaml

from src.models.baseline import KairosBaselineConfig
from src.train.baseline_trainer import BaselineTrainConfig, train_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Kairos baseline.")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    args = parser.parse_args()
    payload = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    model_cfg = KairosBaselineConfig(**payload.pop("model"))
    result = train_baseline(BaselineTrainConfig(model=model_cfg, **payload))
    print(result)


if __name__ == "__main__":
    main()
