#!/usr/bin/env python3
"""Train the VQ-VAE from YAML config."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import yaml

from src.models.vqvae import VQVAEConfig
from src.train.vqvae_trainer import VQVAETrainConfig, train_vqvae


def main() -> None:
    parser = argparse.ArgumentParser(description="Train KAIROS-X VQ-VAE.")
    parser.add_argument("--config", type=str, default="configs/vqvae.yaml")
    args = parser.parse_args()
    payload = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    model_cfg = VQVAEConfig(**payload.pop("model"))
    result = train_vqvae(VQVAETrainConfig(model=model_cfg, **payload))
    print(result)


if __name__ == "__main__":
    main()
