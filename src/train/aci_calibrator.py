"""Calibration utilities for adaptive conformal inference."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.preference.filtering import ACIConfig, AdaptiveConformalController


@dataclass(slots=True)
class ACICalibrationConfig:
    """Config for fitting an ACI controller."""

    predictions_path: str
    output_path: str
    prediction_key: str = "predictions"
    target_key: str = "targets"
    aci: ACIConfig = field(default_factory=ACIConfig)


def calibrate_aci(config: ACICalibrationConfig) -> dict[str, object]:
    """Fit and persist a conformal controller from prediction residuals."""
    payload = np.load(config.predictions_path)
    predictions = payload[config.prediction_key]
    targets = payload[config.target_key]
    controller = AdaptiveConformalController(config.aci)
    controller.fit(predictions, targets)
    controller.save(config.output_path)
    summary = {
        "alpha": controller.alpha,
        "num_scores": len(controller.nonconformity_scores),
        "output_path": str(Path(config.output_path).resolve()),
    }
    Path(config.output_path).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def cli_main() -> None:
    """CLI wrapper."""
    parser = argparse.ArgumentParser(description="Fit adaptive conformal calibration stats.")
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    calibrate_aci(ACICalibrationConfig(predictions_path=args.predictions, output_path=args.output))
