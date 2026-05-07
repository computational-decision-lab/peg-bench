"""Adaptive conformal filtering for preference pairs and inference control."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import json
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class ACIConfig:
    """Adaptive conformal configuration."""

    alpha: float = 0.1
    alpha_target: float = 0.1
    gamma: float = 0.01
    uncertainty_threshold: float = 0.5
    min_pair_gap: float = 0.1


@dataclass(slots=True)
class ConfidenceBand:
    """Prediction interval for a utility or outcome estimate."""

    lower: float
    upper: float

    @property
    def width(self) -> float:
        return float(self.upper - self.lower)

    def overlaps(self, other: "ConfidenceBand") -> bool:
        return not (self.upper < other.lower or other.upper < self.lower)


class AdaptiveConformalController:
    """Adaptive conformal inference with pair filtering and action scaling."""

    def __init__(self, config: ACIConfig | None = None) -> None:
        self.config = config or ACIConfig()
        self.alpha = self.config.alpha
        self.nonconformity_scores: list[float] = []

    def update(self, error: float) -> float:
        """Online update of the significance level."""
        self.alpha = float(np.clip(self.alpha + self.config.gamma * (error - self.config.alpha_target), 1e-3, 0.5))
        return self.alpha

    def fit(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """Calibrate on a validation stream."""
        scores = np.abs(predictions - targets)
        self.nonconformity_scores = scores.astype(np.float32).tolist()

    def interval(self, center: float) -> ConfidenceBand:
        """Construct a conformal interval around a point estimate."""
        if not self.nonconformity_scores:
            radius = 0.0
        else:
            quantile = float(np.quantile(self.nonconformity_scores, 1.0 - self.alpha))
            radius = max(quantile, 0.0)
        return ConfidenceBand(lower=center - radius, upper=center + radius)

    def keep_preference(self, winner_value: float, loser_value: float) -> tuple[bool, ConfidenceBand, ConfidenceBand]:
        """Decide whether a pair is informative enough for training."""
        winner_band = self.interval(winner_value)
        loser_band = self.interval(loser_value)
        keep = (not winner_band.overlaps(loser_band)) and abs(winner_value - loser_value) >= self.config.min_pair_gap
        return keep, winner_band, loser_band

    def scale_action(self, action: float, uncertainty: float) -> float:
        """Reduce or abstain when uncertainty is too high."""
        if uncertainty >= self.config.uncertainty_threshold:
            return 0.0
        scale = max(0.0, 1.0 - uncertainty / max(self.config.uncertainty_threshold, 1e-8))
        return float(action * scale)

    def save(self, path: str | Path) -> None:
        """Persist calibration state to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"config": asdict(self.config), "alpha": self.alpha, "scores": self.nonconformity_scores}
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "AdaptiveConformalController":
        """Load saved conformal state."""
        payload = json.loads(Path(path).read_text())
        controller = cls(ACIConfig(**payload["config"]))
        controller.alpha = float(payload["alpha"])
        controller.nonconformity_scores = list(payload["scores"])
        return controller
