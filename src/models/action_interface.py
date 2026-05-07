"""Shared forecast-to-action wrapper for PEG-Bench-style evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(slots=True)
class ActionInterfaceConfig:
    """Config for converting forecasts into benchmark position targets."""

    cost_bps: float
    delta: float = 0.0
    rho: float = 0.9


@dataclass(slots=True)
class ActionInterfaceResult:
    """Raw and wrapped action views plus lightweight audit statistics."""

    threshold_decimal: float
    raw_positions: np.ndarray
    wrapped_positions: np.ndarray
    raw_deltas: np.ndarray
    wrapped_deltas: np.ndarray

    def summary(self) -> dict[str, float]:
        """Return compact scalar diagnostics for audit/report payloads."""
        raw_nonzero = float(np.mean(np.abs(self.raw_positions) > 0.0)) if len(self.raw_positions) else 0.0
        wrapped_nonzero = float(np.mean(np.abs(self.wrapped_positions) > 0.0)) if len(self.wrapped_positions) else 0.0
        raw_turnover = float(np.sum(np.abs(self.raw_deltas))) if len(self.raw_deltas) else 0.0
        wrapped_turnover = float(np.sum(np.abs(self.wrapped_deltas))) if len(self.wrapped_deltas) else 0.0
        return {
            "threshold_decimal": float(self.threshold_decimal),
            "raw_nonzero_rate": raw_nonzero,
            "wrapped_nonzero_rate": wrapped_nonzero,
            "raw_mean_abs_position": float(np.mean(np.abs(self.raw_positions))) if len(self.raw_positions) else 0.0,
            "wrapped_mean_abs_position": float(np.mean(np.abs(self.wrapped_positions))) if len(self.wrapped_positions) else 0.0,
            "raw_turnover": raw_turnover,
            "wrapped_turnover": wrapped_turnover,
        }


def cost_threshold_decimal(cost_bps: float, delta: float = 0.0) -> float:
    """Convert a basis-point cost level into a decimal action threshold."""
    return max(float(cost_bps) / 10_000.0 + float(delta), 0.0)


def raw_forecast_positions(y_pred: np.ndarray | list[float], cost_bps: float, delta: float = 0.0) -> np.ndarray:
    """Map return forecasts into {-1, 0, +1} position targets."""
    predictions = np.asarray(y_pred, dtype=np.float32)
    threshold = cost_threshold_decimal(cost_bps, delta=delta)
    raw = np.sign(predictions)
    raw[np.abs(predictions) <= threshold] = 0.0
    return np.clip(raw, -1.0, 1.0).astype(np.float32)


def smooth_positions(raw_positions: np.ndarray | list[float], rho: float = 0.9) -> np.ndarray:
    """Apply simple exponential smoothing to target positions."""
    positions = np.asarray(raw_positions, dtype=np.float32).reshape(-1)
    if len(positions) == 0:
        return positions
    weight = float(np.clip(rho, 0.0, 0.999999))
    smoothed = np.zeros_like(positions, dtype=np.float32)
    previous = 0.0
    for index, value in enumerate(positions):
        previous = weight * previous + (1.0 - weight) * float(value)
        smoothed[index] = float(np.clip(previous, -1.0, 1.0))
    return smoothed.reshape(np.asarray(raw_positions, dtype=np.float32).shape)


def position_targets_to_deltas(positions: np.ndarray | list[float]) -> np.ndarray:
    """Convert target positions into delta-position simulator actions."""
    targets = np.asarray(positions, dtype=np.float32).reshape(-1)
    if len(targets) == 0:
        return targets
    previous = np.concatenate([np.zeros(1, dtype=np.float32), targets[:-1]], axis=0)
    return (targets - previous).astype(np.float32).reshape(np.asarray(positions, dtype=np.float32).shape)


def forecast_to_action(y_pred, cost_bps, delta: float = 0.0, rho: float = 0.9):
    """
    Convert return forecasts into benchmark position targets in [-1, 1].

    The benchmark wrapper first applies a cost-aware deadzone and sign mapping,
    then smooths the resulting target positions to reduce churn.
    """
    raw_positions = raw_forecast_positions(y_pred, cost_bps=cost_bps, delta=delta)
    wrapped_positions = smooth_positions(raw_positions, rho=rho)
    return np.clip(wrapped_positions, -1.0, 1.0).astype(np.float32)


def build_action_interface_result(
    y_pred: np.ndarray | list[float],
    *,
    cost_bps: float,
    delta: float = 0.0,
    rho: float = 0.9,
) -> ActionInterfaceResult:
    """Build raw/wrapped position targets and simulator deltas for one forecast series."""
    raw_positions = raw_forecast_positions(y_pred, cost_bps=cost_bps, delta=delta)
    wrapped_positions = forecast_to_action(y_pred, cost_bps=cost_bps, delta=delta, rho=rho)
    return ActionInterfaceResult(
        threshold_decimal=cost_threshold_decimal(cost_bps, delta=delta),
        raw_positions=raw_positions,
        wrapped_positions=wrapped_positions,
        raw_deltas=position_targets_to_deltas(raw_positions),
        wrapped_deltas=position_targets_to_deltas(wrapped_positions),
    )


def serialize_action_interface_result(result: ActionInterfaceResult) -> dict[str, object]:
    """Serialize action-interface diagnostics without embedding full arrays."""
    payload = asdict(result)
    payload["raw_positions"] = None
    payload["wrapped_positions"] = None
    payload["raw_deltas"] = None
    payload["wrapped_deltas"] = None
    payload["summary"] = result.summary()
    return payload
