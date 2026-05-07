"""Trajectory-level utility and business metrics."""

from __future__ import annotations

from dataclasses import dataclass

from src.simulation.execution import TrajectoryRecord


@dataclass(slots=True)
class UtilityConfig:
    """Config for scalar utility construction."""

    lambda_dd: float = 0.5
    lambda_turn: float = 0.1
    liquidation_penalty: float = 5.0


class TrajectoryUtilityCalculator:
    """Compute scalar and report-friendly trajectory utilities."""

    def __init__(self, config: UtilityConfig | None = None) -> None:
        self.config = config or UtilityConfig()

    def scalar_utility(self, trajectory: TrajectoryRecord) -> float:
        """Scalar utility used for the default oracle mode."""
        utility = (
            trajectory.metrics.sharpe
            - self.config.lambda_dd * trajectory.metrics.max_drawdown
            - self.config.lambda_turn * trajectory.metrics.turnover
        )
        if trajectory.metrics.liquidation:
            utility -= self.config.liquidation_penalty
        return float(utility)
