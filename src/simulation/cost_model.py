"""Research-grade but simplified transaction-cost models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class CostModelConfig:
    """Config for maker/taker fees and impact."""

    fee_bps: float = 3.0
    impact_coef: float = 0.1
    impact_mode: str = "sqrt"
    min_depth: float = 1e-6
    max_slippage_bps: float = 25.0


class CapacityAwareCostModel:
    """Fee + liquidity-aware impact model derived from visible depth."""

    def __init__(self, config: CostModelConfig | None = None) -> None:
        self.config = config or CostModelConfig()

    def estimate_impact_bps(self, trade_size: float, visible_depth: float, volatility: float) -> float:
        """Estimate impact in basis points."""
        visible_depth = max(visible_depth, self.config.min_depth)
        relative_size = abs(trade_size) / visible_depth
        if self.config.impact_mode == "piecewise":
            impact = self.config.impact_coef * relative_size * 10_000.0
        else:
            impact = self.config.impact_coef * np.sqrt(relative_size) * max(volatility, 1e-6) * 10_000.0
        return float(np.clip(impact, 0.0, self.config.max_slippage_bps))

    def total_cost_decimal(self, trade_size: float, visible_depth: float, volatility: float) -> float:
        """Return total cost as a decimal fraction."""
        total_bps = self.config.fee_bps + self.estimate_impact_bps(trade_size, visible_depth, volatility)
        return total_bps / 10_000.0
