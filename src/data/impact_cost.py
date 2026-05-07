"""
Impact cost model for transaction cost estimation.
Implements market impact cost from paper Section 7.2.4.

执行位置: ☁️ 云端（AutoDL服务器）
数据范围: 2025-10-01 to 2025-10-30 (30 days)

Market impact cost model:
    impact_cost = k * sqrt(trade_size / avg_volume) * volatility

where:
    - k: impact coefficient (calibrated from data)
    - trade_size: size of the trade in base currency
    - avg_volume: average volume at that price level
    - volatility: recent price volatility

Total transaction cost = fee_cost + impact_cost
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class ImpactCostModel:
    """
    Market impact cost model for LOB trading.

    Attributes:
        impact_coef: Impact coefficient k (default: 0.1)
        volatility_window: Window for volatility estimation (default: 20 bars)
        min_volume: Minimum volume to avoid division by zero (default: 1e-8)
    """

    def __init__(
        self,
        impact_coef: float = 0.1,
        volatility_window: int = 20,
        min_volume: float = 1e-8,
    ):
        self.impact_coef = impact_coef
        self.volatility_window = volatility_window
        self.min_volume = min_volume

    def estimate_impact(
        self,
        trade_size: float,
        avg_volume: float,
        volatility: float,
    ) -> float:
        """
        Estimate market impact cost for a single trade.

        Args:
            trade_size: Size of the trade in base currency
            avg_volume: Average volume at the price level
            volatility: Recent price volatility (std of returns)

        Returns:
            Impact cost in basis points (bps)
        """
        if trade_size <= 0 or avg_volume <= 0:
            return 0.0

        # Ensure minimum volume to avoid division by zero
        avg_volume = max(avg_volume, self.min_volume)

        # Square-root impact model
        relative_size = trade_size / avg_volume
        impact_bps = self.impact_coef * np.sqrt(relative_size) * volatility * 10000

        return float(impact_bps)

    def estimate_batch_impact(
        self,
        trade_sizes: np.ndarray,
        avg_volumes: np.ndarray,
        volatilities: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate market impact costs for a batch of trades.

        Args:
            trade_sizes: (N,) array of trade sizes
            avg_volumes: (N,) array of average volumes
            volatilities: (N,) array of volatilities

        Returns:
            (N,) array of impact costs in bps
        """
        # Ensure minimum volumes
        avg_volumes = np.maximum(avg_volumes, self.min_volume)

        # Compute relative sizes
        relative_sizes = trade_sizes / avg_volumes

        # Square-root impact model
        impact_bps = self.impact_coef * np.sqrt(relative_sizes) * volatilities * 10000

        # Set zero impact for zero trade sizes
        impact_bps[trade_sizes <= 0] = 0.0

        return impact_bps

    def compute_volatility(
        self,
        returns: np.ndarray,
        window: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute rolling volatility from returns.

        Args:
            returns: (N,) array of returns
            window: Rolling window size (default: self.volatility_window)

        Returns:
            (N,) array of volatilities
        """
        if window is None:
            window = self.volatility_window

        n = len(returns)
        volatilities = np.zeros(n, dtype=np.float32)

        for i in range(n):
            start = max(0, i - window + 1)
            window_returns = returns[start:i+1]
            if len(window_returns) > 1:
                volatilities[i] = np.std(window_returns)
            else:
                volatilities[i] = 0.0

        return volatilities


class TransactionCostModel:
    """
    Combined transaction cost model: fees + market impact.

    Attributes:
        fee_bps: Trading fee in basis points (default: 3.0 for Binance)
        impact_model: ImpactCostModel instance
    """

    def __init__(
        self,
        fee_bps: float = 3.0,
        impact_coef: float = 0.1,
        volatility_window: int = 20,
    ):
        self.fee_bps = fee_bps
        self.impact_model = ImpactCostModel(
            impact_coef=impact_coef,
            volatility_window=volatility_window,
        )

    def compute_total_cost(
        self,
        trade_size: float,
        avg_volume: float,
        volatility: float,
    ) -> tuple[float, float, float]:
        """
        Compute total transaction cost.

        Args:
            trade_size: Size of the trade in base currency
            avg_volume: Average volume at the price level
            volatility: Recent price volatility

        Returns:
            Tuple of (total_cost_bps, fee_cost_bps, impact_cost_bps)
        """
        fee_cost = self.fee_bps
        impact_cost = self.impact_model.estimate_impact(
            trade_size, avg_volume, volatility
        )
        total_cost = fee_cost + impact_cost

        return total_cost, fee_cost, impact_cost

    def compute_batch_costs(
        self,
        trade_sizes: np.ndarray,
        avg_volumes: np.ndarray,
        volatilities: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute total transaction costs for a batch of trades.

        Args:
            trade_sizes: (N,) array of trade sizes
            avg_volumes: (N,) array of average volumes
            volatilities: (N,) array of volatilities

        Returns:
            Tuple of (total_costs, fee_costs, impact_costs), each (N,) array in bps
        """
        n = len(trade_sizes)
        fee_costs = np.full(n, self.fee_bps, dtype=np.float32)
        impact_costs = self.impact_model.estimate_batch_impact(
            trade_sizes, avg_volumes, volatilities
        )
        total_costs = fee_costs + impact_costs

        return total_costs, fee_costs, impact_costs


def calibrate_impact_coefficient(
    trade_sizes: np.ndarray,
    avg_volumes: np.ndarray,
    volatilities: np.ndarray,
    realized_impacts: np.ndarray,
) -> float:
    """
    Calibrate impact coefficient from historical data.

    Uses least squares to fit k in:
        realized_impact = k * sqrt(trade_size / avg_volume) * volatility

    Args:
        trade_sizes: (N,) array of historical trade sizes
        avg_volumes: (N,) array of average volumes
        volatilities: (N,) array of volatilities
        realized_impacts: (N,) array of realized impact costs in bps

    Returns:
        Calibrated impact coefficient k
    """
    # Compute features: sqrt(trade_size / avg_volume) * volatility
    avg_volumes = np.maximum(avg_volumes, 1e-8)
    relative_sizes = trade_sizes / avg_volumes
    features = np.sqrt(relative_sizes) * volatilities * 10000

    # Filter out invalid data
    valid_mask = (features > 0) & (realized_impacts >= 0)
    features = features[valid_mask]
    realized_impacts = realized_impacts[valid_mask]

    if len(features) == 0:
        return 0.1  # Default value

    # Least squares: k = sum(y * x) / sum(x^2)
    k = np.sum(realized_impacts * features) / np.sum(features ** 2)

    # Ensure reasonable bounds
    k = np.clip(k, 0.01, 1.0)

    return float(k)


# Example usage and testing
if __name__ == "__main__":
    # Test impact cost model
    model = ImpactCostModel(impact_coef=0.1)

    # Single trade
    trade_size = 1.0  # 1 BTC
    avg_volume = 10.0  # 10 BTC average volume
    volatility = 0.001  # 0.1% volatility

    impact = model.estimate_impact(trade_size, avg_volume, volatility)
    print(f"Single trade impact: {impact:.2f} bps")

    # Batch trades
    trade_sizes = np.array([0.5, 1.0, 2.0, 5.0])
    avg_volumes = np.array([10.0, 10.0, 10.0, 10.0])
    volatilities = np.array([0.001, 0.001, 0.001, 0.001])

    impacts = model.estimate_batch_impact(trade_sizes, avg_volumes, volatilities)
    print(f"\nBatch trade impacts: {impacts}")

    # Test transaction cost model
    cost_model = TransactionCostModel(fee_bps=3.0, impact_coef=0.1)

    total, fee, impact = cost_model.compute_total_cost(1.0, 10.0, 0.001)
    print(f"\nTotal cost: {total:.2f} bps (fee: {fee:.2f}, impact: {impact:.2f})")

    # Test calibration
    np.random.seed(42)
    n_samples = 1000
    true_k = 0.15
    trade_sizes = np.random.uniform(0.1, 5.0, n_samples)
    avg_volumes = np.random.uniform(5.0, 20.0, n_samples)
    volatilities = np.random.uniform(0.0005, 0.002, n_samples)

    # Generate synthetic realized impacts
    realized_impacts = true_k * np.sqrt(trade_sizes / avg_volumes) * volatilities * 10000
    realized_impacts += np.random.normal(0, 0.1, n_samples)  # Add noise

    calibrated_k = calibrate_impact_coefficient(
        trade_sizes, avg_volumes, volatilities, realized_impacts
    )
    print(f"\nCalibration: true k={true_k:.3f}, calibrated k={calibrated_k:.3f}")
