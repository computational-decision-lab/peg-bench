"""Path-dependent execution simulator and trajectory bookkeeping."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from src.eval.metrics import TradingMetrics, summarize_trading_metrics
from src.simulation.cost_model import CapacityAwareCostModel, CostModelConfig


@dataclass(slots=True)
class FillEvent:
    """Single execution fill emitted by the simulator."""

    step: int
    timestamp: int
    delta_position: float
    target_position: float
    fee_cost: float
    impact_cost: float
    total_cost: float


@dataclass(slots=True)
class SimulatorConfig:
    """Execution simulator config."""

    max_position: float = 1.0
    liquidation_drawdown: float = 0.20
    hard_stop_drawdown: float = 0.10
    periods_per_year: int = 365 * 24 * 60 * 60
    enable_risk_controls: bool = True
    min_trade_size: float = 0.0
    cost: CostModelConfig = field(default_factory=CostModelConfig)


@dataclass
class TrajectoryRecord:
    """Trajectory rollout result used for utility calculation and preference construction."""

    trajectory_id: str
    context_id: int
    variant: str
    risk_mode: str
    state_tokens: np.ndarray
    lob_states: np.ndarray
    market_returns: np.ndarray
    timestamps: np.ndarray
    actions: np.ndarray
    positions: np.ndarray
    fills: list[FillEvent]
    transaction_costs: np.ndarray
    gross_returns: np.ndarray
    net_returns: np.ndarray
    metrics: TradingMetrics
    regime_tag: str

    def to_serializable(self) -> dict[str, object]:
        """Convert nested dataclasses into a pickle/json friendly structure."""
        payload = dict(self.__dict__)
        payload["fills"] = [asdict(fill) for fill in self.fills]
        payload["metrics"] = asdict(self.metrics)
        return payload


class ExecutionSimulator:
    """Market-friction simulator for continuous delta-position actions."""

    def __init__(self, config: SimulatorConfig | None = None) -> None:
        self.config = config or SimulatorConfig()
        self.cost_model = CapacityAwareCostModel(self.config.cost)

    def _visible_depth(self, lob_state: np.ndarray) -> float:
        log_volumes = np.nan_to_num(lob_state[1], nan=0.0, posinf=0.0, neginf=0.0)
        return float(np.exp(log_volumes).sum())

    def _volatility_proxy(self, returns: np.ndarray, index: int, window: int = 20) -> float:
        start = max(0, index - window + 1)
        return float(np.std(returns[start : index + 1])) if index >= 0 else 0.0

    def simulate(
        self,
        trajectory_id: str,
        context_id: int,
        variant: str,
        risk_mode: str,
        state_tokens: np.ndarray,
        lob_states: np.ndarray,
        market_returns: np.ndarray,
        timestamps: np.ndarray,
        actions: np.ndarray,
        regime_tag: str,
    ) -> TrajectoryRecord:
        """Simulate a trajectory with fees, impact and drawdown-based controls."""
        lob_states = np.nan_to_num(np.asarray(lob_states, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        market_returns = np.nan_to_num(np.asarray(market_returns, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        actions = np.nan_to_num(np.asarray(actions, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        timestamps = np.asarray(timestamps, dtype=np.int64)
        position = 0.0
        positions = np.zeros_like(actions, dtype=np.float32)
        costs = np.zeros_like(actions, dtype=np.float32)
        gross_returns = np.zeros_like(actions, dtype=np.float32)
        net_returns = np.zeros_like(actions, dtype=np.float32)
        fills: list[FillEvent] = []
        liquidation = False

        for step, delta in enumerate(actions):
            target_position = float(np.clip(position + delta, -self.config.max_position, self.config.max_position))
            trade_size = target_position - position
            if abs(trade_size) < self.config.min_trade_size:
                trade_size = 0.0
                target_position = position
            visible_depth = self._visible_depth(lob_states[step])
            volatility = self._volatility_proxy(market_returns, step)
            total_cost = self.cost_model.total_cost_decimal(trade_size, visible_depth, volatility)
            fee_cost = self.config.cost.fee_bps / 10_000.0 * abs(trade_size)
            impact_cost = max(0.0, total_cost * abs(trade_size) - fee_cost)

            position = target_position
            positions[step] = position
            gross_returns[step] = position * market_returns[step]
            costs[step] = total_cost * abs(trade_size)
            net_returns[step] = gross_returns[step] - costs[step]
            fills.append(
                FillEvent(
                    step=step,
                    timestamp=int(timestamps[step]),
                    delta_position=float(trade_size),
                    target_position=float(target_position),
                    fee_cost=float(fee_cost),
                    impact_cost=float(impact_cost),
                    total_cost=float(costs[step]),
                )
            )

            if self.config.enable_risk_controls:
                current_drawdown = summarize_trading_metrics(
                    net_returns[: step + 1],
                    turnover=float(np.abs(np.diff(np.concatenate([[0.0], positions[: step + 1]])).sum())),
                    liquidation=False,
                    periods_per_year=self.config.periods_per_year,
                ).max_drawdown
                if current_drawdown >= self.config.liquidation_drawdown:
                    liquidation = True
                    # Preserve the triggering step's realized loss/costs; only future steps stay flat.
                    if step + 1 < len(net_returns):
                        net_returns[step + 1 :] = 0.0
                        gross_returns[step + 1 :] = 0.0
                        positions[step + 1 :] = 0.0
                    break
                if current_drawdown >= self.config.hard_stop_drawdown:
                    position = 0.0

        turnover = float(np.abs(np.diff(np.concatenate([[0.0], positions]))).sum())
        metrics = summarize_trading_metrics(
            net_returns,
            turnover=turnover,
            liquidation=liquidation,
            periods_per_year=self.config.periods_per_year,
        )
        return TrajectoryRecord(
            trajectory_id=trajectory_id,
            context_id=context_id,
            variant=variant,
            risk_mode=risk_mode,
            state_tokens=state_tokens.astype(np.int64),
            lob_states=lob_states.astype(np.float32),
            market_returns=market_returns.astype(np.float32),
            timestamps=timestamps.astype(np.int64),
            actions=actions.astype(np.float32),
            positions=positions.astype(np.float32),
            fills=fills,
            transaction_costs=costs.astype(np.float32),
            gross_returns=gross_returns.astype(np.float32),
            net_returns=net_returns.astype(np.float32),
            metrics=metrics,
            regime_tag=regime_tag,
        )
