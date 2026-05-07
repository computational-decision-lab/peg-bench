"""Lexicographic oracle and preference templates."""

from __future__ import annotations

from dataclasses import dataclass

from src.simulation.execution import TrajectoryRecord


@dataclass(slots=True)
class OracleDecision:
    """Winner/loser decision returned by the oracle."""

    winner: TrajectoryRecord
    loser: TrajectoryRecord
    margin: float
    criterion: str


class LexicographicOracle:
    """Compare trajectories according to risk-conditioned business rules."""

    def compare(self, left: TrajectoryRecord, right: TrajectoryRecord, mode: str) -> OracleDecision | None:
        """Return a winner/loser ordering or `None` if no meaningful preference exists."""
        if mode == "aggressive":
            return self._aggressive(left, right)
        if mode == "balanced":
            return self._balanced(left, right)
        if mode == "conservative":
            return self._conservative(left, right)
        if mode == "low_turnover":
            return self._low_turnover(left, right)
        raise ValueError(f"Unsupported oracle mode: {mode}")

    def _liquidation_first(self, left: TrajectoryRecord, right: TrajectoryRecord) -> OracleDecision | None:
        if left.metrics.liquidation == right.metrics.liquidation:
            return None
        winner, loser = (right, left) if left.metrics.liquidation else (left, right)
        return OracleDecision(winner=winner, loser=loser, margin=1.0, criterion="liquidation")

    def _aggressive(self, left: TrajectoryRecord, right: TrajectoryRecord) -> OracleDecision | None:
        failure = self._liquidation_first(left, right)
        if failure is not None:
            return failure
        margin = abs(left.metrics.cumulative_return - right.metrics.cumulative_return)
        if margin <= 1e-6:
            return None
        winner, loser = (left, right) if left.metrics.cumulative_return > right.metrics.cumulative_return else (right, left)
        return OracleDecision(winner=winner, loser=loser, margin=float(margin), criterion="cumulative_return")

    def _balanced(self, left: TrajectoryRecord, right: TrajectoryRecord) -> OracleDecision | None:
        failure = self._liquidation_first(left, right)
        if failure is not None:
            return failure
        sharpe_diff = left.metrics.sharpe - right.metrics.sharpe
        if abs(sharpe_diff) > 1e-4:
            winner, loser = (left, right) if sharpe_diff > 0 else (right, left)
            return OracleDecision(winner=winner, loser=loser, margin=float(abs(sharpe_diff)), criterion="sharpe")
        mdd_diff = right.metrics.max_drawdown - left.metrics.max_drawdown
        if abs(mdd_diff) > 1e-6:
            winner, loser = (left, right) if mdd_diff > 0 else (right, left)
            return OracleDecision(winner=winner, loser=loser, margin=float(abs(mdd_diff)), criterion="max_drawdown")
        return None

    def _conservative(self, left: TrajectoryRecord, right: TrajectoryRecord) -> OracleDecision | None:
        failure = self._liquidation_first(left, right)
        if failure is not None:
            return failure
        calmar_diff = left.metrics.calmar - right.metrics.calmar
        if abs(calmar_diff) <= 1e-6:
            return None
        winner, loser = (left, right) if calmar_diff > 0 else (right, left)
        return OracleDecision(winner=winner, loser=loser, margin=float(abs(calmar_diff)), criterion="calmar")

    def _low_turnover(self, left: TrajectoryRecord, right: TrajectoryRecord) -> OracleDecision | None:
        failure = self._liquidation_first(left, right)
        if failure is not None:
            return failure
        score_left = left.metrics.net_pnl - 0.1 * left.metrics.turnover
        score_right = right.metrics.net_pnl - 0.1 * right.metrics.turnover
        diff = score_left - score_right
        if abs(diff) <= 1e-6:
            return None
        winner, loser = (left, right) if diff > 0 else (right, left)
        return OracleDecision(winner=winner, loser=loser, margin=float(abs(diff)), criterion="net_pnl_minus_turnover")
