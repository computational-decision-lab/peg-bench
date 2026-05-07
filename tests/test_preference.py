from __future__ import annotations

import unittest
import numpy as np

from src.eval.metrics import TradingMetrics
from src.preference.comparator import PreferenceComparator
from src.preference.pair_builder import resolve_lexicographic_template
from src.simulation.execution import TrajectoryRecord


class PreferenceComparatorTest(unittest.TestCase):
    def _trajectory(self, name: str, sharpe: float, pnl: float, turnover: float, mdd: float) -> TrajectoryRecord:
        return TrajectoryRecord(
            trajectory_id=name,
            context_id=0,
            variant=name,
            risk_mode="balanced",
            state_tokens=np.zeros((4, 8), dtype=np.int64),
            lob_states=np.zeros((4, 4, 10, 2), dtype=np.float32),
            market_returns=np.zeros(4, dtype=np.float32),
            timestamps=np.arange(4, dtype=np.int64),
            actions=np.zeros(4, dtype=np.float32),
            positions=np.zeros(4, dtype=np.float32),
            fills=[],
            transaction_costs=np.zeros(4, dtype=np.float32),
            gross_returns=np.zeros(4, dtype=np.float32),
            net_returns=np.zeros(4, dtype=np.float32),
            metrics=TradingMetrics(
                cumulative_return=pnl,
                annualized_return=pnl,
                sharpe=sharpe,
                calmar=sharpe / (mdd + 1e-8),
                max_drawdown=mdd,
                turnover=turnover,
                net_pnl=pnl,
                liquidation=False,
            ),
            regime_tag="normal",
        )

    def test_balanced_prefers_higher_sharpe(self) -> None:
        comparator = PreferenceComparator()
        better = self._trajectory("better", sharpe=2.0, pnl=1.0, turnover=0.5, mdd=0.1)
        worse = self._trajectory("worse", sharpe=1.0, pnl=1.2, turnover=0.5, mdd=0.1)
        compared = comparator.compare(better, worse)
        self.assertIsNotNone(compared)
        winner, _, _ = compared
        self.assertEqual(winner.trajectory_id, "better")


class PreferenceTemplateResolutionTest(unittest.TestCase):
    def test_auto_template_tracks_risk_mode(self) -> None:
        self.assertEqual(resolve_lexicographic_template("auto", "aggressive"), "aggressive")
        self.assertEqual(resolve_lexicographic_template("auto", "balanced"), "balanced")
        self.assertEqual(resolve_lexicographic_template("auto", "low_turnover"), "low_turnover")

    def test_explicit_template_is_preserved(self) -> None:
        self.assertEqual(resolve_lexicographic_template("balanced", "aggressive"), "balanced")


if __name__ == "__main__":
    unittest.main()
