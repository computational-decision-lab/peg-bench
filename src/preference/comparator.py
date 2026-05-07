"""Scalar and lexicographic preference comparison logic."""

from __future__ import annotations

from dataclasses import dataclass

from src.preference.filtering import AdaptiveConformalController, ConfidenceBand
from src.preference.dataset import PreferenceMetadata
from src.simulation.execution import TrajectoryRecord
from src.simulation.oracle import LexicographicOracle, OracleDecision
from src.simulation.utility import TrajectoryUtilityCalculator


@dataclass(slots=True)
class ComparatorConfig:
    """Preference construction config."""

    mode: str = "lexicographic"
    lexicographic_template: str = "balanced"
    min_utility_gap: float = 0.1


class PreferenceComparator:
    """Compare trajectories under scalar or lexicographic objectives."""

    def __init__(
        self,
        config: ComparatorConfig | None = None,
        utility_calculator: TrajectoryUtilityCalculator | None = None,
        conformal: AdaptiveConformalController | None = None,
    ) -> None:
        self.config = config or ComparatorConfig()
        self.utility = utility_calculator or TrajectoryUtilityCalculator()
        self.oracle = LexicographicOracle()
        self.conformal = conformal

    def compare(
        self,
        left: TrajectoryRecord,
        right: TrajectoryRecord,
    ) -> tuple[TrajectoryRecord, TrajectoryRecord, PreferenceMetadata] | None:
        """Compare a trajectory pair and return serialized metadata."""
        if self.config.mode == "scalar":
            return self._compare_scalar(left, right)
        return self._compare_lexicographic(left, right)

    def _conformal_intervals(self, left_value: float, right_value: float) -> tuple[ConfidenceBand, ConfidenceBand, bool]:
        if self.conformal is None:
            null_band = ConfidenceBand(lower=left_value, upper=left_value)
            return null_band, ConfidenceBand(lower=right_value, upper=right_value), True
        keep, left_band, right_band = self.conformal.keep_preference(left_value, right_value)
        return left_band, right_band, keep

    def _compare_scalar(
        self,
        left: TrajectoryRecord,
        right: TrajectoryRecord,
    ) -> tuple[TrajectoryRecord, TrajectoryRecord, PreferenceMetadata] | None:
        left_value = self.utility.scalar_utility(left)
        right_value = self.utility.scalar_utility(right)
        diff = left_value - right_value
        if abs(diff) < self.config.min_utility_gap:
            return None
        left_band, right_band, keep = self._conformal_intervals(left_value, right_value)
        if not keep:
            return None
        winner, loser = (left, right) if diff > 0 else (right, left)
        metadata = PreferenceMetadata(
            utility_difference=float(abs(diff)),
            preference_mode="scalar",
            source_trajectory_ids=(winner.trajectory_id, loser.trajectory_id),
            source_variants=(winner.variant, loser.variant),
            regime_tag=winner.regime_tag,
            winner_interval=(left_band.lower, left_band.upper) if winner is left else (right_band.lower, right_band.upper),
            loser_interval=(right_band.lower, right_band.upper) if winner is left else (left_band.lower, left_band.upper),
            criterion="scalar_utility",
        )
        return winner, loser, metadata

    def _compare_lexicographic(
        self,
        left: TrajectoryRecord,
        right: TrajectoryRecord,
    ) -> tuple[TrajectoryRecord, TrajectoryRecord, PreferenceMetadata] | None:
        decision: OracleDecision | None = self.oracle.compare(left, right, self.config.lexicographic_template)
        if decision is None or decision.margin < self.config.min_utility_gap:
            return None
        left_value = self.utility.scalar_utility(left)
        right_value = self.utility.scalar_utility(right)
        left_band, right_band, keep = self._conformal_intervals(left_value, right_value)
        if not keep:
            return None
        metadata = PreferenceMetadata(
            utility_difference=float(decision.margin),
            preference_mode=self.config.lexicographic_template,
            source_trajectory_ids=(decision.winner.trajectory_id, decision.loser.trajectory_id),
            source_variants=(decision.winner.variant, decision.loser.variant),
            regime_tag=decision.winner.regime_tag,
            winner_interval=(left_band.lower, left_band.upper) if decision.winner is left else (right_band.lower, right_band.upper),
            loser_interval=(right_band.lower, right_band.upper) if decision.winner is left else (left_band.lower, left_band.upper),
            criterion=decision.criterion,
        )
        return decision.winner, decision.loser, metadata
