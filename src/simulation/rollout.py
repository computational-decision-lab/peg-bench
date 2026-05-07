"""Trajectory rollout engine from baseline predictions or continuous policies."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from src.models.baseline import KairosBaselineModel
from src.models.policy import GaussianActionPolicy
from src.preference.filtering import AdaptiveConformalController
from src.simulation.execution import ExecutionSimulator, SimulatorConfig, TrajectoryRecord


RISK_TO_ID = {"aggressive": 0, "balanced": 1, "conservative": 2, "low_turnover": 3}


@dataclass(slots=True)
class RolloutConfig:
    """Config for candidate trajectory generation."""

    seq_len: int = 96
    horizon: int = 96
    top_decile: float = 0.1
    noise_std: float = 0.15
    low_turnover_scale: float = 0.35
    conservative_scale: float = 0.2
    device: str = "cpu"
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)


def detect_regime_tag(returns: np.ndarray) -> str:
    """Assign a simple volatility regime tag."""
    if len(returns) == 0:
        return "empty"
    vol = float(np.std(returns))
    if vol > np.percentile(np.abs(returns), 80):
        return "high_volatility"
    if abs(np.mean(returns)) > vol:
        return "trend"
    return "normal"


class TrajectoryRolloutEngine:
    """Build candidate trajectories from a baseline model or a learned policy."""

    def __init__(
        self,
        config: RolloutConfig | None = None,
        conformal: AdaptiveConformalController | None = None,
    ) -> None:
        self.config = config or RolloutConfig()
        self.device = torch.device(self.config.device)
        self.simulator = ExecutionSimulator(self.config.simulator)
        self.conformal = conformal

    def _context_windows(self, codes: np.ndarray, start: int, horizon: int) -> np.ndarray:
        windows = []
        for current in range(start, start + horizon):
            if current < self.config.seq_len:
                continue
            windows.append(codes[current - self.config.seq_len : current])
        return np.asarray(windows, dtype=np.int64)

    def _baseline_predictions(
        self,
        model: KairosBaselineModel,
        windows: np.ndarray,
    ) -> np.ndarray:
        if len(windows) == 0:
            return np.zeros(0, dtype=np.float32)
        tensor = torch.from_numpy(windows).long().to(self.device)
        with torch.no_grad():
            outputs = model(input_ids=tensor)
        return outputs["return_pred"].cpu().numpy().astype(np.float32)

    def _policy_actions(
        self,
        policy: GaussianActionPolicy,
        windows: np.ndarray,
        risk_mode: str,
    ) -> np.ndarray:
        if len(windows) == 0:
            return np.zeros(0, dtype=np.float32)
        input_ids = torch.from_numpy(windows).long().to(self.device)
        risk_ids = torch.full((len(windows),), RISK_TO_ID[risk_mode], dtype=torch.long, device=self.device)
        with torch.no_grad():
            actions = policy.act(input_ids=input_ids, risk_ids=risk_ids, deterministic=True).cpu().numpy()
        return actions.astype(np.float32)

    def _candidate_actions(self, predictions: np.ndarray, risk_mode: str) -> dict[str, np.ndarray]:
        if len(predictions) == 0:
            names = ["dense_005", "dense_010", "dense_020", "top_decile", "bottom_decile", "long_only", "random", "conservative", "low_turnover"]
            return {name: np.zeros(0, dtype=np.float32) for name in names}
        upper = np.quantile(predictions, 1.0 - self.config.top_decile)
        lower = np.quantile(predictions, self.config.top_decile)
        base = np.tanh(predictions)
        random_noise = np.random.normal(0.0, self.config.noise_std, size=len(predictions))
        top = np.where(predictions >= upper, base, 0.0)
        bottom = np.where(predictions <= lower, -np.abs(base), 0.0)
        long_only = np.clip(base, 0.0, 1.0)
        conservative = base * self.config.conservative_scale
        low_turnover = base * self.config.low_turnover_scale
        dense_005 = base * 0.05
        dense_010 = base * 0.10
        dense_020 = base * 0.20

        variants = {
            "dense_005": dense_005.astype(np.float32),
            "dense_010": dense_010.astype(np.float32),
            "dense_020": dense_020.astype(np.float32),
            "top_decile": top.astype(np.float32),
            "bottom_decile": bottom.astype(np.float32),
            "long_only": long_only.astype(np.float32),
            "random": np.tanh(predictions + random_noise).astype(np.float32),
            "conservative": conservative.astype(np.float32),
            "low_turnover": low_turnover.astype(np.float32),
        }
        if risk_mode == "aggressive":
            variants["dense_040"] = (base * 0.40).astype(np.float32)
            variants["long_only"] = np.clip(base * 1.2, 0.0, 1.0).astype(np.float32)
        elif risk_mode == "conservative":
            variants = {k: v for k, v in variants.items() if k not in {"dense_020", "dense_040", "random"}}
        elif risk_mode == "low_turnover":
            variants = {k: v for k, v in variants.items() if k not in {"dense_020", "dense_040", "bottom_decile"}}
        return variants

    def build_candidate_trajectories(
        self,
        context_id: int,
        codes: np.ndarray,
        lob: np.ndarray,
        returns: np.ndarray,
        timestamps: np.ndarray,
        baseline_model: KairosBaselineModel | None = None,
        policy: GaussianActionPolicy | None = None,
        risk_mode: str = "balanced",
    ) -> list[TrajectoryRecord]:
        """Generate candidate trajectories for a single historical context."""
        start = context_id
        horizon = min(self.config.horizon, len(codes) - start)
        windows = self._context_windows(codes, start, horizon)
        if len(windows) == 0:
            return []

        if policy is not None:
            predictions = self._policy_actions(policy, windows, risk_mode)
        elif baseline_model is not None:
            predictions = self._baseline_predictions(baseline_model, windows)
        else:
            raise ValueError("Either a baseline model or a policy must be provided.")

        variants = self._candidate_actions(predictions, risk_mode)
        trajectories: list[TrajectoryRecord] = []
        for variant_name, actions in variants.items():
            if self.conformal is not None and len(actions) > 0:
                actions = np.asarray(
                    [self.conformal.scale_action(float(action), abs(float(action))) for action in actions],
                    dtype=np.float32,
                )
            effective_horizon = min(len(actions), len(lob) - start, len(returns) - start)
            if effective_horizon <= 0:
                continue
            state_tokens = windows[:effective_horizon]
            lob_states = lob[start : start + effective_horizon]
            market_returns = returns[start : start + effective_horizon]
            ts = timestamps[start : start + effective_horizon]
            regime_tag = detect_regime_tag(market_returns)
            trajectories.append(
                self.simulator.simulate(
                    trajectory_id=f"{context_id}:{variant_name}:{risk_mode}",
                    context_id=context_id,
                    variant=variant_name,
                    risk_mode=risk_mode,
                    state_tokens=state_tokens,
                    lob_states=lob_states,
                    market_returns=market_returns,
                    timestamps=ts,
                    actions=actions[:effective_horizon],
                    regime_tag=regime_tag,
                )
            )
        return trajectories
