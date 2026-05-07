"""Shared PEG-Bench realism ladder and report-card helpers."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from src.eval.gap_metrics import compute_gap_report
from src.eval.metrics import (
    average_holding_period,
    cvar_at_level,
    directional_accuracy,
    mean_absolute_error,
    mean_squared_error,
    rank_ic,
    summarize_trading_metrics,
)
from src.simulation.cost_model import CostModelConfig
from src.simulation.execution import ExecutionSimulator, SimulatorConfig


def _rolling_stat(values: np.ndarray, window: int, reducer) -> np.ndarray:
    result = np.zeros(len(values), dtype=np.float64)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result[index] = reducer(values[start : index + 1])
    return result


def _infer_regime_masks(market_returns: np.ndarray, lob_states: np.ndarray) -> dict[str, np.ndarray]:
    abs_returns = np.abs(market_returns)
    rolling_vol = _rolling_stat(market_returns, 32, np.std)
    rolling_mean = _rolling_stat(market_returns, 32, np.mean)
    visible_depth = np.exp(lob_states[:, 1]).sum(axis=(1, 2))
    vol_threshold = np.percentile(rolling_vol, 75) if len(rolling_vol) else 0.0
    up_threshold = np.percentile(rolling_mean, 75) if len(rolling_mean) else 0.0
    down_threshold = np.percentile(rolling_mean, 25) if len(rolling_mean) else 0.0
    low_depth_threshold = np.percentile(visible_depth, 25) if len(visible_depth) else 0.0
    abs_move_threshold = np.percentile(abs_returns, 75) if len(abs_returns) else 0.0
    high_vol = rolling_vol >= vol_threshold
    trend_up = rolling_mean >= up_threshold
    trend_down = rolling_mean <= down_threshold
    liquidity_stress = visible_depth <= low_depth_threshold
    large_moves = abs_returns >= abs_move_threshold
    normal = ~(high_vol | trend_up | trend_down | liquidity_stress | large_moves)
    min_normal_points = min(max(8, len(market_returns) // 10), len(market_returns))
    if int(normal.sum()) < min_normal_points and len(market_returns) > 0:
        calm_mask = (
            (abs_returns <= np.percentile(abs_returns, 50))
            & (rolling_vol <= np.percentile(rolling_vol, 50))
            & (visible_depth >= np.percentile(visible_depth, 50))
        )
        if int(calm_mask.sum()) >= min_normal_points:
            normal = calm_mask
        else:
            abs_rank = np.argsort(np.argsort(abs_returns))
            vol_rank = np.argsort(np.argsort(rolling_vol))
            depth_rank = np.argsort(np.argsort(-visible_depth))
            calm_score = abs_rank + vol_rank + depth_rank
            keep = min_normal_points if min_normal_points > 0 else len(market_returns)
            cutoff = np.partition(calm_score, keep - 1)[keep - 1] if keep > 0 else np.max(calm_score)
            normal = calm_score <= cutoff
    return {
        "normal": normal,
        "high_vol": high_vol,
        "trend_up": trend_up,
        "trend_down": trend_down,
        "liquidity_stress": liquidity_stress,
        "large_moves": large_moves,
    }


def _extract_stress_intervals(mask: np.ndarray, min_length: int = 4) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start = None
    for index, flagged in enumerate(mask):
        if flagged and start is None:
            start = index
        elif not flagged and start is not None:
            if index - start >= min_length:
                intervals.append((start, index))
            start = None
    if start is not None and len(mask) - start >= min_length:
        intervals.append((start, len(mask)))
    return intervals


def _compute_robustness_panel(
    net_returns: np.ndarray,
    actions: np.ndarray,
    market_returns: np.ndarray,
    lob_states: np.ndarray,
    periods_per_year: int,
    failure_drawdown: float,
) -> dict[str, object]:
    regime_masks = _infer_regime_masks(market_returns, lob_states)
    regime_metrics: dict[str, dict[str, float]] = {}
    regime_utilities: list[float] = []
    regime_counts: dict[str, int] = {}
    for name, mask in regime_masks.items():
        if int(mask.sum()) < 8:
            continue
        metrics = summarize_trading_metrics(
            net_returns[mask],
            turnover=float(np.abs(actions[mask]).sum()),
            liquidation=False,
            periods_per_year=periods_per_year,
        )
        regime_metrics[name] = asdict(metrics)
        regime_utilities.append(metrics.sharpe)
        regime_counts[name] = int(mask.sum())
    worst_regime_sharpe = min(regime_utilities) if regime_utilities else None
    regime_variance = float(np.var(np.asarray(regime_utilities, dtype=np.float64))) if regime_utilities else None

    abs_returns = np.abs(market_returns)
    rolling_vol = _rolling_stat(market_returns, 32, np.std)
    visible_depth = np.exp(lob_states[:, 1]).sum(axis=(1, 2))
    stress_mask = (
        (abs_returns >= np.percentile(abs_returns, 95))
        | (rolling_vol >= np.percentile(rolling_vol, 95))
        | (visible_depth <= np.percentile(visible_depth, 5))
    )
    intervals = _extract_stress_intervals(stress_mask)
    survival_flags: list[bool] = []
    stress_metrics: list[dict[str, object]] = []
    for start, stop in intervals:
        metrics = summarize_trading_metrics(
            net_returns[start:stop],
            turnover=float(np.abs(actions[start:stop]).sum()),
            liquidation=False,
            periods_per_year=periods_per_year,
        )
        survived = bool(metrics.max_drawdown < failure_drawdown)
        survival_flags.append(survived)
        payload = asdict(metrics)
        payload.update({"start": start, "stop": stop, "survived": survived})
        stress_metrics.append(payload)
    drawdown_violation_rate = float(np.mean(np.asarray([item["max_drawdown"] >= failure_drawdown for item in stress_metrics], dtype=np.float64))) if stress_metrics else None
    return {
        "worst_regime_sharpe": worst_regime_sharpe,
        "stress_day_survival": float(np.mean(survival_flags)) if survival_flags else None,
        "regime_variance": regime_variance,
        "regime_metrics": regime_metrics,
        "regime_counts": regime_counts,
        "regime_utilities": regime_utilities,
        "stress_intervals": stress_metrics,
        "stress_days": [{"start": item["start"], "stop": item["stop"]} for item in stress_metrics],
        "drawdown_violation_rate": drawdown_violation_rate,
    }


def _simulate_stage(
    stage: str,
    cost_bps: float,
    state_tokens: np.ndarray,
    lob_states: np.ndarray,
    market_returns: np.ndarray,
    timestamps: np.ndarray,
    actions: np.ndarray,
    base_config: SimulatorConfig,
) -> tuple[dict[str, float], dict[str, object] | None, object]:
    enable_impact = stage in {"R2", "R3", "R4"}
    enable_risk = stage in {"R3", "R4"}
    sim = ExecutionSimulator(
        SimulatorConfig(
            max_position=base_config.max_position,
            liquidation_drawdown=base_config.liquidation_drawdown,
            hard_stop_drawdown=base_config.hard_stop_drawdown,
            periods_per_year=base_config.periods_per_year,
            enable_risk_controls=enable_risk,
            min_trade_size=base_config.min_trade_size,
            cost=CostModelConfig(
                fee_bps=cost_bps,
                impact_coef=base_config.cost.impact_coef if enable_impact else 0.0,
                impact_mode=base_config.cost.impact_mode,
                min_depth=base_config.cost.min_depth,
                max_slippage_bps=base_config.cost.max_slippage_bps,
            ),
        )
    )
    trajectory = sim.simulate(
        trajectory_id=f"{stage}:{cost_bps}",
        context_id=0,
        variant=stage,
        risk_mode="benchmark",
        state_tokens=state_tokens,
        lob_states=lob_states,
        market_returns=market_returns,
        timestamps=timestamps,
        actions=actions,
        regime_tag="shared_eval",
    )
    robustness = None
    if stage == "R4":
        robustness = _compute_robustness_panel(
            net_returns=trajectory.net_returns,
            actions=trajectory.actions,
            market_returns=trajectory.market_returns,
            lob_states=trajectory.lob_states,
            periods_per_year=base_config.periods_per_year,
            failure_drawdown=base_config.liquidation_drawdown,
        )
    return asdict(trajectory.metrics), robustness, trajectory


def _prediction_panel(predictions: np.ndarray | None, targets: np.ndarray | None) -> dict[str, object]:
    if predictions is None or targets is None:
        return {
            "directional_accuracy": None,
            "rank_ic": None,
            "mse": None,
            "mae": None,
            "status": "protocol_target",
            "reason": "offline predictive columns are not defined for this method output",
        }
    return {
        "directional_accuracy": directional_accuracy(predictions, targets),
        "rank_ic": rank_ic(predictions, targets),
        "mse": mean_squared_error(predictions, targets),
        "mae": mean_absolute_error(predictions, targets),
        "status": "computed",
    }


def run_realism_ladder(
    *,
    predictions: np.ndarray | None,
    targets: np.ndarray | None,
    state_tokens: np.ndarray,
    lob_states: np.ndarray,
    market_returns: np.ndarray,
    timestamps: np.ndarray,
    actions: np.ndarray | None = None,
    actions_by_cost: dict[str, np.ndarray] | None = None,
    action_metadata: dict[str, dict[str, object]] | None = None,
    cost_bps_scan: tuple[float, ...] = (0.0, 1.0, 3.0, 5.0, 10.0),
    anchor_cost_bps: float = 3.0,
    simulator_config: SimulatorConfig | None = None,
) -> dict[str, object]:
    """Evaluate a method under PEG-Bench's R0→R4 realism ladder."""
    if actions is None and not actions_by_cost:
        raise ValueError("run_realism_ladder requires `actions` or `actions_by_cost`.")
    base_config = simulator_config or SimulatorConfig()
    ladder: dict[str, object] = {
        "R0": _prediction_panel(predictions, targets),
        "R1": {"cost_scan": {}},
        "R2": {"cost_scan": {}},
        "R3": {"cost_scan": {}},
        "R4": {"cost_scan": {}, "robustness": {}},
    }
    anchor_trajectory = None
    for cost_bps in cost_bps_scan:
        key = str(float(cost_bps))
        stage_actions = actions_by_cost.get(key) if actions_by_cost is not None else actions
        if stage_actions is None:
            raise ValueError(f"Missing actions for cost {key}.")
        for stage in ("R1", "R2", "R3", "R4"):
            metrics, robustness, trajectory = _simulate_stage(
                stage=stage,
                cost_bps=float(cost_bps),
                state_tokens=state_tokens,
                lob_states=lob_states,
                market_returns=market_returns,
                timestamps=timestamps,
                actions=stage_actions,
                base_config=base_config,
            )
            ladder[stage]["cost_scan"][key] = metrics
            if stage == "R4":
                ladder["R4"]["robustness"][key] = robustness
                if key == str(float(anchor_cost_bps)):
                    anchor_trajectory = trajectory
    anchor_key = str(float(anchor_cost_bps))
    if anchor_key not in ladder["R4"]["cost_scan"]:
        anchor_key = sorted(ladder["R4"]["cost_scan"].keys(), key=float)[0]
    anchor_metrics = ladder["R4"]["cost_scan"][anchor_key]
    anchor_robustness = ladder["R4"]["robustness"][anchor_key]
    if anchor_trajectory is None:
        _, _, anchor_trajectory = _simulate_stage(
            stage="R4",
            cost_bps=float(anchor_key),
            state_tokens=state_tokens,
            lob_states=lob_states,
            market_returns=market_returns,
            timestamps=timestamps,
            actions=actions_by_cost.get(anchor_key) if actions_by_cost is not None else actions,
            base_config=base_config,
        )
    report_card = {
        "prediction": ladder["R0"],
        "execution": {
            "anchor_level": "R4",
            "anchor_cost_bps": float(anchor_key),
            "post_cost_sharpe": anchor_metrics["sharpe"],
            "max_drawdown": anchor_metrics["max_drawdown"],
            "turnover": anchor_metrics["turnover"],
            "calmar": anchor_metrics["calmar"],
            "net_return": anchor_metrics["net_pnl"],
            "cost_ladder": ladder["R4"]["cost_scan"],
            "r1_cost_ladder": ladder["R1"]["cost_scan"],
            "r2_cost_ladder": ladder["R2"]["cost_scan"],
            "r3_cost_ladder": ladder["R3"]["cost_scan"],
            "r4_cost_ladder": ladder["R4"]["cost_scan"],
            "cvar95": cvar_at_level(anchor_trajectory.net_returns),
            "average_holding_period": average_holding_period(anchor_trajectory.positions),
        },
        "gap": compute_gap_report({}),
        "robustness": anchor_robustness,
        "auditability": {
            "cost_ladder_bps": [float(cost) for cost in cost_bps_scan],
            "shared_evaluator": "R0-R4 ladder",
            "confidence_intervals": "protocol_target",
            "split_protocol": "single-slice motivating evidence; frozen 90-day 60/15/15 remains protocol target",
            "action_interface": action_metadata.get(anchor_key) if action_metadata else None,
        },
    }
    summary = {
        "directional_accuracy": ladder["R0"].get("directional_accuracy"),
        "rank_ic": ladder["R0"].get("rank_ic"),
        "mse": ladder["R0"].get("mse"),
        "mae": ladder["R0"].get("mae"),
        "cost_scan": ladder["R4"]["cost_scan"],
        "realism_ladder": ladder,
        "report_card": report_card,
        "anchor_cost_bps": float(anchor_key),
        "action_interface": action_metadata or {},
    }
    return summary
