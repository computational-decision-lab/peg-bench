"""Backtesting for continuous Gaussian policies trained by DPO."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from src.data.datasets import DatasetSpec, segmented_slice_time_indices
from src.eval.metrics import directional_accuracy, mean_absolute_error, mean_squared_error, rank_ic
from src.eval.peg_bench import _prediction_panel, run_realism_ladder
from src.models.baseline import KairosBaselineConfig, KairosBaselineModel
from src.models.policy import GaussianActionPolicy, GaussianPolicyConfig
from src.preference.filtering import AdaptiveConformalController
from src.simulation.cost_model import CostModelConfig
from src.simulation.execution import ExecutionSimulator, SimulatorConfig
from src.simulation.rollout import RISK_TO_ID
from src.utils.device import resolve_torch_device


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PolicyBacktestConfig:
    """Config for DPO policy backtesting."""

    dpo_checkpoint: str
    reference_checkpoint: str
    output_path: str
    selection_encoded_path: str | None = None
    selection_dataset_path: str | None = None
    evaluation_encoded_path: str | None = None
    evaluation_dataset_path: str | None = None
    encoded_path: str | None = None
    dataset_path: str | None = None
    seq_len: int = 64
    stride: int = 16
    target_key: str = "return_20"
    risk_mode: str = "auto"
    cost_bps_scan: tuple[float, ...] = (0.0, 1.0, 3.0, 5.0, 10.0)
    aci_path: str | None = None
    device: str = "cpu"
    scale_grid: tuple[float, ...] = (0.02, 0.05, 0.1, 0.2)
    deadzone_grid: tuple[float, ...] = (0.0, 0.05, 0.1)
    selection_cost_bps: float = 3.0
    selection_metric: str = "sharpe"
    min_selection_turnover: float = 0.0
    fallback_to_any_candidate: bool = True
    debug_fast_r4_only: bool = False
    progress_every: int = 0
    inference_batch_size: int = 2048
    diagnostics_output_path: str | None = None
    split_alignment_output_path: str | None = None
    sanity_check_output_path: str | None = None


def _load_policy(config: PolicyBacktestConfig, device: torch.device) -> GaussianActionPolicy:
    ref_payload = torch.load(config.reference_checkpoint, map_location=device)
    baseline_cfg = KairosBaselineConfig(**ref_payload["config"])
    backbone = KairosBaselineModel(baseline_cfg).to(device)
    backbone.load_state_dict(ref_payload["model_state_dict"], strict=False)
    policy = GaussianActionPolicy(GaussianPolicyConfig(baseline=baseline_cfg), backbone=backbone).to(device)
    dpo_payload = torch.load(config.dpo_checkpoint, map_location=device)
    policy.load_state_dict(dpo_payload["policy_state_dict"], strict=False)
    policy.eval()
    return policy


def _simulate_cost_scan(actions: np.ndarray, tokens: np.ndarray, lob: np.ndarray, returns: np.ndarray, timestamps: np.ndarray, risk_mode: str, costs: tuple[float, ...]) -> dict[str, dict[str, float]]:
    repeated_tokens = np.repeat(tokens[:, None, :], repeats=1, axis=1)
    result: dict[str, dict[str, float]] = {}
    for cost_bps in costs:
        simulator = ExecutionSimulator(SimulatorConfig(cost=CostModelConfig(fee_bps=cost_bps)))
        traj = simulator.simulate(
            trajectory_id=f"dpo:{risk_mode}:{cost_bps}",
            context_id=0,
            variant="policy_mean",
            risk_mode=risk_mode,
            state_tokens=repeated_tokens[:, 0],
            lob_states=lob,
            market_returns=returns,
            timestamps=timestamps,
            actions=actions.astype(np.float32),
            regime_tag="full_test",
        )
        result[str(cost_bps)] = asdict(traj.metrics)
    return result


def _array_stats(name: str, values: np.ndarray) -> dict[str, object]:
    arr = np.asarray(values)
    flat = arr.reshape(-1) if arr.size else arr.astype(np.float64).reshape(-1)
    if arr.dtype.kind in {"i", "u"}:
        finite_mask = np.isfinite(flat.astype(np.float64))
        nan_count = 0
        inf_count = 0
    else:
        finite_mask = np.isfinite(flat)
        nan_count = int(np.isnan(flat).sum())
        inf_count = int(np.isinf(flat).sum())
    finite_values = flat[finite_mask]
    return {
        "name": name,
        "shape": list(arr.shape),
        "length": int(len(flat)),
        "finite_ratio": float(finite_mask.mean()) if len(flat) else 1.0,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "min": float(np.min(finite_values)) if len(finite_values) else None,
        "max": float(np.max(finite_values)) if len(finite_values) else None,
        "mean": float(np.mean(finite_values)) if len(finite_values) else None,
        "std": float(np.std(finite_values)) if len(finite_values) else None,
        "first_nonfinite_indices": np.where(~finite_mask)[0][:5].astype(int).tolist() if len(flat) else [],
    }


def _sanitize_actions(actions: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    arr = np.asarray(actions, dtype=np.float32).reshape(-1)
    finite_mask = np.isfinite(arr)
    diagnostics = {
        "action_shape": list(np.asarray(actions).shape),
        "action_length": int(len(arr)),
        "action_finite_ratio": float(finite_mask.mean()) if len(arr) else 1.0,
        "action_nan_count": int(np.isnan(arr).sum()),
        "action_inf_count": int(np.isinf(arr).sum()),
        "first_nonfinite_indices": np.where(~finite_mask)[0][:5].astype(int).tolist() if len(arr) else [],
        "action_nonfinite_sanitized": False,
    }
    if not finite_mask.all():
        arr = np.where(finite_mask, arr, 0.0).astype(np.float32)
        diagnostics["action_nonfinite_sanitized"] = True
    return arr, diagnostics


def _sanitize_split_arrays(
    *,
    split_name: str,
    tokens: np.ndarray,
    lob: np.ndarray,
    returns: np.ndarray,
    timestamps: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    tokens_arr = np.asarray(tokens)
    lob_arr = np.asarray(lob, dtype=np.float32)
    returns_arr = np.asarray(returns, dtype=np.float32).reshape(-1)
    timestamps_arr = np.asarray(timestamps, dtype=np.int64).reshape(-1)

    base_len = min(len(tokens_arr), len(lob_arr), len(returns_arr), len(timestamps_arr))
    tokens_arr = tokens_arr[:base_len]
    lob_arr = lob_arr[:base_len]
    returns_arr = returns_arr[:base_len]
    timestamps_arr = timestamps_arr[:base_len]

    lob_valid = np.isfinite(lob_arr.reshape(base_len, -1)).all(axis=1)
    returns_valid = np.isfinite(returns_arr)
    timestamps_valid = np.isfinite(timestamps_arr.astype(np.float64))
    valid_mask = lob_valid & returns_valid & timestamps_valid

    diagnostics = {
        "split_name": split_name,
        "action_source": "policy_mean",
        "token_length": int(len(tokens_arr)),
        "lob_length": int(len(lob_arr)),
        "return_length": int(len(returns_arr)),
        "timestamp_length": int(len(timestamps_arr)),
        "after_alignment_length": int(base_len),
        "number_of_valid_samples": int(valid_mask.sum()),
        "validation_slice_empty": bool(base_len == 0),
        "first_timestamp": int(timestamps_arr[0]) if len(timestamps_arr) else None,
        "last_timestamp": int(timestamps_arr[-1]) if len(timestamps_arr) else None,
        "tokens": _array_stats(f"{split_name}.tokens", tokens_arr.astype(np.float64)),
        "lob": _array_stats(f"{split_name}.lob", lob_arr),
        "returns": _array_stats(f"{split_name}.returns", returns_arr),
        "timestamps": _array_stats(f"{split_name}.timestamps", timestamps_arr.astype(np.float64)),
        "invalid_reasons": [],
    }
    if not lob_valid.all():
        diagnostics["invalid_reasons"].append("lob_nonfinite_masked")
    if not returns_valid.all():
        diagnostics["invalid_reasons"].append("returns_nonfinite_masked")
    if not timestamps_valid.all():
        diagnostics["invalid_reasons"].append("timestamps_nonfinite_masked")
    if int(valid_mask.sum()) < 8:
        diagnostics["invalid_reasons"].append("too_few_valid_samples")

    return {
        "tokens": tokens_arr[valid_mask],
        "lob": np.nan_to_num(lob_arr[valid_mask], nan=0.0, posinf=0.0, neginf=0.0),
        "returns": returns_arr[valid_mask],
        "timestamps": timestamps_arr[valid_mask],
    }, diagnostics


def _simulate_cost_scan_with_diagnostics(
    *,
    actions: np.ndarray,
    tokens: np.ndarray,
    lob: np.ndarray,
    returns: np.ndarray,
    timestamps: np.ndarray,
    risk_mode: str,
    split_name: str,
    costs: tuple[float, ...],
) -> tuple[dict[str, dict[str, float]], list[dict[str, object]]]:
    sanitized_actions, action_diag = _sanitize_actions(actions)
    sanitized_split, split_diag = _sanitize_split_arrays(
        split_name=split_name,
        tokens=tokens,
        lob=lob,
        returns=returns,
        timestamps=timestamps,
    )
    diagnostics: list[dict[str, object]] = []
    if "too_few_valid_samples" in split_diag["invalid_reasons"]:
        for cost_bps in costs:
            diagnostics.append(
                {
                    "risk_mode": risk_mode,
                    "cost_bps": float(cost_bps),
                    "split": split_name,
                    "candidate_status": "invalid",
                    "reason": "too_few_valid_samples",
                    "action_diagnostics": action_diag,
                    "split_diagnostics": split_diag,
                }
            )
        return {str(float(cost_bps)): {
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "sharpe": 0.0,
            "calmar": 0.0,
            "max_drawdown": 0.0,
            "turnover": 0.0,
            "net_pnl": 0.0,
            "liquidation": False,
        } for cost_bps in costs}, diagnostics

    repeated_tokens = np.repeat(sanitized_split["tokens"][:, None, :], repeats=1, axis=1)
    result: dict[str, dict[str, float]] = {}
    for cost_bps in costs:
        simulator = ExecutionSimulator(SimulatorConfig(cost=CostModelConfig(fee_bps=cost_bps)))
        traj = simulator.simulate(
            trajectory_id=f"dpo:{risk_mode}:{cost_bps}",
            context_id=0,
            variant="policy_mean",
            risk_mode=risk_mode,
            state_tokens=repeated_tokens[:, 0],
            lob_states=sanitized_split["lob"],
            market_returns=sanitized_split["returns"],
            timestamps=sanitized_split["timestamps"],
            actions=sanitized_actions[: len(sanitized_split["returns"])].astype(np.float32),
            regime_tag="full_test",
        )
        metrics = asdict(traj.metrics)
        metric_diagnostics = {
            "gross_pnl": _array_stats("gross_returns", traj.gross_returns),
            "costs": _array_stats("transaction_costs", traj.transaction_costs),
            "net_pnl_vector": _array_stats("net_returns", traj.net_returns),
            "equity_curve": _array_stats("equity_curve", np.cumsum(np.nan_to_num(traj.net_returns, nan=0.0, posinf=0.0, neginf=0.0))),
            "positions": _array_stats("positions", traj.positions),
        }
        metric_invalid_reasons: list[str] = []
        if not np.isfinite(traj.net_returns).all():
            metric_invalid_reasons.append("non_finite_pnl")
        if not np.isfinite(np.cumsum(np.nan_to_num(traj.net_returns, nan=0.0, posinf=0.0, neginf=0.0))).all():
            metric_invalid_reasons.append("non_finite_equity")
        if len(traj.net_returns) == 0:
            metric_invalid_reasons.append("empty_series")
        elif np.std(np.nan_to_num(traj.net_returns, nan=0.0, posinf=0.0, neginf=0.0)) <= 1e-12:
            metric_invalid_reasons.append("zero_std")
        diagnostics.append(
            {
                "risk_mode": risk_mode,
                "cost_bps": float(cost_bps),
                "split": split_name,
                "candidate_status": "valid" if not metric_invalid_reasons else "sanitized",
                "reason": metric_invalid_reasons or ["finite"],
                "action_diagnostics": action_diag,
                "split_diagnostics": split_diag,
                "metric_diagnostics": metric_diagnostics,
                "metrics": metrics,
            }
        )
        LOGGER.info(
            "policy_backtest split=%s cost_bps=%.1f sharpe=%.6f turnover=%.6f",
            split_name,
            float(cost_bps),
            float(metrics.get("sharpe", 0.0)),
            float(metrics.get("turnover", 0.0)),
        )
        result[str(float(cost_bps))] = metrics
    return result, diagnostics


def _resolve_split_paths(config: PolicyBacktestConfig) -> tuple[tuple[str, str], tuple[str, str]]:
    legacy_paths = None
    if config.encoded_path and config.dataset_path:
        legacy_paths = (config.encoded_path, config.dataset_path)
    selection = (
        config.selection_encoded_path,
        config.selection_dataset_path,
    )
    evaluation = (
        config.evaluation_encoded_path,
        config.evaluation_dataset_path,
    )
    if selection[0] is None or selection[1] is None:
        if legacy_paths is None:
            raise ValueError("PolicyBacktestConfig requires selection split paths or legacy encoded_path/dataset_path.")
        selection = legacy_paths
    if evaluation[0] is None or evaluation[1] is None:
        if legacy_paths is None:
            raise ValueError("PolicyBacktestConfig requires evaluation split paths or legacy encoded_path/dataset_path.")
        evaluation = legacy_paths
    return (str(selection[0]), str(selection[1])), (str(evaluation[0]), str(evaluation[1]))


def _load_split_arrays(
    *,
    encoded_path: str,
    dataset_path: str,
    spec: DatasetSpec,
    split_name: str,
) -> dict[str, object]:
    token_payload = np.load(encoded_path)
    raw = np.load(dataset_path)
    codes = token_payload["codes"].astype(np.int64, copy=False)
    segment_lengths = raw["segment_lengths"].astype(np.int64) if "segment_lengths" in raw else None
    indices = segmented_slice_time_indices(
        len(codes),
        spec.seq_len,
        spec.stride,
        segment_lengths=segment_lengths,
    )
    if len(indices) == 0:
        raise ValueError(f"No windows available for policy backtest split={split_name}.")
    offsets = np.arange(spec.seq_len, dtype=np.int64)[None, :]
    window_indices = indices[:, None] - spec.seq_len + offsets
    tokens_np = codes[window_indices]
    aligned_lob = raw["lob"][indices]
    aligned_returns = raw[spec.target_key][indices]
    timestamps = raw["timestamps"][indices]
    return {
        "split": split_name,
        "encoded_path": str(Path(encoded_path).resolve()),
        "dataset_path": str(Path(dataset_path).resolve()),
        "num_windows": int(len(tokens_np)),
        "tokens": tokens_np,
        "lob": aligned_lob,
        "returns": aligned_returns,
        "timestamps": timestamps,
    }


def _run_sanity_check(
    *,
    policy: GaussianActionPolicy,
    split_payload: dict[str, object],
    device: torch.device,
    cost_bps: float,
    means_np: np.ndarray | None = None,
    stds_np: np.ndarray | None = None,
) -> dict[str, object]:
    tokens_np = np.asarray(split_payload["tokens"])
    returns_np = np.asarray(split_payload["returns"])
    lob_np = np.asarray(split_payload["lob"])
    timestamps_np = np.asarray(split_payload["timestamps"])
    if means_np is None or stds_np is None:
        means_np, stds_np = _predict_policy_statistics(
            policy=policy,
            tokens_np=tokens_np,
            risk_mode="aggressive",
            device=device,
            batch_size=2048,
        )
    rng = np.random.default_rng(7)
    candidates = {
        "all_zero_action": np.zeros(len(returns_np), dtype=np.float32),
        "constant_small_long": np.full(len(returns_np), 0.01, dtype=np.float32),
        "random_finite_action": np.clip(rng.normal(0.0, 0.02, size=len(returns_np)), -0.05, 0.05).astype(np.float32),
        "dpo_policy_action": _build_actions(
            means_np=means_np,
            stds_np=stds_np,
            scale=0.05,
            deadzone=0.0,
            controller=None,
        ),
    }
    results: dict[str, object] = {}
    for name, actions in candidates.items():
        cost_scan, diagnostics = _simulate_cost_scan_with_diagnostics(
            actions=actions,
            tokens=tokens_np,
            lob=lob_np,
            returns=returns_np,
            timestamps=timestamps_np,
            risk_mode="sanity",
            split_name="selection",
            costs=(cost_bps,),
        )
        metrics = cost_scan[str(float(cost_bps))]
        results[name] = {
            "finite_metrics": bool(all(np.isfinite(float(metrics[key])) for key in ("sharpe", "net_pnl", "max_drawdown", "turnover", "calmar"))),
            "sharpe_3bps": metrics["sharpe"],
            "net_return_3bps": metrics["net_pnl"],
            "maxdd": metrics["max_drawdown"],
            "turnover": metrics["turnover"],
            "calmar": metrics["calmar"],
            "valid_sample_count": diagnostics[0]["split_diagnostics"]["number_of_valid_samples"] if diagnostics else 0,
        }
    return results


def _predict_policy_statistics(
    *,
    policy: GaussianActionPolicy,
    tokens_np: np.ndarray,
    risk_mode: str,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    risk_id = RISK_TO_ID.get(risk_mode, 1)
    means_batches: list[np.ndarray] = []
    std_batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(tokens_np), batch_size):
            stop = min(start + batch_size, len(tokens_np))
            token_batch = torch.from_numpy(tokens_np[start:stop]).long().to(device)
            risk_ids = torch.full((stop - start,), risk_id, dtype=torch.long, device=device)
            means, stds = policy(input_ids=token_batch, risk_ids=risk_ids)
            means_batches.append(means.detach().cpu().numpy().astype(np.float32))
            std_batches.append(stds.detach().cpu().numpy().astype(np.float32))
    means_np = np.concatenate(means_batches, axis=0) if means_batches else np.zeros(0, dtype=np.float32)
    stds_np = np.concatenate(std_batches, axis=0) if std_batches else np.zeros(0, dtype=np.float32)
    return means_np, stds_np


def _build_actions(
    *,
    means_np: np.ndarray,
    stds_np: np.ndarray,
    scale: float,
    deadzone: float,
    controller: AdaptiveConformalController | None,
) -> np.ndarray:
    actions = means_np * float(scale)
    actions = np.where(np.abs(actions) < float(deadzone), 0.0, actions)
    if controller is not None:
        uncertainty = stds_np / (1.0 + stds_np)
        actions = np.asarray(
            [controller.scale_action(float(action), float(unc)) for action, unc in zip(actions, uncertainty)],
            dtype=np.float32,
        )
    return actions.astype(np.float32)


def _score_prediction_proxy(scores: np.ndarray, targets: np.ndarray) -> dict[str, float | str]:
    return {
        "directional_accuracy": directional_accuracy(scores, targets),
        "rank_ic": rank_ic(scores, targets),
        "mse": mean_squared_error(scores, targets),
        "mae": mean_absolute_error(scores, targets),
        "score_mean": float(np.mean(scores)) if len(scores) else 0.0,
        "score_std": float(np.std(scores)) if len(scores) else 0.0,
        "target_mean": float(np.mean(targets)) if len(targets) else 0.0,
        "target_std": float(np.std(targets)) if len(targets) else 0.0,
        "status": "computed",
    }


def _anchor_cost_key(cost_scan: dict[str, dict[str, float]], anchor_cost_bps: float) -> str:
    preferred = str(float(anchor_cost_bps))
    if preferred in cost_scan:
        return preferred
    for key in cost_scan:
        if float(key) == float(anchor_cost_bps):
            return key
    if not cost_scan:
        raise ValueError("Cost scan is empty; cannot resolve anchor cost key.")
    return sorted(cost_scan.keys(), key=float)[0]


def run_policy_backtest(config: PolicyBacktestConfig) -> dict[str, object]:
    """Run a post-cost backtest using validation-selected execution settings."""
    device = resolve_torch_device(config.device)
    policy = _load_policy(config, device)
    spec = DatasetSpec(seq_len=config.seq_len, stride=config.stride, target_key=config.target_key)
    (selection_encoded_path, selection_dataset_path), (evaluation_encoded_path, evaluation_dataset_path) = _resolve_split_paths(config)
    selection = _load_split_arrays(
        encoded_path=selection_encoded_path,
        dataset_path=selection_dataset_path,
        spec=spec,
        split_name="selection",
    )
    evaluation = _load_split_arrays(
        encoded_path=evaluation_encoded_path,
        dataset_path=evaluation_dataset_path,
        spec=spec,
        split_name="evaluation",
    )

    split_alignment = {
        "seq_len": config.seq_len,
        "stride": config.stride,
        "target_key": config.target_key,
        "selection_split": {
            "dataset_path": selection["dataset_path"],
            "encoded_path": selection["encoded_path"],
            "num_windows": selection["num_windows"],
        },
        "evaluation_split": {
            "dataset_path": evaluation["dataset_path"],
            "encoded_path": evaluation["encoded_path"],
            "num_windows": evaluation["num_windows"],
        },
    }
    if config.split_alignment_output_path:
        output = Path(config.split_alignment_output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(split_alignment, indent=2))

    controller = AdaptiveConformalController.load(config.aci_path) if config.aci_path else None
    candidate_modes = [config.risk_mode] if config.risk_mode != "auto" else ["aggressive", "balanced", "conservative", "low_turnover"]
    candidates: list[dict[str, object]] = []
    selection_predictions_by_mode: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    best_summary: dict[str, object] | None = None
    best_score = -1e18
    best_any_summary: dict[str, object] | None = None
    best_any_score = -1e18
    eligible_candidates = 0
    total_candidates = len(candidate_modes) * len(config.scale_grid) * len(config.deadzone_grid)
    candidate_index = 0
    diagnostics_payload: list[dict[str, object]] = []

    for risk_mode in candidate_modes:
        selection_means, selection_stds = _predict_policy_statistics(
            policy=policy,
            tokens_np=selection["tokens"],
            risk_mode=risk_mode,
            device=device,
            batch_size=config.inference_batch_size,
        )
        selection_predictions_by_mode[risk_mode] = (selection_means, selection_stds)

        for scale in config.scale_grid:
            for deadzone in config.deadzone_grid:
                candidate_index += 1
                actions = _build_actions(
                    means_np=selection_means,
                    stds_np=selection_stds,
                    scale=scale,
                    deadzone=deadzone,
                    controller=controller,
                )
                cost_scan, candidate_diagnostics = _simulate_cost_scan_with_diagnostics(
                    actions=actions,
                    tokens=selection["tokens"],
                    lob=selection["lob"],
                    returns=selection["returns"],
                    timestamps=selection["timestamps"],
                    risk_mode=risk_mode,
                    split_name="selection",
                    costs=(float(config.selection_cost_bps),),
                )
                diagnostics_payload.extend(
                    [
                        {
                            **item,
                            "candidate_config": {
                                "risk_mode": risk_mode,
                                "scale": float(scale),
                                "deadzone": float(deadzone),
                                "cost_bps": float(config.selection_cost_bps),
                                "split": "selection",
                                "action_source": "policy_mean",
                                "action_length": int(len(actions)),
                                "return_length": int(len(selection["returns"])),
                                "timestamp_length": int(len(selection["timestamps"])),
                            },
                        }
                        for item in candidate_diagnostics
                    ]
                )
                anchor_metrics = cost_scan[str(config.selection_cost_bps)]
                score = float(anchor_metrics.get(config.selection_metric, anchor_metrics.get("sharpe", 0.0)))
                if not np.isfinite(score):
                    score = -1e9
                anchor_turnover = float(anchor_metrics.get("turnover", 0.0))
                if not np.isfinite(anchor_turnover):
                    anchor_turnover = 0.0
                eligible = anchor_turnover >= float(config.min_selection_turnover)
                item = {
                    "risk_mode": risk_mode,
                    "scale": scale,
                    "deadzone": deadzone,
                    "selection_score": score,
                    "selection_metric": config.selection_metric,
                    "anchor_turnover": anchor_turnover,
                    "eligible_for_selection": eligible,
                    "cost_scan": cost_scan,
                }
                candidates.append(item)
                if score > best_any_score:
                    best_any_score = score
                    best_any_summary = item
                if eligible:
                    eligible_candidates += 1
                if eligible and score > best_score:
                    best_score = score
                    best_summary = item
                if config.progress_every > 0 and (
                    candidate_index == 1
                    or candidate_index == total_candidates
                    or candidate_index % config.progress_every == 0
                ):
                    LOGGER.info(
                        "policy_backtest progress=%s/%s risk=%s scale=%.6f deadzone=%.6f score=%.6f turnover=%.6f eligible=%s",
                        candidate_index,
                        total_candidates,
                        risk_mode,
                        scale,
                        deadzone,
                        score,
                        anchor_turnover,
                        eligible,
                    )

    fallback_used = False
    if best_summary is None:
        if not config.fallback_to_any_candidate:
            raise ValueError(
                "No candidate satisfied the minimum selection turnover "
                f"({config.min_selection_turnover})."
            )
        if best_any_summary is None:
            raise ValueError("No policy backtest candidates were generated.")
        best_summary = best_any_summary
        fallback_used = True

    if config.diagnostics_output_path:
        output = Path(config.diagnostics_output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(diagnostics_payload, indent=2))

    selected_risk_mode = str(best_summary["risk_mode"])
    LOGGER.info(
        "policy_backtest selected risk=%s scale=%.6f deadzone=%.6f evaluation_windows=%s batch=%s fast_r4_only=%s",
        selected_risk_mode,
        float(best_summary["scale"]),
        float(best_summary["deadzone"]),
        int(len(evaluation["tokens"])),
        int(config.inference_batch_size),
        bool(config.debug_fast_r4_only),
    )
    selection_means, selection_stds = selection_predictions_by_mode.get(selected_risk_mode, (None, None))
    if selection_means is None or selection_stds is None:
        selection_means, selection_stds = _predict_policy_statistics(
            policy=policy,
            tokens_np=selection["tokens"],
            risk_mode=selected_risk_mode,
            device=device,
            batch_size=config.inference_batch_size,
        )
    evaluation_means, evaluation_stds = _predict_policy_statistics(
        policy=policy,
        tokens_np=evaluation["tokens"],
        risk_mode=selected_risk_mode,
        device=device,
        batch_size=config.inference_batch_size,
    )
    LOGGER.info(
        "policy_backtest evaluation_forward_done windows=%s",
        int(len(evaluation_means)),
    )
    selection_actions = _build_actions(
        means_np=selection_means,
        stds_np=selection_stds,
        scale=float(best_summary["scale"]),
        deadzone=float(best_summary["deadzone"]),
        controller=controller,
    )
    evaluation_actions = _build_actions(
        means_np=evaluation_means,
        stds_np=evaluation_stds,
        scale=float(best_summary["scale"]),
        deadzone=float(best_summary["deadzone"]),
        controller=controller,
    )

    if config.sanity_check_output_path:
        output = Path(config.sanity_check_output_path)
        reuse_existing_sanity = False
        if config.debug_fast_r4_only and output.exists():
            try:
                existing_sanity = json.loads(output.read_text())
                required_sanity_keys = ("all_zero_action", "constant_small_long", "random_finite_action")
                reuse_existing_sanity = all(bool(existing_sanity.get(name, {}).get("finite_metrics")) for name in required_sanity_keys)
            except Exception:
                reuse_existing_sanity = False
        if not reuse_existing_sanity:
            sanity = _run_sanity_check(
                policy=policy,
                split_payload=selection,
                device=device,
                cost_bps=float(config.selection_cost_bps),
                means_np=selection_means,
                stds_np=selection_stds,
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(sanity, indent=2))

    if config.debug_fast_r4_only:
        LOGGER.info("policy_backtest debug_fast_r4_only cost_scan_start costs=%s", list(config.cost_bps_scan))
        evaluation_cost_scan, evaluation_diagnostics = _simulate_cost_scan_with_diagnostics(
            actions=evaluation_actions,
            tokens=evaluation["tokens"],
            lob=evaluation["lob"],
            returns=evaluation["returns"],
            timestamps=evaluation["timestamps"],
            risk_mode=selected_risk_mode,
            split_name="evaluation",
            costs=tuple(float(value) for value in config.cost_bps_scan),
        )
        diagnostics_payload.extend(
            [
                {
                    **item,
                    "candidate_config": {
                        "risk_mode": selected_risk_mode,
                        "scale": float(best_summary["scale"]),
                        "deadzone": float(best_summary["deadzone"]),
                        "cost_bps": float(item.get("cost_bps", config.selection_cost_bps)),
                        "split": item.get("split", "evaluation"),
                        "action_source": "policy_mean",
                        "action_length": int(len(evaluation_actions)),
                        "return_length": int(len(evaluation["returns"])),
                        "timestamp_length": int(len(evaluation["timestamps"])),
                    },
                }
                for item in evaluation_diagnostics
            ]
        )
        if config.diagnostics_output_path:
            output = Path(config.diagnostics_output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(diagnostics_payload, indent=2))
        LOGGER.info("policy_backtest debug_fast_r4_only cost_scan_done")
        anchor_key = _anchor_cost_key(evaluation_cost_scan, float(config.selection_cost_bps))
        anchor_metrics = evaluation_cost_scan[anchor_key]
        prediction_panel = _prediction_panel(
            np.nan_to_num(evaluation_means, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(evaluation["returns"], nan=0.0, posinf=0.0, neginf=0.0),
        )
        summary: dict[str, object] = {
            "selection_split": {key: value for key, value in selection.items() if key not in {"tokens", "lob", "returns", "timestamps"}},
            "evaluation_split": {key: value for key, value in evaluation.items() if key not in {"tokens", "lob", "returns", "timestamps"}},
            "selected_risk_mode": selected_risk_mode,
            "selected_scale": best_summary["scale"],
            "selected_deadzone": best_summary["deadzone"],
            "selection_score": best_summary["selection_score"],
            "selection_metric": config.selection_metric,
            "cost_scan": evaluation_cost_scan,
            "realism_ladder": {
                "status": "debug_fast_r4_only",
                "note": "Smoke-gate fast path. Selection and evaluation finite-metric verification are preserved; full R1-R3 ladder is skipped to reduce debug cost.",
                "R0": prediction_panel,
                "R4": {
                    "cost_scan": evaluation_cost_scan,
                    "robustness": {},
                },
            },
            "report_card": {
                "prediction": prediction_panel,
                "execution": {
                    "anchor_level": "R4_debug_fast_only",
                    "anchor_cost_bps": float(anchor_key),
                    "post_cost_sharpe": anchor_metrics["sharpe"],
                    "max_drawdown": anchor_metrics["max_drawdown"],
                    "turnover": anchor_metrics["turnover"],
                    "calmar": anchor_metrics["calmar"],
                    "net_return": anchor_metrics["net_pnl"],
                    "cost_ladder": evaluation_cost_scan,
                    "debug_mode": "fast_r4_only",
                },
                "gap": {
                    "status": "not_computed_debug_fast_r4_only"
                },
                "robustness": None,
                "auditability": {
                    "cost_ladder_bps": [float(cost) for cost in config.cost_bps_scan],
                    "shared_evaluator": "policy_backtest_debug_fast_r4_only",
                    "confidence_intervals": "not_computed_debug_fast_r4_only",
                    "split_protocol": "selection diagnostics remain validation-only for candidate choice; evaluation split is used only to confirm finite execution metrics for smoke gate.",
                    "action_interface": {
                        "selected_risk_mode": selected_risk_mode,
                        "selected_scale": float(best_summary["scale"]),
                        "selected_deadzone": float(best_summary["deadzone"]),
                        "mode": "policy_mean",
                    },
                },
            },
            "prediction_proxy": {
                "selection": _score_prediction_proxy(selection_means, selection["returns"]),
                "evaluation": _score_prediction_proxy(evaluation_means, evaluation["returns"]),
                "note": "Computed from deterministic policy mean against forward returns. This is a unified R0 proxy for the execution-first policy, not a pure return-forecast baseline output.",
            },
            "selection_action_summary": {
                "mean_abs_action": float(np.mean(np.abs(selection_actions))) if len(selection_actions) else 0.0,
                "nonzero_rate": float(np.mean(np.abs(selection_actions) > 0.0)) if len(selection_actions) else 0.0,
            },
            "evaluation_action_summary": {
                "mean_abs_action": float(np.mean(np.abs(evaluation_actions))) if len(evaluation_actions) else 0.0,
                "nonzero_rate": float(np.mean(np.abs(evaluation_actions) > 0.0)) if len(evaluation_actions) else 0.0,
                "turnover_proxy": float(np.sum(np.abs(np.diff(np.concatenate([np.zeros(1, dtype=np.float32), evaluation_actions]))))) if len(evaluation_actions) else 0.0,
            },
            "selection_constraints": {
                "selection_cost_bps": config.selection_cost_bps,
                "selection_metric": config.selection_metric,
                "min_selection_turnover": config.min_selection_turnover,
                "eligible_candidate_count": eligible_candidates,
                "fallback_to_any_candidate": config.fallback_to_any_candidate,
                "fallback_used": fallback_used,
            },
            "top_candidates": sorted(candidates, key=lambda item: item["selection_score"], reverse=True)[:8],
            "debug_mode": "fast_r4_only",
        }
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2))
        return summary

    ladder_summary = run_realism_ladder(
        predictions=np.nan_to_num(evaluation_means, nan=0.0, posinf=0.0, neginf=0.0),
        targets=np.nan_to_num(evaluation["returns"], nan=0.0, posinf=0.0, neginf=0.0),
        state_tokens=evaluation["tokens"],
        lob_states=np.nan_to_num(evaluation["lob"], nan=0.0, posinf=0.0, neginf=0.0),
        market_returns=np.nan_to_num(evaluation["returns"], nan=0.0, posinf=0.0, neginf=0.0),
        timestamps=evaluation["timestamps"],
        actions=evaluation_actions,
        cost_bps_scan=config.cost_bps_scan,
        anchor_cost_bps=config.selection_cost_bps,
    )
    summary: dict[str, object] = {
        "selection_split": {key: value for key, value in selection.items() if key not in {"tokens", "lob", "returns", "timestamps"}},
        "evaluation_split": {key: value for key, value in evaluation.items() if key not in {"tokens", "lob", "returns", "timestamps"}},
        "selected_risk_mode": selected_risk_mode,
        "selected_scale": best_summary["scale"],
        "selected_deadzone": best_summary["deadzone"],
        "selection_score": best_summary["selection_score"],
        "selection_metric": config.selection_metric,
        "cost_scan": ladder_summary["cost_scan"],
        "realism_ladder": ladder_summary["realism_ladder"],
        "report_card": ladder_summary["report_card"],
        "prediction_proxy": {
            "selection": _score_prediction_proxy(selection_means, selection["returns"]),
            "evaluation": _score_prediction_proxy(evaluation_means, evaluation["returns"]),
            "note": "Computed from deterministic policy mean against forward returns. This is a unified R0 proxy for the execution-first policy, not a pure return-forecast baseline output.",
        },
        "selection_action_summary": {
            "mean_abs_action": float(np.mean(np.abs(selection_actions))) if len(selection_actions) else 0.0,
            "nonzero_rate": float(np.mean(np.abs(selection_actions) > 0.0)) if len(selection_actions) else 0.0,
        },
        "evaluation_action_summary": {
            "mean_abs_action": float(np.mean(np.abs(evaluation_actions))) if len(evaluation_actions) else 0.0,
            "nonzero_rate": float(np.mean(np.abs(evaluation_actions) > 0.0)) if len(evaluation_actions) else 0.0,
            "turnover_proxy": float(np.sum(np.abs(np.diff(np.concatenate([np.zeros(1, dtype=np.float32), evaluation_actions]))))) if len(evaluation_actions) else 0.0,
        },
        "selection_constraints": {
            "selection_cost_bps": config.selection_cost_bps,
            "selection_metric": config.selection_metric,
            "min_selection_turnover": config.min_selection_turnover,
            "eligible_candidate_count": eligible_candidates,
            "fallback_to_any_candidate": config.fallback_to_any_candidate,
            "fallback_used": fallback_used,
        },
        "top_candidates": sorted(candidates, key=lambda item: item["selection_score"], reverse=True)[:8],
    }
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    return summary
