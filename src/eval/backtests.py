"""Backtesting helpers for baseline and aligned policies."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from src.data.datasets import DatasetSpec, LOBLatentSequenceDataset, LOBTokenSequenceDataset
from src.eval.peg_bench import run_realism_ladder
from src.models.baseline import KairosBaselineConfig, KairosBaselineModel
from src.models.action_interface import build_action_interface_result, serialize_action_interface_result
from src.utils.device import resolve_torch_device


@dataclass(slots=True)
class BacktestConfig:
    """Backtest config."""

    checkpoint_path: str
    encoded_path: str
    dataset_path: str
    output_path: str
    input_mode: str = "codes"
    seq_len: int = 96
    stride: int = 50
    target_key: str = "return_20"
    cost_bps_scan: tuple[float, ...] = (0.0, 1.0, 3.0, 5.0, 10.0)
    device: str = "cpu"


def _load_model(checkpoint_path: Path, device: torch.device) -> KairosBaselineModel:
    payload = torch.load(checkpoint_path, map_location=device)
    config = KairosBaselineConfig(**payload["config"])
    model = KairosBaselineModel(config).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.eval()
    return model


def run_baseline_backtest(config: BacktestConfig) -> dict[str, object]:
    """Run cost-sensitive baseline evaluation."""
    spec = DatasetSpec(seq_len=config.seq_len, stride=config.stride, target_key=config.target_key)
    if config.input_mode == "codes":
        dataset = LOBTokenSequenceDataset(config.encoded_path, config.dataset_path, spec)
    else:
        dataset = LOBLatentSequenceDataset(config.encoded_path, config.dataset_path, spec)
    raw_payload = np.load(config.dataset_path)

    device = resolve_torch_device(config.device)
    model = _load_model(Path(config.checkpoint_path), device)

    inputs = []
    targets = []
    for index in range(len(dataset)):
        sample, target = dataset[index]
        inputs.append(sample.numpy())
        targets.append(float(target.item()))
    if not inputs:
        raise ValueError("Dataset produced no windows for backtesting.")
    inputs_np = np.stack(inputs)
    targets_np = np.asarray(targets, dtype=np.float32)

    if config.input_mode == "codes":
        tensor = torch.from_numpy(inputs_np).long().to(device)
        with torch.no_grad():
            predictions = model(input_ids=tensor)["return_pred"].cpu().numpy()
    else:
        tensor = torch.from_numpy(inputs_np).float().to(device)
        with torch.no_grad():
            predictions = model(latents=tensor)["return_pred"].cpu().numpy()

    aligned_lob = raw_payload["lob"][spec.seq_len :: spec.stride][: len(predictions)]
    aligned_returns = raw_payload[config.target_key][spec.seq_len :: spec.stride][: len(predictions)]
    timestamps = raw_payload["timestamps"][spec.seq_len :: spec.stride][: len(predictions)]
    repeated_tokens = np.repeat(inputs_np[:, None, :], repeats=1, axis=1)
    actions_by_cost: dict[str, np.ndarray] = {}
    action_interface: dict[str, dict[str, object]] = {}
    for cost in config.cost_bps_scan:
        key = str(float(cost))
        action_result = build_action_interface_result(predictions, cost_bps=float(cost))
        actions_by_cost[key] = action_result.wrapped_deltas
        payload = serialize_action_interface_result(action_result)
        payload["cost_bps"] = float(cost)
        payload["delta"] = 0.0
        payload["rho"] = 0.9
        action_interface[key] = payload
    summary = run_realism_ladder(
        predictions=predictions,
        targets=targets_np,
        state_tokens=repeated_tokens[:, 0],
        lob_states=aligned_lob,
        market_returns=aligned_returns,
        timestamps=timestamps,
        actions_by_cost=actions_by_cost,
        cost_bps_scan=config.cost_bps_scan,
        anchor_cost_bps=3.0,
        action_metadata=action_interface,
    )

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    np.savez_compressed(output_path.with_suffix(".npz"), predictions=predictions, targets=targets_np)
    return summary


def cli_main() -> None:
    """CLI wrapper."""
    parser = argparse.ArgumentParser(description="Run baseline backtest across transaction-cost settings.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--encoded", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--input-mode", type=str, default="codes", choices=["codes", "latents"])
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    run_baseline_backtest(
        BacktestConfig(
            checkpoint_path=args.checkpoint,
            encoded_path=args.encoded,
            dataset_path=args.dataset,
            output_path=args.output,
            input_mode=args.input_mode,
            device=args.device,
        )
    )
