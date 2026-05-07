"""Preference dataset construction from trajectory rollout collections."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.models.baseline import KairosBaselineConfig, KairosBaselineModel
from src.preference.comparator import ComparatorConfig, PreferenceComparator
from src.preference.dataset import PreferencePairRecord, save_preference_dataset
from src.preference.filtering import ACIConfig, AdaptiveConformalController
from src.simulation.rollout import RISK_TO_ID, RolloutConfig, TrajectoryRolloutEngine
from src.simulation.utility import TrajectoryUtilityCalculator, UtilityConfig


@dataclass(slots=True)
class PreferenceBuilderConfig:
    """Config for offline preference pair generation."""

    encoded_path: str
    dataset_path: str
    checkpoint_path: str
    output_path: str
    mode: str = "lexicographic"
    lexicographic_template: str = "balanced"
    min_utility_gap: float = 0.1
    start_index: int = 96
    max_contexts: int = 64
    seq_len: int = 96
    horizon: int = 96
    stride: int = 96
    target_key: str = "return_20"
    device: str = "cpu"
    use_aci_filtering: bool = True


def resolve_lexicographic_template(template: str, risk_mode: str) -> str:
    """Map the configured template to the oracle template used for one risk mode."""
    if template == "auto":
        return risk_mode
    return template


def _load_baseline(checkpoint_path: Path, device: torch.device) -> KairosBaselineModel:
    payload = torch.load(checkpoint_path, map_location=device)
    config = KairosBaselineConfig(**payload["config"])
    model = KairosBaselineModel(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def build_preference_pairs(config: PreferenceBuilderConfig) -> dict[str, object]:
    """Generate preference pairs from baseline rollouts."""
    encoded = np.load(config.encoded_path)
    dataset = np.load(config.dataset_path)
    codes = encoded["codes"]
    lob = dataset["lob"]
    if config.target_key not in dataset:
        available = sorted(str(key) for key in dataset.files)
        raise KeyError(f"Target key {config.target_key!r} is missing from {config.dataset_path}. Available keys: {available}")
    returns = dataset[config.target_key]
    timestamps = dataset["timestamps"]

    device = torch.device(config.device)
    baseline = _load_baseline(Path(config.checkpoint_path), device)

    conformal = None
    if config.use_aci_filtering:
        conformal = AdaptiveConformalController(ACIConfig(min_pair_gap=config.min_utility_gap))
        if len(returns) > 1:
            conformal.fit(returns[:-1], returns[1:])
    rollout = TrajectoryRolloutEngine(
        RolloutConfig(seq_len=config.seq_len, horizon=config.horizon, device=config.device),
        conformal=conformal,
    )
    risk_modes = ["aggressive", "balanced", "conservative", "low_turnover"]
    records: list[PreferencePairRecord] = []
    template_usage = {mode: 0 for mode in risk_modes}
    stop = min(len(codes) - config.horizon, config.start_index + config.max_contexts * config.stride)
    context_indices = range(config.start_index, stop, config.stride)
    for context_id in context_indices:
        for risk_mode in risk_modes:
            resolved_template = resolve_lexicographic_template(config.lexicographic_template, risk_mode)
            comparator = PreferenceComparator(
                ComparatorConfig(
                    mode=config.mode,
                    lexicographic_template=resolved_template,
                    min_utility_gap=config.min_utility_gap,
                ),
                utility_calculator=TrajectoryUtilityCalculator(UtilityConfig()),
                conformal=conformal,
            )
            trajectories = rollout.build_candidate_trajectories(
                context_id=context_id,
                codes=codes,
                lob=lob,
                returns=returns,
                timestamps=timestamps,
                baseline_model=baseline,
                risk_mode=risk_mode,
            )
            for left_index in range(len(trajectories)):
                for right_index in range(left_index + 1, len(trajectories)):
                    compared = comparator.compare(trajectories[left_index], trajectories[right_index])
                    if compared is None:
                        continue
                    template_usage[resolved_template] = template_usage.get(resolved_template, 0) + 1
                    winner, loser, metadata = compared
                    records.append(
                        PreferencePairRecord(
                            context_id=context_id,
                            risk_mode=risk_mode,
                            risk_id=RISK_TO_ID[risk_mode],
                            winner_state_tokens=winner.state_tokens,
                            winner_lob_states=winner.lob_states,
                            winner_market_returns=winner.market_returns,
                            winner_actions=winner.actions,
                            loser_state_tokens=loser.state_tokens,
                            loser_lob_states=loser.lob_states,
                            loser_market_returns=loser.market_returns,
                            loser_actions=loser.actions,
                            metadata=metadata,
                        )
                    )

    save_preference_dataset(records, config.output_path)
    stats = {
        "num_pairs": len(records),
        "risk_mode_counts": {
            mode: sum(1 for record in records if record.risk_mode == mode) for mode in risk_modes
        },
        "target_key": config.target_key,
        "lexicographic_template": config.lexicographic_template,
        "template_usage_counts": template_usage,
        "output_path": str(Path(config.output_path).resolve()),
    }
    stats_path = Path(config.output_path).with_suffix(".json")
    stats_path.write_text(json.dumps(stats, indent=2))
    return stats


def cli_main() -> None:
    """CLI wrapper for offline preference generation."""
    parser = argparse.ArgumentParser(description="Build OG-PA preference pairs from baseline rollouts.")
    parser.add_argument("--encoded", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--mode", type=str, default="lexicographic", choices=["scalar", "lexicographic"])
    parser.add_argument("--template", type=str, default="balanced")
    parser.add_argument("--max-contexts", type=int, default=64)
    parser.add_argument("--target-key", type=str, default="return_20")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--disable-aci-filter", action="store_true")
    args = parser.parse_args()
    build_preference_pairs(
        PreferenceBuilderConfig(
            encoded_path=args.encoded,
            dataset_path=args.dataset,
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            mode=args.mode,
            lexicographic_template=args.template,
            max_contexts=args.max_contexts,
            target_key=args.target_key,
            device=args.device,
            use_aci_filtering=not args.disable_aci_filter,
        )
    )


if __name__ == "__main__":
    cli_main()
