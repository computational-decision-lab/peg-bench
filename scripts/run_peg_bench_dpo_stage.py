#!/usr/bin/env python3
"""Run a PEG-Bench -> OG-PA/DPO stage from existing panel artifacts."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json

import numpy as np

from src.eval.policy_backtest import PolicyBacktestConfig, run_policy_backtest
from src.preference.pair_builder import PreferenceBuilderConfig, build_preference_pairs
from src.train.aci_calibrator import ACICalibrationConfig, calibrate_aci
from src.train.dpo_trainer import ContinuousDPOTrainer, DPOTrainConfig


def _mode_paths(standardized_root: Path, compare_root: Path, mode: str) -> dict[str, Path]:
    panel_name = f"{mode}_train.npz"
    val_name = f"{mode}_val.npz"
    test_name = f"{mode}_test.npz"
    return {
        "train_panel": standardized_root / "panels" / panel_name,
        "val_panel": standardized_root / "panels" / val_name,
        "test_panel": standardized_root / "panels" / test_name,
        "encoded_train": compare_root / "vqvae_64code" / "encoded_tokens.npz",
        "encoded_val": compare_root / "vqvae_64code" / "encoded_val.npz",
        "encoded_test": compare_root / "vqvae_64code" / "encoded_test.npz",
        "baseline_checkpoint": compare_root / "methods" / "kairos_64code" / "checkpoints" / "best_baseline.pt",
        "predictions": compare_root / "methods" / "kairos_64code" / "predictions.npz",
    }


def _assert_exists(paths: dict[str, Path]) -> None:
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required PEG-Bench artifacts:\n" + "\n".join(missing))


def _build_aci(predictions_path: Path, output_root: Path) -> dict[str, object]:
    payload = np.load(predictions_path)
    calibration_input = output_root / "aci" / "val_predictions_for_aci.npz"
    calibration_input.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        calibration_input,
        predictions=payload["val_predictions"].astype(np.float32),
        targets=payload["val_targets"].astype(np.float32),
    )
    return calibrate_aci(
        ACICalibrationConfig(
            predictions_path=str(calibration_input),
            output_path=str(output_root / "aci" / "calibration.json"),
        )
    )


def _build_preferences_with_fallback(
    *,
    train_panel: Path,
    encoded_train: Path,
    baseline_checkpoint: Path,
    output_root: Path,
    seq_len: int,
    stride: int,
    horizon: int,
    target_key: str,
    max_contexts: int,
    device: str,
    preference_mode: str,
    preference_template: str,
    preference_gaps: tuple[float, ...],
    preference_aci_options: tuple[bool, ...],
    include_scalar_fallback: bool,
) -> dict[str, object]:
    pref_path = output_root / "preferences" / "pairs.pkl"
    pref_path.parent.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, object]] = []
    for gap in preference_gaps:
        for aci in preference_aci_options:
            attempts.append(
                {
                    "mode": preference_mode,
                    "template": preference_template,
                    "gap": float(gap),
                    "aci": bool(aci),
                }
            )
    if include_scalar_fallback:
        attempts.append({"mode": "scalar", "template": "balanced", "gap": 0.00, "aci": False})
    latest: dict[str, object] | None = None
    for attempt in attempts:
        stats = build_preference_pairs(
            PreferenceBuilderConfig(
                encoded_path=str(encoded_train),
                dataset_path=str(train_panel),
                checkpoint_path=str(baseline_checkpoint),
                output_path=str(pref_path),
                mode=str(attempt["mode"]),
                lexicographic_template=str(attempt["template"]),
                min_utility_gap=float(attempt["gap"]),
                start_index=seq_len,
                max_contexts=max_contexts,
                seq_len=seq_len,
                horizon=horizon,
                stride=stride,
                target_key=target_key,
                device=device,
                use_aci_filtering=bool(attempt["aci"]),
            )
        )
        stats["attempt"] = attempt
        latest = stats
        if int(stats["num_pairs"]) > 0:
            break
    if latest is None or int(latest["num_pairs"]) <= 0:
        raise RuntimeError("Preference construction produced zero pairs.")
    return latest


def run_stage(
    *,
    standardized_root: Path,
    compare_root: Path,
    output_root: Path,
    mode: str,
    seq_len: int,
    stride: int,
    horizon: int,
    target_key: str,
    max_contexts: int,
    dpo_epochs: int,
    dpo_batch_size: int,
    device: str,
    preference_mode: str,
    preference_template: str,
    preference_gaps: tuple[float, ...],
    preference_aci_options: tuple[bool, ...],
    include_scalar_fallback: bool,
    backtest_progress_every: int,
    seed: int,
    selection_metric: str,
) -> dict[str, object]:
    paths = _mode_paths(standardized_root, compare_root, mode)
    _assert_exists(paths)
    output_root.mkdir(parents=True, exist_ok=True)

    aci_summary = _build_aci(paths["predictions"], output_root)
    preference_summary = _build_preferences_with_fallback(
        train_panel=paths["train_panel"],
        encoded_train=paths["encoded_train"],
        baseline_checkpoint=paths["baseline_checkpoint"],
        output_root=output_root,
        seq_len=seq_len,
        stride=stride,
        horizon=horizon,
        target_key=target_key,
        max_contexts=max_contexts,
        device=device,
        preference_mode=preference_mode,
        preference_template=preference_template,
        preference_gaps=preference_gaps,
        preference_aci_options=preference_aci_options,
        include_scalar_fallback=include_scalar_fallback,
    )

    dpo_summary = ContinuousDPOTrainer(
        DPOTrainConfig(
            preferences_path=str(output_root / "preferences" / "pairs.pkl"),
            reference_checkpoint=str(paths["baseline_checkpoint"]),
            output_dir=str(output_root / "dpo"),
            epochs=dpo_epochs,
            batch_size=dpo_batch_size,
            device=device,
            seed=seed,
        )
    ).train()

    backtest_summary = run_policy_backtest(
        PolicyBacktestConfig(
            dpo_checkpoint=str(output_root / "dpo" / "best_dpo.pt"),
            reference_checkpoint=str(paths["baseline_checkpoint"]),
            selection_encoded_path=str(paths["encoded_val"]),
            selection_dataset_path=str(paths["val_panel"]),
            evaluation_encoded_path=str(paths["encoded_test"]),
            evaluation_dataset_path=str(paths["test_panel"]),
            output_path=str(output_root / "eval" / "dpo_backtest.json"),
            seq_len=seq_len,
            stride=stride,
            target_key=target_key,
            aci_path=str(output_root / "aci" / "calibration.json"),
            device=device,
            progress_every=backtest_progress_every,
            selection_metric=selection_metric,
        )
    )
    (output_root / "eval" / "dpo_prediction_proxy.json").write_text(
        json.dumps(backtest_summary.get("prediction_proxy", {}), indent=2)
    )
    (output_root / "preference_pair_stats.json").write_text(json.dumps(preference_summary, indent=2))
    (output_root / "eval" / "benchmark_report_row.json").write_text(
        json.dumps(
            {
                "method": "ogpa_dpo",
                "mode": mode,
                "target_key": target_key,
                "report_card": backtest_summary.get("report_card", {}),
                "realism_ladder": backtest_summary.get("realism_ladder", {}),
                "selection_constraints": backtest_summary.get("selection_constraints", {}),
            },
            indent=2,
        )
    )

    summary = {
        "mode": mode,
        "standardized_root": str(standardized_root.resolve()),
        "compare_root": str(compare_root.resolve()),
        "output_root": str(output_root.resolve()),
        "seq_len": seq_len,
        "stride": stride,
        "horizon": horizon,
        "target_key": target_key,
        "max_contexts": max_contexts,
        "seed": seed,
        "selection_metric": selection_metric,
        "aci": aci_summary,
        "preferences": preference_summary,
        "dpo": dpo_summary,
        "backtest": backtest_summary,
    }
    summary_path = output_root / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PEG-Bench DPO stage from existing panel artifacts.")
    parser.add_argument("--standardized-root", type=str, required=True)
    parser.add_argument("--compare-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True, choices=["time", "volume"])
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--target-key", type=str, default="return_20")
    parser.add_argument("--max-contexts", type=int, default=192)
    parser.add_argument("--dpo-epochs", type=int, default=8)
    parser.add_argument("--dpo-batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--preference-mode", type=str, default="lexicographic", choices=["lexicographic", "scalar"])
    parser.add_argument("--preference-template", type=str, default="auto")
    parser.add_argument("--preference-gaps", type=float, nargs="+", default=[0.10, 0.0])
    parser.add_argument("--preference-aci-options", type=str, nargs="+", default=["true", "false"], choices=["true", "false"])
    parser.add_argument("--disable-scalar-fallback", action="store_true")
    parser.add_argument("--backtest-progress-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--selection-metric", type=str, default="sharpe", choices=["sharpe", "net_pnl", "calmar"])
    args = parser.parse_args()

    summary = run_stage(
        standardized_root=Path(args.standardized_root),
        compare_root=Path(args.compare_root),
        output_root=Path(args.output_root),
        mode=args.mode,
        seq_len=args.seq_len,
        stride=args.stride,
        horizon=args.horizon,
        target_key=args.target_key,
        max_contexts=args.max_contexts,
        dpo_epochs=args.dpo_epochs,
        dpo_batch_size=args.dpo_batch_size,
        device=args.device,
        preference_mode=args.preference_mode,
        preference_template=args.preference_template,
        preference_gaps=tuple(float(value) for value in args.preference_gaps),
        preference_aci_options=tuple(value == "true" for value in args.preference_aci_options),
        include_scalar_fallback=not args.disable_scalar_fallback,
        backtest_progress_every=args.backtest_progress_every,
        seed=args.seed,
        selection_metric=args.selection_metric,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
