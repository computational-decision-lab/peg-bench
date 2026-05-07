#!/usr/bin/env python3
"""Re-run PEG-Bench DPO backtests from existing checkpoints without retraining."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import logging

from src.eval.policy_backtest import PolicyBacktestConfig, run_policy_backtest


def _mode_defaults(compare_root: Path, standardized_root: Path, dpo_root: Path, mode: str) -> dict[str, Path]:
    return {
        "dpo_checkpoint": dpo_root / "dpo" / "best_dpo.pt",
        "reference_checkpoint": compare_root / "methods" / "kairos_64code" / "checkpoints" / "best_baseline.pt",
        "selection_encoded_path": compare_root / "vqvae_64code" / "encoded_val.npz",
        "selection_dataset_path": standardized_root / "panels" / f"{mode}_val.npz",
        "evaluation_encoded_path": compare_root / "vqvae_64code" / "encoded_test.npz",
        "evaluation_dataset_path": standardized_root / "panels" / f"{mode}_test.npz",
        "aci_path": dpo_root / "aci" / "calibration.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay PEG-Bench DPO backtests with custom execution-layer settings.")
    parser.add_argument("--compare-root", type=str, required=True)
    parser.add_argument("--standardized-root", type=str, required=True)
    parser.add_argument("--dpo-root", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True, choices=["time", "volume"])
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--target-key", type=str, default="return_20")
    parser.add_argument("--risk-mode", type=str, default="auto")
    parser.add_argument("--selection-cost-bps", type=float, default=3.0)
    parser.add_argument("--selection-metric", type=str, default="sharpe", choices=["sharpe", "net_pnl", "calmar"])
    parser.add_argument("--min-selection-turnover", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--disable-aci", action="store_true")
    parser.add_argument("--scale-grid", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2])
    parser.add_argument("--deadzone-grid", type=float, nargs="+", default=[0.0, 0.05, 0.1])
    parser.add_argument("--cost-bps", type=float, nargs="+", default=[0.0, 1.0, 3.0, 5.0, 10.0])
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--fast-r4-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--inference-batch-size", type=int, default=2048)
    parser.add_argument("--diagnostics-output", type=str, default=None)
    parser.add_argument("--split-alignment-output", type=str, default=None)
    parser.add_argument("--sanity-check-output", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

    compare_root = Path(args.compare_root)
    standardized_root = Path(args.standardized_root)
    dpo_root = Path(args.dpo_root)
    defaults = _mode_defaults(compare_root, standardized_root, dpo_root, args.mode)

    aci_path = None if args.disable_aci else str(defaults["aci_path"])
    summary = run_policy_backtest(
        PolicyBacktestConfig(
            dpo_checkpoint=str(defaults["dpo_checkpoint"]),
            reference_checkpoint=str(defaults["reference_checkpoint"]),
            selection_encoded_path=str(defaults["selection_encoded_path"]),
            selection_dataset_path=str(defaults["selection_dataset_path"]),
            evaluation_encoded_path=str(defaults["evaluation_encoded_path"]),
            evaluation_dataset_path=str(defaults["evaluation_dataset_path"]),
            output_path=args.output,
            seq_len=args.seq_len,
            stride=args.stride,
            target_key=args.target_key,
            risk_mode=args.risk_mode,
            cost_bps_scan=tuple(float(value) for value in args.cost_bps),
            aci_path=aci_path,
            device=args.device,
            scale_grid=tuple(float(value) for value in args.scale_grid),
            deadzone_grid=tuple(float(value) for value in args.deadzone_grid),
            selection_cost_bps=float(args.selection_cost_bps),
            selection_metric=str(args.selection_metric),
            min_selection_turnover=float(args.min_selection_turnover),
            fallback_to_any_candidate=not args.no_fallback,
            debug_fast_r4_only=bool(args.fast_r4_only),
            progress_every=int(args.progress_every),
            inference_batch_size=int(args.inference_batch_size),
            diagnostics_output_path=args.diagnostics_output,
            split_alignment_output_path=args.split_alignment_output,
            sanity_check_output_path=args.sanity_check_output,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
