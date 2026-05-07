#!/usr/bin/env python3
"""Run fairness-tightened 30d PEG-Bench DPO stages across multiple seeds."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from typing import Any

import numpy as np

from scripts.run_peg_bench_dpo_stage import run_stage


def _metric(summary: dict[str, Any], path: list[str]) -> float | None:
    current: Any = summary
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if isinstance(current, (int, float)):
        return float(current)
    return None


def _aggregate(values: list[float | None]) -> dict[str, float | int] | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    array = np.asarray(present, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
        "count": int(array.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 30d PEG-Bench DPO stage over multiple seeds.")
    parser.add_argument("--standardized-root", type=str, required=True)
    parser.add_argument("--compare-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True, choices=["time", "volume"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 23])
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--max-contexts", type=int, default=192)
    parser.add_argument("--dpo-epochs", type=int, default=8)
    parser.add_argument("--dpo-batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--preference-mode", type=str, default="lexicographic", choices=["lexicographic", "scalar"])
    parser.add_argument("--preference-template", type=str, default="auto")
    parser.add_argument("--preference-gaps", type=float, nargs="+", default=[0.10, 0.0])
    parser.add_argument("--preference-aci-options", type=str, nargs="+", default=["true", "false"], choices=["true", "false"])
    parser.add_argument("--disable-scalar-fallback", action="store_true")
    parser.add_argument("--selection-metric", type=str, default="sharpe", choices=["sharpe", "net_pnl", "calmar"])
    parser.add_argument("--backtest-progress-every", type=int, default=0)
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    seed_runs: dict[str, dict[str, Any]] = {}
    for seed in args.seeds:
        seed_root = output_root / f"seed_{seed}"
        summary_path = seed_root / "run_summary.json"
        if args.skip_completed and summary_path.exists():
            seed_runs[str(seed)] = json.loads(summary_path.read_text())
            continue
        summary = run_stage(
            standardized_root=Path(args.standardized_root),
            compare_root=Path(args.compare_root),
            output_root=seed_root,
            mode=args.mode,
            seq_len=args.seq_len,
            stride=args.stride,
            horizon=args.horizon,
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
            seed=int(seed),
            selection_metric=args.selection_metric,
        )
        seed_runs[str(seed)] = summary

    aggregate = {
        "mode": args.mode,
        "seeds": list(args.seeds),
        "selection_metric": args.selection_metric,
        "anchor_sharpe_3bps": _aggregate(
            [_metric(summary, ["backtest", "cost_scan", "3.0", "sharpe"]) for summary in seed_runs.values()]
        ),
        "anchor_net_pnl_3bps": _aggregate(
            [_metric(summary, ["backtest", "cost_scan", "3.0", "net_pnl"]) for summary in seed_runs.values()]
        ),
        "max_drawdown_3bps": _aggregate(
            [_metric(summary, ["backtest", "cost_scan", "3.0", "max_drawdown"]) for summary in seed_runs.values()]
        ),
        "turnover_3bps": _aggregate(
            [_metric(summary, ["backtest", "cost_scan", "3.0", "turnover"]) for summary in seed_runs.values()]
        ),
        "r0_proxy_rank_ic": _aggregate(
            [_metric(summary, ["backtest", "prediction_proxy", "evaluation", "rank_ic"]) for summary in seed_runs.values()]
        ),
        "r0_proxy_directional_accuracy": _aggregate(
            [_metric(summary, ["backtest", "prediction_proxy", "evaluation", "directional_accuracy"]) for summary in seed_runs.values()]
        ),
        "selected_scale": {
            seed: _metric(summary, ["backtest", "selected_scale"]) for seed, summary in seed_runs.items()
        },
        "selected_deadzone": {
            seed: _metric(summary, ["backtest", "selected_deadzone"]) for seed, summary in seed_runs.items()
        },
        "selected_risk_mode": {
            seed: summary.get("backtest", {}).get("selected_risk_mode") for seed, summary in seed_runs.items()
        },
    }

    (output_root / "multiseed_summary.json").write_text(json.dumps(aggregate, indent=2))
    (output_root / "multiseed_runs.json").write_text(json.dumps(seed_runs, indent=2))
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
