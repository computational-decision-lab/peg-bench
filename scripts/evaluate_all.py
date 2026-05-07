#!/usr/bin/env python3
"""Run baseline evaluation and generate a compact markdown report."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import yaml
from pathlib import Path

from src.eval.backtests import BacktestConfig, run_baseline_backtest
from src.eval.plots import plot_cost_sensitivity
from src.eval.report import generate_markdown_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate KAIROS-X artifacts and generate a report.")
    parser.add_argument("--backtest-config", type=str, default="configs/backtest.yaml")
    parser.add_argument("--report", type=str, default="outputs/reports/summary.md")
    args = parser.parse_args()

    payload = yaml.safe_load(open(args.backtest_config, "r", encoding="utf-8"))
    payload["cost_bps_scan"] = tuple(payload.get("cost_bps_scan", [1.0, 3.0, 10.0]))
    summary = run_baseline_backtest(BacktestConfig(**payload))
    plot_cost_sensitivity(summary["cost_scan"], Path(args.report).with_name("cost_sensitivity.png"))
    generate_markdown_report(
        args.report,
        sections={"Baseline Backtest": summary},
        approximations=[
            "Execution cost uses a capacity-aware fee + impact approximation rather than exchange-native matching.",
            "Trajectory rollout replays historical bars and does not model latent counterfactual order flow.",
            "ACI uses residual-based conformal intervals as a research approximation under non-stationarity.",
        ],
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
