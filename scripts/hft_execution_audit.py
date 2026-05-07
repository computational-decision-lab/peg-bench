#!/usr/bin/env python3
"""Execution sensitivity audit with hftbacktest availability recorded.

This is not a full exchange L2 replay conversion. It uses frozen PEG action
paths and PEG returns, then brackets the action path with latency and queue-fill
haircuts. The output is marked as an audit sensitivity table, not a ranked
model result.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.benchmark_baselines import _load_rows, _window_core_arrays
from src.eval.metrics import summarize_trading_metrics
from src.models.action_interface import build_action_interface_result
from src.simulation.cost_model import CostModelConfig


CASES = {
    "time_h20": {
        "mode": "time",
        "target_key": "return_20",
        "compare_root": "/root/autodl-tmp/peg_bench_runs/tardis_90d_time_table2_compare_v1",
        "standardized_root": "/root/autodl-tmp/peg_bench_runs/tardis_90d_time_standardized_inputs",
        "methods": ("kairos_64code", "lightgbm_lob"),
    },
    "volume_h20": {
        "mode": "volume",
        "target_key": "return_20",
        "compare_root": "/root/autodl-tmp/peg_bench_runs/tardis_90d_volume_table2_compare_v1",
        "standardized_root": "/root/autodl-tmp/peg_bench_runs/tardis_90d_volume_standardized_inputs",
        "methods": ("kairos_64code", "lightgbm_lob"),
    },
}


PERIODS_PER_YEAR = 365 * 24 * 60 * 60
def rolling_std(values: np.ndarray, window: int = 20) -> np.ndarray:
    result = np.zeros(len(values), dtype=np.float32)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result[index] = float(np.std(values[start : index + 1])) if index >= 0 else 0.0
    return result


def visible_depth(lob_states: np.ndarray) -> np.ndarray:
    return np.exp(np.nan_to_num(lob_states[:, 1], nan=0.0, posinf=0.0, neginf=0.0)).sum(axis=(1, 2)).astype(np.float32)


def fast_r4_replay(
    *,
    positions: np.ndarray,
    market_returns: np.ndarray,
    lob_states: np.ndarray,
    cost_bps: float = 3.0,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Vectorized fixed-path replay for latency/queue sensitivity sweeps.

    This audit is intentionally not the ranked PEG evaluator. It brackets fixed
    action paths under latency and queue-fill haircuts while retaining the same
    fee plus square-root-impact cost formula.
    """
    cfg = CostModelConfig(fee_bps=float(cost_bps))
    executed_positions = np.nan_to_num(np.asarray(positions, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    returns = np.nan_to_num(np.asarray(market_returns, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    depth = visible_depth(lob_states)
    vol = rolling_std(returns)
    executed_deltas = np.diff(np.concatenate([np.zeros(1, dtype=np.float32), executed_positions])).astype(np.float32)
    relative_size = np.abs(executed_deltas) / np.maximum(depth, cfg.min_depth)
    impact_bps = cfg.impact_coef * np.sqrt(relative_size) * np.maximum(vol, 1e-6) * 10_000.0
    impact_bps = np.clip(impact_bps, 0.0, cfg.max_slippage_bps).astype(np.float32)
    total_cost = ((float(cost_bps) + impact_bps) / 10_000.0) * np.abs(executed_deltas)
    net_returns = (executed_positions * returns - total_cost).astype(np.float32)
    turnover = float(np.abs(executed_deltas).sum())
    metrics = summarize_trading_metrics(
        net_returns,
        turnover=turnover,
        liquidation=False,
        periods_per_year=PERIODS_PER_YEAR,
    )
    return asdict(metrics), net_returns, executed_deltas


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_case(spec: dict[str, Any]) -> dict[str, Any]:
    index_path = Path(spec["standardized_root"]) / "manifests" / f"{spec['mode']}_test_window_index_seq64.csv"
    panel_path = Path(spec["standardized_root"]) / "panels" / f"{spec['mode']}_test.npz"
    rows = _load_rows(index_path)
    arrays = _window_core_arrays(panel_path, rows, spec["target_key"])
    return arrays


def shift_positions(positions: np.ndarray, latency_bars: int) -> np.ndarray:
    if latency_bars <= 0:
        return positions.astype(np.float32)
    if latency_bars >= len(positions):
        return np.zeros_like(positions, dtype=np.float32)
    return np.concatenate([np.zeros(latency_bars, dtype=np.float32), positions[:-latency_bars].astype(np.float32)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)

    try:
        import hftbacktest  # type: ignore

        hft_status = "installed"
        hft_version = getattr(hftbacktest, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover
        hft_status = f"import_failed:{exc!r}"
        hft_version = None

    rows: list[dict[str, Any]] = []
    for panel, spec in CASES.items():
        arrays = load_case(spec)
        for method in spec["methods"]:
            pred_path = Path(spec["compare_root"]) / "methods" / method / "predictions.npz"
            payload = np.load(pred_path)
            predictions = np.nan_to_num(payload["test_predictions"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            base_positions = build_action_interface_result(predictions, cost_bps=3.0, rho=0.9).wrapped_positions
            for latency_bars in (0, 1, 5, 10, 25):
                delayed = shift_positions(base_positions, latency_bars)
                for queue_fill_probability in (1.0, 0.75, 0.5, 0.25):
                    executed_positions = delayed * float(queue_fill_probability)
                    metrics, _, _ = fast_r4_replay(
                        positions=executed_positions,
                        market_returns=arrays["targets"],
                        lob_states=arrays["lob_states"],
                        cost_bps=3.0,
                    )
                    rows.append(
                        {
                            "panel": panel,
                            "method": method,
                            "latency_bars": latency_bars,
                            "queue_fill_probability": queue_fill_probability,
                            "audit_type": "fixed_action_path_latency_queue_sensitivity",
                            "hftbacktest_import_status": hft_status,
                            "hftbacktest_version": hft_version,
                            "replay_engine": "fast_vectorized_peg_fee_impact_drawdown_replay",
                            "ranked_leaderboard": "no",
                            "r4_sharpe": metrics.get("sharpe"),
                            "r4_net_pnl": metrics.get("net_pnl"),
                            "r4_max_drawdown": metrics.get("max_drawdown"),
                            "r4_turnover": metrics.get("turnover"),
                            "worst_regime_sharpe": None,
                            "stress_day_survival": None,
                        }
                    )
    write_csv(out / "hftbacktest_fixed_action_sensitivity.csv", rows)
    (out / "run_metadata.json").write_text(
        json.dumps(
            {
                "hftbacktest_import_status": hft_status,
                "hftbacktest_version": hft_version,
                "interpretation": "Audit sensitivity over fixed PEG action paths with hftbacktest availability recorded; not a ranked model and not a full raw L2 HftBacktest replay conversion.",
                "replay_engine": "fast_vectorized_peg_fee_impact_replay_no_ranked_risk_controls",
            },
            indent=2,
        )
    )
    print(out)


if __name__ == "__main__":
    main()
