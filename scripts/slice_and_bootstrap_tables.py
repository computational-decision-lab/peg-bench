#!/usr/bin/env python3
"""Reviewer-facing slice breakdowns and action-path bootstrap CIs.

The script is read-only with respect to the frozen benchmark runs. It consumes
test predictions plus frozen window indices and emits supplemental CSV tables:

* per_asset_venue_breakdown.csv
* action_path_metric_bootstrap_ci.csv
* action_path_gap_bootstrap_ci.csv
* target_baseline_summary.csv

TABL-lite and iTransformer-lite rows are explicitly named as benchmark-native
lite implementations rather than official external-repository reproductions.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.benchmark_baselines import _load_rows, _window_core_arrays
from src.eval.metrics import (
    directional_accuracy,
    mean_absolute_error,
    mean_squared_error,
    rank_ic,
    summarize_trading_metrics,
)
from src.eval.peg_bench import _compute_robustness_panel
from src.models.action_interface import build_action_interface_result
from src.simulation.cost_model import CostModelConfig


PANEL_SPECS: dict[str, dict[str, str]] = {
    "time_h5": {
        "mode": "time",
        "target_key": "return_5",
        "compare_root": "/root/autodl-tmp/peg_bench_runs/tardis_90d_time_h5_table2_compare_v1",
        "standardized_root": "/root/autodl-tmp/peg_bench_runs/tardis_90d_time_standardized_inputs",
    },
    "time_h20": {
        "mode": "time",
        "target_key": "return_20",
        "compare_root": "/root/autodl-tmp/peg_bench_runs/tardis_90d_time_table2_compare_v1",
        "standardized_root": "/root/autodl-tmp/peg_bench_runs/tardis_90d_time_standardized_inputs",
    },
    "time_h50": {
        "mode": "time",
        "target_key": "return_50",
        "compare_root": "/root/autodl-tmp/peg_bench_runs/tardis_90d_time_h50_table2_compare_v1",
        "standardized_root": "/root/autodl-tmp/peg_bench_runs/tardis_90d_time_standardized_inputs",
    },
    "volume_h5": {
        "mode": "volume",
        "target_key": "return_5",
        "compare_root": "/root/autodl-tmp/peg_bench_runs/tardis_90d_volume_h5_table2_compare_v1",
        "standardized_root": "/root/autodl-tmp/peg_bench_runs/tardis_90d_volume_standardized_inputs",
    },
    "volume_h20": {
        "mode": "volume",
        "target_key": "return_20",
        "compare_root": "/root/autodl-tmp/peg_bench_runs/tardis_90d_volume_table2_compare_v1",
        "standardized_root": "/root/autodl-tmp/peg_bench_runs/tardis_90d_volume_standardized_inputs",
    },
    "volume_h50": {
        "mode": "volume",
        "target_key": "return_50",
        "compare_root": "/root/autodl-tmp/peg_bench_runs/tardis_90d_volume_h50_table2_compare_v1",
        "standardized_root": "/root/autodl-tmp/peg_bench_runs/tardis_90d_volume_standardized_inputs",
    },
}


PERIODS_PER_YEAR = 365 * 24 * 60 * 60
LIQUIDATION_DRAWDOWN = 0.20
HARD_STOP_DRAWDOWN = 0.10


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ci(values: list[float]) -> tuple[float | None, float | None, float | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return None, None, None
    return float(np.mean(arr)), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def index_path(spec: dict[str, str], split: str = "test", seq_len: int = 64) -> Path:
    return Path(spec["standardized_root"]) / "manifests" / f"{spec['mode']}_{split}_window_index_seq{seq_len}.csv"


def panel_path(spec: dict[str, str], split: str = "test") -> Path:
    return Path(spec["standardized_root"]) / "panels" / f"{spec['mode']}_{split}.npz"


def load_panel(spec: dict[str, str]) -> dict[str, Any]:
    rows = _load_rows(index_path(spec))
    arrays = _window_core_arrays(panel_path(spec), rows, spec["target_key"])
    arrays["rows"] = rows
    arrays["exchange"] = np.asarray([row["exchange"] for row in rows])
    arrays["symbol"] = np.asarray([row["symbol"] for row in rows])
    arrays["segment_key"] = np.asarray([row["segment_key"] for row in rows])
    return arrays


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
    cost_bps: float,
) -> dict[str, Any]:
    cfg = CostModelConfig(fee_bps=float(cost_bps))
    desired = np.nan_to_num(np.asarray(positions, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    returns = np.nan_to_num(np.asarray(market_returns, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    depth = visible_depth(lob_states)
    vol = rolling_std(returns)
    executed_positions = np.zeros_like(desired, dtype=np.float32)
    executed_deltas = np.zeros_like(desired, dtype=np.float32)
    net_returns = np.zeros_like(desired, dtype=np.float32)
    position = 0.0
    cumulative = 0.0
    running_peak = 0.0
    liquidation = False
    for step, target in enumerate(desired):
        target_position = float(np.clip(target, -1.0, 1.0))
        trade_size = target_position - position
        depth_step = max(float(depth[step]), cfg.min_depth)
        relative_size = abs(trade_size) / depth_step
        impact_bps = cfg.impact_coef * np.sqrt(relative_size) * max(float(vol[step]), 1e-6) * 10_000.0
        impact_bps = float(np.clip(impact_bps, 0.0, cfg.max_slippage_bps))
        total_cost = (float(cost_bps) + impact_bps) / 10_000.0 * abs(trade_size)
        position = target_position
        executed_positions[step] = position
        executed_deltas[step] = trade_size
        net = position * float(returns[step]) - total_cost
        net_returns[step] = net
        cumulative += net
        running_peak = max(running_peak, cumulative)
        drawdown = running_peak - cumulative
        if drawdown >= LIQUIDATION_DRAWDOWN:
            liquidation = True
            if step + 1 < len(net_returns):
                executed_positions[step + 1 :] = 0.0
                executed_deltas[step + 1 :] = 0.0
                net_returns[step + 1 :] = 0.0
            break
        if drawdown >= HARD_STOP_DRAWDOWN:
            position = 0.0
    turnover = float(np.abs(executed_deltas).sum())
    metrics = summarize_trading_metrics(
        net_returns,
        turnover=turnover,
        liquidation=liquidation,
        periods_per_year=PERIODS_PER_YEAR,
    )
    robustness = _compute_robustness_panel(
        net_returns=net_returns,
        actions=executed_deltas,
        market_returns=returns,
        lob_states=lob_states,
        periods_per_year=PERIODS_PER_YEAR,
        failure_drawdown=LIQUIDATION_DRAWDOWN,
    )
    return {
        "net_returns": net_returns,
        "actions": executed_deltas,
        "positions": executed_positions,
        "metrics": metrics,
        "robustness": robustness,
    }


def ordered_mask(mask: np.ndarray, arrays: dict[str, Any]) -> np.ndarray:
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return indices
    return indices[np.argsort(arrays["timestamps"][indices], kind="mergesort")]


def replay_slice(
    *,
    arrays: dict[str, Any],
    predictions: np.ndarray,
    mask: np.ndarray,
    cost_bps: float,
) -> dict[str, Any] | None:
    segment_outputs: list[dict[str, Any]] = []
    pred_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    for segment in sorted({str(item) for item in arrays["segment_key"][mask].tolist()}):
        segment_mask = mask & (arrays["segment_key"] == segment)
        order = ordered_mask(segment_mask, arrays)
        if len(order) == 0:
            continue
        segment_pred = np.nan_to_num(predictions[order].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        action = build_action_interface_result(segment_pred, cost_bps=cost_bps, rho=0.9)
        segment_outputs.append(
            fast_r4_replay(
                positions=action.wrapped_positions,
                market_returns=arrays["targets"][order],
                lob_states=arrays["lob_states"][order],
                cost_bps=cost_bps,
            )
        )
        pred_parts.append(segment_pred)
        target_parts.append(arrays["targets"][order].astype(np.float32))
    if not segment_outputs:
        return None
    net_returns = np.concatenate([item["net_returns"] for item in segment_outputs]).astype(np.float32)
    actions = np.concatenate([item["actions"] for item in segment_outputs]).astype(np.float32)
    targets = np.concatenate(target_parts).astype(np.float32)
    preds = np.concatenate(pred_parts).astype(np.float32)
    lob_states = np.concatenate(
        [arrays["lob_states"][ordered_mask(mask & (arrays["segment_key"] == segment), arrays)] for segment in sorted({str(item) for item in arrays["segment_key"][mask].tolist()})],
        axis=0,
    ).astype(np.float32)
    metrics = summarize_trading_metrics(
        net_returns,
        turnover=float(np.abs(actions).sum()),
        liquidation=bool(any(item["metrics"].liquidation for item in segment_outputs)),
        periods_per_year=PERIODS_PER_YEAR,
    )
    robustness = _compute_robustness_panel(
        net_returns=net_returns,
        actions=actions,
        market_returns=targets,
        lob_states=lob_states,
        periods_per_year=PERIODS_PER_YEAR,
        failure_drawdown=LIQUIDATION_DRAWDOWN,
    )
    return {
        "predictions": preds,
        "targets": targets,
        "net_returns": net_returns,
        "actions": actions,
        "lob_states": lob_states,
        "metrics": metrics,
        "robustness": robustness,
    }


def method_prediction_paths(panel: str, spec: dict[str, str], target_root: Path | None) -> list[tuple[str, Path, str]]:
    paths: list[tuple[str, Path, str]] = []
    methods_root = Path(spec["compare_root"]) / "methods"
    for method_dir in sorted(path for path in methods_root.iterdir() if path.is_dir()):
        pred_path = method_dir / "predictions.npz"
        if pred_path.exists():
            paths.append((method_dir.name, pred_path, "core_90d_method"))
    if target_root is not None:
        panel_root = target_root / panel / "methods"
        if panel_root.exists():
            for method_dir in sorted(path for path in panel_root.iterdir() if path.is_dir()):
                pred_path = method_dir / "predictions.npz"
                if pred_path.exists():
                    source = "reviewer_target_baseline_lite" if method_dir.name.endswith("_lite") else "reviewer_target_baseline"
                    paths.append((method_dir.name, pred_path, source))
    return paths


def load_predictions(path: Path) -> np.ndarray:
    payload = np.load(path)
    return np.nan_to_num(payload["test_predictions"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def metric_row(
    *,
    panel: str,
    method: str,
    source: str,
    grouping: str,
    group_value: str,
    replay: dict[str, Any],
    cost_bps: float,
) -> dict[str, Any]:
    metrics = replay["metrics"]
    robustness = replay["robustness"]
    return {
        "panel": panel,
        "method": method,
        "method_source": source,
        "grouping": grouping,
        "group_value": group_value,
        "cost_bps": cost_bps,
        "n_windows": len(replay["targets"]),
        "directional_accuracy": directional_accuracy(replay["predictions"], replay["targets"]),
        "rank_ic": rank_ic(replay["predictions"], replay["targets"]),
        "mse": mean_squared_error(replay["predictions"], replay["targets"]),
        "mae": mean_absolute_error(replay["predictions"], replay["targets"]),
        "r4_sharpe_fast": metrics.sharpe,
        "r4_max_drawdown_fast": metrics.max_drawdown,
        "r4_turnover_fast": metrics.turnover,
        "r4_net_pnl_fast": metrics.net_pnl,
        "worst_regime_sharpe_fast": robustness.get("worst_regime_sharpe"),
        "stress_day_survival_fast": robustness.get("stress_day_survival"),
        "replay_note": "fast O(n) PEG action-interface replay with fee/impact and drawdown stops; supplemental slice diagnostic",
    }


def compute_slice_breakdown(out: Path, target_root: Path | None, cost_bps: float) -> None:
    rows: list[dict[str, Any]] = []
    for panel, spec in PANEL_SPECS.items():
        arrays = load_panel(spec)
        for method, pred_path, source in method_prediction_paths(panel, spec, target_root):
            predictions = load_predictions(pred_path)
            if len(predictions) != len(arrays["targets"]):
                continue
            masks: list[tuple[str, str, np.ndarray]] = [("all", "all", np.ones(len(predictions), dtype=bool))]
            for symbol in sorted({str(item) for item in arrays["symbol"].tolist()}):
                masks.append(("asset", symbol, arrays["symbol"] == symbol))
            for exchange in sorted({str(item) for item in arrays["exchange"].tolist()}):
                masks.append(("venue", exchange, arrays["exchange"] == exchange))
            for asset_venue in sorted({f"{exchange}:{symbol}" for exchange, symbol in zip(arrays["exchange"], arrays["symbol"])}):
                exchange, symbol = asset_venue.split(":", 1)
                masks.append(("asset_venue", asset_venue, (arrays["exchange"] == exchange) & (arrays["symbol"] == symbol)))
            for grouping, group_value, mask in masks:
                replay = replay_slice(arrays=arrays, predictions=predictions, mask=mask, cost_bps=cost_bps)
                if replay is not None:
                    rows.append(
                        metric_row(
                            panel=panel,
                            method=method,
                            source=source,
                            grouping=grouping,
                            group_value=group_value,
                            replay=replay,
                            cost_bps=cost_bps,
                        )
                    )
    write_csv(out / "per_asset_venue_breakdown.csv", rows)


def bootstrap_metric_values(blocks: list[dict[str, Any]], draws: int, rng: np.random.Generator) -> dict[str, tuple[float | None, float | None, float | None]]:
    metric_samples: dict[str, list[float]] = {"r4_sharpe": [], "max_drawdown": [], "worst_regime_sharpe": []}
    if not blocks:
        return {key: (None, None, None) for key in metric_samples}
    for _ in range(draws):
        sample = list(rng.choice(blocks, size=len(blocks), replace=True))
        net_returns = np.concatenate([item["net_returns"] for item in sample]).astype(np.float32)
        actions = np.concatenate([item["actions"] for item in sample]).astype(np.float32)
        targets = np.concatenate([item["targets"] for item in sample]).astype(np.float32)
        lob_states = np.concatenate([item["lob_states"] for item in sample], axis=0).astype(np.float32)
        metrics = summarize_trading_metrics(net_returns, turnover=float(np.abs(actions).sum()), liquidation=False, periods_per_year=PERIODS_PER_YEAR)
        robustness = _compute_robustness_panel(
            net_returns=net_returns,
            actions=actions,
            market_returns=targets,
            lob_states=lob_states,
            periods_per_year=PERIODS_PER_YEAR,
            failure_drawdown=LIQUIDATION_DRAWDOWN,
        )
        metric_samples["r4_sharpe"].append(metrics.sharpe)
        metric_samples["max_drawdown"].append(metrics.max_drawdown)
        metric_samples["worst_regime_sharpe"].append(float(robustness.get("worst_regime_sharpe") or 0.0))
    return {key: ci(values) for key, values in metric_samples.items()}


def segment_blocks(arrays: dict[str, Any], predictions: np.ndarray, cost_bps: float) -> dict[str, dict[str, Any]]:
    blocks: dict[str, dict[str, Any]] = {}
    for segment in sorted({str(item) for item in arrays["segment_key"].tolist()}):
        mask = arrays["segment_key"] == segment
        order = ordered_mask(mask, arrays)
        if len(order) == 0:
            continue
        segment_pred = predictions[order].astype(np.float32)
        action = build_action_interface_result(segment_pred, cost_bps=cost_bps, rho=0.9)
        replay = fast_r4_replay(
            positions=action.wrapped_positions,
            market_returns=arrays["targets"][order],
            lob_states=arrays["lob_states"][order],
            cost_bps=cost_bps,
        )
        blocks[segment] = {
            "predictions": segment_pred,
            "targets": arrays["targets"][order].astype(np.float32),
            "net_returns": replay["net_returns"],
            "actions": replay["actions"],
            "lob_states": arrays["lob_states"][order].astype(np.float32),
        }
    return blocks


def ranking_gap(scores: dict[str, tuple[float, float]]) -> float:
    methods = sorted(scores)
    concordant = 0
    discordant = 0
    for i, left in enumerate(methods[:-1]):
        for right in methods[i + 1 :]:
            prod = (scores[left][0] - scores[right][0]) * (scores[left][1] - scores[right][1])
            if abs(prod) < 1e-18:
                continue
            if prod > 0:
                concordant += 1
            else:
                discordant += 1
    denom = max(concordant + discordant, 1)
    return float((concordant - discordant) / denom)


def compute_action_path_bootstrap(out: Path, target_root: Path | None, cost_bps: float, draws: int) -> None:
    rng = np.random.default_rng(20260504)
    metric_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    for panel, spec in PANEL_SPECS.items():
        arrays = load_panel(spec)
        method_blocks: dict[str, tuple[str, dict[str, dict[str, Any]]]] = {}
        for method, pred_path, source in method_prediction_paths(panel, spec, target_root):
            if source != "core_90d_method":
                continue
            predictions = load_predictions(pred_path)
            if len(predictions) != len(arrays["targets"]):
                continue
            blocks = segment_blocks(arrays, predictions, cost_bps)
            method_blocks[method] = (source, blocks)
            full_replay = replay_slice(arrays=arrays, predictions=predictions, mask=np.ones(len(predictions), dtype=bool), cost_bps=cost_bps)
            if full_replay is None:
                continue
            ci_map = bootstrap_metric_values(list(blocks.values()), draws, rng)
            for metric_name, (mean, low, high) in ci_map.items():
                point = getattr(full_replay["metrics"], "sharpe" if metric_name == "r4_sharpe" else "max_drawdown", None)
                if metric_name == "worst_regime_sharpe":
                    point = full_replay["robustness"].get("worst_regime_sharpe")
                metric_rows.append(
                    {
                        "panel": panel,
                        "method": method,
                        "method_source": source,
                        "metric": metric_name,
                        "point_fast": point,
                        "bootstrap_mean": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "block_unit": "asset_venue_day_action_path",
                        "num_blocks": len(blocks),
                        "draws": draws,
                        "cost_bps": cost_bps,
                    }
                )
        common_segments = sorted(set.intersection(*(set(blocks) for _, blocks in method_blocks.values()))) if method_blocks else []
        if not common_segments:
            continue
        gap_samples: dict[str, list[float]] = {"ranking_gap": [], "reversal_rate_at_1": [], "metric_gap": []}
        for _ in range(draws):
            sampled_segments = list(rng.choice(common_segments, size=len(common_segments), replace=True))
            scores: dict[str, tuple[float, float]] = {}
            for method, (_, blocks) in method_blocks.items():
                sample = [blocks[key] for key in sampled_segments]
                preds = np.concatenate([item["predictions"] for item in sample]).astype(np.float32)
                targets = np.concatenate([item["targets"] for item in sample]).astype(np.float32)
                net_returns = np.concatenate([item["net_returns"] for item in sample]).astype(np.float32)
                actions = np.concatenate([item["actions"] for item in sample]).astype(np.float32)
                metrics = summarize_trading_metrics(net_returns, turnover=float(np.abs(actions).sum()), liquidation=False, periods_per_year=PERIODS_PER_YEAR)
                scores[method] = (rank_ic(preds, targets), metrics.sharpe)
            offline = [name for name, _ in sorted(scores.items(), key=lambda item: item[1][0], reverse=True)]
            online = [name for name, _ in sorted(scores.items(), key=lambda item: item[1][1], reverse=True)]
            gap_samples["reversal_rate_at_1"].append(0.0 if offline[0] == online[0] else 1.0)
            gap_samples["ranking_gap"].append(ranking_gap(scores))
            x = np.asarray([scores[name][0] for name in sorted(scores)], dtype=np.float64)
            y = np.asarray([scores[name][1] for name in sorted(scores)], dtype=np.float64)
            gap_samples["metric_gap"].append(float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-12 and np.std(y) > 1e-12 else 0.0)
        for metric_name, values in gap_samples.items():
            mean, low, high = ci(values)
            gap_rows.append(
                {
                    "panel": panel,
                    "metric": metric_name,
                    "bootstrap_mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "block_unit": "asset_venue_day_action_path",
                    "num_blocks": len(common_segments),
                    "draws": draws,
                    "cost_bps": cost_bps,
                }
            )
    write_csv(out / "action_path_metric_bootstrap_ci.csv", metric_rows)
    write_csv(out / "action_path_gap_bootstrap_ci.csv", gap_rows)


def target_baseline_summary(out: Path, target_root: Path | None) -> None:
    rows: list[dict[str, Any]] = []
    if target_root is None or not target_root.exists():
        write_csv(out / "target_baseline_summary.csv", rows)
        return
    for result_path in sorted(target_root.glob("*/results_table.csv")):
        panel = result_path.parent.name
        with result_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                method = row.get("method", "")
                row["panel"] = panel
                row["baseline_status"] = (
                    "benchmark-native lite implementation; not official external repository reproduction"
                    if method.endswith("_lite")
                    else "completed reviewer target baseline"
                )
                rows.append(row)
    write_csv(out / "target_baseline_summary.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--target-root")
    parser.add_argument("--cost-bps", type=float, default=3.0)
    parser.add_argument("--draws", type=int, default=300)
    args = parser.parse_args()
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    target_root = Path(args.target_root) if args.target_root else None
    compute_slice_breakdown(out, target_root, args.cost_bps)
    compute_action_path_bootstrap(out, target_root, args.cost_bps, args.draws)
    target_baseline_summary(out, target_root)
    (out / "run_metadata.json").write_text(
        json.dumps(
            {
                "cost_bps": args.cost_bps,
                "draws": args.draws,
                "target_root": str(target_root) if target_root else None,
                "replay_note": "Supplemental O(n) PEG action-interface replay with fee/impact and drawdown stops.",
            },
            indent=2,
        )
    )
    print(out)


if __name__ == "__main__":
    main()
