#!/usr/bin/env python3
"""Reviewer-requested PEG-Bench evidence completion.

This script is intentionally read-mostly: it consumes the frozen 90D panels,
window indices, method predictions, daily reports, and wrapper audit artifacts
that already exist on the remote machine. It writes a separate completion
bundle without modifying the original experiment outputs.
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

from src.eval.benchmark_baselines import _load_rows, _predict_metrics, _window_core_arrays
from src.eval.gap_metrics import GapInput, compute_gap_report
from src.eval.metrics import max_drawdown, rank_ic, sharpe_ratio
from src.eval.peg_bench import _compute_robustness_panel, _simulate_stage, run_realism_ladder
from src.models.action_interface import build_action_interface_result
from src.simulation.execution import SimulatorConfig


PANELS: dict[str, dict[str, str]] = {
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


COSTS = (0.0, 1.0, 3.0, 5.0, 10.0)


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


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def ci(values: list[float] | np.ndarray) -> tuple[float | None, float | None, float | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return None, None, None
    return float(np.mean(arr)), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def panel_index_path(spec: dict[str, str], split: str = "test", seq_len: int = 64) -> Path:
    return Path(spec["standardized_root"]) / "manifests" / f"{spec['mode']}_{split}_window_index_seq{seq_len}.csv"


def panel_npz_path(spec: dict[str, str], split: str = "test") -> Path:
    return Path(spec["standardized_root"]) / "panels" / f"{spec['mode']}_{split}.npz"


def load_panel_arrays(spec: dict[str, str]) -> dict[str, Any]:
    rows = _load_rows(panel_index_path(spec))
    arrays = _window_core_arrays(panel_npz_path(spec), rows, spec["target_key"])
    arrays["rows"] = rows
    arrays["dates"] = np.asarray([row["date"] for row in rows])
    arrays["exchanges"] = np.asarray([row["exchange"] for row in rows])
    arrays["symbols"] = np.asarray([row["symbol"] for row in rows])
    arrays["asset_venues"] = np.asarray([f'{row["exchange"]}:{row["symbol"]}' for row in rows])
    return arrays


def load_predictions(compare_root: Path, method: str) -> dict[str, np.ndarray]:
    payload = np.load(compare_root / "methods" / method / "predictions.npz")
    return {
        "pred": np.nan_to_num(payload["test_predictions"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0),
        "target": np.nan_to_num(payload["test_targets"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0),
    }


def methods_for_panel(spec: dict[str, str]) -> list[str]:
    root = Path(spec["compare_root"]) / "methods"
    return sorted(path.name for path in root.iterdir() if (path / "predictions.npz").exists())


def build_actions(predictions: np.ndarray, cost: float, *, rho: float = 0.9) -> np.ndarray:
    return build_action_interface_result(predictions, cost_bps=cost, rho=rho).wrapped_deltas


def evaluate_method(pred: np.ndarray, arrays: dict[str, Any]) -> dict[str, Any]:
    actions_by_cost = {str(float(cost)): build_actions(pred, cost) for cost in COSTS}
    return run_realism_ladder(
        predictions=pred,
        targets=arrays["targets"],
        state_tokens=np.zeros(len(pred), dtype=np.int64),
        lob_states=arrays["lob_states"],
        market_returns=arrays["targets"],
        timestamps=arrays["timestamps"],
        actions_by_cost=actions_by_cost,
        cost_bps_scan=COSTS,
        anchor_cost_bps=3.0,
        action_metadata={},
    )


def simulate_net_returns(pred: np.ndarray, arrays: dict[str, Any], *, cost: float = 3.0):
    actions = build_actions(pred, cost)
    _, robustness, trajectory = _simulate_stage(
        stage="R4",
        cost_bps=cost,
        state_tokens=np.zeros(len(pred), dtype=np.int64),
        lob_states=arrays["lob_states"],
        market_returns=arrays["targets"],
        timestamps=arrays["timestamps"],
        actions=actions,
        base_config=SimulatorConfig(),
    )
    return np.asarray(trajectory.net_returns, dtype=np.float64), actions, robustness


def gap_from_scores(scores: dict[str, tuple[float, float]], robustness: dict[str, list[float]] | None = None) -> dict[str, Any]:
    panel = {
        method: GapInput(
            offline_score=float(vals[0]),
            full_realism_utility=float(vals[1]),
            regime_utilities=(robustness or {}).get(method),
        )
        for method, vals in scores.items()
        if np.isfinite(vals[0]) and np.isfinite(vals[1])
    }
    return compute_gap_report(panel, k=1)


def complete_cost_ladder(output_root: Path, evaluated: dict[str, dict[str, dict[str, Any]]]) -> None:
    rows: list[dict[str, Any]] = []
    for panel, methods in evaluated.items():
        for method, summary in methods.items():
            execution = summary["report_card"]["execution"]
            robustness_by_cost = summary["realism_ladder"]["R4"]["robustness"]
            for cost, metrics in execution["r4_cost_ladder"].items():
                robust = robustness_by_cost.get(cost) or {}
                rows.append(
                    {
                        "panel": panel,
                        "method": method,
                        "cost_bps": float(cost),
                        "r4_sharpe": metrics.get("sharpe"),
                        "r4_net_pnl": metrics.get("net_pnl"),
                        "r4_max_drawdown": metrics.get("max_drawdown"),
                        "r4_turnover": metrics.get("turnover"),
                        "r4_calmar": metrics.get("calmar"),
                        "worst_regime_sharpe": robust.get("worst_regime_sharpe"),
                        "stress_day_survival": robust.get("stress_day_survival"),
                        "drawdown_violation_rate": robust.get("drawdown_violation_rate"),
                    }
                )
    write_csv(output_root / "cost_ladder_full_core_methods.csv", rows)

    leader_rows: list[dict[str, Any]] = []
    for panel in sorted({row["panel"] for row in rows}):
        for cost in COSTS:
            subset = [row for row in rows if row["panel"] == panel and float(row["cost_bps"]) == float(cost)]
            if not subset:
                continue
            winner = max(subset, key=lambda row: float(row["r4_sharpe"]))
            leader_rows.append(
                {
                    "panel": panel,
                    "cost_bps": cost,
                    "r4_winner": winner["method"],
                    "r4_winner_sharpe": winner["r4_sharpe"],
                    "r4_winner_max_drawdown": winner["r4_max_drawdown"],
                    "r4_winner_turnover": winner["r4_turnover"],
                }
            )
    write_csv(output_root / "cost_ladder_winners_by_panel.csv", leader_rows)


def complete_per_asset_venue(
    output_root: Path,
    panels: dict[str, dict[str, str]],
    arrays_by_panel: dict[str, dict[str, Any]],
) -> None:
    rows: list[dict[str, Any]] = []
    grouping_specs = {
        "asset": ("symbols",),
        "venue": ("exchanges",),
        "asset_venue": ("asset_venues",),
    }
    for panel, spec in panels.items():
        arrays = arrays_by_panel[panel]
        for method in methods_for_panel(spec):
            pred = load_predictions(Path(spec["compare_root"]), method)["pred"]
            for group_type, keys in grouping_specs.items():
                labels = arrays[keys[0]]
                for label in sorted(set(labels.tolist())):
                    mask = labels == label
                    if int(mask.sum()) < 16:
                        continue
                    sub_arrays = {
                        "targets": arrays["targets"][mask],
                        "lob_states": arrays["lob_states"][mask],
                        "timestamps": arrays["timestamps"][mask],
                    }
                    order = np.argsort(sub_arrays["timestamps"])
                    sub_pred = pred[mask][order]
                    for key in sub_arrays:
                        sub_arrays[key] = sub_arrays[key][order]
                    summary = evaluate_method(sub_pred, sub_arrays)
                    execution = summary["report_card"]["execution"]
                    robust = summary["report_card"]["robustness"] or {}
                    pred_metrics = summary["report_card"]["prediction"]
                    rows.append(
                        {
                            "panel": panel,
                            "method": method,
                            "group_type": group_type,
                            "group": str(label),
                            "num_windows": int(mask.sum()),
                            "rank_ic": pred_metrics.get("rank_ic"),
                            "directional_accuracy": pred_metrics.get("directional_accuracy"),
                            "r4_sharpe_3bps": execution.get("post_cost_sharpe"),
                            "r4_net_return_3bps": execution.get("net_return"),
                            "r4_max_drawdown_3bps": execution.get("max_drawdown"),
                            "r4_turnover_3bps": execution.get("turnover"),
                            "worst_regime_sharpe": robust.get("worst_regime_sharpe"),
                            "stress_day_survival": robust.get("stress_day_survival"),
                        }
                    )
    write_csv(output_root / "per_asset_venue_breakdown.csv", rows)


def complete_day_block_ci(output_root: Path, panels: dict[str, dict[str, str]]) -> None:
    rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260504)
    for panel, spec in panels.items():
        daily: dict[str, dict[str, dict[str, float]]] = {}
        methods = methods_for_panel(spec)
        for method in methods:
            report_dir = Path(spec["compare_root"]) / "methods" / method / "reports"
            for report_path in sorted(report_dir.glob("*.json")):
                payload = json.loads(report_path.read_text())
                date = str(payload["date"])
                card = payload["report_card"]
                daily.setdefault(date, {})[method] = {
                    "rank_ic": float(card["prediction"].get("rank_ic") or 0.0),
                    "r4_sharpe": float(card["execution"].get("post_cost_sharpe") or 0.0),
                    "max_drawdown": float(card["execution"].get("max_drawdown") or 0.0),
                    "worst_regime_sharpe": float(card["robustness"].get("worst_regime_sharpe") or 0.0),
                }
        dates = sorted(date for date, vals in daily.items() if all(method in vals for method in methods))
        for method in methods:
            for metric in ("r4_sharpe", "max_drawdown", "worst_regime_sharpe"):
                point = float(np.mean([daily[d][method][metric] for d in dates]))
                draws = []
                for _ in range(500):
                    sample = rng.choice(dates, size=len(dates), replace=True)
                    draws.append(float(np.mean([daily[d][method][metric] for d in sample])))
                mean, lo, hi = ci(draws)
                rows.append(
                    {
                        "panel": panel,
                        "method": method,
                        "metric": metric,
                        "point": point,
                        "bootstrap_mean": mean,
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "num_days": len(dates),
                        "bootstrap": "day_block",
                        "draws": 500,
                    }
                )
        point_gap = gap_from_scores(
            {
                method: (
                    float(np.mean([daily[d][method]["rank_ic"] for d in dates])),
                    float(np.mean([daily[d][method]["r4_sharpe"] for d in dates])),
                )
                for method in methods
            }
        )
        gap_draws: dict[str, list[float]] = {"ranking_gap": [], "reversal_rate_at_k": [], "metric_gap": []}
        for _ in range(500):
            sample = rng.choice(dates, size=len(dates), replace=True)
            report = gap_from_scores(
                {
                    method: (
                        float(np.mean([daily[d][method]["rank_ic"] for d in sample])),
                        float(np.mean([daily[d][method]["r4_sharpe"] for d in sample])),
                    )
                    for method in methods
                }
            )
            for metric in gap_draws:
                value = report.get(metric)
                if value is not None:
                    gap_draws[metric].append(float(value))
        for metric, draws in gap_draws.items():
            mean, lo, hi = ci(draws)
            gap_rows.append(
                {
                    "panel": panel,
                    "metric": metric,
                    "point": point_gap.get(metric),
                    "bootstrap_mean": mean,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "num_days": len(dates),
                    "bootstrap": "day_block",
                    "draws": 500,
                }
            )
    write_csv(output_root / "day_block_metric_ci.csv", rows)
    write_csv(output_root / "day_block_gap_ci.csv", gap_rows)


def complete_action_path_bootstrap(
    output_root: Path,
    arrays_by_panel: dict[str, dict[str, Any]],
    net_returns_by_panel: dict[str, dict[str, np.ndarray]],
    predictions_by_panel: dict[str, dict[str, np.ndarray]],
    *,
    draws: int,
    block_len: int,
) -> None:
    rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260505)
    for panel, returns_by_method in net_returns_by_panel.items():
        methods = sorted(returns_by_method)
        n = min(len(arr) for arr in returns_by_method.values())
        starts = np.arange(0, max(n - block_len + 1, 1), dtype=np.int64)
        indices_draws: list[np.ndarray] = []
        blocks_per_draw = int(np.ceil(n / block_len))
        for _ in range(draws):
            chosen = rng.choice(starts, size=blocks_per_draw, replace=True)
            pieces = [np.arange(start, min(start + block_len, n), dtype=np.int64) for start in chosen]
            indices_draws.append(np.concatenate(pieces)[:n])

        for method in methods:
            values_sharpe = []
            values_mdd = []
            returns = returns_by_method[method][:n]
            for idx in indices_draws:
                sample_returns = returns[idx]
                values_sharpe.append(sharpe_ratio(sample_returns))
                values_mdd.append(max_drawdown(sample_returns))
            for metric, values in (("r4_sharpe", values_sharpe), ("max_drawdown", values_mdd)):
                mean, lo, hi = ci(values)
                point = sharpe_ratio(returns) if metric == "r4_sharpe" else max_drawdown(returns)
                rows.append(
                    {
                        "panel": panel,
                        "method": method,
                        "metric": metric,
                        "point": point,
                        "bootstrap_mean": mean,
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "bootstrap": "action_path_block",
                        "draws": draws,
                        "block_len": block_len,
                    }
                )

        targets = arrays_by_panel[panel]["targets"][:n]
        for metric_name in ("ranking_gap", "reversal_rate_at_k", "metric_gap"):
            values = []
            for idx in indices_draws:
                scores = {}
                for method in methods:
                    scores[method] = (
                        rank_ic(predictions_by_panel[panel][method][:n][idx], targets[idx]),
                        sharpe_ratio(net_returns_by_panel[panel][method][:n][idx]),
                    )
                report = gap_from_scores(scores)
                if report.get(metric_name) is not None:
                    values.append(float(report[metric_name]))
            point_scores = {
                method: (
                    rank_ic(predictions_by_panel[panel][method][:n], targets),
                    sharpe_ratio(net_returns_by_panel[panel][method][:n]),
                )
                for method in methods
            }
            point = gap_from_scores(point_scores).get(metric_name)
            mean, lo, hi = ci(values)
            gap_rows.append(
                {
                    "panel": panel,
                    "metric": metric_name,
                    "point": point,
                    "bootstrap_mean": mean,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "bootstrap": "action_path_block",
                    "draws": draws,
                    "block_len": block_len,
                }
            )
    write_csv(output_root / "action_path_metric_ci.csv", rows)
    write_csv(output_root / "action_path_gap_ci.csv", gap_rows)


def complete_wrapper_matrix(output_root: Path) -> None:
    wrapper_roots = sorted(Path("/root/autodl-tmp/peg_bench_runs/final_90d_missing_panel_runs").glob("wrapper_audit_*"))
    rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    for root in wrapper_roots:
        csv_path = root / "wrapper_audit_results.csv"
        if not csv_path.exists():
            continue
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                panel = str(row.get("panel") or root.name.replace("wrapper_audit_", "").replace("_v2fast", "").replace("_v1", ""))
                row["panel"] = panel
                row["source_file"] = str(csv_path)
                rows.append(row)
                detail_path = Path(row["detail_json"])
                if detail_path.exists():
                    detail = json.loads(detail_path.read_text())
                    ladder = detail.get("test_summary", {}).get("report_card", {}).get("execution", {}).get("r4_cost_ladder", {})
                    for cost, metrics in ladder.items():
                        cost_rows.append(
                            {
                                "panel": panel,
                                "method": row["method"],
                                "variant": row["variant"],
                                "cost_bps": float(cost),
                                "r4_sharpe": metrics.get("sharpe"),
                                "r4_net_pnl": metrics.get("net_pnl"),
                                "r4_max_drawdown": metrics.get("max_drawdown"),
                                "r4_turnover": metrics.get("turnover"),
                                "source_detail": str(detail_path),
                            }
                        )
    write_csv(output_root / "wrapper_matrix_all_rows.csv", rows)
    write_csv(output_root / "wrapper_matrix_cost_ladder.csv", cost_rows)

    best_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["panel"], row["method"]), []).append(row)
    for (panel, method), group in sorted(grouped.items()):
        best = max(group, key=lambda r: float(r.get("test_sharpe_3bps") or -1e18))
        raw = next((r for r in group if r.get("variant") == "raw_sign"), None)
        best_rows.append(
            {
                "panel": panel,
                "method": method,
                "raw_sharpe_3bps": raw.get("test_sharpe_3bps") if raw else None,
                "best_variant": best.get("variant"),
                "best_sharpe_3bps": best.get("test_sharpe_3bps"),
                "delta_best_minus_raw": (
                    float(best.get("test_sharpe_3bps"))
                    - float(raw.get("test_sharpe_3bps"))
                    if raw and raw.get("test_sharpe_3bps") not in (None, "")
                    else None
                ),
                "candidate_count": len(group),
            }
        )
    write_csv(output_root / "wrapper_raw_vs_best_by_panel_method.csv", best_rows)


def complete_execution_sensitivity(output_root: Path, arrays_by_panel: dict[str, dict[str, Any]], predictions_by_panel: dict[str, dict[str, np.ndarray]]) -> None:
    rows: list[dict[str, Any]] = []
    cases = [
        ("time_h20", "kairos_64code"),
        ("time_h20", "lightgbm_lob"),
        ("volume_h20", "lightgbm_lob"),
        ("volume_h20", "kairos_64code"),
    ]
    for panel, method in cases:
        if panel not in arrays_by_panel or method not in predictions_by_panel[panel]:
            continue
        arrays = arrays_by_panel[panel]
        pred = predictions_by_panel[panel][method]
        base_positions = build_action_interface_result(pred, cost_bps=3.0, rho=0.9).wrapped_positions
        for latency_bars in (0, 1, 5, 10):
            if latency_bars == 0:
                delayed = base_positions.copy()
            else:
                delayed = np.concatenate([np.zeros(latency_bars, dtype=np.float32), base_positions[:-latency_bars]])
            for fill_prob in (1.0, 0.75, 0.5):
                scaled = delayed * fill_prob
                actions = np.diff(np.concatenate([[0.0], scaled])).astype(np.float32)
                metrics, robust, _ = _simulate_stage(
                    stage="R4",
                    cost_bps=3.0,
                    state_tokens=np.zeros(len(pred), dtype=np.int64),
                    lob_states=arrays["lob_states"],
                    market_returns=arrays["targets"],
                    timestamps=arrays["timestamps"],
                    actions=actions,
                    base_config=SimulatorConfig(),
                )
                rows.append(
                    {
                        "panel": panel,
                        "method": method,
                        "latency_bars": latency_bars,
                        "queue_fill_probability": fill_prob,
                        "audit_engine": "internal_latency_queue_sensitivity",
                        "hftbacktest_status": "package_missing_not_claimed",
                        "r4_sharpe": metrics.get("sharpe"),
                        "r4_net_pnl": metrics.get("net_pnl"),
                        "r4_max_drawdown": metrics.get("max_drawdown"),
                        "r4_turnover": metrics.get("turnover"),
                        "worst_regime_sharpe": (robust or {}).get("worst_regime_sharpe"),
                    }
                )
    write_csv(output_root / "external_execution_sensitivity_internal.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--action-bootstrap-draws", type=int, default=120)
    parser.add_argument("--action-block-len", type=int, default=2048)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    arrays_by_panel: dict[str, dict[str, Any]] = {}
    evaluated: dict[str, dict[str, dict[str, Any]]] = {}
    predictions_by_panel: dict[str, dict[str, np.ndarray]] = {}
    net_returns_by_panel: dict[str, dict[str, np.ndarray]] = {}
    metadata: dict[str, Any] = {"panels": PANELS, "costs": COSTS}

    for panel, spec in PANELS.items():
        print(f"[reviewer_completion] loading {panel}", flush=True)
        arrays = load_panel_arrays(spec)
        arrays_by_panel[panel] = arrays
        evaluated[panel] = {}
        predictions_by_panel[panel] = {}
        net_returns_by_panel[panel] = {}
        for method in methods_for_panel(spec):
            print(f"[reviewer_completion] evaluate {panel} {method}", flush=True)
            pred = load_predictions(Path(spec["compare_root"]), method)["pred"]
            predictions_by_panel[panel][method] = pred
            evaluated[panel][method] = evaluate_method(pred, arrays)
            net_returns, _, _ = simulate_net_returns(pred, arrays, cost=3.0)
            net_returns_by_panel[panel][method] = net_returns

    complete_cost_ladder(output_root, evaluated)
    complete_day_block_ci(output_root, PANELS)
    complete_action_path_bootstrap(
        output_root,
        arrays_by_panel,
        net_returns_by_panel,
        predictions_by_panel,
        draws=args.action_bootstrap_draws,
        block_len=args.action_block_len,
    )
    complete_wrapper_matrix(output_root)
    complete_per_asset_venue(output_root, PANELS, arrays_by_panel)
    complete_execution_sensitivity(output_root, arrays_by_panel, predictions_by_panel)
    dump_json(output_root / "run_metadata.json", metadata)
    print(f"[reviewer_completion] wrote {output_root}", flush=True)


if __name__ == "__main__":
    main()
