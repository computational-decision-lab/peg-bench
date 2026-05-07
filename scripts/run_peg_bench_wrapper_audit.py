#!/usr/bin/env python3
"""Run a frozen forecast-to-action wrapper audit on existing PEG-Bench forecasts."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv
import json
from typing import Any

import numpy as np

from src.eval.benchmark_baselines import _load_rows, _predict_metrics, _window_core_arrays, ensure_window_index
from src.eval.peg_bench import _simulate_stage, run_realism_ladder
from src.models.action_interface import cost_threshold_decimal, position_targets_to_deltas, smooth_positions
from src.simulation.execution import SimulatorConfig


DEFAULT_VARIANTS = (
    "raw_sign",
    "threshold_deadzone",
    "cost_threshold",
    "smoothing",
    "turnover_penalty",
    "cost_threshold_plus_smoothing_plus_turnover",
)


def _panel_paths(standardized_root: Path, mode: str, split: str, seq_len: int) -> tuple[str, str]:
    panel = standardized_root / "panels" / f"{mode}_{split}.npz"
    index = standardized_root / "manifests" / f"{mode}_{split}_window_index_seq{seq_len}.csv"
    return str(panel.resolve()), str(index.resolve())


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _unique_sorted(values: list[float]) -> list[float]:
    rounded = {round(float(value), 10) for value in values if float(value) >= 0.0}
    return sorted(rounded)


def _deadzone_grid(predictions: np.ndarray, *, dense: bool) -> list[float]:
    absolute = np.abs(np.asarray(predictions, dtype=np.float32).reshape(-1))
    if len(absolute) == 0:
        return [0.0]
    quantiles = [0.0, 0.5, 0.75, 0.9] if dense else [0.0, 0.5, 0.9]
    values = [float(np.quantile(absolute, quantile)) for quantile in quantiles]
    scale = max(float(np.quantile(absolute, 0.9)), 1e-8)
    values.append(0.1 * scale)
    return _unique_sorted(values)


def _rho_grid(dense: bool) -> list[float]:
    return [0.0, 0.5, 0.75, 0.9] if dense else [0.5, 0.75, 0.9]


def _turnover_lambda_grid(dense: bool) -> list[float]:
    return [0.0, 0.1, 0.25] if dense else [0.1, 0.25]


def _variant_candidates(variant: str, val_predictions: np.ndarray) -> list[dict[str, float]]:
    dense_deadzones = _deadzone_grid(val_predictions, dense=True)
    compact_deadzones = _deadzone_grid(val_predictions, dense=False)
    dense_rhos = _rho_grid(dense=True)
    compact_rhos = _rho_grid(dense=False)
    dense_lambdas = _turnover_lambda_grid(dense=True)
    compact_lambdas = _turnover_lambda_grid(dense=False)

    if variant == "raw_sign":
        return [{"deadzone": 0.0, "delta": 0.0, "rho": 0.0, "turnover_lambda": 0.0}]
    if variant == "threshold_deadzone":
        return [{"deadzone": value, "delta": 0.0, "rho": 0.0, "turnover_lambda": 0.0} for value in dense_deadzones]
    if variant == "cost_threshold":
        return [{"deadzone": 0.0, "delta": value, "rho": 0.0, "turnover_lambda": 0.0} for value in compact_deadzones]
    if variant == "smoothing":
        return [{"deadzone": 0.0, "delta": 0.0, "rho": value, "turnover_lambda": 0.0} for value in dense_rhos]
    if variant == "turnover_penalty":
        return [
            {"deadzone": deadzone, "delta": 0.0, "rho": 0.0, "turnover_lambda": penalty}
            for deadzone in dense_deadzones
            for penalty in dense_lambdas
        ]
    if variant == "cost_threshold_plus_smoothing_plus_turnover":
        return [
            {"deadzone": 0.0, "delta": delta, "rho": rho, "turnover_lambda": penalty}
            for delta in compact_deadzones
            for rho in compact_rhos
            for penalty in compact_lambdas
        ]
    raise ValueError(f"Unsupported wrapper variant: {variant}")


def _build_positions(variant: str, predictions: np.ndarray, *, cost_bps: float, params: dict[str, float]) -> np.ndarray:
    forecasts = np.asarray(predictions, dtype=np.float32).reshape(-1)
    signed = np.sign(forecasts).astype(np.float32)

    if variant == "raw_sign":
        positions = signed
    elif variant in {"threshold_deadzone", "turnover_penalty"}:
        positions = signed.copy()
        threshold = float(params.get("deadzone", 0.0))
        positions[np.abs(forecasts) <= threshold] = 0.0
    elif variant == "cost_threshold":
        positions = signed.copy()
        threshold = cost_threshold_decimal(cost_bps, delta=float(params.get("delta", 0.0)))
        positions[np.abs(forecasts) <= threshold] = 0.0
    elif variant == "smoothing":
        positions = smooth_positions(signed, rho=float(params.get("rho", 0.0)))
    elif variant == "cost_threshold_plus_smoothing_plus_turnover":
        threshold = cost_threshold_decimal(cost_bps, delta=float(params.get("delta", 0.0)))
        raw_positions = signed.copy()
        raw_positions[np.abs(forecasts) <= threshold] = 0.0
        positions = smooth_positions(raw_positions, rho=float(params.get("rho", 0.0)))
    else:
        raise ValueError(f"Unsupported wrapper variant: {variant}")

    return np.clip(np.asarray(positions, dtype=np.float32), -1.0, 1.0)


def _action_payload(
    variant: str,
    predictions: np.ndarray,
    *,
    cost_bps: float,
    params: dict[str, float],
) -> tuple[np.ndarray, dict[str, object]]:
    positions = _build_positions(variant, predictions, cost_bps=cost_bps, params=params)
    deltas = position_targets_to_deltas(positions)
    payload = {
        "variant": variant,
        "cost_bps": float(cost_bps),
        "params": {key: float(value) for key, value in params.items()},
        "summary": {
            "mean_abs_position": float(np.mean(np.abs(positions))) if len(positions) else 0.0,
            "nonzero_rate": float(np.mean(np.abs(positions) > 0.0)) if len(positions) else 0.0,
            "turnover": float(np.sum(np.abs(deltas))) if len(deltas) else 0.0,
        },
    }
    return deltas, payload


def _evaluate_predictions(
    *,
    predictions: np.ndarray,
    targets: np.ndarray,
    lob_states: np.ndarray,
    timestamps: np.ndarray,
    variant: str,
    params: dict[str, float],
    cost_bps_scan: tuple[float, ...],
    anchor_cost_bps: float,
) -> dict[str, object]:
    actions_by_cost: dict[str, np.ndarray] = {}
    action_metadata: dict[str, dict[str, object]] = {}
    for cost_bps in cost_bps_scan:
        key = str(float(cost_bps))
        actions, payload = _action_payload(
            variant,
            predictions,
            cost_bps=float(cost_bps),
            params=params,
        )
        actions_by_cost[key] = actions
        action_metadata[key] = payload
    return run_realism_ladder(
        predictions=np.asarray(predictions, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.float32),
        state_tokens=np.zeros(len(predictions), dtype=np.int64),
        lob_states=np.asarray(lob_states, dtype=np.float32),
        market_returns=np.asarray(targets, dtype=np.float32),
        timestamps=np.asarray(timestamps, dtype=np.int64),
        actions_by_cost=actions_by_cost,
        action_metadata=action_metadata,
        cost_bps_scan=cost_bps_scan,
        anchor_cost_bps=anchor_cost_bps,
    )


def _evaluate_selection_candidate(
    *,
    predictions: np.ndarray,
    targets: np.ndarray,
    lob_states: np.ndarray,
    timestamps: np.ndarray,
    variant: str,
    params: dict[str, float],
    selection_cost_bps: float,
) -> dict[str, object]:
    actions, payload = _action_payload(
        variant,
        predictions,
        cost_bps=float(selection_cost_bps),
        params=params,
    )
    metrics, robustness, _ = _simulate_stage(
        stage="R4",
        cost_bps=float(selection_cost_bps),
        state_tokens=np.zeros(len(predictions), dtype=np.int64),
        lob_states=np.asarray(lob_states, dtype=np.float32),
        market_returns=np.asarray(targets, dtype=np.float32),
        timestamps=np.asarray(timestamps, dtype=np.int64),
        actions=actions,
        base_config=SimulatorConfig(),
    )
    return {
        "prediction": _predict_metrics(np.asarray(predictions, dtype=np.float32), np.asarray(targets, dtype=np.float32)),
        "report_card": {
            "execution": {
                "anchor_level": "R4",
                "anchor_cost_bps": float(selection_cost_bps),
                "post_cost_sharpe": float(metrics["sharpe"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "turnover": float(metrics["turnover"]),
                "calmar": float(metrics["calmar"]),
                "net_return": float(metrics["net_pnl"]),
            },
            "robustness": robustness,
            "auditability": {"action_interface": payload},
        },
    }


def _select_candidate(
    *,
    variant: str,
    val_predictions: np.ndarray,
    val_payload: dict[str, np.ndarray],
    selection_cost_bps: float,
    selection_metric: str,
) -> tuple[dict[str, float], dict[str, object], float]:
    best_params: dict[str, float] | None = None
    best_summary: dict[str, object] | None = None
    best_score = -1e18
    for params in _variant_candidates(variant, val_predictions):
        summary = _evaluate_selection_candidate(
            predictions=val_predictions,
            targets=val_payload["targets"],
            lob_states=val_payload["lob_states"],
            timestamps=val_payload["timestamps"],
            variant=variant,
            params=params,
            selection_cost_bps=selection_cost_bps,
        )
        execution = summary["report_card"]["execution"]
        sharpe = float(execution.get("post_cost_sharpe", 0.0) or 0.0)
        turnover = float(execution.get("turnover", 0.0) or 0.0)
        if not np.isfinite(sharpe):
            sharpe = -1e9
        if not np.isfinite(turnover):
            turnover = 0.0
        turnover_rate = turnover / max(len(val_predictions), 1)
        if selection_metric == "sharpe":
            score = sharpe
        elif selection_metric == "sharpe_minus_turnover":
            score = sharpe - float(params.get("turnover_lambda", 0.0)) * turnover_rate
        else:
            raise ValueError(f"Unsupported selection metric: {selection_metric}")
        if score > best_score:
            best_score = float(score)
            best_params = params
            best_summary = summary
    if best_params is None or best_summary is None:
        raise RuntimeError(f"No valid candidate found for variant {variant}.")
    return best_params, best_summary, best_score


def _load_split_payload(
    *,
    standardized_root: Path,
    mode: str,
    split: str,
    seq_len: int,
    stride: int,
    target_key: str,
) -> dict[str, np.ndarray]:
    panel_path, index_path = _panel_paths(standardized_root, mode, split, seq_len)
    ensured_index = ensure_window_index(
        panel_path=panel_path,
        output_path=index_path,
        seq_len=seq_len,
        stride=stride,
        target_key=target_key,
    )
    rows = _load_rows(ensured_index)
    arrays = _window_core_arrays(panel_path, rows, target_key)
    return {
        "targets": np.nan_to_num(arrays["targets"], nan=0.0, posinf=0.0, neginf=0.0),
        "lob_states": np.nan_to_num(arrays["lob_states"], nan=0.0, posinf=0.0, neginf=0.0),
        "timestamps": arrays["timestamps"],
    }


def _load_method_predictions(compare_root: Path, method: str) -> dict[str, np.ndarray]:
    predictions_path = compare_root / "methods" / method / "predictions.npz"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions for method {method}: {predictions_path}")
    payload = np.load(predictions_path)
    return {
        "val_predictions": np.nan_to_num(payload["val_predictions"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0),
        "val_targets": np.nan_to_num(payload["val_targets"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0),
        "test_predictions": np.nan_to_num(payload["test_predictions"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0),
        "test_targets": np.nan_to_num(payload["test_targets"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0),
        "predictions_path": str(predictions_path.resolve()),
    }


def _anchor_metrics(summary: dict[str, object]) -> dict[str, float]:
    execution = summary["report_card"]["execution"]
    return {
        "sharpe_3bps": float(execution.get("post_cost_sharpe", 0.0) or 0.0),
        "net_return_3bps": float(execution.get("net_return", 0.0) or 0.0),
        "max_drawdown_3bps": float(execution.get("max_drawdown", 0.0) or 0.0),
        "turnover_3bps": float(execution.get("turnover", 0.0) or 0.0),
        "calmar_3bps": float(execution.get("calmar", 0.0) or 0.0),
    }


def _save_variant_artifacts(
    *,
    output_root: Path,
    method: str,
    variant: str,
    params: dict[str, float],
    selection_summary: dict[str, object],
    selection_score: float,
    test_summary: dict[str, object],
    test_predictions: np.ndarray,
    test_targets: np.ndarray,
    anchor_cost_bps: float,
) -> dict[str, str]:
    variant_root = output_root / method / variant
    variant_root.mkdir(parents=True, exist_ok=True)
    detail_path = variant_root / "detail.json"
    action_positions = _build_positions(variant, test_predictions, cost_bps=anchor_cost_bps, params=params)
    action_deltas = position_targets_to_deltas(action_positions)
    np.savez_compressed(
        variant_root / "actions_anchor_3bps.npz",
        predictions=test_predictions.astype(np.float32),
        targets=test_targets.astype(np.float32),
        positions=action_positions.astype(np.float32),
        deltas=action_deltas.astype(np.float32),
    )
    detail_path.write_text(
        json.dumps(
            {
                "method": method,
                "variant": variant,
                "selected_params": params,
                "selection_score": selection_score,
                "selection_summary": selection_summary,
                "test_summary": test_summary,
            },
            indent=2,
        )
    )
    return {
        "detail_json": str(detail_path.resolve()),
        "action_npz": str((variant_root / "actions_anchor_3bps.npz").resolve()),
    }


def _markdown_summary(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Wrapper Audit Report",
        "",
        "Frozen validation-selected wrapper variants on existing prediction-first forecasts.",
        "",
        "## Anchor Comparison @3bps",
        "",
        "| method | variant | val_score | test_sharpe | test_turnover | test_calmar |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {variant} | {val_score:.4f} | {test_sharpe_3bps:.4f} | {test_turnover_3bps:.4f} | {test_calmar_3bps:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `raw_sign`: sign-only action path with no deadzone and no smoothing.",
            "- `threshold_deadzone`: fixed deadzone selected on validation.",
            "- `cost_threshold`: cost-aware deadzone with validation-selected additive delta.",
            "- `smoothing`: smoothed sign actions with validation-selected rho.",
            "- `turnover_penalty`: fixed deadzone selected with validation Sharpe minus turnover-rate penalty.",
            "- `cost_threshold_plus_smoothing_plus_turnover`: full wrapper using cost-aware deadzone, smoothing, and turnover-aware validation selection.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit forecast-to-action wrapper variants on frozen PEG-Bench forecasts.")
    parser.add_argument("--standardized-root", type=str, required=True)
    parser.add_argument("--compare-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True, choices=["time", "volume"])
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--target-key", type=str, default="return_20")
    parser.add_argument("--methods", type=str, required=True)
    parser.add_argument("--variants", type=str, default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--selection-cost-bps", type=float, default=3.0)
    parser.add_argument("--cost-bps", type=float, nargs="+", default=[0.0, 1.0, 3.0, 5.0, 10.0])
    parser.add_argument("--selection-metric", type=str, default="sharpe", choices=["sharpe", "sharpe_minus_turnover"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    del args.seed
    del args.device

    standardized_root = Path(args.standardized_root)
    compare_root = Path(args.compare_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    cost_scan = tuple(float(value) for value in args.cost_bps)

    val_payload = _load_split_payload(
        standardized_root=standardized_root,
        mode=args.mode,
        split="val",
        seq_len=args.seq_len,
        stride=args.stride,
        target_key=args.target_key,
    )
    test_payload = _load_split_payload(
        standardized_root=standardized_root,
        mode=args.mode,
        split="test",
        seq_len=args.seq_len,
        stride=args.stride,
        target_key=args.target_key,
    )

    rows: list[dict[str, object]] = []
    val_params: dict[str, dict[str, object]] = {}
    for method in methods:
        method_predictions = _load_method_predictions(compare_root, method)
        if len(method_predictions["val_predictions"]) != len(val_payload["targets"]):
            raise ValueError(
                f"Validation length mismatch for {method}: "
                f"{len(method_predictions['val_predictions'])} predictions vs {len(val_payload['targets'])} targets"
            )
        if len(method_predictions["test_predictions"]) != len(test_payload["targets"]):
            raise ValueError(
                f"Test length mismatch for {method}: "
                f"{len(method_predictions['test_predictions'])} predictions vs {len(test_payload['targets'])} targets"
            )

        method_val_params: dict[str, object] = {}
        for variant in variants:
            selection_metric = "sharpe_minus_turnover" if "turnover" in variant else args.selection_metric
            params, val_summary, val_score = _select_candidate(
                variant=variant,
                val_predictions=method_predictions["val_predictions"],
                val_payload=val_payload,
                selection_cost_bps=float(args.selection_cost_bps),
                selection_metric=selection_metric,
            )
            test_summary = _evaluate_predictions(
                predictions=method_predictions["test_predictions"],
                targets=test_payload["targets"],
                lob_states=test_payload["lob_states"],
                timestamps=test_payload["timestamps"],
                variant=variant,
                params=params,
                cost_bps_scan=cost_scan,
                anchor_cost_bps=float(args.selection_cost_bps),
            )
            artifact_paths = _save_variant_artifacts(
                output_root=output_root,
                method=method,
                variant=variant,
                params=params,
                selection_summary=val_summary,
                selection_score=val_score,
                test_summary=test_summary,
                test_predictions=method_predictions["test_predictions"],
                test_targets=test_payload["targets"],
                anchor_cost_bps=float(args.selection_cost_bps),
            )
            val_anchor = _anchor_metrics(val_summary)
            test_anchor = _anchor_metrics(test_summary)
            row = {
                "mode": args.mode,
                "target_key": args.target_key,
                "method": method,
                "variant": variant,
                "selection_metric": selection_metric,
                "val_score": float(val_score),
                "selected_deadzone": float(params.get("deadzone", 0.0)),
                "selected_delta": float(params.get("delta", 0.0)),
                "selected_rho": float(params.get("rho", 0.0)),
                "selected_turnover_lambda": float(params.get("turnover_lambda", 0.0)),
                "val_sharpe_3bps": val_anchor["sharpe_3bps"],
                "val_turnover_3bps": val_anchor["turnover_3bps"],
                "test_sharpe_3bps": test_anchor["sharpe_3bps"],
                "test_net_return_3bps": test_anchor["net_return_3bps"],
                "test_max_drawdown_3bps": test_anchor["max_drawdown_3bps"],
                "test_turnover_3bps": test_anchor["turnover_3bps"],
                "test_calmar_3bps": test_anchor["calmar_3bps"],
                "detail_json": artifact_paths["detail_json"],
                "action_npz": artifact_paths["action_npz"],
            }
            rows.append(row)
            method_val_params[variant] = {
                "selection_metric": selection_metric,
                "selected_params": params,
                "selection_score": val_score,
                "val_anchor_metrics": val_anchor,
                "detail_json": artifact_paths["detail_json"],
            }
            val_params[method] = method_val_params
            print(
                f"[wrapper_audit] method={method} variant={variant} "
                f"val_score={val_score:.4f} test_sharpe_3bps={test_anchor['sharpe_3bps']:.4f}",
                flush=True,
            )
            _write_csv(output_root / "wrapper_audit_results.partial.csv", rows)
            (output_root / "wrapper_params_val.partial.json").write_text(json.dumps(val_params, indent=2))

    rows.sort(key=lambda item: (str(item["method"]), str(item["variant"])))
    _write_csv(output_root / "wrapper_audit_results.csv", rows)
    (output_root / "wrapper_params_val.json").write_text(json.dumps(val_params, indent=2))
    (output_root / "raw_vs_wrapped_report.md").write_text(_markdown_summary(rows))
    (output_root / "run_summary.json").write_text(
        json.dumps(
            {
                "mode": args.mode,
                "target_key": args.target_key,
                "methods": methods,
                "variants": variants,
                "selection_cost_bps": float(args.selection_cost_bps),
                "cost_bps_scan": list(cost_scan),
                "standardized_root": str(standardized_root.resolve()),
                "compare_root": str(compare_root.resolve()),
                "output_root": str(output_root.resolve()),
                "rows": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
