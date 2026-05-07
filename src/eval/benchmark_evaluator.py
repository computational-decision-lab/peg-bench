"""Benchmark-level evaluator and report-card aggregation for PEG-Bench."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
from datetime import datetime, timedelta

import numpy as np

from src.eval.gap_metrics import GapInput, compute_gap_report
from src.eval.report import generate_markdown_report


@dataclass(slots=True)
class BenchmarkMethodSpec:
    """Set of daily report artifacts for a single method."""

    name: str
    reports: tuple[str, ...]


@dataclass(slots=True)
class BenchmarkEvaluatorConfig:
    """Config for aggregating daily method reports into a benchmark report card."""

    output_path: str
    markdown_path: str | None = None
    evaluation_split: str = "test"
    top_k: int = 1
    bootstrap_samples: int = 1_000
    random_seed: int = 7


DATE_RE = re.compile(r"(20\d{2}-\d{2}-\d{2})")


def _infer_date(path: str, payload: dict[str, object]) -> str | None:
    for key in ("date", "session_date", "evaluation_date"):
        value = payload.get(key)
        if isinstance(value, str) and DATE_RE.fullmatch(value):
            return value
    match = DATE_RE.search(path)
    return match.group(1) if match else None


def _load_json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text())


def _date_range(start: str, end: str) -> list[str]:
    start_dt = datetime.strptime(start, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end, "%Y-%m-%d").date()
    values: list[str] = []
    current = start_dt
    while current <= end_dt:
        values.append(current.isoformat())
        current += timedelta(days=1)
    return values


def _load_split_lookup_from_split_plan(path: str | Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for split in ("train", "val", "test"):
                start = str(row.get(f"{split}_start", "")).strip()
                end = str(row.get(f"{split}_end", "")).strip()
                if not start or not end:
                    continue
                for day in _date_range(start, end):
                    previous = lookup.get(day)
                    if previous is not None and previous != split:
                        raise ValueError(
                            f"Split-plan conflict for date {day}: saw both {previous!r} and {split!r}."
                        )
                    lookup[day] = split
    return lookup


def _extract_report_card(payload: dict[str, object]) -> dict[str, object]:
    report_card = payload.get("report_card")
    if isinstance(report_card, dict):
        return report_card
    return payload


def _bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float] | None:
    if len(values) == 0:
        return None
    if len(values) == 1:
        return (float(values[0]), float(values[0]))
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def _summarize_scalar(values: list[float], samples: int, seed: int) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return {"mean": None, "ci95": None, "num_days": 0}
    ci = _bootstrap_ci(array, samples=samples, seed=seed)
    return {
        "mean": float(np.mean(array)),
        "ci95": list(ci) if ci is not None else None,
        "num_days": int(len(array)),
    }


def _daily_metric(record: dict[str, object], section: str, key: str) -> float | None:
    payload = record.get(section, {})
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    return None if value is None else float(value)


def _aggregate_cost_ladder(records: list[dict[str, object]], samples: int, seed: int) -> dict[str, object]:
    ladder: dict[str, dict[str, list[float]]] = {}
    for record in records:
        execution = record.get("execution", {})
        if not isinstance(execution, dict):
            continue
        cost_ladder = execution.get("cost_ladder", {})
        if not isinstance(cost_ladder, dict):
            continue
        for cost, metrics in cost_ladder.items():
            if not isinstance(metrics, dict):
                continue
            bucket = ladder.setdefault(cost, {})
            for metric_name in ("sharpe", "max_drawdown", "turnover", "calmar", "net_pnl"):
                value = metrics.get(metric_name)
                if value is None:
                    continue
                bucket.setdefault(metric_name, []).append(float(value))
    summary: dict[str, object] = {}
    for cost, metrics in ladder.items():
        summary[cost] = {
            name: _summarize_scalar(values, samples=samples, seed=seed)
            for name, values in metrics.items()
        }
    return summary


def _aggregate_stage_cost_ladder(
    records: list[dict[str, object]],
    *,
    stage: str,
    samples: int,
    seed: int,
) -> dict[str, object]:
    ladder: dict[str, dict[str, list[float]]] = {}
    for record in records:
        realism_ladder = record.get("realism_ladder", {})
        if not isinstance(realism_ladder, dict):
            continue
        stage_payload = realism_ladder.get(stage, {})
        if not isinstance(stage_payload, dict):
            continue
        cost_scan = stage_payload.get("cost_scan", {})
        if not isinstance(cost_scan, dict):
            continue
        for cost, metrics in cost_scan.items():
            if not isinstance(metrics, dict):
                continue
            bucket = ladder.setdefault(cost, {})
            for metric_name in ("sharpe", "max_drawdown", "turnover", "calmar", "net_pnl"):
                value = metrics.get(metric_name)
                if value is None:
                    continue
                bucket.setdefault(metric_name, []).append(float(value))
    summary: dict[str, object] = {}
    for cost, metrics in ladder.items():
        summary[cost] = {
            name: _summarize_scalar(values, samples=samples, seed=seed)
            for name, values in metrics.items()
        }
    return summary


def _extract_daily_record(
    method: str,
    path: str,
    split_lookup: dict[str, str],
    evaluation_split: str,
) -> dict[str, object] | None:
    payload = _load_json(path)
    report_card = _extract_report_card(payload)
    date = _infer_date(path, payload)
    split = split_lookup.get(date, payload.get("split", "unknown")) if date is not None else payload.get("split", "unknown")
    if split != evaluation_split:
        return None
    prediction = report_card.get("prediction", {}) if isinstance(report_card, dict) else {}
    execution = report_card.get("execution", {}) if isinstance(report_card, dict) else {}
    robustness = report_card.get("robustness", {}) if isinstance(report_card, dict) else {}
    realism_ladder = payload.get("realism_ladder")
    if not isinstance(realism_ladder, dict) and isinstance(report_card, dict):
        realism_ladder = report_card.get("realism_ladder", {})
    return {
        "method": method,
        "date": date,
        "split": split,
        "path": str(Path(path).resolve()),
        "prediction": prediction,
        "execution": execution,
        "robustness": robustness,
        "realism_ladder": realism_ladder,
    }


def _coverage_summary(
    records: list[dict[str, object]],
    split_lookup: dict[str, str],
    evaluation_split: str,
) -> dict[str, object]:
    expected_dates = sorted(day for day, split in split_lookup.items() if split == evaluation_split)
    observed_dates = sorted({str(item["date"]) for item in records if item.get("date")})
    observed_set = set(observed_dates)
    expected_set = set(expected_dates)
    missing_dates = sorted(expected_set - observed_set)
    extra_dates = sorted(observed_set - expected_set)
    coverage_ratio = 1.0 if not expected_dates else len(observed_set & expected_set) / len(expected_dates)
    return {
        "expected_num_days": int(len(expected_dates)),
        "observed_num_days": int(len(observed_dates)),
        "coverage_ratio": float(coverage_ratio),
        "expected_dates": expected_dates,
        "observed_dates": observed_dates,
        "missing_dates": missing_dates,
        "extra_dates": extra_dates,
    }


def evaluate_benchmark_methods(
    method_specs: list[BenchmarkMethodSpec],
    split_lookup: dict[str, str],
    config: BenchmarkEvaluatorConfig,
) -> dict[str, object]:
    """Aggregate daily report-card JSON files into a benchmark-level comparison."""
    all_daily: list[dict[str, object]] = []
    methods: dict[str, list[dict[str, object]]] = {spec.name: [] for spec in method_specs}
    for spec in method_specs:
        for report in spec.reports:
            record = _extract_daily_record(spec.name, report, split_lookup, config.evaluation_split)
            if record is None:
                continue
            methods[spec.name].append(record)
            all_daily.append(record)

    if not any(records for records in methods.values()):
        raise ValueError(f"No reports matched evaluation split {config.evaluation_split!r}.")

    method_reports: dict[str, object] = {}
    gap_panel: dict[str, GapInput] = {}
    leaderboard: list[dict[str, object]] = []

    for index, (method_name, records) in enumerate(sorted(methods.items(), key=lambda item: item[0])):
        coverage = _coverage_summary(records, split_lookup=split_lookup, evaluation_split=config.evaluation_split)
        prediction_rank_ic = [_daily_metric(item, "prediction", "rank_ic") for item in records]
        prediction_rank_ic = [value for value in prediction_rank_ic if value is not None]
        prediction_dir_acc = [_daily_metric(item, "prediction", "directional_accuracy") for item in records]
        prediction_dir_acc = [value for value in prediction_dir_acc if value is not None]
        prediction_mse = [_daily_metric(item, "prediction", "mse") for item in records]
        prediction_mse = [value for value in prediction_mse if value is not None]
        prediction_mae = [_daily_metric(item, "prediction", "mae") for item in records]
        prediction_mae = [value for value in prediction_mae if value is not None]

        execution_sharpe = [_daily_metric(item, "execution", "post_cost_sharpe") for item in records]
        execution_sharpe = [value for value in execution_sharpe if value is not None]
        execution_drawdown = [_daily_metric(item, "execution", "max_drawdown") for item in records]
        execution_drawdown = [value for value in execution_drawdown if value is not None]
        execution_turnover = [_daily_metric(item, "execution", "turnover") for item in records]
        execution_turnover = [value for value in execution_turnover if value is not None]
        execution_calmar = [_daily_metric(item, "execution", "calmar") for item in records]
        execution_calmar = [value for value in execution_calmar if value is not None]
        execution_net_return = [_daily_metric(item, "execution", "net_return") for item in records]
        execution_net_return = [value for value in execution_net_return if value is not None]
        execution_cvar = [_daily_metric(item, "execution", "cvar95") for item in records]
        execution_cvar = [value for value in execution_cvar if value is not None]
        holding = [_daily_metric(item, "execution", "average_holding_period") for item in records]
        holding = [value for value in holding if value is not None]

        worst_regime = []
        survival = []
        regime_utilities: list[float] = []
        for record in records:
            robustness = record.get("robustness", {})
            if not isinstance(robustness, dict):
                continue
            if robustness.get("worst_regime_sharpe") is not None:
                worst_regime.append(float(robustness["worst_regime_sharpe"]))
            if robustness.get("stress_day_survival") is not None:
                survival.append(float(robustness["stress_day_survival"]))
            values = robustness.get("regime_utilities")
            if isinstance(values, list):
                regime_utilities.extend(float(item) for item in values)

        prediction_block = {
            "rank_ic": _summarize_scalar(prediction_rank_ic, samples=config.bootstrap_samples, seed=config.random_seed + index),
            "directional_accuracy": _summarize_scalar(
                prediction_dir_acc,
                samples=config.bootstrap_samples,
                seed=config.random_seed + index + 101,
            ),
            "mse": _summarize_scalar(prediction_mse, samples=config.bootstrap_samples, seed=config.random_seed + index + 202),
            "mae": _summarize_scalar(prediction_mae, samples=config.bootstrap_samples, seed=config.random_seed + index + 303),
        }
        execution_block = {
            "post_cost_sharpe": _summarize_scalar(
                execution_sharpe,
                samples=config.bootstrap_samples,
                seed=config.random_seed + index + 404,
            ),
            "max_drawdown": _summarize_scalar(
                execution_drawdown,
                samples=config.bootstrap_samples,
                seed=config.random_seed + index + 505,
            ),
            "turnover": _summarize_scalar(
                execution_turnover,
                samples=config.bootstrap_samples,
                seed=config.random_seed + index + 606,
            ),
            "calmar": _summarize_scalar(
                execution_calmar,
                samples=config.bootstrap_samples,
                seed=config.random_seed + index + 707,
            ),
            "net_return": _summarize_scalar(
                execution_net_return,
                samples=config.bootstrap_samples,
                seed=config.random_seed + index + 808,
            ),
            "cvar95": _summarize_scalar(
                execution_cvar,
                samples=config.bootstrap_samples,
                seed=config.random_seed + index + 909,
            ),
            "average_holding_period": _summarize_scalar(
                holding,
                samples=config.bootstrap_samples,
                seed=config.random_seed + index + 1001,
            ),
            "cost_ladder": _aggregate_cost_ladder(
                records,
                samples=config.bootstrap_samples,
                seed=config.random_seed + index + 1111,
            ),
            "r1_cost_ladder": _aggregate_stage_cost_ladder(
                records,
                stage="R1",
                samples=config.bootstrap_samples,
                seed=config.random_seed + index + 1121,
            ),
            "r2_cost_ladder": _aggregate_stage_cost_ladder(
                records,
                stage="R2",
                samples=config.bootstrap_samples,
                seed=config.random_seed + index + 1131,
            ),
            "r3_cost_ladder": _aggregate_stage_cost_ladder(
                records,
                stage="R3",
                samples=config.bootstrap_samples,
                seed=config.random_seed + index + 1141,
            ),
            "r4_cost_ladder": _aggregate_stage_cost_ladder(
                records,
                stage="R4",
                samples=config.bootstrap_samples,
                seed=config.random_seed + index + 1151,
            ),
        }
        robustness_block = {
            "worst_regime_sharpe": _summarize_scalar(
                worst_regime,
                samples=config.bootstrap_samples,
                seed=config.random_seed + index + 1212,
            ),
            "stress_day_survival": _summarize_scalar(
                survival,
                samples=config.bootstrap_samples,
                seed=config.random_seed + index + 1313,
            ),
        }
        method_reports[method_name] = {
            "prediction": prediction_block,
            "execution": execution_block,
            "robustness": robustness_block,
            "auditability": {
                "num_days": int(len(records)),
                "evaluation_split": config.evaluation_split,
                "source_paths": [item["path"] for item in records],
                "dates": [item["date"] for item in records],
                **coverage,
            },
        }
        offline_score = prediction_block["rank_ic"]["mean"]
        if offline_score is None:
            offline_score = prediction_block["directional_accuracy"]["mean"]
        full_realism = execution_block["post_cost_sharpe"]["mean"]
        if offline_score is None:
            offline_score = 0.0
        if full_realism is None:
            full_realism = 0.0
        if records:
            gap_panel[method_name] = GapInput(
                offline_score=float(offline_score),
                full_realism_utility=float(full_realism),
                regime_utilities=regime_utilities,
            )
        leaderboard.append(
            {
                "method": method_name,
                "post_cost_sharpe": execution_block["post_cost_sharpe"]["mean"],
                "rank_ic": prediction_block["rank_ic"]["mean"],
                "max_drawdown": execution_block["max_drawdown"]["mean"],
                "turnover": execution_block["turnover"]["mean"],
                "num_days": len(records),
            }
        )

    leaderboard.sort(
        key=lambda item: float("-inf") if item["post_cost_sharpe"] is None else float(item["post_cost_sharpe"]),
        reverse=True,
    )
    result = {
        "evaluation_split": config.evaluation_split,
        "leaderboard": leaderboard,
        "methods": method_reports,
        "gap": compute_gap_report(gap_panel, k=config.top_k) if gap_panel else {"status": "no_methods_with_reports"},
        "auditability": {
            "bootstrap_samples": config.bootstrap_samples,
            "random_seed": config.random_seed,
            "num_methods": len(method_reports),
            "num_daily_reports": len(all_daily),
            "expected_dates_in_split": sorted(day for day, split in split_lookup.items() if split == config.evaluation_split),
        },
    }

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))

    markdown_path = Path(config.markdown_path) if config.markdown_path else output_path.with_suffix(".md")
    generate_markdown_report(
        markdown_path,
        sections={
            "Leaderboard": leaderboard,
            "Methods": method_reports,
            "Gap": result["gap"],
            "Auditability": result["auditability"],
        },
        approximations=[
            "Benchmark evaluator aggregates per-day JSON artifacts and inherits the simulator approximation used in each report.",
            "Confidence intervals use day-level bootstrap over the available daily report set.",
            "Gap metrics rank offline prediction against shared R4 execution metrics using post-cost Sharpe as the full-realism anchor.",
        ],
        protocol_targets=[
            "Frozen 90-day 60/15/15 protocol is only fully populated once all asset-venue daily reports exist.",
            "Execution logs remain method outputs rather than precomputed benchmark artifacts at preprocessing time.",
        ],
    )
    return result


def load_method_specs_from_json(path: str | Path) -> tuple[list[BenchmarkMethodSpec], dict[str, str], dict[str, object]]:
    """Load method-report manifest used by the benchmark evaluator CLI."""
    payload = _load_json(path)
    raw_methods = payload.get("methods", [])
    method_specs = [
        BenchmarkMethodSpec(name=str(item["name"]), reports=tuple(str(report) for report in item["reports"]))
        for item in raw_methods
    ]
    split_lookup: dict[str, str] = {}
    if "split_plan_path" in payload:
        split_lookup = _load_split_lookup_from_split_plan(payload["split_plan_path"])
    elif "split_index_path" in payload:
        split_payload = _load_json(payload["split_index_path"])
        split_lookup = {day: str(meta["split"]) for day, meta in split_payload.items()}
    elif "bundle_manifest" in payload:
        manifest = _load_json(payload["bundle_manifest"])
        split_payload = manifest.get("split_index", {})
        split_lookup = {day: str(meta["split"]) for day, meta in split_payload.items()}
    elif "split_lookup" in payload:
        split_lookup = {str(day): str(split) for day, split in payload["split_lookup"].items()}
    return method_specs, split_lookup, payload
