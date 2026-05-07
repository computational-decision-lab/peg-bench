#!/usr/bin/env python3
"""Audit current PEG-Bench readiness and build a formal 30d report card."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.gap_metrics import compute_gap_report_from_method_reports
from src.eval.regime import summarize_regime_panel
from src.eval.representation_audit import audit_from_diagnostic_summary
from src.eval.stress import summarize_stress_panel


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _load_optional_json(path: str | Path) -> dict[str, Any] | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    return json.loads(file_path.read_text())


def _read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _write_json(path: str | Path, payload: dict[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(output.resolve())


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return str(output.resolve())
    fieldnames = list(rows[0].keys())
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(output.resolve())


def _write_markdown(path: str | Path, text: str) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return str(output.resolve())


def _benchmark_report_layout(outputs_root: Path) -> dict[str, dict[int, dict[str, Path]]]:
    layout = {
        "time": {
            5: {"benchmark": outputs_root / "time_h5_compare_v1" / "benchmark_report.json", "results": outputs_root / "time_h5_compare_v1" / "results_table.json"},
            20: {"benchmark": outputs_root / "time" / "benchmark_report.json", "results": outputs_root / "time" / "results_table.json"},
            50: {"benchmark": outputs_root / "time_h50_compare_v1" / "benchmark_report.json", "results": outputs_root / "time_h50_compare_v1" / "results_table.json"},
        },
        "volume": {
            5: {"benchmark": outputs_root / "volume_h5_compare_v1" / "benchmark_report.json", "results": outputs_root / "volume_h5_compare_v1" / "results_table.json"},
            20: {"benchmark": outputs_root / "volume" / "benchmark_report.json", "results": outputs_root / "volume" / "results_table.json"},
            50: {"benchmark": outputs_root / "volume_h50_compare_v1" / "benchmark_report.json", "results": outputs_root / "volume_h50_compare_v1" / "results_table.json"},
        },
    }
    return layout


def _load_benchmark_panels(outputs_root: Path) -> dict[str, dict[int, dict[str, Any]]]:
    loaded: dict[str, dict[int, dict[str, Any]]] = {}
    for mode, horizons in _benchmark_report_layout(outputs_root).items():
        loaded[mode] = {}
        for horizon, paths in horizons.items():
            benchmark = _load_optional_json(paths["benchmark"])
            results = _load_optional_json(paths["results"])
            loaded[mode][horizon] = {
                "benchmark": benchmark,
                "results": results,
                "benchmark_path": str(paths["benchmark"].resolve()),
                "results_path": str(paths["results"].resolve()),
            }
    return loaded


def _maybe_mean(payload: Any) -> float | None:
    if isinstance(payload, dict):
        value = payload.get("mean")
        return None if value is None else float(value)
    if payload is None:
        return None
    return float(payload)


def _report_paths(method_payload: dict[str, Any]) -> list[str]:
    auditability = method_payload.get("auditability", {})
    if not isinstance(auditability, dict):
        return []
    raw = auditability.get("source_paths", [])
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw]


def _has_text(path: Path, snippets: list[str]) -> bool:
    text = _read_text(path)
    return all(snippet in text for snippet in snippets)


def _flatten_report_card(report_card: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def visit(section: str, prefix: list[str], value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                visit(section, prefix + [str(key)], nested)
            return
        if isinstance(value, list):
            rows.append(
                {
                    "section": section,
                    "path": ".".join(prefix),
                    "value": json.dumps(value, ensure_ascii=False),
                }
            )
            return
        rows.append(
            {
                "section": section,
                "path": ".".join(prefix),
                "value": value,
            }
        )

    for section, payload in report_card.items():
        if section == "scope":
            continue
        visit(section, [section], payload)
    return rows


def _prediction_rows(benchmark_report: dict[str, Any] | None) -> dict[str, Any]:
    if not benchmark_report:
        return {}
    result: dict[str, Any] = {}
    for method_name, method_payload in sorted(benchmark_report.get("methods", {}).items()):
        prediction = method_payload.get("prediction", {})
        if not isinstance(prediction, dict):
            continue
        result[method_name] = {
            "directional_accuracy": _maybe_mean(prediction.get("directional_accuracy")),
            "rank_ic": _maybe_mean(prediction.get("rank_ic")),
            "mse": _maybe_mean(prediction.get("mse")),
            "mae": _maybe_mean(prediction.get("mae")),
        }
    return result


def _execution_rows(benchmark_report: dict[str, Any] | None, mode: str, horizon: int) -> dict[str, Any]:
    if not benchmark_report:
        return {}
    result: dict[str, Any] = {}
    for method_name, method_payload in sorted(benchmark_report.get("methods", {}).items()):
        execution = method_payload.get("execution", {})
        if not isinstance(execution, dict):
            continue
        report_paths = _report_paths(method_payload)
        stress_summary = summarize_stress_panel(report_paths)
        result[method_name] = {
            "mode": mode,
            "horizon": horizon,
            "post_cost_sharpe": _maybe_mean(execution.get("post_cost_sharpe")),
            "max_drawdown": _maybe_mean(execution.get("max_drawdown")),
            "turnover": _maybe_mean(execution.get("turnover")),
            "calmar": _maybe_mean(execution.get("calmar")),
            "net_return": _maybe_mean(execution.get("net_return")),
            "cvar95": _maybe_mean(execution.get("cvar95")),
            "average_holding_period": _maybe_mean(execution.get("average_holding_period")),
            "cost_ladder": execution.get("cost_ladder", {}),
            "r1_cost_ladder": execution.get("r1_cost_ladder", {}),
            "r2_cost_ladder": execution.get("r2_cost_ladder", {}),
            "r3_cost_ladder": execution.get("r3_cost_ladder", {}),
            "r4_cost_ladder": execution.get("r4_cost_ladder", execution.get("cost_ladder", {})),
            "drawdown_violation_rate": stress_summary.get("drawdown_violation_rate"),
            "liquidation_violation_rate": stress_summary.get("liquidation_violation_rate"),
            "position_limit": 1.0,
            "simulator_grade": "research-grade",
        }
    return result


def _robustness_rows(benchmark_report: dict[str, Any] | None, mode: str, horizon: int) -> dict[str, Any]:
    if not benchmark_report:
        return {}
    result: dict[str, Any] = {}
    for method_name, method_payload in sorted(benchmark_report.get("methods", {}).items()):
        robustness = method_payload.get("robustness", {})
        report_paths = _report_paths(method_payload)
        regime_summary = summarize_regime_panel(report_paths)
        stress_summary = summarize_stress_panel(report_paths)
        result[method_name] = {
            "mode": mode,
            "horizon": horizon,
            "worst_regime_sharpe": _maybe_mean(robustness.get("worst_regime_sharpe")),
            "stress_survival_rate": _maybe_mean(robustness.get("stress_day_survival")),
            "regime_variance": regime_summary.get("regime_variance"),
            "available_regimes": regime_summary.get("available_regimes", []),
            "missing_regimes": regime_summary.get("missing_regimes", []),
            "stress_interval_count": stress_summary.get("stress_interval_count"),
            "drawdown_violation_rate": stress_summary.get("drawdown_violation_rate"),
            "stress_status": stress_summary.get("status"),
            "regime_status": regime_summary.get("status"),
        }
    return result


def _dpo_execution_payload(run_summary: dict[str, Any], mode: str) -> dict[str, Any]:
    backtest = run_summary.get("backtest", {})
    report_card = backtest.get("report_card", {})
    execution = report_card.get("execution", {})
    robustness = report_card.get("robustness", {})
    prediction_proxy = backtest.get("prediction_proxy", {})
    return {
        "mode": mode,
        "selected_risk_mode": backtest.get("selected_risk_mode"),
        "selected_scale": backtest.get("selected_scale"),
        "selected_deadzone": backtest.get("selected_deadzone"),
        "post_cost_sharpe": execution.get("post_cost_sharpe"),
        "max_drawdown": execution.get("max_drawdown"),
        "turnover": execution.get("turnover"),
        "calmar": execution.get("calmar"),
        "net_return": execution.get("net_return"),
        "cvar95": execution.get("cvar95"),
        "average_holding_period": execution.get("average_holding_period"),
        "cost_ladder": backtest.get("cost_scan", {}),
        "worst_regime_sharpe": robustness.get("worst_regime_sharpe"),
        "stress_survival_rate": robustness.get("stress_day_survival"),
        "r0_prediction_metrics": prediction_proxy.get("evaluation", report_card.get("prediction", {})),
        "prediction_proxy": prediction_proxy,
        "selection_split": backtest.get("selection_split", {}),
        "evaluation_split": backtest.get("evaluation_split", {}),
        "note": prediction_proxy.get(
            "note",
            "Execution-first evidence. Compare against baseline Sharpe/cost ladder, not baseline RankIC.",
        ),
    }


def _method_has_stage_ladders(method_payload: dict[str, Any]) -> bool:
    execution = method_payload.get("execution", {})
    if not isinstance(execution, dict):
        return False
    return all(
        isinstance(execution.get(name), dict) and len(execution.get(name, {})) > 0
        for name in ("r1_cost_ladder", "r2_cost_ladder", "r3_cost_ladder", "r4_cost_ladder")
    )


def _cost_ladder_complete(method_payload: dict[str, Any]) -> bool:
    execution = method_payload.get("execution", {})
    if not isinstance(execution, dict):
        return False
    observed = {str(key) for key in execution.get("cost_ladder", {}).keys()}
    return observed == {"0.0", "1.0", "3.0", "5.0", "10.0"}


def build_readiness_and_report(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    outputs_root = Path(args.outputs_root)
    final_root = Path(args.final_root)
    panels = _load_benchmark_panels(outputs_root)
    diagnostic = _load_json(outputs_root / "metadata" / "diagnostic_summary_30d.json")
    representation_audit = audit_from_diagnostic_summary(outputs_root / "metadata" / "diagnostic_summary_30d.json")
    time_report = panels["time"][20]["benchmark"] or {"methods": {}, "leaderboard": [], "gap": {}}
    volume_report = panels["volume"][20]["benchmark"] or {"methods": {}, "leaderboard": [], "gap": {}}
    time_results = panels["time"][20]["results"] or []
    volume_results = panels["volume"][20]["results"] or []
    time_dpo_root = Path(args.time_dpo_root) if args.time_dpo_root else outputs_root / "time_dpo_gap0_noaci_v1"
    volume_dpo_root = Path(args.volume_dpo_root) if args.volume_dpo_root else outputs_root / "volume_dpo_gap0_noaci_v1"
    time_dpo = _load_optional_json(time_dpo_root / "run_summary.json") or {}
    volume_dpo = _load_optional_json(volume_dpo_root / "run_summary.json") or {}

    baseline_model_source = ROOT / "src" / "models" / "baseline.py"
    action_interface_source = ROOT / "src" / "models" / "action_interface.py"
    baseline_train_source = ROOT / "src" / "train" / "baseline_trainer.py"
    features_source = ROOT / "src" / "data" / "features.py"
    order_book_source = ROOT / "src" / "data" / "order_book.py"
    peg_bench_source = ROOT / "src" / "eval" / "peg_bench.py"
    benchmark_baseline_source = ROOT / "src" / "eval" / "benchmark_baselines.py"
    execution_source = ROOT / "src" / "simulation" / "execution.py"
    cost_source = ROOT / "src" / "simulation" / "cost_model.py"
    vq_source = ROOT / "src" / "models" / "vqvae.py"
    pair_builder_source = ROOT / "src" / "preference" / "pair_builder.py"
    oracle_source = ROOT / "src" / "simulation" / "oracle.py"
    dpo_trainer_source = ROOT / "src" / "train" / "dpo_trainer.py"

    completed: list[str] = []
    missing: list[str] = []
    blocking: list[str] = []

    split_days = diagnostic["time"]["protocol"]["split_days"]
    benchmark_units = set(diagnostic["time"]["protocol"]["benchmark_units"])
    has_full_universe = (
        diagnostic["time"]["coverage"]["complete_asset_days"] == 120
        and diagnostic["volume"]["coverage"]["complete_asset_days"] == 120
    )
    if has_full_universe:
        completed.append("30d universe covers Binance/OKX x BTCUSDT/ETHUSDT with complete asset-day coverage.")
    else:
        missing.append("30d universe coverage is incomplete.")
        blocking.append("Current data universe is incomplete.")

    expected_units = {"raw_l2_events", "reconstructed_top10_lob", "time_bars_1s", "volume_bars", "execution_logs"}
    if expected_units.issubset(benchmark_units):
        completed.append("Benchmark contract artifacts include raw L2, reconstructed top-10 LOB, bars and execution logs.")
    else:
        missing.append("Benchmark contract artifacts are missing one or more required units.")
        blocking.append("Raw-to-execution artifact chain is incomplete.")

    if sum(split_days.values()) == 30 and all(value > 0 for value in split_days.values()):
        completed.append(
            f"30d benchmark contract uses a strict chronological {split_days['train']}/{split_days['val']}/{split_days['test']} split."
        )
    else:
        missing.append("30d chronological split metadata is missing or invalid.")
        blocking.append("Current 30d split contract is not auditable.")

    if _has_text(order_book_source, ["def apply_l2", "def apply_trade", "def merge_event_streams"]):
        completed.append("State construction replays L2/trades into a reconstructed LOB instead of reading bars only.")
    else:
        missing.append("LOB replay path is missing or not auditable.")

    if "depth: int = 10" in _read_text(features_source):
        completed.append("Top-10 order book tensorization is implemented.")
    else:
        missing.append("Top-10 order book tensorization is missing.")

    feature_text = _read_text(features_source)
    if all(snippet in feature_text for snippet in ["rel_log_price", "log_volume", "ofi", "toxicity", "microprice"]):
        completed.append("Feature stack includes log price/volume, OFI, spread/microprice and toxicity proxy.")
    else:
        missing.append("Feature stack is missing one or more required microstructure features.")

    if diagnostic["time"]["protocol"]["context_len"] == 96:
        completed.append("Protocol target context length is set to 96 bars.")
    time_notes_text = " ".join(str(item.get("notes", "")) for item in time_results if isinstance(item, dict))
    if "seq=64" in time_notes_text:
        missing.append("Some 30d comparison runs use seq_len=64 for token/Kairos paths while the protocol target remains context=96.")

    if diagnostic["time"]["protocol"]["horizons"] != [5, 20, 50]:
        missing.append("Required horizons 5/20/50 are not fully implemented.")
        blocking.append("Horizon metadata is incomplete.")
    elif all(panels[mode][horizon]["benchmark"] for mode in ("time", "volume") for horizon in (5, 20, 50)):
        completed.append("Horizons 5/20/50 are implemented and benchmark reports exist for time and volume.")
    else:
        missing.append("One or more horizon benchmark reports are missing.")
        blocking.append("Current 30d horizon coverage is incomplete.")

    vq_text = _read_text(vq_source)
    if "num_embeddings: int = 64" in vq_text:
        completed.append("VQ-VAE codebook size is 64.")
    else:
        missing.append("VQ codebook size is not 64.")
        blocking.append("KAIROS representation is off-contract on codebook size.")

    if "VectorQuantizerEMA" in vq_text and "hierarchical" not in vq_text.lower():
        missing.append("Current VQ-VAE is single-level; hierarchical VQ-VAE is not implemented.")
    else:
        completed.append("Hierarchical VQ-VAE is implemented.")

    unique_codes = [int(representation_audit["time"]["active_codes"]), int(representation_audit["volume"]["active_codes"])]
    if min(unique_codes) < 10:
        missing.append(f"Codebook collapse detected: active codes counts are {unique_codes}.")
        blocking.append("VQ codebook collapsed below 10 active codes.")
    else:
        completed.append(f"VQ code usage is above collapse threshold: active codes counts are {unique_codes}.")

    if _has_text(baseline_model_source, ["_causal_mask", "self.transformer(hidden, mask=mask)"]):
        completed.append("Prediction model is autoregressive via a causal mask.")
    else:
        missing.append("Autoregressive transformer masking is missing.")

    if _has_text(baseline_model_source, ["baseline_loss", "mse", "rank_ic"]) and _has_text(
        baseline_train_source,
        ["rankic_weight", "baseline_loss"],
    ):
        completed.append("Prediction loss combines MSE and RankIC.")
    else:
        missing.append("Prediction loss does not clearly combine MSE and RankIC.")

    action_text = _read_text(action_interface_source)
    baseline_actions_are_bounded = "np.clip(wrapped_positions" in action_text
    baseline_has_threshold_wrapper = "raw_forecast_positions" in action_text and "cost_threshold_decimal" in action_text
    if baseline_actions_are_bounded and baseline_has_threshold_wrapper and _has_text(benchmark_baseline_source, ["build_action_interface_result"]):
        completed.append("Prediction-first models share the standard sign-threshold action wrapper.")
    else:
        missing.append("Cost-aware sign/threshold action wrapper is not implemented for the current baseline pipeline.")
        blocking.append("Prediction-first models do not yet share a standard action wrapper.")

    if "smooth_positions" in action_text:
        completed.append("Action smoothing is implemented.")
    else:
        missing.append("Action smoothing is not implemented.")
        blocking.append("Current action interface lacks smoothing over positions.")

    if _has_text(peg_bench_source, ['"R0"', '"R1"', '"R2"', '"R3"', '"R4"']):
        completed.append("Shared realism ladder code defines R0 through R4.")
    else:
        missing.append("Shared realism ladder is incomplete in code.")
        blocking.append("R0-R4 evaluator code is incomplete.")

    available_main_methods = list(time_report.get("methods", {}).values()) + list(volume_report.get("methods", {}).values())
    if available_main_methods and all(_method_has_stage_ladders(item) for item in available_main_methods):
        completed.append("Main 30d benchmark artifacts persist explicit R1/R2/R3/R4 stage ladders.")
    else:
        missing.append("Current baseline artifacts do not yet expose standalone R1/R2/R3/R4 benchmark tables.")
        blocking.append("Stage-wise benchmark artifacts are incomplete.")

    if _has_text(cost_source, ["fee_bps", "estimate_impact_bps"]) and _has_text(
        execution_source,
        ["enable_risk_controls", "max_position", "turnover =", "positions = np.zeros_like"],
    ):
        completed.append("Execution simulator includes fee, impact/slippage proxy, turnover and position tracking.")
    else:
        missing.append("Execution simulator is missing fee/slippage/risk primitives.")
        blocking.append("Execution simulator is incomplete.")

    if "queue" not in _read_text(execution_source).lower() and "latency" not in _read_text(execution_source).lower():
        missing.append("Simulator has no queue or latency model; it remains research-grade, not exchange-grade.")

    gap_ready = True
    for mode in ("time", "volume"):
        for horizon in (5, 20, 50):
            benchmark = panels[mode][horizon]["benchmark"]
            if not benchmark or benchmark.get("gap", {}).get("status") != "computed":
                gap_ready = False
    if gap_ready:
        completed.append("Gap metrics are computed on current 30d baseline reports.")
    else:
        missing.append("Gap metrics are missing from one or more current benchmark reports.")
        blocking.append("Gap metrics are incomplete.")

    time_robustness = next(iter(time_report["methods"].values()))["robustness"] if time_report.get("methods") else {}
    volume_robustness = next(iter(volume_report["methods"].values()))["robustness"] if volume_report.get("methods") else {}
    if "worst_regime_sharpe" in time_robustness and "stress_day_survival" in time_robustness and "worst_regime_sharpe" in volume_robustness:
        completed.append("R4 robustness metrics include worst-regime Sharpe and stress survival.")
    else:
        missing.append("R4 robustness metrics are incomplete.")
        blocking.append("R4 robustness layer is incomplete.")

    if time_report.get("methods"):
        time_regime_summary = summarize_regime_panel(_report_paths(next(iter(time_report["methods"].values()))))
    else:
        time_regime_summary = {"available_regimes": []}
    if volume_report.get("methods"):
        volume_regime_summary = summarize_regime_panel(_report_paths(next(iter(volume_report["methods"].values()))))
    else:
        volume_regime_summary = {"available_regimes": []}
    if "normal" in time_regime_summary.get("available_regimes", []) or "normal" in volume_regime_summary.get("available_regimes", []):
        completed.append("Normal regime slice is present in persisted robustness artifacts.")
    else:
        missing.append("Normal regime slice is not persisted; current regime coverage is missing an explicit normal bucket.")
        blocking.append("Regime/stress persistence is incomplete.")

    anchor_methods = {"no_trade", "buy_and_hold", "ofi_threshold"}
    main_prediction_methods = set(time_report.get("methods", {}).keys()) | set(volume_report.get("methods", {}).keys())
    if anchor_methods.issubset(main_prediction_methods):
        completed.append("Anchor baselines no_trade, buy_and_hold, and ofi_threshold are present in the 30d panel.")
    else:
        missing.append("Anchor baselines no_trade / buy_and_hold / ofi_threshold are missing from the current 30d panel.")
        blocking.append("Baseline panel is missing required anchors.")

    if available_main_methods and all(_cost_ladder_complete(item) for item in available_main_methods if isinstance(item, dict)):
        completed.append("The full 0/1/3/5/10 bps cost ladder is present for the main time and volume panels.")
    else:
        missing.append("The full 0/1/3/5/10 bps cost ladder is missing from one or more main panels.")
        blocking.append("Cost ladder coverage is incomplete.")

    if _has_text(pair_builder_source, ["build_preference_pairs", "TrajectoryRolloutEngine"]) and _has_text(
        oracle_source,
        ["max_drawdown", "criterion"],
    ) and _has_text(dpo_trainer_source, ["preferences_path", "val_sharpe", "GaussianActionPolicy"]):
        completed.append("OG-PA/DPO path includes rollouts, preference pairs, drawdown-aware ordering and DPO training.")
    else:
        missing.append("OG-PA/DPO chain is incomplete.")
        blocking.append("DPO / OG-PA chain is incomplete.")

    if time_dpo and volume_dpo:
        completed.append("Time and volume DPO execution-first artifacts are available.")
    else:
        missing.append("Missing time or volume DPO execution artifacts.")
        blocking.append("DPO / OG-PA results are incomplete.")

    dpo_prediction_proxy_ok = all(
        isinstance(run_summary.get("backtest", {}).get("prediction_proxy", {}).get("evaluation"), dict)
        and run_summary.get("backtest", {}).get("prediction_proxy", {}).get("evaluation", {}).get("status") == "computed"
        for run_summary in (time_dpo, volume_dpo)
        if run_summary
    )
    if time_dpo and volume_dpo and dpo_prediction_proxy_ok:
        completed.append("DPO execution-first runs now expose a unified R0 proxy report on the evaluation split.")
    else:
        missing.append("DPO execution-first runs still lack a unified R0 proxy report on the evaluation split.")

    current_level = "R0"
    if _has_text(peg_bench_source, ['"R1"']) and _has_text(cost_source, ["fee_bps"]):
        current_level = "R1"
    if _has_text(peg_bench_source, ['"R2"']) and _has_text(cost_source, ["estimate_impact_bps"]):
        current_level = "R2"
    if _has_text(peg_bench_source, ['"R3"']) and _has_text(execution_source, ["enable_risk_controls"]):
        current_level = "R3"
    if _has_text(peg_bench_source, ['"R4"']) and time_report.get("methods") and volume_report.get("methods"):
        current_level = "R4"

    readiness = {
        "current_level": current_level,
        "completed": completed,
        "missing": missing,
        "blocking_items": blocking,
        "can_claim_final_benchmark": len(blocking) == 0 and current_level == "R4",
    }

    prediction_section: dict[str, Any] = {
        "R0": {},
        "DPO_execution_first": {
            "time": time_dpo.get("backtest", {}).get("prediction_proxy", {}).get("evaluation", {}) if time_dpo else {},
            "volume": volume_dpo.get("backtest", {}).get("prediction_proxy", {}).get("evaluation", {}) if volume_dpo else {},
        },
        "dpo_note": "DPO / OG-PA is execution-first. The R0 block below is a policy-mean proxy against forward returns, so it is more comparable than before but still not identical to a pure return-forecast baseline output.",
    }
    execution_section: dict[str, Any] = {
        "ladder_status": {
            "R0": "prediction metrics",
            "R1": "fees only",
            "R2": "fees + slippage",
            "R3": "fees + slippage + risk controls",
            "R4": "R3 + regime/stress evaluation",
        },
        "baselines": {},
        "dpo_execution_first": {
            "time": _dpo_execution_payload(time_dpo, mode="time") if time_dpo else {},
            "volume": _dpo_execution_payload(volume_dpo, mode="volume") if volume_dpo else {},
        },
    }
    gap_section: dict[str, Any] = {}
    robustness_section: dict[str, Any] = {"baselines": {}, "dpo_execution_first": {}}
    cost_ladder_results: dict[str, Any] = {"baselines": {}, "dpo_execution_first": execution_section["dpo_execution_first"]}
    regime_stress_results: dict[str, Any] = {"baselines": {}, "dpo_execution_first": {}}
    model_leaderboards: dict[str, Any] = {"prediction": {}, "execution": {}, "dpo_execution_first": execution_section["dpo_execution_first"]}

    for mode in ("time", "volume"):
        execution_section["baselines"][mode] = {}
        robustness_section["baselines"][mode] = {}
        cost_ladder_results["baselines"][mode] = {}
        regime_stress_results["baselines"][mode] = {}
        gap_section[mode] = {}
        for horizon in (5, 20, 50):
            benchmark_report = panels[mode][horizon]["benchmark"]
            panel_key = f"{mode}_h{horizon}"
            prediction_section["R0"][panel_key] = _prediction_rows(benchmark_report)
            execution_section["baselines"][mode][f"h{horizon}"] = _execution_rows(benchmark_report, mode=mode, horizon=horizon)
            robustness_section["baselines"][mode][f"h{horizon}"] = _robustness_rows(benchmark_report, mode=mode, horizon=horizon)
            gap_section[mode][f"h{horizon}"] = (
                compute_gap_report_from_method_reports(benchmark_report.get("methods", {}), k=min(3, max(1, len(benchmark_report.get("methods", {})))))
                if benchmark_report
                else {"status": "missing"}
            )
            cost_ladder_results["baselines"][mode][f"h{horizon}"] = execution_section["baselines"][mode][f"h{horizon}"]
            regime_stress_results["baselines"][mode][f"h{horizon}"] = robustness_section["baselines"][mode][f"h{horizon}"]
            if benchmark_report:
                execution_leaderboard = benchmark_report.get("leaderboard", [])
                prediction_leaderboard = sorted(
                    [
                        {
                            "method": method_name,
                            "rank_ic": _maybe_mean(method_payload.get("prediction", {}).get("rank_ic")),
                            "directional_accuracy": _maybe_mean(method_payload.get("prediction", {}).get("directional_accuracy")),
                        }
                        for method_name, method_payload in benchmark_report.get("methods", {}).items()
                    ],
                    key=lambda item: float("-inf") if item["rank_ic"] is None else float(item["rank_ic"]),
                    reverse=True,
                )
                model_leaderboards["prediction"][panel_key] = prediction_leaderboard
                model_leaderboards["execution"][panel_key] = execution_leaderboard

    if time_dpo:
        regime_stress_results["dpo_execution_first"]["time"] = time_dpo.get("backtest", {}).get("report_card", {}).get("robustness", {})
    if volume_dpo:
        regime_stress_results["dpo_execution_first"]["volume"] = volume_dpo.get("backtest", {}).get("report_card", {}).get("robustness", {})

    report_card: dict[str, Any] = {
        "scope": {
            "name": "PEG-Bench 30d final report card",
            "current_level": current_level,
            "can_claim_final_benchmark": readiness["can_claim_final_benchmark"],
            "data_window": diagnostic["time"]["protocol"]["date_window"],
            "split_days_realized": split_days,
            "paper_contract_60_15_15": split_days == {"train": 60, "val": 15, "test": 15},
        },
        "prediction": prediction_section,
        "execution": execution_section,
        "gap": gap_section,
        "robustness": robustness_section,
        "auditability": {
            "benchmark_contract": {
                "venues": ["binance", "okx"],
                "symbols": ["BTCUSDT", "ETHUSDT"],
                "l2_trades_unified_schema": True,
                "raw_l2": "raw_l2_events" in benchmark_units,
                "reconstructed_lob": "reconstructed_top10_lob" in benchmark_units,
                "volume_bars": "volume_bars" in benchmark_units,
                "execution_logs": "execution_logs" in benchmark_units,
                "current_split_days": split_days,
            },
            "representation": representation_audit,
            "prediction_model": {
                "autoregressive": True,
                "loss_mse": True,
                "loss_rankic": True,
                "objective": "prediction-first",
            },
            "action_interface": {
                "bounded_minus1_to_1": baseline_actions_are_bounded,
                "sign_threshold_wrapper": baseline_has_threshold_wrapper,
                "smoothing": "smooth_positions" in action_text,
                "source_file": str(action_interface_source.resolve()),
            },
            "execution_simulator": {
                "fee": True,
                "slippage_proxy": True,
                "risk_controls": True,
                "queue_model": False,
                "latency_model": False,
                "simulator_grade": "research-grade simulator",
            },
            "horizon_reports": {
                mode: {
                    f"h{horizon}": {
                        "benchmark_report": panels[mode][horizon]["benchmark_path"],
                        "results_table": panels[mode][horizon]["results_path"],
                        "available": panels[mode][horizon]["benchmark"] is not None,
                    }
                    for horizon in (5, 20, 50)
                }
                for mode in ("time", "volume")
            },
            "missing_modules": missing,
            "blocking_items": blocking,
        },
    }

    readiness_path = _write_json(final_root / "readiness_audit.json", readiness)
    report_card_json_path = _write_json(final_root / "report_card.json", report_card)
    report_card_csv_path = _write_csv(final_root / "report_card.csv", _flatten_report_card(report_card))
    _write_json(final_root / "gap_metrics.json", gap_section)
    _write_json(final_root / "cost_ladder_results.json", cost_ladder_results)
    _write_json(final_root / "regime_stress_results.json", regime_stress_results)
    _write_json(final_root / "model_leaderboards.json", model_leaderboards)

    best_time_baseline = max(
        (
            (item.get("post_cost_sharpe"), name)
            for name, item in execution_section["baselines"]["time"]["h20"].items()
            if item.get("post_cost_sharpe") is not None
        ),
        default=(None, None),
    )
    best_volume_baseline = max(
        (
            (item.get("post_cost_sharpe"), name)
            for name, item in execution_section["baselines"]["volume"]["h20"].items()
            if item.get("post_cost_sharpe") is not None
        ),
        default=(None, None),
    )
    time_dpo_sharpe = execution_section["dpo_execution_first"]["time"].get("post_cost_sharpe")
    volume_dpo_sharpe = execution_section["dpo_execution_first"]["volume"].get("post_cost_sharpe")
    dpo_beats_time = best_time_baseline[0] is not None and time_dpo_sharpe is not None and float(time_dpo_sharpe) > float(best_time_baseline[0])
    dpo_beats_volume = best_volume_baseline[0] is not None and volume_dpo_sharpe is not None and float(volume_dpo_sharpe) > float(best_volume_baseline[0])

    stage_lines = [
        "# 30d Stage Conclusion",
        "",
        f"- Current highest completed level: `{current_level}`",
        f"- 30d final benchmark claim: `{readiness['can_claim_final_benchmark']}`",
        f"- Prediction-Execution Gap claim: `{'yes' if gap_ready else 'partial'}`",
        f"- DPO / OG-PA execution improvement claim: `time={dpo_beats_time}, volume={dpo_beats_volume}`",
        f"- Full R4 claim: `{'yes' if readiness['can_claim_final_benchmark'] and current_level == 'R4' else 'partial'}`",
        "",
        "## Complete",
        "",
        *[f"- {item}" for item in completed],
        "",
        "## Partial / Missing",
        "",
        *[f"- {item}" for item in missing],
        "",
        "## Blocking",
        "",
        *([f"- {item}" for item in blocking] if blocking else ["- None"]),
        "",
        "## Key Answers",
        "",
        f"- Prediction-first baselines: `{', '.join(sorted(set(time_report.get('methods', {}).keys()) | set(volume_report.get('methods', {}).keys())))}`",
        f"- Best time execution baseline @h20: `{best_time_baseline[1]}` ({best_time_baseline[0]})",
        f"- Best volume execution baseline @h20: `{best_volume_baseline[1]}` ({best_volume_baseline[0]})",
        f"- DPO time Sharpe @3bps: `{time_dpo_sharpe}`",
        f"- DPO volume Sharpe @3bps: `{volume_dpo_sharpe}`",
        "- Extending to 90d would still require the frozen 60/15/15 contract, wider regime coverage, and a full paper-grade method panel.",
    ]
    _write_markdown(final_root / "stage_conclusion.md", "\n".join(stage_lines) + "\n")

    md_lines = [
        "# PEG-Bench 30d Report Card",
        "",
        "## Readiness",
        "",
        "```json",
        json.dumps(readiness, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Prediction",
        "",
        "```json",
        json.dumps(report_card["prediction"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Execution",
        "",
        "```json",
        json.dumps(report_card["execution"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Gap",
        "",
        "```json",
        json.dumps(report_card["gap"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Robustness",
        "",
        "```json",
        json.dumps(report_card["robustness"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Auditability",
        "",
        "```json",
        json.dumps(report_card["auditability"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Output Files",
        "",
        f"- `{readiness_path}`",
        f"- `{report_card_json_path}`",
        f"- `{report_card_csv_path}`",
        f"- `{(final_root / 'gap_metrics.json').resolve()}`",
        f"- `{(final_root / 'cost_ladder_results.json').resolve()}`",
        f"- `{(final_root / 'regime_stress_results.json').resolve()}`",
        f"- `{(final_root / 'model_leaderboards.json').resolve()}`",
        f"- `{(final_root / 'stage_conclusion.md').resolve()}`",
    ]
    _write_markdown(final_root / "report_card.md", "\n".join(md_lines) + "\n")
    return readiness, report_card


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit current PEG-Bench readiness and build a formal 30d report card.")
    parser.add_argument(
        "--outputs-root",
        type=str,
        default=str((ROOT / "outputs" / "peg_bench_remote").resolve()),
        help="Root directory containing synchronized 30d PEG-Bench artifacts.",
    )
    parser.add_argument(
        "--final-root",
        type=str,
        default=str((ROOT / "outputs" / "final_30d").resolve()),
        help="Output directory for readiness and report card artifacts.",
    )
    parser.add_argument(
        "--time-dpo-root",
        type=str,
        default=None,
        help="Override directory containing the time DPO run_summary.json.",
    )
    parser.add_argument(
        "--volume-dpo-root",
        type=str,
        default=None,
        help="Override directory containing the volume DPO run_summary.json.",
    )
    args = parser.parse_args()
    readiness, report_card = build_readiness_and_report(args)
    print(json.dumps(readiness, indent=2, ensure_ascii=False))
    print(f"Wrote report card to {Path(args.final_root).resolve()}")
    print(f"Report card sections: {', '.join(report_card.keys())}")


if __name__ == "__main__":
    main()
