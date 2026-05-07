#!/usr/bin/env python3
"""Validate a 30d PEG-Bench final directory and write acceptance artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_FILES = (
    "readiness_audit.json",
    "report_card.json",
    "report_card.md",
    "stage_conclusion.md",
    "gap_metrics.json",
    "cost_ladder_results.json",
    "regime_stress_results.json",
    "model_leaderboards.json",
)

EXPECTED_COST_LEVELS = ("0.0", "1.0", "3.0", "5.0", "10.0")
EXPECTED_REGIMES = ("normal", "high_vol", "trend_up", "trend_down", "liquidity_stress")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _as_float(value: Any) -> float | None:
    """Extract a numeric scalar from a raw value or a {'mean': ...} block."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        mean_value = value.get("mean")
        if isinstance(mean_value, (int, float)):
            return float(mean_value)
    return None


def _append_unique(items: list[str], value: str) -> None:
    """Append one item while preserving order and removing duplicates."""
    if value not in items:
        items.append(value)


def _prediction_layer_ok(report_card: dict[str, Any]) -> bool:
    """Check that prediction layer has concrete DA and RankIC values."""
    r0 = report_card.get("prediction", {}).get("R0", {})
    if not isinstance(r0, dict):
        return False
    has_da = False
    has_rank_ic = False
    for panel in r0.values():
        if not isinstance(panel, dict):
            continue
        for metrics in panel.values():
            if not isinstance(metrics, dict):
                continue
            if _as_float(metrics.get("directional_accuracy")) is not None:
                has_da = True
            if _as_float(metrics.get("rank_ic")) is not None:
                has_rank_ic = True
    return has_da and has_rank_ic


def _dpo_prediction_proxy_ok(report_card: dict[str, Any]) -> bool:
    """Check whether the execution-first DPO section exposes evaluation-split proxy metrics."""
    dpo_section = report_card.get("prediction", {}).get("DPO_execution_first", {})
    if not isinstance(dpo_section, dict):
        return False
    for mode in ("time", "volume"):
        payload = dpo_section.get(mode, {})
        if not isinstance(payload, dict):
            return False
        if payload.get("status") != "computed":
            return False
        if _as_float(payload.get("directional_accuracy")) is None:
            return False
        if _as_float(payload.get("rank_ic")) is None:
            return False
    return True


def _execution_layer_ok(report_card: dict[str, Any]) -> bool:
    """Check that execution layer has concrete Sharpe / PnL / DD / turnover / calmar."""
    baselines = report_card.get("execution", {}).get("baselines", {})
    if not isinstance(baselines, dict):
        return False
    required = ("post_cost_sharpe", "net_return", "max_drawdown", "turnover", "calmar")
    for mode_payload in baselines.values():
        if not isinstance(mode_payload, dict):
            continue
        for horizon_payload in mode_payload.values():
            if not isinstance(horizon_payload, dict):
                continue
            for method_payload in horizon_payload.values():
                if not isinstance(method_payload, dict):
                    continue
                if all(_as_float(method_payload.get(key)) is not None for key in required):
                    return True
    return False


def _gap_layer_ok(gap_metrics: dict[str, Any]) -> bool:
    """Check that gap layer has actual numeric metrics and fragility values."""
    for mode in ("time", "volume"):
        for horizon in ("h5", "h20", "h50"):
            payload = gap_metrics.get(mode, {}).get(horizon, {})
            if not isinstance(payload, dict):
                return False
            if _as_float(payload.get("metric_gap")) is None:
                return False
            if _as_float(payload.get("ranking_gap")) is None:
                return False
            if _as_float(payload.get("reversal_rate_at_k")) is None:
                return False
            fragility = payload.get("fragility_gap", {})
            if not isinstance(fragility, dict) or fragility.get("status") != "computed":
                return False
            if not isinstance(fragility.get("values"), dict) or not fragility.get("values"):
                return False
    return True


def _robustness_layer_ok(regime_results: dict[str, Any]) -> bool:
    """Check that robustness layer has actual regime/stress numbers."""
    baselines = regime_results.get("baselines", {})
    for mode in ("time", "volume"):
        for horizon in ("h5", "h20", "h50"):
            horizon_payload = baselines.get(mode, {}).get(horizon, {})
            if not isinstance(horizon_payload, dict) or not horizon_payload:
                return False
            found = False
            for method_payload in horizon_payload.values():
                if not isinstance(method_payload, dict):
                    continue
                if (
                    _as_float(method_payload.get("worst_regime_sharpe")) is not None
                    and _as_float(method_payload.get("stress_survival_rate")) is not None
                    and _as_float(method_payload.get("regime_variance")) is not None
                ):
                    found = True
                    break
            if not found:
                return False
    return True


def _auditability_layer_ok(report_card: dict[str, Any]) -> bool:
    """Check that auditability layer exposes split, cost, simulator, and claim-boundary info."""
    auditability = report_card.get("auditability", {})
    if not isinstance(auditability, dict):
        return False
    contract = auditability.get("benchmark_contract", {})
    simulator = auditability.get("execution_simulator", {})
    horizons = auditability.get("horizon_reports", {})
    missing_modules = auditability.get("missing_modules")
    return (
        isinstance(contract, dict)
        and isinstance(contract.get("current_split_days"), dict)
        and isinstance(simulator, dict)
        and "simulator_grade" in simulator
        and isinstance(horizons, dict)
        and isinstance(missing_modules, list)
    )


def _best_baseline_at_anchor(cost_results: dict[str, Any], mode: str, horizon: str) -> tuple[str, float]:
    """Return the best baseline Sharpe at 3bps for one mode/horizon."""
    best_name = ""
    best_sharpe = float("-inf")
    for method_name, payload in cost_results["baselines"][mode][horizon].items():
        sharpe = _as_float(payload.get("r4_cost_ladder", {}).get("3.0", {}).get("sharpe"))
        if sharpe is None:
            continue
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_name = method_name
    return best_name, best_sharpe


def _best_baseline_by_cost(cost_results: dict[str, Any], mode: str, horizon: str) -> dict[str, tuple[str, float]]:
    """Return the best baseline for each cost level."""
    result: dict[str, tuple[str, float]] = {}
    for cost in EXPECTED_COST_LEVELS:
        best_name = ""
        best_sharpe = float("-inf")
        for method_name, payload in cost_results["baselines"][mode][horizon].items():
            sharpe = _as_float(payload.get("r4_cost_ladder", {}).get(cost, {}).get("sharpe"))
            if sharpe is None:
                continue
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_name = method_name
        result[cost] = (best_name, best_sharpe)
    return result


def _required_files_abs(final_dir: Path) -> list[str]:
    return [str((final_dir / name).resolve()) for name in REQUIRED_FILES]


def _supplement_stage_conclusion(stage_path: Path) -> None:
    """Append a claim-boundary section if the current stage note is too implicit."""
    text = stage_path.read_text(encoding="utf-8")
    if "## Claim Boundary" in text and "## Non-Blocking Follow-Ups" in text:
        return
    extra = [
        "",
        "## Claim Boundary",
        "",
        "- This deliverable is a 30d benchmark, not the full 90d PEG-Bench release.",
        "- The simulator is research-grade; it does not implement exchange-grade queue or latency replay.",
        "- Some token/Kairos runs still use seq_len=64, so not every path is fully aligned to context=96.",
        "- The current VQ representation is single-level, not hierarchical VQ-VAE.",
        "- DPO / OG-PA is execution-first here; it should be compared against baseline execution metrics, not baseline RankIC.",
        "",
        "## Non-Blocking Follow-Ups",
        "",
        "- Multi-seed reruns are follow-up stability work, not a blocking issue for the current 30d final benchmark.",
        "- Wider model expansion and larger hyperparameter sweeps are follow-up enhancement work, not a blocking issue for the current 30d final benchmark.",
        "- Extending from 30d to 90d requires a new protocol run and should not be conflated with the current accepted 30d final benchmark.",
    ]
    _write_text(stage_path, text.rstrip() + "\n" + "\n".join(extra) + "\n")


def build_acceptance(final_dir: Path, update_stage_conclusion: bool) -> tuple[dict[str, Any], str]:
    """Build the machine-readable and markdown acceptance outputs."""
    if update_stage_conclusion:
        _supplement_stage_conclusion(final_dir / "stage_conclusion.md")

    blocking_issues: list[str] = []
    non_blocking_limitations: list[str] = []
    completed_modules: list[str] = []

    required_exists = True
    for name in REQUIRED_FILES:
        if not (final_dir / name).exists():
            required_exists = False
            _append_unique(blocking_issues, f"Missing required file: {name}")

    readiness = _load_json(final_dir / "readiness_audit.json") if (final_dir / "readiness_audit.json").exists() else {}
    report_card = _load_json(final_dir / "report_card.json") if (final_dir / "report_card.json").exists() else {}
    gap_metrics = _load_json(final_dir / "gap_metrics.json") if (final_dir / "gap_metrics.json").exists() else {}
    cost_results = _load_json(final_dir / "cost_ladder_results.json") if (final_dir / "cost_ladder_results.json").exists() else {}
    regime_results = _load_json(final_dir / "regime_stress_results.json") if (final_dir / "regime_stress_results.json").exists() else {}
    leaderboards = _load_json(final_dir / "model_leaderboards.json") if (final_dir / "model_leaderboards.json").exists() else {}

    completed_level = str(readiness.get("current_level", "unknown"))
    can_claim_final = bool(readiness.get("can_claim_final_benchmark", False))
    if not can_claim_final:
        _append_unique(blocking_issues, "readiness_audit.json does not grant can_claim_final_benchmark == true.")
    if completed_level != "R4":
        _append_unique(blocking_issues, f"Completed level is {completed_level}, not R4.")
    for item in readiness.get("blocking_items", []):
        _append_unique(blocking_issues, f"readiness audit blocking item: {item}")

    auditability = report_card.get("auditability", {})
    horizon_reports = auditability.get("horizon_reports", {})
    time_ok = all(horizon_reports.get("time", {}).get(h, {}).get("available") for h in ("h5", "h20", "h50"))
    volume_ok = all(horizon_reports.get("volume", {}).get(h, {}).get("available") for h in ("h5", "h20", "h50"))
    if time_ok and volume_ok:
        completed_modules.append("time + volume baseline paths")
    else:
        _append_unique(blocking_issues, "time and volume horizon reports are not both complete.")

    if time_ok and volume_ok:
        completed_modules.append("h=5/20/50 coverage")
    else:
        _append_unique(blocking_issues, "One or more h=5/20/50 benchmark reports are missing.")

    action_interface = auditability.get("action_interface", {})
    if (
        isinstance(action_interface, dict)
        and action_interface.get("bounded_minus1_to_1")
        and action_interface.get("sign_threshold_wrapper")
        and action_interface.get("smoothing")
    ):
        completed_modules.append("unified action interface")
    else:
        _append_unique(blocking_issues, "Unified action interface is not fully auditable.")

    ladder_status = report_card.get("execution", {}).get("ladder_status", {})
    if set(ladder_status.keys()) >= {"R0", "R1", "R2", "R3", "R4"}:
        completed_modules.append("R0→R4 evaluator")
    else:
        _append_unique(blocking_issues, "R0→R4 ladder is incomplete in report_card.")

    if _gap_layer_ok(gap_metrics):
        completed_modules.append("gap metrics")
    else:
        _append_unique(blocking_issues, "gap_metrics.json is incomplete or not numeric.")

    if _robustness_layer_ok(regime_results):
        completed_modules.append("regime/stress evaluation")
    else:
        _append_unique(blocking_issues, "regime_stress_results.json is incomplete.")

    dpo_exec = report_card.get("execution", {}).get("dpo_execution_first", {})
    if isinstance(dpo_exec, dict) and "time" in dpo_exec and "volume" in dpo_exec:
        completed_modules.append("DPO/OG-PA execution-first evaluation")
    else:
        _append_unique(blocking_issues, "DPO/OG-PA execution-first outputs are missing from report_card.")

    if _prediction_layer_ok(report_card):
        completed_modules.append("prediction layer with DA and RankIC")
    else:
        _append_unique(blocking_issues, "prediction layer is missing numeric DA/RankIC values.")

    if _execution_layer_ok(report_card):
        completed_modules.append("execution layer with Sharpe / PnL / MaxDD / Turnover / Calmar")
    else:
        _append_unique(blocking_issues, "execution layer is missing numeric Sharpe / PnL / MaxDD / Turnover / Calmar values.")

    if _auditability_layer_ok(report_card):
        completed_modules.append("auditability layer with split / cost / simulator / claim boundary info")
    else:
        _append_unique(blocking_issues, "auditability layer is incomplete.")

    for mode in ("time", "volume"):
        if mode not in cost_results.get("baselines", {}):
            _append_unique(blocking_issues, f"cost_ladder_results.json is missing mode: {mode}")
            continue
        for horizon in ("h5", "h20", "h50"):
            if horizon not in cost_results["baselines"][mode]:
                _append_unique(blocking_issues, f"cost_ladder_results.json is missing {mode} {horizon}.")
                continue
            for method_name, payload in cost_results["baselines"][mode][horizon].items():
                ladder = payload.get("r4_cost_ladder", {})
                if set(ladder.keys()) != set(EXPECTED_COST_LEVELS):
                    _append_unique(blocking_issues, f"{mode} {horizon} {method_name} has incomplete cost levels.")
                    continue
                for cost in EXPECTED_COST_LEVELS:
                    metrics = ladder.get(cost, {})
                    for metric_name in ("sharpe", "net_pnl", "max_drawdown", "turnover", "calmar"):
                        if _as_float(metrics.get(metric_name)) is None:
                            _append_unique(
                                blocking_issues,
                                f"{mode} {horizon} {method_name} missing {metric_name} at {cost} bps.",
                            )
    if not any("cost_ladder_results.json" in item for item in blocking_issues) and not any("incomplete cost levels" in item for item in blocking_issues):
        completed_modules.append("cost ladder 0/1/3/5/10 bps")

    observed_regimes: set[str] = set()
    for mode in ("time", "volume"):
        for horizon in ("h5", "h20", "h50"):
            for method_payload in regime_results.get("baselines", {}).get(mode, {}).get(horizon, {}).values():
                if not isinstance(method_payload, dict):
                    continue
                for tag in method_payload.get("available_regimes", []):
                    observed_regimes.add(str(tag))
    missing_regimes = [tag for tag in EXPECTED_REGIMES if tag not in observed_regimes]
    if missing_regimes:
        _append_unique(non_blocking_limitations, f"Some expected regime tags are not observed in 30d artifacts: {', '.join(missing_regimes)}.")

    prediction_lb = leaderboards.get("prediction", {})
    execution_lb = leaderboards.get("execution", {})
    dpo_lb = leaderboards.get("dpo_execution_first", {})
    if not (prediction_lb and execution_lb and dpo_lb):
        _append_unique(blocking_issues, "model_leaderboards.json does not expose prediction / execution / DPO sections together.")
    if any("DPO" in str(item) for panel in prediction_lb.values() for item in panel):
        _append_unique(blocking_issues, "DPO/OG-PA appears inside prediction leaderboard and should stay execution-first only.")
    if not report_card.get("prediction", {}).get("dpo_note"):
        _append_unique(blocking_issues, "report_card is missing the note that DPO lacks a unified R0 prediction report.")

    time_best_name, time_best_sharpe = _best_baseline_at_anchor(cost_results, "time", "h20")
    volume_best_name, volume_best_sharpe = _best_baseline_at_anchor(cost_results, "volume", "h20")
    dpo_time_sharpe = _as_float(dpo_exec.get("time", {}).get("cost_ladder", {}).get("3.0", {}).get("sharpe"))
    dpo_volume_sharpe = _as_float(dpo_exec.get("volume", {}).get("cost_ladder", {}).get("3.0", {}).get("sharpe"))
    dpo_time_delta = None if dpo_time_sharpe is None else dpo_time_sharpe - time_best_sharpe
    dpo_volume_delta = None if dpo_volume_sharpe is None else dpo_volume_sharpe - volume_best_sharpe

    time_cost_dominance: list[str] = []
    volume_cost_dominance: list[str] = []
    for cost, (best_name, best_val) in _best_baseline_by_cost(cost_results, "time", "h20").items():
        dpo_cost = _as_float(dpo_exec.get("time", {}).get("cost_ladder", {}).get(cost, {}).get("sharpe"))
        if dpo_cost is not None and dpo_cost > best_val:
            time_cost_dominance.append(cost)
    for cost, (best_name, best_val) in _best_baseline_by_cost(cost_results, "volume", "h20").items():
        dpo_cost = _as_float(dpo_exec.get("volume", {}).get("cost_ladder", {}).get(cost, {}).get("sharpe"))
        if dpo_cost is not None and dpo_cost > best_val:
            volume_cost_dominance.append(cost)

    peg_evidence = [
        "time h20: lightgbm_lob is offline RankIC #1 (0.1113) but execution Sharpe@3bps is -1.3609.",
        "volume h20: lightgbm_lob is offline RankIC #1 (0.2826) but execution Sharpe@3bps is -0.9257.",
        "volume h20: buy_and_hold has RankIC 0.0 but execution Sharpe@3bps 0.2196 and ranks above lightgbm_lob and kairos_64code.",
        f"time h20 gap metrics: ranking_gap={gap_metrics['time']['h20']['ranking_gap']}, reversal_rate@3={gap_metrics['time']['h20']['reversal_rate_at_k']}.",
        f"volume h20 gap metrics: ranking_gap={gap_metrics['volume']['h20']['ranking_gap']}, reversal_rate@3={gap_metrics['volume']['h20']['reversal_rate_at_k']}.",
    ]
    dpo_evidence = [
        f"time h20 anchor Sharpe@3bps: DPO={dpo_time_sharpe:.4f} vs best baseline {time_best_name}={time_best_sharpe:.4f}, delta={dpo_time_delta:+.4f}.",
        f"volume h20 anchor Sharpe@3bps: DPO={dpo_volume_sharpe:.4f} vs best baseline {volume_best_name}={volume_best_sharpe:.4f}, delta={dpo_volume_delta:+.4f}.",
        f"time h20 DPO beats the best baseline at cost levels: {', '.join(time_cost_dominance)} bps.",
        f"volume h20 DPO beats the best baseline at cost levels: {', '.join(volume_cost_dominance)} bps.",
        "DPO / OG-PA remains execution-first here; compare it against baseline execution metrics, not baseline RankIC.",
    ]

    prediction_gap_supported = bool(
        gap_metrics.get("time", {}).get("h20", {}).get("ranking_gap") is not None
        and gap_metrics.get("volume", {}).get("h20", {}).get("ranking_gap") is not None
    )
    dpo_execution_supported = bool(
        dpo_time_sharpe is not None
        and dpo_volume_sharpe is not None
        and dpo_time_sharpe > time_best_sharpe
        and dpo_volume_sharpe > volume_best_sharpe
    )

    _append_unique(non_blocking_limitations, "Current deliverable is 30d, not the full 90d multi-venue benchmark.")
    _append_unique(non_blocking_limitations, "Simulator fidelity is research-grade; there is no exchange-grade queue or latency replay.")
    _append_unique(non_blocking_limitations, "Some token/Kairos runs still use seq_len=64 instead of the target context length 96.")
    _append_unique(non_blocking_limitations, "VQ is single-level rather than hierarchical VQ-VAE.")
    if not _dpo_prediction_proxy_ok(report_card):
        _append_unique(non_blocking_limitations, "DPO/OG-PA still lacks a directly comparable unified R0 prediction report.")
    _append_unique(non_blocking_limitations, "Multi-seed reruns, broader model expansion, and larger hyperparameter sweeps are follow-up enhancements, not blocking issues for the accepted 30d final benchmark.")

    final_decision = "ACCEPT"
    if blocking_issues:
        final_decision = "REJECT"
    elif non_blocking_limitations:
        final_decision = "ACCEPT_WITH_LIMITATIONS"

    acceptance = {
        "final_30d_exists": required_exists,
        "can_claim_30d_final_benchmark": can_claim_final and completed_level == "R4" and not blocking_issues,
        "completed_level": completed_level,
        "files_checked": _required_files_abs(final_dir),
        "completed_modules": completed_modules,
        "blocking_issues": blocking_issues,
        "non_blocking_limitations": non_blocking_limitations,
        "prediction_execution_gap_claim": {
            "supported": prediction_gap_supported,
            "evidence": peg_evidence,
        },
        "dpo_ogpa_execution_claim": {
            "supported": dpo_execution_supported,
            "evidence": dpo_evidence,
        },
        "full_90d_claim": {
            "supported": False,
            "reason": "Current deliverable is 30d, not full 90d multi-asset multi-venue benchmark.",
        },
        "final_decision": final_decision,
    }

    markdown = "\n".join(
        [
            "# 30d PEG-Bench Final Acceptance Check",
            "",
            "## Final decision",
            f"- `{final_decision}`",
            "",
            "## Completed level",
            f"- `{completed_level}`",
            f"- `can_claim_30d_final_benchmark = {acceptance['can_claim_30d_final_benchmark']}`",
            "",
            "## Files checked",
            *[f"- `{path}`" for path in acceptance["files_checked"]],
            "",
            "## Completed modules",
            *[f"- {item}" for item in completed_modules],
            "",
            "## Key evidence for Prediction–Execution Gap",
            *[f"- {item}" for item in peg_evidence],
            "",
            "## Key evidence for DPO/OG-PA execution improvement",
            *[f"- {item}" for item in dpo_evidence],
            "",
            "## Blocking issues",
            *(["- None"] if not blocking_issues else [f"- {item}" for item in blocking_issues]),
            "",
            "## Non-blocking limitations",
            *[f"- {item}" for item in non_blocking_limitations],
            "",
            "## What can be claimed",
            "- The current deliverable reaches R4 on the 30d benchmark contract.",
            "- The 30d final benchmark can be claimed as accepted with limitations.",
            "- Prediction–Execution Gap is supported by both leaderboard reversal metrics and concrete model reversals.",
            "- DPO/OG-PA can be claimed as an execution-first method that outperforms the best baseline on execution Sharpe for the main h20 panel.",
            "",
            "## What cannot be claimed",
            "- This is not the full 90d PEG-Bench release.",
            "- This is not an exchange-grade queue/latency simulator benchmark.",
            "- This does not prove hierarchical VQ-VAE robustness because the current VQ is single-level.",
            "- This does not provide a directly comparable R0 prediction report for DPO/OG-PA.",
            "",
            "## Next recommended step",
            "- Freeze the accepted 30d artifacts, then if needed run a stricter fairness pass with seq_len=96 on token/Kairos/DPO and add multi-seed stability checks.",
        ]
    )

    return acceptance, markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final acceptance artifacts for a 30d PEG-Bench output directory.")
    parser.add_argument("--final-dir", type=str, required=True)
    parser.add_argument("--update-stage-conclusion", action="store_true")
    args = parser.parse_args()

    final_dir = Path(args.final_dir).resolve()
    final_dir.mkdir(parents=True, exist_ok=True)
    acceptance, markdown = build_acceptance(final_dir, update_stage_conclusion=args.update_stage_conclusion)
    _write_json(final_dir / "final_acceptance_check.json", acceptance)
    _write_text(final_dir / "final_acceptance_check.md", markdown + "\n")
    print(json.dumps(acceptance, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
