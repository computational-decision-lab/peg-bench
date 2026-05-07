#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REMOTE_REPO = Path("/root/prediction-execution-gap")
REMOTE_TMP = Path("/root/autodl-tmp")
OUT_DEFAULT = REMOTE_REPO / "outputs" / "final_guaranteed_evidence_pass"

PANEL_REPORTS = [
    ("time_h5", "Time h5", Path("/root/autodl-tmp/peg_bench_runs/tardis_90d_time_h5_table2_compare_v1/benchmark_report.json")),
    ("time_h20", "Time h20", Path("/root/autodl-tmp/peg_bench_runs/tardis_90d_time_table2_compare_v1/benchmark_report.json")),
    ("time_h50", "Time h50", Path("/root/autodl-tmp/peg_bench_runs/tardis_90d_time_h50_table2_compare_v1/benchmark_report.json")),
    ("volume_h5", "Volume h5", Path("/root/autodl-tmp/peg_bench_runs/tardis_90d_volume_h5_table2_compare_v1/benchmark_report.json")),
    ("volume_h20", "Volume h20", Path("/root/autodl-tmp/peg_bench_runs/final_90d_missing_panel_runs/protocol_clean_volume_h20_seq96/benchmark_report.json")),
    ("volume_h50", "Volume h50", Path("/root/autodl-tmp/peg_bench_runs/tardis_90d_volume_h50_table2_compare_v1/benchmark_report.json")),
]

CORE_METHODS = ["lightgbm_lob", "kairos_64code", "ridge_tokens", "histgbm_tokens", "deeplob_timebars"]
DISPLAY = {
    "lightgbm_lob": "LightGBM",
    "kairos_64code": "Kairos-64",
    "ridge_tokens": "Ridge",
    "histgbm_tokens": "HistGBM",
    "deeplob_timebars": "DeepLOB",
}
TASKS = [
    "p0_1_main_results",
    "p0_2_gap_metrics",
    "p0_3_cost_ladder",
    "p0_4_bootstrap_ci",
    "p0_5_wrapper_matrix",
    "p0_6_formal_boundary",
    "p1_7_per_asset_venue",
    "p1_8_hft_audit",
    "p1_9_stability",
]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def ensure_dirs(out: Path) -> None:
    for sub in ["logs", "blocker_reports", "p0_main_results", "p0_gap_metrics", "p0_cost_ladder", "p0_bootstrap_ci", "p0_wrapper_matrix", "p0_formal_boundary", "p1_per_asset_venue", "p1_hft_audit", "p1_stability"]:
        (out / sub).mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8", errors="ignore") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8", errors="ignore") as f:
        return json.load(f)


def ffloat(v: Any, default: float = 0.0) -> float:
    try:
        if isinstance(v, dict):
            if "mean" in v:
                return float(v["mean"])
            if "value" in v:
                return float(v["value"])
        if v in ("", None):
            return default
        return float(v)
    except Exception:
        return default


def fmt(v: Any, nd: int = 6) -> str:
    try:
        x = float(v)
    except Exception:
        return "" if v is None else str(v)
    if abs(x) >= 1000 or (abs(x) < 1e-4 and x != 0):
        return f"{x:.6e}"
    return f"{x:.{nd}f}".rstrip("0").rstrip(".")


def display(method: str) -> str:
    return DISPLAY.get(method, method)


def metric_mean(obj: dict[str, Any], key: str) -> float:
    return ffloat(obj.get(key), 0.0)


def report_for(panel_key: str) -> tuple[str, Path, dict[str, Any]]:
    for key, label, path in PANEL_REPORTS:
        if key == panel_key:
            return label, path, read_json(path)
    raise KeyError(panel_key)


def all_reports() -> list[tuple[str, str, Path, dict[str, Any]]]:
    out = []
    for key, label, path in PANEL_REPORTS:
        out.append((key, label, path, read_json(path)))
    return out


def latex_table(path: Path, rows: list[dict[str, Any]], caption: str, label: str) -> None:
    if not rows:
        path.write_text("% no rows\n", encoding="utf-8")
        return
    headers = list(rows[0])
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\begin{tabular}{@{}" + "l" * len(headers) + "@{}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(row.get(h, "")) for h in headers) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def sha256_manifest(out: Path) -> None:
    rows = []
    for p in sorted(out.rglob("*")):
        if not p.is_file() or p.name == "sha256_manifest.txt":
            continue
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        rows.append(f"{h.hexdigest()}  {p.relative_to(out)}")
    (out / "sha256_manifest.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def find_candidates(names: list[str], roots: list[Path]) -> list[Path]:
    hits: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for name in names:
            direct = root / name
            if direct.exists():
                hits.append(direct)
            try:
                hits.extend(root.rglob(name))
            except Exception:
                pass
    seen = set()
    uniq = []
    for p in hits:
        s = str(p)
        if s not in seen:
            seen.add(s)
            uniq.append(p)
    return sorted(uniq, key=lambda p: (p.stat().st_mtime if p.exists() else 0, str(p)), reverse=True)


def blocker(out: Path, task: str, reason: str, attempted: list[str]) -> None:
    path = out / "blocker_reports" / f"{task}_blocker.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Blocker Report\n\n"
        f"- task: {task}\n"
        f"- time: {now()}\n"
        f"- reason: {reason}\n"
        "\n## Attempted Fixes\n"
        + "".join(f"- {a}\n" for a in attempted)
        + "\n",
        encoding="utf-8",
    )


def preflight(out: Path) -> None:
    ensure_dirs(out)
    inventory = []
    for key, label, path in PANEL_REPORTS:
        inventory.append({
            "artifact_type": "canonical_benchmark_report",
            "panel": key,
            "path": str(path),
            "exists": str(path.exists()).lower(),
            "size_bytes": path.stat().st_size if path.exists() else "",
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)) if path.exists() else "",
        })
    patterns = [
        "cost_ladder_panel_results_90d.csv",
        "action_path_metric_bootstrap_ci.csv",
        "action_path_gap_bootstrap_ci.csv",
        "wrapper_audit_results.csv",
        "wrapper_audit_results.partial.csv",
        "per_asset_venue_breakdown.csv",
        "hftbacktest_fixed_action_sensitivity.csv",
        "results_table.csv",
        "dpo_backtest*.json",
    ]
    for pat in patterns:
        for p in find_candidates([pat], [REMOTE_REPO / "outputs", REMOTE_TMP / "pegbench_ddl_sprint_v25", REMOTE_TMP / "peg_bench_runs"])[0:80]:
            inventory.append({
                "artifact_type": pat,
                "panel": "",
                "path": str(p),
                "exists": "true",
                "size_bytes": p.stat().st_size,
                "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)),
            })
    write_csv(out / "source_inventory.csv", inventory)

    scripts = [
        "reproduce_main_tables.py",
        "run_peg_bench_wrapper_audit.py",
        "slice_and_bootstrap_tables.py",
        "hft_execution_audit.py",
        "run_reviewer_target_baselines.py",
        "run_peg_bench_dpo_backtest.py",
        "final_guaranteed_evidence_pass.py",
    ]
    lines = ["# Script Existence Check\n"]
    for s in scripts:
        p = REMOTE_REPO / "scripts" / s
        lines.append(f"- `{s}`: {'present' if p.exists() else 'missing'}")
    lines.append("\nNo training scripts are invoked by this pass; only read-only aggregation/reconstruction is allowed.\n")
    (out / "script_existence_check.md").write_text("\n".join(lines), encoding="utf-8")

    lines = ["# Benchmark Report Path Resolution\n"]
    for key, label, path in PANEL_REPORTS:
        lines.append(f"- {key}: `{path}` exists={path.exists()}")
    (out / "benchmark_report_path_resolution.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    plan = [
        "# Final Task Plan",
        "",
        "Execution order: P0-1 -> P0-6, then P1-7 -> P1-9. P2 is not run.",
        "",
    ]
    for idx, task in enumerate(TASKS, 1):
        plan.append(f"{idx}. {task}: read-only aggregation/reconstruction, 45 minute max in runner.")
    (out / "task_plan_final.md").write_text("\n".join(plan) + "\n", encoding="utf-8")


def p0_1_main_results(out: Path) -> None:
    d = out / "p0_main_results"
    rows, winner_rows, scatter_rows, trace = [], [], [], []
    for panel, label, path, report in all_reports():
        methods = report.get("methods", {})
        r0_candidates = []
        for method, obj in methods.items():
            pred = obj.get("prediction", {})
            execm = obj.get("execution", {})
            rob = obj.get("robustness", {})
            rank_ic = metric_mean(pred, "rank_ic")
            da = metric_mean(pred, "directional_accuracy")
            sharpe = metric_mean(execm, "post_cost_sharpe")
            turnover = metric_mean(execm, "turnover")
            r0_candidates.append((method, rank_ic, da))
            scatter_rows.append({
                "panel": panel,
                "method": method,
                "display_method": display(method),
                "rank_ic": rank_ic,
                "directional_accuracy": da,
                "r4_sharpe_3bps": sharpe,
                "max_drawdown": metric_mean(execm, "max_drawdown"),
                "turnover": turnover,
                "worst_regime_sharpe": metric_mean(rob, "worst_regime_sharpe"),
                "stress_day_survival": metric_mean(rob, "stress_day_survival"),
                "source_artifact": str(path),
            })
        r0_method, r0_rankic, r0_da = max(r0_candidates, key=lambda x: x[1])
        active_rows = []
        for row in report.get("leaderboard", []):
            method = row.get("method", "")
            if method not in CORE_METHODS:
                continue
            if abs(ffloat(row.get("turnover"), 0.0)) <= 1e-12:
                continue
            active_rows.append(row)
        if not active_rows:
            active_rows = [r for r in report.get("leaderboard", []) if r.get("method") in CORE_METHODS]
        r4 = max(active_rows, key=lambda x: ffloat(x.get("post_cost_sharpe"), -1e18))
        r4_method = r4.get("method", "")
        r4_sharpe = ffloat(r4.get("post_cost_sharpe"), 0.0)
        verdict = ("Same" if r4_method == r0_method else "Changed") + (", positive" if r4_sharpe > 0 else ", negative")
        rows.append({
            "Panel": label,
            "R0 selected": display(r0_method),
            "RankIC": fmt(r0_rankic, 3),
            "DA (%)": fmt(100 * r0_da, 1),
            "R4 selected": display(str(r4_method)),
            "R4 Sharpe": fmt(r4_sharpe, 3),
            "Deployment verdict": verdict,
        })
        winner_rows.append({
            "panel": panel,
            "panel_label": label,
            "r0_top_method": r0_method,
            "r0_top_rank_ic": r0_rankic,
            "r0_top_directional_accuracy": r0_da,
            "clean_active_r4_top_method": r4_method,
            "clean_active_r4_sharpe_3bps": r4_sharpe,
            "winner_changed": str(r4_method != r0_method).lower(),
            "r4_negative_after_costs": str(r4_sharpe < 0).lower(),
            "source_artifact": str(path),
        })
        trace.append({
            "panel": panel,
            "source_artifact": str(path),
            "r0_rule": "max prediction.rank_ic across methods in canonical benchmark_report",
            "r4_rule": "max post_cost_sharpe among available core methods with nonzero turnover",
        })
    write_csv(d / "r0_r4_comparison.csv", rows)
    latex_table(d / "r0_r4_comparison.tex", rows, "R0 predictor selection versus clean active R4 deployment selection.", "tab:90d_r0_r4_comparison")
    write_csv(d / "winner_shift_source.csv", winner_rows)
    write_csv(d / "rankic_vs_r4_sharpe_source.csv", scatter_rows)
    write_csv(d / "source_trace_r0_r4.csv", trace)
    changed = sum(1 for r in winner_rows if r["winner_changed"] == "true")
    unchanged_negative = sum(1 for r in winner_rows if r["winner_changed"] == "false" and r["r4_negative_after_costs"] == "true")
    lightgbm_r0 = sum(1 for r in winner_rows if r["r0_top_method"] == "lightgbm_lob")
    (d / "source_trace_r0_r4.md").write_text(
        "# R0/R4 Source Trace\n\n"
        f"- panels: {len(winner_rows)}\n"
        f"- LightGBM R0 top count: {lightgbm_r0}/6\n"
        f"- clean active R4 winner changed: {changed}/6\n"
        f"- unchanged but negative after costs: {unchanged_negative}\n"
        "- rule: no values copied from paper prose.\n",
        encoding="utf-8",
    )


def p0_2_gap_metrics(out: Path) -> None:
    d = out / "p0_gap_metrics"
    rows, trace = [], []
    for panel, label, path, report in all_reports():
        gap = report.get("gap", {})
        rr = gap.get("reversal_rate_at_k", "")
        if isinstance(rr, dict):
            rr_val = rr.get("1", rr.get(1, rr.get("k1", "")))
            if rr_val == "" and len(rr) == 1:
                rr_val = next(iter(rr.values()))
        else:
            rr_val = rr
        rows.append({
            "Panel": label,
            "Metric Gap": fmt(gap.get("metric_gap", ""), 3),
            "Ranking Gap": fmt(gap.get("ranking_gap", ""), 3),
            "RR@1": fmt(rr_val, 3),
        })
        trace.append({"panel": panel, "source_artifact": str(path), "gap_keys": ",".join(gap.keys())})
    write_csv(d / "gap_metrics.csv", rows)
    latex_table(d / "gap_metrics.tex", rows, "Prediction--execution gap metrics under the official deterministic ranking policy.", "tab:90d_gap_metrics")
    write_csv(d / "gap_metrics_source_trace.csv", trace)
    all_rr = all(str(r["RR@1"]) in {"1", "1.0", "1.000"} for r in rows)
    all_neg = all(ffloat(r["Ranking Gap"], 0) < 0 for r in rows)
    (d / "gap_metrics_source_trace.md").write_text(
        "# Gap Metrics Source Trace\n\n"
        f"- rows: {len(rows)}\n"
        f"- RR@1 all one: {all_rr}\n"
        f"- Ranking Gap all negative: {all_neg}\n"
        "- Table 2 clean-active winner shift and Table 3 RR@1 use different ranking universes/policies; this is a caption/definition distinction, not a contradiction.\n",
        encoding="utf-8",
    )


def cost_metrics_for(execution: dict[str, Any], cost: str) -> dict[str, Any]:
    ladders = ["cost_ladder", "r4_cost_ladder", "r3_cost_ladder", "r2_cost_ladder", "r1_cost_ladder"]
    keys = [cost, f"{float(cost):.1f}", str(int(float(cost))) if float(cost).is_integer() else cost]
    for ladder_name in ladders:
        ladder = execution.get(ladder_name, {})
        if isinstance(ladder, list):
            for row in ladder:
                if str(row.get("cost_bps", row.get("cost", ""))) in keys:
                    return row
        if isinstance(ladder, dict):
            for k in keys:
                if k in ladder and isinstance(ladder[k], dict):
                    return ladder[k]
    if float(cost) == 3.0:
        return execution
    return {}


def p0_3_cost_ladder(out: Path) -> None:
    d = out / "p0_cost_ladder"
    costs = ["0", "1", "3", "5", "10"]
    panel_rows, leader_rows, gap_rows, trace = [], [], [], []
    for panel, label, path, report in all_reports():
        methods = report.get("methods", {})
        r0_top = max(methods, key=lambda m: metric_mean(methods[m].get("prediction", {}), "rank_ic"))
        for method in CORE_METHODS:
            if method not in methods:
                continue
            execm = methods[method].get("execution", {})
            for cost in costs:
                cm = cost_metrics_for(execm, cost)
                if not cm:
                    continue
                panel_rows.append({
                    "panel": panel,
                    "method": method,
                    "cost_bps": cost,
                    "post_cost_sharpe": cm.get("post_cost_sharpe", cm.get("sharpe", "")),
                    "max_drawdown": cm.get("max_drawdown", ""),
                    "turnover": cm.get("turnover", ""),
                    "calmar": cm.get("calmar", ""),
                    "net_return": cm.get("net_return", cm.get("net_pnl", "")),
                    "source_artifact": str(path),
                })
        for cost in costs:
            candidates = [r for r in panel_rows if r["panel"] == panel and r["cost_bps"] == cost and abs(ffloat(r.get("turnover"), 0.0)) > 1e-12]
            if not candidates:
                candidates = [r for r in panel_rows if r["panel"] == panel and r["cost_bps"] == cost]
            if not candidates:
                continue
            best = max(candidates, key=lambda r: ffloat(r.get("post_cost_sharpe"), -1e18))
            leader_rows.append({"panel": panel, "cost_bps": cost, "top_clean_active_method": best["method"], "top_clean_active_sharpe": best["post_cost_sharpe"], "source_artifact": str(path)})
            gap_rows.append({"panel": panel, "cost_bps": cost, "r0_top_method": r0_top, "r4_top_active_method": best["method"], "winner_changed_vs_r0": "yes" if best["method"] != r0_top else "no", "top_active_sharpe": best["post_cost_sharpe"], "source_artifact": str(path)})
        trace.append({"panel": panel, "source_artifact": str(path), "scope": "available core methods only; values read from report execution cost ladder, not inferred from 3bps"})
    write_csv(d / "cost_ladder_panel_results_90d.csv", panel_rows)
    write_csv(d / "cost_ladder_leaderboards_90d.csv", leader_rows)
    write_csv(d / "cost_ladder_gap_metrics_90d.csv", gap_rows)
    write_csv(d / "cost_ladder_source_trace.csv", trace)
    rows = [{"Scope": "six panels x available core methods", "Panel rows": len(panel_rows), "Leaderboard rows": len(leader_rows), "Gap rows": len(gap_rows)}]
    latex_table(d / "cost_ladder_summary.tex", rows, "Cost ladder coverage for available core methods.", "tab:cost_ladder_summary")
    (d / "cost_ladder_audit.md").write_text(
        "# Cost Ladder Audit\n\n- costs: 0/1/3/5/10 bps\n- scope: six completed panels and available core methods only\n- no future method families included\n- non-3bps values are read from normalized report cost ladders, not inferred from 3bps.\n",
        encoding="utf-8",
    )


def copy_first(outdir: Path, candidates: list[Path], dst_name: str) -> Path | None:
    if not candidates:
        return None
    src = candidates[0]
    dst = outdir / dst_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return src


def p0_4_bootstrap_ci(out: Path) -> None:
    d = out / "p0_bootstrap_ci"
    roots = [REMOTE_REPO / "outputs", REMOTE_TMP / "pegbench_ddl_sprint_v25", REMOTE_TMP / "peg_bench_runs"]
    mapping = {
        "day_block_metric_ci.csv": ["day_block_metric_ci.csv", "day_block_metric_ci_from_daily_reports.csv", "action_path_metric_bootstrap_ci.csv"],
        "day_block_gap_ci.csv": ["day_block_gap_ci.csv", "day_block_gap_ci_from_daily_reports.csv", "action_path_gap_bootstrap_ci.csv"],
        "action_path_metric_bootstrap_ci.csv": ["action_path_metric_bootstrap_ci.csv", "day_block_metric_ci.csv"],
        "action_path_gap_bootstrap_ci.csv": ["action_path_gap_bootstrap_ci.csv", "day_block_gap_ci.csv"],
    }
    trace, missing = [], []
    for dst, names in mapping.items():
        cands = find_candidates(names, roots)
        src = copy_first(d, cands, dst)
        if src:
            trace.append({"output": dst, "source_artifact": str(src), "rows": len(read_csv(d / dst))})
        else:
            missing.append(dst)
    metric_rows = read_csv(d / "action_path_metric_bootstrap_ci.csv")
    gap_rows = read_csv(d / "action_path_gap_bootstrap_ci.csv")
    latex_table(d / "bootstrap_summary.tex", [{"family": "metric_ci", "rows": len(metric_rows)}, {"family": "gap_ci", "rows": len(gap_rows)}], "Bootstrap CI coverage.", "tab:bootstrap_summary")
    write_csv(d / "bootstrap_source_trace.csv", trace)
    (d / "bootstrap_audit.md").write_text(
        "# Bootstrap Audit\n\n"
        "- resampling unit: day-block / action-path report-card blocks as recorded in CSV `block_unit`.\n"
        "- not raw order-level bootstrap.\n"
        f"- missing outputs: {', '.join(missing) if missing else 'none'}\n",
        encoding="utf-8",
    )
    if missing:
        blocker(out, "p0_4_bootstrap_ci", "Some bootstrap source files unavailable", [f"searched {names}" for names in mapping.values()])


def infer_panel(row: dict[str, Any]) -> str:
    if row.get("panel"):
        return str(row["panel"])
    mode = str(row.get("mode", ""))
    target = str(row.get("target_key", ""))
    if mode and target.startswith("return_"):
        return f"{mode}_h{target.split('_')[-1]}"
    return ""


def p0_5_wrapper_matrix(out: Path) -> None:
    d = out / "p0_wrapper_matrix"
    roots = [REMOTE_REPO / "outputs", REMOTE_TMP / "pegbench_ddl_sprint_v25", REMOTE_TMP / "peg_bench_runs"]
    paths = find_candidates(["wrapper_audit_results.csv", "wrapper_audit_results.partial.csv", "wrapper_all_available_rows_live.csv"], roots)
    rows = []
    for p in paths:
        for row in read_csv(p):
            row = dict(row)
            row.setdefault("source_artifact", str(p))
            row["partial_flag"] = row.get("partial_flag") or ("partial" if ".partial." in p.name else "complete")
            row["panel"] = infer_panel(row)
            turnover = ffloat(row.get("test_turnover_3bps"), 0.0)
            sharpe = ffloat(row.get("test_sharpe_3bps"), 0.0)
            row["no_trade_collapse_flag"] = "true" if abs(turnover) < 1e-12 and abs(sharpe) < 1e-12 else "false"
            rows.append(row)
    write_csv(d / "wrapper_matrix_all_rows.csv", rows)
    best, raw = {}, {}
    for row in rows:
        key = (row.get("panel", ""), row.get("method", ""))
        if row.get("variant") == "raw_sign":
            raw[key] = row
        if key not in best or ffloat(row.get("test_sharpe_3bps"), -1e18) > ffloat(best[key].get("test_sharpe_3bps"), -1e18):
            best[key] = row
    best_rows, delta_rows, activity = [], [], []
    for key, row in sorted(best.items()):
        panel, method = key
        raw_row = raw.get(key, {})
        best_rows.append({
            "panel": panel, "method": method, "best_variant": row.get("variant", ""),
            "best_test_sharpe_3bps": row.get("test_sharpe_3bps", ""), "best_test_turnover_3bps": row.get("test_turnover_3bps", ""),
            "best_test_max_drawdown_3bps": row.get("test_max_drawdown_3bps", ""), "best_test_net_return_3bps": row.get("test_net_return_3bps", ""),
            "val_score": row.get("val_score", ""), "selected_delta": row.get("selected_delta", ""), "selected_deadzone": row.get("selected_deadzone", ""),
            "selected_rho": row.get("selected_rho", ""), "selected_turnover_lambda": row.get("selected_turnover_lambda", ""),
            "partial_flag": row.get("partial_flag", ""), "no_trade_collapse_flag": row.get("no_trade_collapse_flag", ""), "source_artifact": row.get("source_artifact", ""),
        })
        delta_rows.append({
            "panel": panel, "method": method, "raw_test_sharpe_3bps": raw_row.get("test_sharpe_3bps", ""),
            "best_variant": row.get("variant", ""), "best_test_sharpe_3bps": row.get("test_sharpe_3bps", ""),
            "delta_sharpe_3bps": ffloat(row.get("test_sharpe_3bps"), 0.0) - ffloat(raw_row.get("test_sharpe_3bps"), 0.0) if raw_row else "",
            "raw_turnover_3bps": raw_row.get("test_turnover_3bps", ""), "best_turnover_3bps": row.get("test_turnover_3bps", ""),
            "partial_flag": row.get("partial_flag", ""), "source_artifact": row.get("source_artifact", ""),
        })
    for row in rows:
        turnover = ffloat(row.get("test_turnover_3bps"), 0.0)
        activity.append({
            "panel": row.get("panel", ""), "method": row.get("method", ""), "variant": row.get("variant", ""),
            "test_sharpe_3bps": row.get("test_sharpe_3bps", ""), "turnover": row.get("test_turnover_3bps", ""),
            "max_drawdown": row.get("test_max_drawdown_3bps", ""), "net_return": row.get("test_net_return_3bps", ""),
            "zero_action_fraction": "1.0" if abs(turnover) < 1e-12 else "",
            "nonzero_action_fraction": "0.0" if abs(turnover) < 1e-12 else "",
            "action_path_source": row.get("action_npz", ""),
            "source_artifact": row.get("source_artifact", ""),
            "no_trade_collapse_flag": row.get("no_trade_collapse_flag", ""),
        })
    write_csv(d / "wrapper_best_by_panel_method.csv", best_rows)
    write_csv(d / "wrapper_raw_vs_best_summary.csv", delta_rows)
    write_csv(d / "wrapper_action_activity_audit.csv", activity)
    latex_table(d / "wrapper_matrix_summary.tex", [{"rows": len(rows), "best_rows": len(best_rows), "raw_vs_best_rows": len(delta_rows)}], "Wrapper matrix coverage.", "tab:wrapper_matrix_summary")
    (d / "wrapper_source_trace.md").write_text(
        "# Wrapper Source Trace\n\n"
        f"- source files: {len(paths)}\n- rows: {len(rows)}\n"
        "- raw/wrapped/clean-active/canonical R4 are not mixed; wrapper rows are validation-selected wrapper audit rows.\n"
        "- sanity checks around 0.629/0.927 are not hard-filled; exact values come from source rows.\n",
        encoding="utf-8",
    )


def p0_6_formal_boundary(out: Path) -> None:
    d = out / "p0_formal_boundary"
    roots = [REMOTE_REPO / "outputs", REMOTE_TMP / "pegbench_ddl_sprint_v25", REMOTE_TMP / "peg_bench_runs"]
    paths = []
    for pat in ["dpo_backtest*.json", "*fixliq*.json"]:
        paths += find_candidates([pat], roots)
    seen, rows = set(), []
    for p in paths:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        try:
            obj = read_json(p)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        selected = obj.get("selected_candidate") if isinstance(obj.get("selected_candidate"), dict) else {}
        evals = obj.get("evaluation") if isinstance(obj.get("evaluation"), dict) else {}
        report_card = obj.get("report_card", {}) if isinstance(obj.get("report_card"), dict) else {}
        cost3 = evals.get("3.0") or evals.get("3") or report_card.get("execution", {})
        if not isinstance(cost3, dict):
            cost3 = {}
        risk = selected.get("risk_mode", obj.get("risk_mode", obj.get("selected_risk_mode", "")))
        scale = selected.get("scale", obj.get("scale", obj.get("selected_scale", "")))
        deadzone = selected.get("deadzone", obj.get("deadzone", obj.get("selected_deadzone", "")))
        sharpe = cost3.get("sharpe", cost3.get("post_cost_sharpe", obj.get("cost3_sharpe", "")))
        turnover = cost3.get("turnover", obj.get("cost3_turnover", ""))
        net = cost3.get("net_pnl", cost3.get("net_return", obj.get("cost3_net_pnl", "")))
        md = cost3.get("max_drawdown", obj.get("cost3_max_drawdown", ""))
        post_fix = "post_fixliq" if ("fixliq" in s or "seq96" in s) else "pre_or_unknown_fixliq"
        status = "diagnostic_failed_or_no_trade" if abs(ffloat(sharpe, 999)) < 1e-12 and abs(ffloat(turnover, 999)) < 1e-12 else "diagnostic_only"
        rows.append({"variant": p.parent.name, "post_fixliq_status": post_fix, "risk_mode": risk, "scale": scale, "deadzone": deadzone, "cost3_sharpe": sharpe, "max_drawdown": md, "turnover": turnover, "net_return_or_pnl": net, "diagnostic_status": status, "source_artifact": s})
    write_csv(d / "formal_method_diagnostic_source_trace.csv", rows)
    write_csv(d / "formal_method_diagnostic_table.csv", rows)
    (d / "formal_method_diagnostic_summary.md").write_text(f"# Formal Method Diagnostic Summary\n\n- rows with source artifacts: {len(rows)}\n- formal method wins: none claimed\n- DPO improves execution: not claimed\n", encoding="utf-8")
    (d / "formal_method_claim_boundary.md").write_text("# Formal Method Claim Boundary\n\nOG-PA/DPO/main_fixliq/rescue2 rows are diagnostic/source-gated only. Post-fix no-trade/all-zero rows are `diagnostic_failed_or_no_trade`. No formal-method wins or DPO execution-improvement claims are supported by this pass.\n", encoding="utf-8")


def p1_7_per_asset_venue(out: Path) -> None:
    d = out / "p1_per_asset_venue"
    roots = [REMOTE_REPO / "outputs", REMOTE_TMP / "pegbench_ddl_sprint_v25", REMOTE_TMP / "peg_bench_runs"]
    src = copy_first(d, find_candidates(["per_asset_venue_breakdown.csv"], roots), "per_asset_venue_breakdown.csv")
    rows = read_csv(d / "per_asset_venue_breakdown.csv")
    write_csv(d / "per_asset_breakdown.csv", [r for r in rows if r.get("grouping") == "asset"])
    write_csv(d / "per_venue_breakdown.csv", [r for r in rows if r.get("grouping") == "venue"])
    latex_table(d / "per_asset_venue_summary.tex", [{"rows": len(rows), "status": "supplemental_fast_replay" if rows else "unavailable"}], "Per-asset and per-venue diagnostic coverage.", "tab:per_asset_venue_summary")
    (d / "per_asset_venue_audit.md").write_text(f"# Per-Asset/Venue Audit\n\n- source: {src or 'unavailable'}\n- rows: {len(rows)}\n- status: supplemental fast-replay diagnostics unless canonical grouped evaluator is present.\n", encoding="utf-8")
    if not rows:
        blocker(out, "p1_7_per_asset_venue", "per_asset_venue_breakdown.csv unavailable", ["searched remote outputs and autodl tmp"])


def p1_8_hft_audit(out: Path) -> None:
    d = out / "p1_hft_audit"
    roots = [REMOTE_REPO / "outputs", REMOTE_TMP / "pegbench_ddl_sprint_v25", REMOTE_TMP / "peg_bench_runs"]
    src = copy_first(d, find_candidates(["hftbacktest_fixed_action_sensitivity.csv"], roots), "hftbacktest_fixed_action_sensitivity.csv")
    rows = read_csv(d / "hftbacktest_fixed_action_sensitivity.csv")
    latex_table(d / "hftbacktest_audit_summary.tex", [{"rows": len(rows), "status": "audit_only"}], "HftBacktest fixed-action latency/queue audit coverage.", "tab:hftbacktest_audit_summary")
    (d / "hftbacktest_audit_boundary.md").write_text(f"# HftBacktest Audit Boundary\n\n- source: {src or 'unavailable'}\n- rows: {len(rows)}\n- audit-only: yes\n- leaderboard: no\n- full raw-L2 replay completed: no\n", encoding="utf-8")
    write_csv(d / "hftbacktest_source_trace.csv", [{"source_artifact": str(src or ""), "rows": len(rows), "boundary": "audit_only_not_leaderboard"}])
    if not rows:
        blocker(out, "p1_8_hft_audit", "hftbacktest_fixed_action_sensitivity.csv unavailable", ["searched remote outputs and autodl tmp"])


def p1_9_stability(out: Path) -> None:
    d = out / "p1_stability"
    roots = [REMOTE_REPO / "outputs", REMOTE_TMP / "pegbench_ddl_sprint_v25", REMOTE_TMP / "peg_bench_runs"]
    paths = [p for p in find_candidates(["results_table.csv"], roots) if "seed" in str(p).lower() or "split" in str(p).lower()]
    rows = []
    for p in paths:
        for row in read_csv(p):
            r = dict(row)
            r["source_artifact"] = str(p)
            r["panel"] = infer_panel(r) or p.parent.name
            rows.append(r)
    write_csv(d / "seed_stability_headline.csv", rows)
    write_csv(d / "split_stability_headline.csv", [r for r in rows if "split" in r.get("source_artifact", "").lower()])
    latex_table(d / "stability_table.tex", [{"rows": len(rows), "status": "existing_sidecar_only"}], "Existing sidecar seed/split stability coverage.", "tab:stability_table")
    write_csv(d / "stability_source_trace.csv", [{"source_artifact": str(p), "rows": len(read_csv(p))} for p in paths])
    (d / "stability_summary.md").write_text(f"# Stability Summary\n\n- rows: {len(rows)}\n- mode: existing sidecar only; no new seed sweep or model retraining was launched.\n", encoding="utf-8")
    if not rows:
        blocker(out, "p1_9_stability", "No existing seed/split sidecar rows found", ["searched for results_table.csv with seed/split in path"])


def final_reports(out: Path) -> None:
    categories = [
        ("completed", "P0 main/gap/cost/bootstrap/wrapper aggregation where source rows exist"),
        ("supplemental", "per-asset/venue fast replay, target lite/baseline rows, seed sidecars"),
        ("audit_only", "HftBacktest fixed-action latency/queue sensitivity"),
        ("target_only", "official TABL/iTransformer/TimesFM/Kronos/PPO/SAC/full raw-L2 HftBacktest not run"),
    ]
    (out / "completed_vs_supplemental_vs_target.md").write_text("# Completed vs Supplemental vs Target\n\n" + "\n".join(f"- {k}: {v}" for k, v in categories) + "\n", encoding="utf-8")
    (out / "remaining_not_completed.md").write_text("# Remaining Not Completed\n\n- official TABL full 90D\n- official iTransformer full 90D\n- TimesFM\n- Kronos official\n- PPO/SAC full benchmark\n- full raw-L2 HftBacktest replay\n- formal-method creative rescue\n", encoding="utf-8")
    (out / "paper_claim_sync_report.md").write_text("# Paper Claim Sync Report\n\nMain empirical claims must use only generated CSV/TEX rows in this bundle. Motivating-slice protocol-different rows and formal-method diagnostics must not be promoted to completed wins.\n", encoding="utf-8")
    (out / "final_submission_readiness.md").write_text("# Final Submission Readiness\n\nP0 evidence pass is designed for NeurIPS E&D submission claims over completed six-panel evidence and available core methods. P1 evidence is supplemental/audit-only where marked. P2 is not run.\n", encoding="utf-8")
    sha256_manifest(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT_DEFAULT))
    parser.add_argument("--task", required=True, choices=["preflight", *TASKS, "final_reports"])
    args = parser.parse_args()
    out = Path(args.out)
    ensure_dirs(out)
    task_fn = {
        "preflight": preflight,
        "p0_1_main_results": p0_1_main_results,
        "p0_2_gap_metrics": p0_2_gap_metrics,
        "p0_3_cost_ladder": p0_3_cost_ladder,
        "p0_4_bootstrap_ci": p0_4_bootstrap_ci,
        "p0_5_wrapper_matrix": p0_5_wrapper_matrix,
        "p0_6_formal_boundary": p0_6_formal_boundary,
        "p1_7_per_asset_venue": p1_7_per_asset_venue,
        "p1_8_hft_audit": p1_8_hft_audit,
        "p1_9_stability": p1_9_stability,
        "final_reports": final_reports,
    }[args.task]
    task_fn(out)


if __name__ == "__main__":
    main()
