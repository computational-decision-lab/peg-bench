#!/usr/bin/env python3
"""Fast reviewer-facing summaries from existing 90D artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


PANELS = {
    "time_h5": "/root/autodl-tmp/peg_bench_runs/tardis_90d_time_h5_table2_compare_v1",
    "time_h20": "/root/autodl-tmp/peg_bench_runs/tardis_90d_time_table2_compare_v1",
    "time_h50": "/root/autodl-tmp/peg_bench_runs/tardis_90d_time_h50_table2_compare_v1",
    "volume_h5": "/root/autodl-tmp/peg_bench_runs/tardis_90d_volume_h5_table2_compare_v1",
    "volume_h20": "/root/autodl-tmp/peg_bench_runs/tardis_90d_volume_table2_compare_v1",
    "volume_h50": "/root/autodl-tmp/peg_bench_runs/tardis_90d_volume_h50_table2_compare_v1",
}


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


def complete_cost_ladder(out: Path) -> None:
    rows: list[dict[str, Any]] = []
    leader_rows: list[dict[str, Any]] = []
    for panel, root_str in PANELS.items():
        report = json.loads((Path(root_str) / "benchmark_report.json").read_text())
        for method, payload in report["methods"].items():
            execution = payload.get("execution", {})
            robustness = payload.get("robustness", {})
            ladder = execution.get("cost_ladder", {})
            for cost, metrics in sorted(ladder.items(), key=lambda item: float(item[0])):
                rows.append(
                    {
                        "panel": panel,
                        "method": method,
                        "cost_bps": float(cost),
                        "r4_sharpe_mean": metrics.get("sharpe", {}).get("mean"),
                        "r4_sharpe_ci_low": (metrics.get("sharpe", {}).get("ci95") or [None, None])[0],
                        "r4_sharpe_ci_high": (metrics.get("sharpe", {}).get("ci95") or [None, None])[1],
                        "max_drawdown_mean": metrics.get("max_drawdown", {}).get("mean"),
                        "turnover_mean": metrics.get("turnover", {}).get("mean"),
                        "net_pnl_mean": metrics.get("net_pnl", {}).get("mean"),
                        "worst_regime_sharpe_mean": robustness.get("worst_regime_sharpe", {}).get("mean"),
                        "stress_day_survival_mean": robustness.get("stress_day_survival", {}).get("mean"),
                    }
                )
        for cost in sorted({row["cost_bps"] for row in rows if row["panel"] == panel}):
            subset = [row for row in rows if row["panel"] == panel and row["cost_bps"] == cost]
            if subset:
                best = max(subset, key=lambda row: float(row["r4_sharpe_mean"] or -1e18))
                leader_rows.append(
                    {
                        "panel": panel,
                        "cost_bps": cost,
                        "winner": best["method"],
                        "winner_r4_sharpe_mean": best["r4_sharpe_mean"],
                    }
                )
    write_csv(out / "cost_ladder_full_from_daily_reports.csv", rows)
    write_csv(out / "cost_ladder_winners_from_daily_reports.csv", leader_rows)


def complete_day_block_ci(out: Path, draws: int = 1000) -> None:
    rng = np.random.default_rng(20260504)
    metric_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    for panel, root_str in PANELS.items():
        root = Path(root_str)
        daily: dict[str, dict[str, dict[str, float]]] = {}
        methods = sorted(path.name for path in (root / "methods").iterdir() if (path / "reports").exists())
        for method in methods:
            for report_path in sorted((root / "methods" / method / "reports").glob("*.json")):
                payload = json.loads(report_path.read_text())
                date = str(payload["date"])
                card = payload["report_card"]
                daily.setdefault(date, {})[method] = {
                    "rank_ic": float(card["prediction"].get("rank_ic") or 0.0),
                    "r4_sharpe": float(card["execution"].get("post_cost_sharpe") or 0.0),
                    "max_drawdown": float(card["execution"].get("max_drawdown") or 0.0),
                    "worst_regime_sharpe": float(card["robustness"].get("worst_regime_sharpe") or 0.0),
                }
        dates = sorted(date for date, item in daily.items() if all(method in item for method in methods))
        for method in methods:
            for metric in ("r4_sharpe", "max_drawdown", "worst_regime_sharpe"):
                point = float(np.mean([daily[date][method][metric] for date in dates]))
                samples = []
                for _ in range(draws):
                    sample_dates = rng.choice(dates, size=len(dates), replace=True)
                    samples.append(float(np.mean([daily[date][method][metric] for date in sample_dates])))
                mean, low, high = ci(samples)
                metric_rows.append(
                    {
                        "panel": panel,
                        "method": method,
                        "metric": metric,
                        "point": point,
                        "bootstrap_mean": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "num_days": len(dates),
                        "draws": draws,
                    }
                )
        for metric in ("ranking_gap", "reversal_rate_at_k", "metric_gap"):
            report = json.loads((root / "benchmark_report.json").read_text())
            point = report.get("gap", {}).get(metric)
            # Daily re-rank bootstrap using mean daily RankIC and mean daily Sharpe.
            values = []
            for _ in range(draws):
                sample_dates = rng.choice(dates, size=len(dates), replace=True)
                scores = {
                    method: (
                        float(np.mean([daily[date][method]["rank_ic"] for date in sample_dates])),
                        float(np.mean([daily[date][method]["r4_sharpe"] for date in sample_dates])),
                    )
                    for method in methods
                }
                offline = [name for name, _ in sorted(scores.items(), key=lambda item: item[1][0], reverse=True)]
                online = [name for name, _ in sorted(scores.items(), key=lambda item: item[1][1], reverse=True)]
                if metric == "reversal_rate_at_k":
                    values.append(0.0 if offline[0] == online[0] else 1.0)
                elif metric == "metric_gap":
                    x = np.asarray([scores[m][0] for m in methods], dtype=float)
                    y = np.asarray([scores[m][1] for m in methods], dtype=float)
                    values.append(float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-12 and np.std(y) > 1e-12 else 0.0)
                else:
                    concordant = 0
                    discordant = 0
                    for i in range(len(methods) - 1):
                        for j in range(i + 1, len(methods)):
                            a, b = methods[i], methods[j]
                            prod = (scores[a][0] - scores[b][0]) * (scores[a][1] - scores[b][1])
                            if abs(prod) < 1e-18:
                                continue
                            if prod > 0:
                                concordant += 1
                            else:
                                discordant += 1
                    denom = max(concordant + discordant, 1)
                    values.append((concordant - discordant) / denom)
            mean, low, high = ci(values)
            gap_rows.append(
                {
                    "panel": panel,
                    "metric": metric,
                    "point": point,
                    "bootstrap_mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "num_days": len(dates),
                    "draws": draws,
                }
            )
    write_csv(out / "day_block_metric_ci_from_daily_reports.csv", metric_rows)
    write_csv(out / "day_block_gap_ci_from_daily_reports.csv", gap_rows)


def complete_wrapper(out: Path) -> None:
    rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    roots = sorted(Path("/root/autodl-tmp/peg_bench_runs/final_90d_missing_panel_runs").glob("wrapper_audit_*"))
    for root in roots:
        csv_path = root / "wrapper_audit_results.csv"
        if not csv_path.exists():
            continue
        inferred_panel = root.name.replace("wrapper_audit_", "")
        for suffix in ("_v2fast", "_v1"):
            inferred_panel = inferred_panel.replace(suffix, "")
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["panel"] = row.get("panel") or inferred_panel
                row["source_file"] = str(csv_path)
                rows.append(row)
                detail = Path(row["detail_json"])
                if detail.exists():
                    payload = json.loads(detail.read_text())
                    ladder = payload.get("test_summary", {}).get("report_card", {}).get("execution", {}).get("r4_cost_ladder", {})
                    for cost, metrics in ladder.items():
                        cost_rows.append(
                            {
                                "panel": row.get("panel"),
                                "method": row.get("method"),
                                "variant": row.get("variant"),
                                "cost_bps": float(cost),
                                "r4_sharpe": metrics.get("sharpe"),
                                "max_drawdown": metrics.get("max_drawdown"),
                                "turnover": metrics.get("turnover"),
                                "net_pnl": metrics.get("net_pnl"),
                            }
                        )
    write_csv(out / "wrapper_matrix_all_rows.csv", rows)
    write_csv(out / "wrapper_matrix_cost_ladder.csv", cost_rows)
    best_rows = []
    keys = sorted({(row.get("panel"), row.get("method")) for row in rows})
    for panel, method in keys:
        group = [row for row in rows if row.get("panel") == panel and row.get("method") == method]
        raw = next((row for row in group if row.get("variant") == "raw_sign"), None)
        best = max(group, key=lambda row: float(row.get("test_sharpe_3bps") or -1e18))
        best_rows.append(
            {
                "panel": panel,
                "method": method,
                "raw_sharpe_3bps": raw.get("test_sharpe_3bps") if raw else None,
                "best_variant": best.get("variant"),
                "best_sharpe_3bps": best.get("test_sharpe_3bps"),
                "delta_best_minus_raw": (
                    float(best.get("test_sharpe_3bps")) - float(raw.get("test_sharpe_3bps"))
                    if raw and raw.get("test_sharpe_3bps") not in (None, "")
                    else None
                ),
                "candidate_count": len(group),
            }
        )
    write_csv(out / "wrapper_raw_vs_best_summary.csv", best_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    complete_cost_ladder(out)
    complete_day_block_ci(out)
    complete_wrapper(out)
    (out / "run_metadata.json").write_text(json.dumps({"panels": PANELS}, indent=2))
    print(out)


if __name__ == "__main__":
    main()
