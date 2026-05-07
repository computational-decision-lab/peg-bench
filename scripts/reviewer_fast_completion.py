#!/usr/bin/env python3
"""Fast completion bundle for remaining reviewer-requested evidence.

This script is designed to finish when the heavier window-level replay scripts
are too slow. It uses already-frozen daily report cards and wrapper/action
artifacts, plus completed reviewer target-baseline outputs.
"""

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


def load_daily_panel(root: Path) -> tuple[list[str], dict[str, dict[str, dict[str, float]]]]:
    methods = sorted(path.name for path in (root / "methods").iterdir() if (path / "reports").exists())
    daily: dict[str, dict[str, dict[str, float]]] = {}
    for method in methods:
        for report_path in sorted((root / "methods" / method / "reports").glob("*.json")):
            payload = json.loads(report_path.read_text())
            card = payload["report_card"]
            date = str(payload["date"])
            daily.setdefault(date, {})[method] = {
                "rank_ic": float(card["prediction"].get("rank_ic") or 0.0),
                "r4_sharpe": float(card["execution"].get("post_cost_sharpe") or 0.0),
                "max_drawdown": float(card["execution"].get("max_drawdown") or 0.0),
                "worst_regime_sharpe": float(card["robustness"].get("worst_regime_sharpe") or 0.0),
            }
    dates = sorted(date for date, payload in daily.items() if all(method in payload for method in methods))
    return dates, daily


def kendall_like(scores: dict[str, tuple[float, float]]) -> float:
    names = sorted(scores)
    concordant = 0
    discordant = 0
    for i, left in enumerate(names[:-1]):
        for right in names[i + 1 :]:
            prod = (scores[left][0] - scores[right][0]) * (scores[left][1] - scores[right][1])
            if abs(prod) < 1e-18:
                continue
            if prod > 0:
                concordant += 1
            else:
                discordant += 1
    return float((concordant - discordant) / max(concordant + discordant, 1))


def action_path_bootstrap_from_days(out: Path, draws: int) -> None:
    rng = np.random.default_rng(20260504)
    metric_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    for panel, root_str in PANELS.items():
        root = Path(root_str)
        dates, daily = load_daily_panel(root)
        methods = sorted(next(iter(daily.values())).keys()) if daily else []
        for method in methods:
            for metric in ("r4_sharpe", "max_drawdown", "worst_regime_sharpe"):
                values = []
                for _ in range(draws):
                    sample_dates = rng.choice(dates, size=len(dates), replace=True)
                    values.append(float(np.mean([daily[date][method][metric] for date in sample_dates])))
                mean, low, high = ci(values)
                metric_rows.append(
                    {
                        "panel": panel,
                        "method": method,
                        "metric": metric,
                        "point": float(np.mean([daily[date][method][metric] for date in dates])),
                        "bootstrap_mean": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "block_unit": "test_day_action_path_report",
                        "num_blocks": len(dates),
                        "draws": draws,
                    }
                )
        gap_samples: dict[str, list[float]] = {"ranking_gap": [], "reversal_rate_at_1": [], "metric_gap": []}
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
            gap_samples["reversal_rate_at_1"].append(0.0 if offline[0] == online[0] else 1.0)
            gap_samples["ranking_gap"].append(kendall_like(scores))
            x = np.asarray([scores[name][0] for name in methods], dtype=np.float64)
            y = np.asarray([scores[name][1] for name in methods], dtype=np.float64)
            gap_samples["metric_gap"].append(float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-12 and np.std(y) > 1e-12 else 0.0)
        for metric, values in gap_samples.items():
            mean, low, high = ci(values)
            gap_rows.append(
                {
                    "panel": panel,
                    "metric": metric,
                    "bootstrap_mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "block_unit": "test_day_action_path_report",
                    "num_blocks": len(dates),
                    "draws": draws,
                }
            )
    write_csv(out / "action_path_metric_bootstrap_ci.csv", metric_rows)
    write_csv(out / "action_path_gap_bootstrap_ci.csv", gap_rows)


def target_baseline_summary(out: Path, target_root: Path | None) -> None:
    rows: list[dict[str, Any]] = []
    if target_root is not None and target_root.exists():
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
    parser.add_argument("--draws", type=int, default=1000)
    args = parser.parse_args()
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    target_root = Path(args.target_root) if args.target_root else None
    action_path_bootstrap_from_days(out, draws=args.draws)
    target_baseline_summary(out, target_root)
    (out / "run_metadata.json").write_text(
        json.dumps(
            {
                "draws": args.draws,
                "target_root": str(target_root) if target_root else None,
                "interpretation": "Fast supplemental completion from frozen test-day action-path report cards and reviewer target outputs.",
            },
            indent=2,
        )
    )
    print(out)


if __name__ == "__main__":
    main()
