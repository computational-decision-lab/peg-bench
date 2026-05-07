"""Coverage utilities for benchmark day-level report collections."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.eval.benchmark_evaluator import BenchmarkMethodSpec, _infer_date, _load_json


def summarize_report_coverage(
    method_specs: list[BenchmarkMethodSpec],
    split_lookup: dict[str, str],
    evaluation_split: str,
) -> dict[str, Any]:
    """Summarize expected-vs-observed daily report coverage for each method."""
    expected_dates = sorted(day for day, split in split_lookup.items() if split == evaluation_split)
    rows: list[dict[str, object]] = []
    methods: dict[str, dict[str, object]] = {}

    for spec in method_specs:
        observed: dict[str, str] = {}
        undated_reports: list[str] = []
        for report in spec.reports:
            payload = _load_json(report)
            date = _infer_date(report, payload)
            if date is None:
                undated_reports.append(str(Path(report).resolve()))
                continue
            observed[date] = str(Path(report).resolve())

        missing_dates = [day for day in expected_dates if day not in observed]
        extra_dates = sorted(day for day in observed if day not in expected_dates)
        coverage_ratio = 1.0 if not expected_dates else (len(expected_dates) - len(missing_dates)) / len(expected_dates)

        for day in expected_dates:
            rows.append(
                {
                    "method": spec.name,
                    "split": evaluation_split,
                    "expected_date": day,
                    "report_found": day in observed,
                    "report_path": observed.get(day, ""),
                }
            )
        for day in extra_dates:
            rows.append(
                {
                    "method": spec.name,
                    "split": "extra",
                    "expected_date": day,
                    "report_found": True,
                    "report_path": observed.get(day, ""),
                }
            )

        methods[spec.name] = {
            "expected_num_days": int(len(expected_dates)),
            "observed_num_days": int(len(observed)),
            "coverage_ratio": float(coverage_ratio),
            "expected_dates": expected_dates,
            "observed_dates": sorted(observed),
            "missing_dates": missing_dates,
            "extra_dates": extra_dates,
            "undated_reports": undated_reports,
        }

    return {
        "evaluation_split": evaluation_split,
        "expected_dates": expected_dates,
        "rows": rows,
        "methods": methods,
    }


def write_report_coverage_outputs(
    summary: dict[str, Any],
    csv_path: str | Path,
    json_path: str | Path,
    markdown_path: str | Path,
) -> dict[str, str]:
    """Persist flat CSV, JSON summary and markdown view of report coverage."""
    csv_target = Path(csv_path)
    json_target = Path(json_path)
    markdown_target = Path(markdown_path)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["method", "split", "expected_date", "report_found", "report_path"]
    with csv_target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary["rows"])

    json_target.write_text(json.dumps(summary, indent=2))

    lines = [
        "# Benchmark Report Coverage",
        "",
        f"- evaluation_split: {summary['evaluation_split']}",
        f"- expected_dates: {', '.join(summary['expected_dates']) if summary['expected_dates'] else 'None'}",
        "",
        "## Method Summary",
        "",
        "| method | expected_num_days | observed_num_days | coverage_ratio | missing_dates | extra_dates | undated_reports |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for method, payload in summary["methods"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    str(payload["expected_num_days"]),
                    str(payload["observed_num_days"]),
                    f"{payload['coverage_ratio']:.4f}",
                    json.dumps(payload["missing_dates"], ensure_ascii=False),
                    json.dumps(payload["extra_dates"], ensure_ascii=False),
                    json.dumps(payload["undated_reports"], ensure_ascii=False),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Flat Coverage Rows",
            "",
            "| method | split | expected_date | report_found | report_path |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    str(row["split"]),
                    str(row["expected_date"]),
                    str(row["report_found"]),
                    str(row["report_path"]),
                ]
            )
            + " |"
        )
    markdown_target.write_text("\n".join(lines) + "\n")

    return {
        "csv_path": str(csv_target.resolve()),
        "json_path": str(json_target.resolve()),
        "markdown_path": str(markdown_target.resolve()),
    }
