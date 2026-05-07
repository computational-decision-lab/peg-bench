"""Helpers for aggregating stress robustness from persisted report cards."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _load_report_card(path: str | Path) -> dict[str, object]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    payload = json.loads(file_path.read_text())
    report_card = payload.get("report_card")
    if isinstance(report_card, dict):
        return report_card
    return payload


def summarize_stress_panel(
    report_paths: list[str] | tuple[str, ...],
    *,
    hard_stop_drawdown: float = 0.10,
    liquidation_drawdown: float = 0.20,
) -> dict[str, object]:
    """Summarize stress survival and drawdown violations from day-level reports."""
    daily_survival: list[float] = []
    daily_drawdown: list[float] = []
    interval_survival: list[float] = []
    interval_drawdowns: list[float] = []

    for report_path in report_paths:
        report_card = _load_report_card(report_path)
        execution = report_card.get("execution", {})
        robustness = report_card.get("robustness", {})
        if isinstance(execution, dict) and execution.get("max_drawdown") is not None:
            daily_drawdown.append(float(execution["max_drawdown"]))
        if isinstance(robustness, dict):
            survival = robustness.get("stress_day_survival")
            if survival is not None:
                daily_survival.append(float(survival))
            intervals = robustness.get("stress_intervals", [])
            if isinstance(intervals, list):
                for interval in intervals:
                    if not isinstance(interval, dict):
                        continue
                    survived = interval.get("survived")
                    if survived is not None:
                        interval_survival.append(1.0 if bool(survived) else 0.0)
                    if interval.get("max_drawdown") is not None:
                        interval_drawdowns.append(float(interval["max_drawdown"]))

    drawdown_violation_rate = None
    liquidation_violation_rate = None
    if daily_drawdown:
        drawdowns = np.asarray(daily_drawdown, dtype=np.float64)
        drawdown_violation_rate = float(np.mean(drawdowns >= hard_stop_drawdown))
        liquidation_violation_rate = float(np.mean(drawdowns >= liquidation_drawdown))

    if interval_survival:
        stress_survival_rate = float(np.mean(np.asarray(interval_survival, dtype=np.float64)))
    elif daily_survival:
        stress_survival_rate = float(np.mean(np.asarray(daily_survival, dtype=np.float64)))
    else:
        stress_survival_rate = None

    if interval_drawdowns:
        max_interval_drawdown = float(np.max(np.asarray(interval_drawdowns, dtype=np.float64)))
    else:
        max_interval_drawdown = None

    if interval_survival or daily_survival:
        status = "computed"
    elif daily_drawdown:
        status = "partial"
    else:
        status = "missing"

    return {
        "status": status,
        "stress_survival_rate": stress_survival_rate,
        "stress_interval_count": int(len(interval_survival)),
        "drawdown_violation_rate": drawdown_violation_rate,
        "liquidation_violation_rate": liquidation_violation_rate,
        "max_interval_drawdown": max_interval_drawdown,
        "hard_stop_drawdown": float(hard_stop_drawdown),
        "liquidation_drawdown": float(liquidation_drawdown),
    }
