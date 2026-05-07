"""Helpers for aggregating regime robustness from persisted report cards."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


REQUIRED_REGIMES = (
    "normal",
    "high_vol",
    "trend_up",
    "trend_down",
    "liquidity_stress",
)


def _load_report_card(path: str | Path) -> dict[str, object]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    payload = json.loads(file_path.read_text())
    report_card = payload.get("report_card")
    if isinstance(report_card, dict):
        return report_card
    return payload


def summarize_regime_panel(report_paths: list[str] | tuple[str, ...]) -> dict[str, object]:
    """Summarize regime coverage and variability from day-level report cards."""
    sharpe_by_regime: dict[str, list[float]] = {}
    regime_day_counts: dict[str, int] = {}

    for report_path in report_paths:
        report_card = _load_report_card(report_path)
        robustness = report_card.get("robustness", {})
        if not isinstance(robustness, dict):
            continue
        regime_metrics = robustness.get("regime_metrics", {})
        if not isinstance(regime_metrics, dict):
            continue
        for regime_name, metrics in regime_metrics.items():
            if not isinstance(metrics, dict):
                continue
            sharpe = metrics.get("sharpe")
            if sharpe is None:
                continue
            sharpe_by_regime.setdefault(str(regime_name), []).append(float(sharpe))
            regime_day_counts[str(regime_name)] = regime_day_counts.get(str(regime_name), 0) + 1

    available_regimes = sorted(sharpe_by_regime)
    missing_regimes = [name for name in REQUIRED_REGIMES if name not in sharpe_by_regime]
    per_regime: dict[str, dict[str, float | int]] = {}
    regime_means: list[float] = []
    for regime_name, values in sorted(sharpe_by_regime.items()):
        arr = np.asarray(values, dtype=np.float64)
        mean_value = float(np.mean(arr))
        regime_means.append(mean_value)
        per_regime[regime_name] = {
            "mean_sharpe": mean_value,
            "min_sharpe": float(np.min(arr)),
            "max_sharpe": float(np.max(arr)),
            "num_samples": int(len(arr)),
            "num_days": int(regime_day_counts.get(regime_name, len(arr))),
        }

    if regime_means:
        regime_variance = float(np.var(np.asarray(regime_means, dtype=np.float64)))
        worst_regime_sharpe = float(np.min(np.asarray(regime_means, dtype=np.float64)))
    else:
        regime_variance = None
        worst_regime_sharpe = None

    if available_regimes and not missing_regimes:
        status = "computed"
    elif available_regimes:
        status = "partial"
    else:
        status = "missing"

    return {
        "status": status,
        "required_regimes": list(REQUIRED_REGIMES),
        "available_regimes": available_regimes,
        "missing_regimes": missing_regimes,
        "per_regime": per_regime,
        "worst_regime_sharpe": worst_regime_sharpe,
        "regime_variance": regime_variance,
    }
