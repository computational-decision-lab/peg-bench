"""PEG-Bench gap metrics over shared offline and full-realism panels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class GapInput:
    """Single method entry for benchmark gap computation."""

    offline_score: float
    full_realism_utility: float
    regime_utilities: list[float] | None = None


def _stable_corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    corr = np.corrcoef(left, right)[0, 1]
    return 0.0 if np.isnan(corr) else float(corr)


def _kendall_tau(left: np.ndarray, right: np.ndarray) -> float:
    """Lightweight Kendall-style rank agreement without scipy."""
    if len(left) < 2:
        return 0.0
    concordant = 0
    discordant = 0
    comparable = 0
    for i in range(len(left) - 1):
        for j in range(i + 1, len(left)):
            dx = left[i] - left[j]
            dy = right[i] - right[j]
            if abs(dx) < 1e-12 or abs(dy) < 1e-12:
                continue
            comparable += 1
            if dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    if comparable == 0:
        return 0.0
    return float((concordant - discordant) / comparable)


def metric_gap(panel: dict[str, GapInput]) -> float:
    """Cross-method correlation between offline score and R4 utility."""
    offline = np.asarray([item.offline_score for item in panel.values()], dtype=np.float64)
    full_realism = np.asarray([item.full_realism_utility for item in panel.values()], dtype=np.float64)
    return _stable_corr(offline, full_realism)


def ranking_gap(panel: dict[str, GapInput]) -> float:
    """Kendall tau between offline and R4 rankings."""
    if len(panel) < 2:
        return 0.0
    offline = np.asarray([item.offline_score for item in panel.values()], dtype=np.float64)
    full_realism = np.asarray([item.full_realism_utility for item in panel.values()], dtype=np.float64)
    tau = _kendall_tau(offline, full_realism)
    return 0.0 if np.isnan(tau) else float(tau)


def reversal_rate_at_k(panel: dict[str, GapInput], k: int = 1) -> float:
    """Fraction of offline top-k methods that drop out of full-realism top-k."""
    if not panel:
        return 0.0
    k = max(1, min(k, len(panel)))
    offline_rank = [name for name, _ in sorted(panel.items(), key=lambda item: item[1].offline_score, reverse=True)]
    full_realism_rank = [name for name, _ in sorted(panel.items(), key=lambda item: item[1].full_realism_utility, reverse=True)]
    offline_top = set(offline_rank[:k])
    full_realism_top = set(full_realism_rank[:k])
    dropped = sum(1 for name in offline_top if name not in full_realism_top)
    return float(dropped / k)


def fragility_gap(panel: dict[str, GapInput]) -> dict[str, float]:
    """Worst-regime shortfall relative to median regime utility for each method."""
    result: dict[str, float] = {}
    for name, item in panel.items():
        if not item.regime_utilities:
            result[name] = 0.0
            continue
        values = np.asarray(item.regime_utilities, dtype=np.float64)
        result[name] = float(np.median(values) - np.min(values))
    return result


def compute_gap_report(panel: dict[str, GapInput], k: int = 1) -> dict[str, object]:
    """Bundle PEG-Bench gap metrics into a single JSON-serializable payload."""
    if len(panel) < 2:
        return {
            "metric_gap": None,
            "ranking_gap": None,
            "reversal_rate_at_k": None,
            "fragility_gap": {"status": "missing", "values": {}},
            "status": "protocol_target",
            "reason": "gap metrics require a multi-method panel under a shared R4 evaluator",
        }
    has_regime_utilities = any(item.regime_utilities for item in panel.values())
    fragility_payload: dict[str, object]
    if has_regime_utilities:
        fragility_payload = {"status": "computed", "values": fragility_gap(panel)}
    else:
        fragility_payload = {
            "status": "missing",
            "values": {},
            "reason": "regime-level utilities were not persisted in the current benchmark artifact set",
        }
    return {
        "metric_gap": metric_gap(panel),
        "ranking_gap": ranking_gap(panel),
        "reversal_rate_at_k": reversal_rate_at_k(panel, k=k),
        "fragility_gap": fragility_payload,
        "status": "computed",
        "k": k,
    }


def build_gap_panel_from_method_reports(methods: dict[str, object]) -> dict[str, GapInput]:
    """Build a gap panel from benchmark-level method report blocks."""
    panel: dict[str, GapInput] = {}
    for method_name, payload in methods.items():
        if not isinstance(payload, dict):
            continue
        prediction = payload.get("prediction", {})
        execution = payload.get("execution", {})
        robustness = payload.get("robustness", {})
        if not isinstance(prediction, dict) or not isinstance(execution, dict):
            continue
        rank_ic_block = prediction.get("rank_ic", {})
        direction_block = prediction.get("directional_accuracy", {})
        sharpe_block = execution.get("post_cost_sharpe", {})
        if isinstance(rank_ic_block, dict):
            offline_score = rank_ic_block.get("mean")
        else:
            offline_score = rank_ic_block
        if offline_score is None:
            if isinstance(direction_block, dict):
                offline_score = direction_block.get("mean")
            else:
                offline_score = direction_block
        if isinstance(sharpe_block, dict):
            full_realism = sharpe_block.get("mean")
        else:
            full_realism = sharpe_block
        if offline_score is None or full_realism is None:
            continue
        regime_utilities = None
        if isinstance(robustness, dict):
            values = robustness.get("regime_utilities")
            if isinstance(values, list):
                regime_utilities = [float(item) for item in values if item is not None]
            elif isinstance(robustness.get("regime_metrics"), dict):
                derived = []
                for metrics in robustness.get("regime_metrics", {}).values():
                    if not isinstance(metrics, dict):
                        continue
                    sharpe = metrics.get("sharpe")
                    if sharpe is None:
                        continue
                    derived.append(float(sharpe))
                if derived:
                    regime_utilities = derived
            if regime_utilities is None:
                worst_regime = robustness.get("worst_regime_sharpe")
                if isinstance(worst_regime, dict):
                    worst_regime = worst_regime.get("mean")
                if worst_regime is not None:
                    regime_utilities = [float(full_realism), float(worst_regime)]
        panel[str(method_name)] = GapInput(
            offline_score=float(offline_score),
            full_realism_utility=float(full_realism),
            regime_utilities=regime_utilities,
        )
    return panel


def compute_gap_report_from_method_reports(methods: dict[str, object], k: int = 1) -> dict[str, object]:
    """Compute gap metrics directly from persisted benchmark method reports."""
    return compute_gap_report(build_gap_panel_from_method_reports(methods), k=k)
