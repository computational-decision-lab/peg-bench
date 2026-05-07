"""Common predictive and trading metrics used across the framework."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class TradingMetrics:
    """Aggregated trajectory or backtest metrics."""

    cumulative_return: float
    annualized_return: float
    sharpe: float
    calmar: float
    max_drawdown: float
    turnover: float
    net_pnl: float
    liquidation: bool


def directional_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Fraction of correctly predicted directions."""
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    valid = np.isfinite(predictions) & np.isfinite(targets)
    if int(valid.sum()) == 0:
        return 0.0
    return float(np.mean((predictions[valid] > 0) == (targets[valid] > 0)))


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average-rank implementation compatible with Spearman IC."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return np.asarray([], dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        average_rank = (start + stop - 1) / 2.0 + 1.0
        ranks[order[start:stop]] = average_rank
        start = stop
    return ranks


def rank_ic(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Spearman rank IC."""
    left = _rankdata(np.asarray(predictions, dtype=np.float64))
    right = _rankdata(np.asarray(targets, dtype=np.float64))
    if len(left) < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    corr = np.corrcoef(left, right)[0, 1]
    return 0.0 if np.isnan(corr) else float(corr)


def mean_squared_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean squared error."""
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    valid = np.isfinite(predictions) & np.isfinite(targets)
    if int(valid.sum()) == 0:
        return 0.0
    return float(np.mean((predictions[valid] - targets[valid]) ** 2))


def mean_absolute_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean absolute error."""
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    valid = np.isfinite(predictions) & np.isfinite(targets)
    if int(valid.sum()) == 0:
        return 0.0
    return float(np.mean(np.abs(predictions[valid] - targets[valid])))


def _finite_returns(returns: np.ndarray) -> np.ndarray:
    values = np.asarray(returns, dtype=np.float64).reshape(-1)
    if len(values) == 0:
        return values
    return values[np.isfinite(values)]


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum path drawdown."""
    cleaned = _finite_returns(returns)
    if len(cleaned) == 0:
        return 0.0
    cumulative = np.cumsum(cleaned)
    running_max = np.maximum.accumulate(cumulative)
    return float(np.max(running_max - cumulative)) if len(cumulative) else 0.0


def annualized_return(returns: np.ndarray, periods_per_year: int = 252 * 24 * 60 * 60) -> float:
    """Conservative observed-horizon return scaling."""
    cleaned = _finite_returns(returns)
    if len(cleaned) == 0:
        return 0.0
    effective_periods = min(periods_per_year, max(len(cleaned), 1))
    mean_value = float(np.mean(cleaned))
    return 0.0 if not np.isfinite(mean_value) else float(mean_value * effective_periods)


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252 * 24 * 60 * 60) -> float:
    """Conservative observed-horizon Sharpe ratio."""
    cleaned = _finite_returns(returns)
    if len(cleaned) == 0:
        return 0.0
    effective_periods = min(periods_per_year, max(len(cleaned), 1))
    mean_value = float(np.mean(cleaned))
    std_value = float(np.std(cleaned))
    if not np.isfinite(mean_value) or not np.isfinite(std_value):
        return 0.0
    if std_value <= 1e-12:
        return 0.0
    return float(mean_value / std_value * np.sqrt(effective_periods))


def calmar_ratio(returns: np.ndarray, periods_per_year: int = 252 * 24 * 60 * 60) -> float:
    """Annualized return divided by max drawdown."""
    ann = annualized_return(returns, periods_per_year)
    mdd = max_drawdown(returns)
    if not np.isfinite(ann) or not np.isfinite(mdd):
        return 0.0
    if mdd <= 1e-12:
        return 0.0
    return float(ann / mdd)


def cvar_at_level(returns: np.ndarray, alpha: float = 0.95) -> float:
    """Positive tail-loss magnitude for the worst `1 - alpha` observations."""
    cleaned = _finite_returns(returns)
    if len(cleaned) == 0:
        return 0.0
    losses = -cleaned
    threshold = np.quantile(losses, alpha)
    tail = losses[losses >= threshold]
    if len(tail) == 0:
        return 0.0
    return float(np.mean(tail))


def average_holding_period(positions: np.ndarray) -> float:
    """Average duration of non-zero same-sign positions."""
    if len(positions) == 0:
        return 0.0
    signs = np.sign(np.asarray(positions, dtype=np.float64))
    runs: list[int] = []
    current_sign = 0.0
    current_len = 0
    for sign in signs:
        if sign == 0:
            if current_len > 0:
                runs.append(current_len)
                current_len = 0
                current_sign = 0.0
            continue
        if sign == current_sign:
            current_len += 1
        else:
            if current_len > 0:
                runs.append(current_len)
            current_sign = sign
            current_len = 1
    if current_len > 0:
        runs.append(current_len)
    if not runs:
        return 0.0
    return float(np.mean(runs))


def summarize_trading_metrics(
    returns: np.ndarray,
    turnover: float,
    liquidation: bool,
    periods_per_year: int = 252 * 24 * 60 * 60,
) -> TradingMetrics:
    """Create the unified metric dataclass from trajectory returns."""
    cleaned = _finite_returns(returns)
    cumulative = float(np.sum(cleaned)) if len(cleaned) else 0.0
    return TradingMetrics(
        cumulative_return=cumulative,
        annualized_return=annualized_return(cleaned, periods_per_year),
        sharpe=sharpe_ratio(cleaned, periods_per_year),
        calmar=calmar_ratio(cleaned, periods_per_year),
        max_drawdown=max_drawdown(cleaned),
        turnover=float(turnover),
        net_pnl=cumulative,
        liquidation=bool(liquidation),
    )
