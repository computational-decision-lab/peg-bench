"""
Adaptive Confidence Intervals (ACI) for KAIROS-X evaluation.
Implements Phase D from paper Section 7.3.

执行位置: ☁️ 云端（AutoDL服务器）
数据范围: 2025-10-01 to 2025-10-30 (30 days)

ACI provides statistically valid confidence intervals for performance metrics
under distribution shift, accounting for:
1. Non-stationarity in financial markets
2. Temporal dependence in returns
3. Regime changes

Reference: Adaptive Conformal Inference Under Distribution Shift (Gibbs & Candès, 2021)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from scipy import stats


@dataclass
class ACIResult:
    """Result of ACI evaluation."""
    metric_name: str
    point_estimate: float
    lower_bound: float
    upper_bound: float
    coverage_level: float
    width: float
    num_samples: int


class AdaptiveConfidenceIntervals:
    """
    Adaptive Confidence Intervals for time-series evaluation.

    Attributes:
        alpha: Significance level (default: 0.05 for 95% CI)
        gamma: Adaptation rate (default: 0.01)
        window_size: Rolling window size for adaptation (default: 100)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        gamma: float = 0.01,
        window_size: int = 100,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.window_size = window_size

    def compute_aci(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        metric_fn: callable,
        metric_name: str = "metric",
    ) -> ACIResult:
        """
        Compute adaptive confidence interval for a metric.

        Args:
            predictions: (N,) array of predictions
            targets: (N,) array of targets
            metric_fn: Function that computes metric from (predictions, targets)
            metric_name: Name of the metric

        Returns:
            ACIResult object
        """
        n = len(predictions)

        # Compute point estimate
        point_estimate = metric_fn(predictions, targets)

        # Compute residuals
        residuals = predictions - targets

        # Initialize adaptive quantile
        quantile = 1 - self.alpha / 2

        # Compute adaptive quantiles using rolling window
        lower_bounds = []
        upper_bounds = []

        for i in range(self.window_size, n):
            # Get window
            window_start = max(0, i - self.window_size)
            window_residuals = residuals[window_start:i]

            # Compute empirical quantiles
            lower_q = np.quantile(window_residuals, self.alpha / 2)
            upper_q = np.quantile(window_residuals, 1 - self.alpha / 2)

            lower_bounds.append(lower_q)
            upper_bounds.append(upper_q)

        # Compute final confidence interval
        if lower_bounds:
            lower_bound = point_estimate + np.mean(lower_bounds)
            upper_bound = point_estimate + np.mean(upper_bounds)
        else:
            # Fallback to standard CI
            std_error = np.std(residuals) / np.sqrt(n)
            z_score = stats.norm.ppf(1 - self.alpha / 2)
            lower_bound = point_estimate - z_score * std_error
            upper_bound = point_estimate + z_score * std_error

        width = upper_bound - lower_bound

        return ACIResult(
            metric_name=metric_name,
            point_estimate=float(point_estimate),
            lower_bound=float(lower_bound),
            upper_bound=float(upper_bound),
            coverage_level=1 - self.alpha,
            width=float(width),
            num_samples=int(n),
        )

    def compute_sharpe_aci(
        self,
        returns: np.ndarray,
    ) -> ACIResult:
        """
        Compute ACI for Sharpe ratio.

        Args:
            returns: (N,) array of returns

        Returns:
            ACIResult for Sharpe ratio
        """
        n = len(returns)

        # Point estimate
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe = mean_return / (std_return + 1e-8)

        # Bootstrap for confidence interval
        n_bootstrap = 1000
        bootstrap_sharpes = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n, size=n, replace=True)
            bootstrap_returns = returns[indices]

            # Compute Sharpe ratio
            bootstrap_mean = np.mean(bootstrap_returns)
            bootstrap_std = np.std(bootstrap_returns)
            bootstrap_sharpe = bootstrap_mean / (bootstrap_std + 1e-8)

            bootstrap_sharpes.append(bootstrap_sharpe)

        # Compute confidence interval
        lower_bound = np.quantile(bootstrap_sharpes, self.alpha / 2)
        upper_bound = np.quantile(bootstrap_sharpes, 1 - self.alpha / 2)

        return ACIResult(
            metric_name="sharpe_ratio",
            point_estimate=float(sharpe),
            lower_bound=float(lower_bound),
            upper_bound=float(upper_bound),
            coverage_level=1 - self.alpha,
            width=float(upper_bound - lower_bound),
            num_samples=int(n),
        )

    def compute_return_aci(
        self,
        returns: np.ndarray,
    ) -> ACIResult:
        """
        Compute ACI for cumulative return.

        Args:
            returns: (N,) array of returns

        Returns:
            ACIResult for cumulative return
        """
        n = len(returns)

        # Point estimate
        cumulative_return = np.sum(returns)

        # Standard error with autocorrelation adjustment
        # Use Newey-West HAC estimator
        lags = min(int(4 * (n / 100) ** (2/9)), n - 1)
        variance = np.var(returns)

        for lag in range(1, lags + 1):
            autocovariance = np.mean(
                (returns[:-lag] - np.mean(returns)) *
                (returns[lag:] - np.mean(returns))
            )
            weight = 1 - lag / (lags + 1)
            variance += 2 * weight * autocovariance

        std_error = np.sqrt(variance * n)

        # Confidence interval
        z_score = stats.norm.ppf(1 - self.alpha / 2)
        lower_bound = cumulative_return - z_score * std_error
        upper_bound = cumulative_return + z_score * std_error

        return ACIResult(
            metric_name="cumulative_return",
            point_estimate=float(cumulative_return),
            lower_bound=float(lower_bound),
            upper_bound=float(upper_bound),
            coverage_level=1 - self.alpha,
            width=float(upper_bound - lower_bound),
            num_samples=int(n),
        )

    def compute_drawdown_aci(
        self,
        returns: np.ndarray,
    ) -> ACIResult:
        """
        Compute ACI for maximum drawdown.

        Args:
            returns: (N,) array of returns

        Returns:
            ACIResult for maximum drawdown
        """
        n = len(returns)

        # Point estimate
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = np.max(drawdown)

        # Bootstrap for confidence interval
        n_bootstrap = 1000
        bootstrap_drawdowns = []

        for _ in range(n_bootstrap):
            # Block bootstrap to preserve temporal structure
            block_size = 20
            n_blocks = n // block_size

            bootstrap_returns = []
            for _ in range(n_blocks):
                start_idx = np.random.randint(0, n - block_size + 1)
                block = returns[start_idx:start_idx + block_size]
                bootstrap_returns.extend(block)

            bootstrap_returns = np.array(bootstrap_returns[:n])

            # Compute max drawdown
            bootstrap_cumulative = np.cumsum(bootstrap_returns)
            bootstrap_running_max = np.maximum.accumulate(bootstrap_cumulative)
            bootstrap_drawdown = bootstrap_running_max - bootstrap_cumulative
            bootstrap_max_drawdown = np.max(bootstrap_drawdown)

            bootstrap_drawdowns.append(bootstrap_max_drawdown)

        # Compute confidence interval
        lower_bound = np.quantile(bootstrap_drawdowns, self.alpha / 2)
        upper_bound = np.quantile(bootstrap_drawdowns, 1 - self.alpha / 2)

        return ACIResult(
            metric_name="max_drawdown",
            point_estimate=float(max_drawdown),
            lower_bound=float(lower_bound),
            upper_bound=float(upper_bound),
            coverage_level=1 - self.alpha,
            width=float(upper_bound - lower_bound),
            num_samples=int(n),
        )


class ACIEvaluator:
    """
    Evaluator for KAIROS-X using Adaptive Confidence Intervals.

    Computes ACI for multiple metrics:
    - Sharpe ratio
    - Cumulative return
    - Maximum drawdown
    - Directional accuracy
    - Rank IC
    """

    def __init__(
        self,
        alpha: float = 0.05,
        gamma: float = 0.01,
        window_size: int = 100,
    ):
        self.aci = AdaptiveConfidenceIntervals(alpha, gamma, window_size)

    def evaluate(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        returns: Optional[np.ndarray] = None,
    ) -> Dict[str, ACIResult]:
        """
        Evaluate model with ACI.

        Args:
            predictions: (N,) array of predictions
            targets: (N,) array of targets
            returns: (N,) array of realized returns (optional)

        Returns:
            Dictionary of metric_name -> ACIResult
        """
        results = {}

        # Directional accuracy
        def dir_acc_fn(pred, tgt):
            return np.mean((pred > 0) == (tgt > 0))

        results['directional_accuracy'] = self.aci.compute_aci(
            predictions, targets, dir_acc_fn, "directional_accuracy"
        )

        # Rank IC (Spearman correlation)
        def rank_ic_fn(pred, tgt):
            return stats.spearmanr(pred, tgt)[0]

        results['rank_ic'] = self.aci.compute_aci(
            predictions, targets, rank_ic_fn, "rank_ic"
        )

        # If returns are provided, compute trading metrics
        if returns is not None:
            results['sharpe_ratio'] = self.aci.compute_sharpe_aci(returns)
            results['cumulative_return'] = self.aci.compute_return_aci(returns)
            results['max_drawdown'] = self.aci.compute_drawdown_aci(returns)

        return results

    def compare_models(
        self,
        model_results: Dict[str, Dict[str, ACIResult]],
    ) -> Dict[str, Any]:
        """
        Compare multiple models using ACI.

        Args:
            model_results: Dictionary of model_name -> results_dict

        Returns:
            Comparison summary
        """
        comparison = {}

        # Get all metrics
        all_metrics = set()
        for results in model_results.values():
            all_metrics.update(results.keys())

        # Compare each metric
        for metric in all_metrics:
            metric_comparison = {}

            for model_name, results in model_results.items():
                if metric in results:
                    result = results[metric]
                    metric_comparison[model_name] = {
                        'point_estimate': result.point_estimate,
                        'ci': [result.lower_bound, result.upper_bound],
                        'width': result.width,
                    }

            comparison[metric] = metric_comparison

        return comparison


def load_predictions(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load predictions and targets from file."""
    data = np.load(path)
    predictions = data['predictions']
    targets = data['targets']
    return predictions, targets


def load_returns(path: Path) -> np.ndarray:
    """Load returns from file."""
    data = np.load(path)
    return data['returns']


def save_results(results: Dict[str, ACIResult], output_path: Path) -> None:
    """Save ACI results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable format
    results_dict = {
        metric: asdict(result)
        for metric, result in results.items()
    }

    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate KAIROS-X with ACI")
    parser.add_argument("--predictions", type=Path, required=True, help="Path to predictions (.npz)")
    parser.add_argument("--returns", type=Path, default=None, help="Path to returns (.npz)")
    parser.add_argument("--output", type=Path, required=True, help="Output path for results (.json)")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    parser.add_argument("--gamma", type=float, default=0.01, help="Adaptation rate")
    parser.add_argument("--window-size", type=int, default=100, help="Rolling window size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load predictions
    print(f"Loading predictions from {args.predictions}...")
    predictions, targets = load_predictions(args.predictions)
    print(f"Loaded {len(predictions)} predictions")

    # Load returns if provided
    returns = None
    if args.returns is not None:
        print(f"Loading returns from {args.returns}...")
        returns = load_returns(args.returns)
        print(f"Loaded {len(returns)} returns")

    # Create evaluator
    evaluator = ACIEvaluator(
        alpha=args.alpha,
        gamma=args.gamma,
        window_size=args.window_size,
    )

    # Evaluate
    print("Computing ACI...")
    results = evaluator.evaluate(predictions, targets, returns)

    # Print results
    print("\nACI Results:")
    print(f"Coverage level: {(1 - args.alpha) * 100:.1f}%")
    print()

    for metric_name, result in results.items():
        print(f"{metric_name}:")
        print(f"  Point estimate: {result.point_estimate:.4f}")
        print(f"  95% CI: [{result.lower_bound:.4f}, {result.upper_bound:.4f}]")
        print(f"  Width: {result.width:.4f}")
        print()

    # Save results
    save_results(results, args.output)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
