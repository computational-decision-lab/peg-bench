"""
Regime-conditional analysis for KAIROS-X evaluation.
Implements regime analysis from paper Section 7.4.

执行位置: 💻 本地（用户机器）
数据范围: 2025-10-01 to 2025-10-30 (30 days)

Analyzes model performance across different market regimes:
1. High volatility vs. low volatility
2. Bull market vs. bear market
3. High volume vs. low volume
4. Trending vs. mean-reverting
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


@dataclass
class RegimeMetrics:
    """Metrics for a specific regime."""
    regime_name: str
    num_samples: int
    sharpe_ratio: float
    cumulative_return: float
    volatility: float
    max_drawdown: float
    directional_accuracy: float
    rank_ic: float


class RegimeDetector:
    """
    Detect market regimes from price data.

    Attributes:
        volatility_window: Window for volatility computation (default: 20)
        volume_window: Window for volume computation (default: 20)
        trend_window: Window for trend detection (default: 50)
    """

    def __init__(
        self,
        volatility_window: int = 20,
        volume_window: int = 20,
        trend_window: int = 50,
    ):
        self.volatility_window = volatility_window
        self.volume_window = volume_window
        self.trend_window = trend_window

    def detect_volatility_regimes(
        self,
        returns: np.ndarray,
        percentile: float = 50.0,
    ) -> np.ndarray:
        """
        Detect high/low volatility regimes.

        Args:
            returns: (N,) array of returns
            percentile: Percentile threshold (default: 50 for median)

        Returns:
            (N,) array of regime labels (0: low vol, 1: high vol)
        """
        n = len(returns)
        volatilities = np.zeros(n)

        for i in range(n):
            start = max(0, i - self.volatility_window + 1)
            window_returns = returns[start:i+1]
            if len(window_returns) > 1:
                volatilities[i] = np.std(window_returns)

        threshold = np.percentile(volatilities, percentile)
        regimes = (volatilities >= threshold).astype(int)

        return regimes

    def detect_trend_regimes(
        self,
        prices: np.ndarray,
    ) -> np.ndarray:
        """
        Detect bull/bear market regimes.

        Args:
            prices: (N,) array of prices

        Returns:
            (N,) array of regime labels (0: bear, 1: bull)
        """
        n = len(prices)
        regimes = np.zeros(n, dtype=int)

        for i in range(self.trend_window, n):
            window_prices = prices[i-self.trend_window:i]

            # Linear regression to detect trend
            x = np.arange(len(window_prices))
            slope, _, _, _, _ = stats.linregress(x, window_prices)

            # Bull market if positive slope
            regimes[i] = 1 if slope > 0 else 0

        return regimes

    def detect_volume_regimes(
        self,
        volumes: np.ndarray,
        percentile: float = 50.0,
    ) -> np.ndarray:
        """
        Detect high/low volume regimes.

        Args:
            volumes: (N,) array of volumes
            percentile: Percentile threshold

        Returns:
            (N,) array of regime labels (0: low volume, 1: high volume)
        """
        n = len(volumes)
        rolling_volumes = np.zeros(n)

        for i in range(n):
            start = max(0, i - self.volume_window + 1)
            rolling_volumes[i] = np.mean(volumes[start:i+1])

        threshold = np.percentile(rolling_volumes, percentile)
        regimes = (rolling_volumes >= threshold).astype(int)

        return regimes

    def detect_mean_reversion_regimes(
        self,
        prices: np.ndarray,
        window: int = 20,
    ) -> np.ndarray:
        """
        Detect trending vs. mean-reverting regimes.

        Uses Hurst exponent:
        - H < 0.5: mean-reverting
        - H > 0.5: trending

        Args:
            prices: (N,) array of prices
            window: Window size for Hurst computation

        Returns:
            (N,) array of regime labels (0: mean-reverting, 1: trending)
        """
        n = len(prices)
        regimes = np.zeros(n, dtype=int)

        for i in range(window, n):
            window_prices = prices[i-window:i]

            # Compute Hurst exponent
            hurst = self._compute_hurst(window_prices)

            # Trending if H > 0.5
            regimes[i] = 1 if hurst > 0.5 else 0

        return regimes

    def _compute_hurst(self, prices: np.ndarray) -> float:
        """Compute Hurst exponent using R/S analysis."""
        n = len(prices)
        if n < 10:
            return 0.5

        # Compute log returns
        log_returns = np.diff(np.log(prices + 1e-8))

        # Compute mean
        mean_return = np.mean(log_returns)

        # Compute cumulative deviations
        deviations = log_returns - mean_return
        cumulative_deviations = np.cumsum(deviations)

        # Compute range
        R = np.max(cumulative_deviations) - np.min(cumulative_deviations)

        # Compute standard deviation
        S = np.std(log_returns)

        # Avoid division by zero
        if S < 1e-8:
            return 0.5

        # R/S ratio
        rs = R / S

        # Hurst exponent: H = log(R/S) / log(n)
        hurst = np.log(rs) / np.log(n)

        # Clip to valid range
        hurst = np.clip(hurst, 0.0, 1.0)

        return hurst


class RegimeAnalyzer:
    """
    Analyze model performance across different market regimes.
    """

    def __init__(self):
        self.detector = RegimeDetector()

    def compute_regime_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        returns: np.ndarray,
        regime_mask: np.ndarray,
        regime_name: str,
    ) -> RegimeMetrics:
        """
        Compute metrics for a specific regime.

        Args:
            predictions: (N,) array of predictions
            targets: (N,) array of targets
            returns: (N,) array of realized returns
            regime_mask: (N,) boolean array indicating regime
            regime_name: Name of the regime

        Returns:
            RegimeMetrics object
        """
        # Filter data by regime
        pred_regime = predictions[regime_mask]
        tgt_regime = targets[regime_mask]
        ret_regime = returns[regime_mask]

        if len(pred_regime) == 0:
            return RegimeMetrics(
                regime_name=regime_name,
                num_samples=0,
                sharpe_ratio=0.0,
                cumulative_return=0.0,
                volatility=0.0,
                max_drawdown=0.0,
                directional_accuracy=0.0,
                rank_ic=0.0,
            )

        # Compute metrics
        cumulative_return = np.sum(ret_regime)
        volatility = np.std(ret_regime)
        sharpe_ratio = cumulative_return / (volatility + 1e-8)

        # Maximum drawdown
        cumulative = np.cumsum(ret_regime)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = np.max(drawdown)

        # Directional accuracy
        dir_acc = np.mean((pred_regime > 0) == (tgt_regime > 0))

        # Rank IC
        rank_ic = stats.spearmanr(pred_regime, tgt_regime)[0]

        return RegimeMetrics(
            regime_name=regime_name,
            num_samples=int(len(pred_regime)),
            sharpe_ratio=float(sharpe_ratio),
            cumulative_return=float(cumulative_return),
            volatility=float(volatility),
            max_drawdown=float(max_drawdown),
            directional_accuracy=float(dir_acc),
            rank_ic=float(rank_ic),
        )

    def analyze_by_volatility(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        returns: np.ndarray,
        price_returns: np.ndarray,
    ) -> Dict[str, RegimeMetrics]:
        """Analyze performance by volatility regime."""
        regimes = self.detector.detect_volatility_regimes(price_returns)

        results = {}
        results['low_volatility'] = self.compute_regime_metrics(
            predictions, targets, returns, regimes == 0, 'Low Volatility'
        )
        results['high_volatility'] = self.compute_regime_metrics(
            predictions, targets, returns, regimes == 1, 'High Volatility'
        )

        return results

    def analyze_by_trend(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        returns: np.ndarray,
        prices: np.ndarray,
    ) -> Dict[str, RegimeMetrics]:
        """Analyze performance by trend regime."""
        regimes = self.detector.detect_trend_regimes(prices)

        results = {}
        results['bear_market'] = self.compute_regime_metrics(
            predictions, targets, returns, regimes == 0, 'Bear Market'
        )
        results['bull_market'] = self.compute_regime_metrics(
            predictions, targets, returns, regimes == 1, 'Bull Market'
        )

        return results

    def analyze_by_volume(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        returns: np.ndarray,
        volumes: np.ndarray,
    ) -> Dict[str, RegimeMetrics]:
        """Analyze performance by volume regime."""
        regimes = self.detector.detect_volume_regimes(volumes)

        results = {}
        results['low_volume'] = self.compute_regime_metrics(
            predictions, targets, returns, regimes == 0, 'Low Volume'
        )
        results['high_volume'] = self.compute_regime_metrics(
            predictions, targets, returns, regimes == 1, 'High Volume'
        )

        return results

    def analyze_by_mean_reversion(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        returns: np.ndarray,
        prices: np.ndarray,
    ) -> Dict[str, RegimeMetrics]:
        """Analyze performance by mean-reversion regime."""
        regimes = self.detector.detect_mean_reversion_regimes(prices)

        results = {}
        results['mean_reverting'] = self.compute_regime_metrics(
            predictions, targets, returns, regimes == 0, 'Mean Reverting'
        )
        results['trending'] = self.compute_regime_metrics(
            predictions, targets, returns, regimes == 1, 'Trending'
        )

        return results

    def analyze_all_regimes(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        returns: np.ndarray,
        prices: np.ndarray,
        price_returns: np.ndarray,
        volumes: np.ndarray,
    ) -> Dict[str, Dict[str, RegimeMetrics]]:
        """Analyze performance across all regime types."""
        results = {}

        results['volatility'] = self.analyze_by_volatility(
            predictions, targets, returns, price_returns
        )
        results['trend'] = self.analyze_by_trend(
            predictions, targets, returns, prices
        )
        results['volume'] = self.analyze_by_volume(
            predictions, targets, returns, volumes
        )
        results['mean_reversion'] = self.analyze_by_mean_reversion(
            predictions, targets, returns, prices
        )

        return results


def plot_regime_comparison(
    regime_results: Dict[str, Dict[str, RegimeMetrics]],
    output_path: Path,
    metric: str = 'sharpe_ratio',
) -> None:
    """
    Plot regime comparison for a specific metric.

    Args:
        regime_results: Dictionary of regime_type -> {regime_name -> RegimeMetrics}
        output_path: Output path for figure
        metric: Metric to plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    regime_types = list(regime_results.keys())

    for idx, regime_type in enumerate(regime_types):
        ax = axes[idx]

        regimes = regime_results[regime_type]
        regime_names = list(regimes.keys())
        values = [getattr(regimes[name], metric) for name in regime_names]

        # Bar plot
        x = np.arange(len(regime_names))
        bars = ax.bar(x, values, alpha=0.7, edgecolor='black')

        # Color bars
        colors = ['blue', 'red']
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_xticks(x)
        ax.set_xticklabels(regime_names, rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(f"{regime_type.replace('_', ' ').title()} Regimes", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved regime comparison plot to {output_path}")
    plt.close()


def load_data(path: Path) -> Dict[str, np.ndarray]:
    """Load evaluation data from npz file."""
    data = np.load(path)
    return {key: data[key] for key in data.keys()}


def save_results(results: Dict[str, Any], output_path: Path) -> None:
    """Save regime analysis results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable format
    results_dict = {}
    for regime_type, regimes in results.items():
        results_dict[regime_type] = {
            regime_name: asdict(metrics)
            for regime_name, metrics in regimes.items()
        }

    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regime-conditional analysis for KAIROS-X")
    parser.add_argument("--data", type=Path, required=True, help="Path to evaluation data (.npz)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load data
    print(f"Loading data from {args.data}...")
    data = load_data(args.data)

    predictions = data['predictions']
    targets = data['targets']
    returns = data['returns']
    prices = data['prices']
    price_returns = data.get('price_returns', np.diff(prices, prepend=prices[0]))
    volumes = data.get('volumes', np.ones_like(prices))

    print(f"Loaded {len(predictions)} samples")

    # Analyze regimes
    analyzer = RegimeAnalyzer()

    print("\nAnalyzing regimes...")
    results = analyzer.analyze_all_regimes(
        predictions, targets, returns, prices, price_returns, volumes
    )

    # Print results
    print("\nRegime Analysis Results:")
    for regime_type, regimes in results.items():
        print(f"\n{regime_type.replace('_', ' ').title()}:")
        for regime_name, metrics in regimes.items():
            print(f"  {metrics.regime_name}:")
            print(f"    Samples: {metrics.num_samples}")
            print(f"    Sharpe: {metrics.sharpe_ratio:.3f}")
            print(f"    Return: {metrics.cumulative_return:.4f}")
            print(f"    Dir Acc: {metrics.directional_accuracy:.3f}")
            print(f"    Rank IC: {metrics.rank_ic:.3f}")

    # Save results
    output_path = args.output_dir / "regime_analysis.json"
    save_results(results, output_path)
    print(f"\nSaved results to {output_path}")

    # Generate plots
    print("\nGenerating plots...")
    for metric in ['sharpe_ratio', 'cumulative_return', 'directional_accuracy', 'rank_ic']:
        plot_regime_comparison(
            results,
            args.output_dir / f"regime_{metric}.png",
            metric=metric,
        )

    print("\nRegime analysis complete!")


if __name__ == "__main__":
    main()
