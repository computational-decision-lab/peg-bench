"""
Select stress test periods from LOB data.
Implements stress period selection from paper Section 7.2.2.

执行位置: ☁️ 云端（AutoDL服务器）
数据范围: 2025-10-01 to 2025-10-30 (30 days)

Stress periods are identified by:
1. High volatility (top 10% of rolling volatility)
2. Large price movements (top 10% of absolute returns)
3. High trading volume (top 10% of volume)
4. Market regime changes (detected by HMM or clustering)

Usage:
    python src/data/select_stress_periods.py \
        --input data/processed/lob_dataset.npz \
        --output data/processed/stress_periods.json \
        --percentile 90
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from scipy import stats


@dataclass
class StressPeriod:
    """A stress test period."""
    start_idx: int
    end_idx: int
    start_timestamp: int
    end_timestamp: int
    stress_type: str  # 'volatility', 'price_move', 'volume', 'regime_change'
    severity: float  # 0-1 score
    metrics: Dict[str, float]


class StressPeriodSelector:
    """
    Select stress test periods from LOB data.

    Attributes:
        volatility_window: Window for volatility computation (default: 20)
        volume_window: Window for volume computation (default: 20)
        percentile: Percentile threshold for stress detection (default: 90)
        min_period_length: Minimum length of stress period (default: 10 bars)
        max_period_length: Maximum length of stress period (default: 100 bars)
    """

    def __init__(
        self,
        volatility_window: int = 20,
        volume_window: int = 20,
        percentile: float = 90.0,
        min_period_length: int = 10,
        max_period_length: int = 100,
    ):
        self.volatility_window = volatility_window
        self.volume_window = volume_window
        self.percentile = percentile
        self.min_period_length = min_period_length
        self.max_period_length = max_period_length

    def compute_volatility(self, returns: np.ndarray) -> np.ndarray:
        """Compute rolling volatility."""
        n = len(returns)
        volatilities = np.zeros(n, dtype=np.float32)

        for i in range(n):
            start = max(0, i - self.volatility_window + 1)
            window_returns = returns[start:i+1]
            if len(window_returns) > 1:
                volatilities[i] = np.std(window_returns)
            else:
                volatilities[i] = 0.0

        return volatilities

    def compute_volume(self, lob_tensors: np.ndarray) -> np.ndarray:
        """
        Compute total volume from LOB tensors.

        Args:
            lob_tensors: (N, 4, levels, 2) array
                Channel 1 is log_volume

        Returns:
            (N,) array of total volumes
        """
        # Extract log volumes (channel 1)
        log_volumes = lob_tensors[:, 1, :, :]  # (N, levels, 2)

        # Convert back to linear scale and sum
        volumes = np.exp(log_volumes)
        total_volumes = np.sum(volumes, axis=(1, 2))  # (N,)

        return total_volumes

    def detect_high_volatility_periods(
        self,
        returns: np.ndarray,
        timestamps: np.ndarray,
    ) -> List[StressPeriod]:
        """Detect periods with high volatility."""
        volatilities = self.compute_volatility(returns)

        # Find threshold
        threshold = np.percentile(volatilities, self.percentile)

        # Find stress periods
        stress_mask = volatilities >= threshold
        periods = self._extract_periods(stress_mask, timestamps, volatilities, 'volatility')

        return periods

    def detect_large_price_moves(
        self,
        returns: np.ndarray,
        timestamps: np.ndarray,
    ) -> List[StressPeriod]:
        """Detect periods with large price movements."""
        abs_returns = np.abs(returns)

        # Find threshold
        threshold = np.percentile(abs_returns, self.percentile)

        # Find stress periods
        stress_mask = abs_returns >= threshold
        periods = self._extract_periods(stress_mask, timestamps, abs_returns, 'price_move')

        return periods

    def detect_high_volume_periods(
        self,
        lob_tensors: np.ndarray,
        timestamps: np.ndarray,
    ) -> List[StressPeriod]:
        """Detect periods with high trading volume."""
        volumes = self.compute_volume(lob_tensors)

        # Compute rolling average volume
        rolling_volumes = np.zeros_like(volumes)
        for i in range(len(volumes)):
            start = max(0, i - self.volume_window + 1)
            rolling_volumes[i] = np.mean(volumes[start:i+1])

        # Find threshold
        threshold = np.percentile(rolling_volumes, self.percentile)

        # Find stress periods
        stress_mask = rolling_volumes >= threshold
        periods = self._extract_periods(stress_mask, timestamps, rolling_volumes, 'volume')

        return periods

    def detect_regime_changes(
        self,
        returns: np.ndarray,
        timestamps: np.ndarray,
    ) -> List[StressPeriod]:
        """
        Detect market regime changes using simple statistical tests.

        Uses rolling mean and variance to detect regime shifts.
        """
        n = len(returns)
        regime_scores = np.zeros(n, dtype=np.float32)

        window = self.volatility_window

        for i in range(window, n):
            # Compare current window with previous window
            prev_window = returns[i-2*window:i-window]
            curr_window = returns[i-window:i]

            if len(prev_window) > 0 and len(curr_window) > 0:
                # T-test for mean difference
                t_stat, _ = stats.ttest_ind(prev_window, curr_window)

                # F-test for variance difference
                var_prev = np.var(prev_window)
                var_curr = np.var(curr_window)
                f_stat = var_curr / (var_prev + 1e-8)

                # Combined score
                regime_scores[i] = abs(t_stat) + abs(np.log(f_stat + 1e-8))

        # Find threshold
        threshold = np.percentile(regime_scores[window:], self.percentile)

        # Find stress periods
        stress_mask = regime_scores >= threshold
        periods = self._extract_periods(stress_mask, timestamps, regime_scores, 'regime_change')

        return periods

    def _extract_periods(
        self,
        stress_mask: np.ndarray,
        timestamps: np.ndarray,
        severity_scores: np.ndarray,
        stress_type: str,
    ) -> List[StressPeriod]:
        """Extract contiguous stress periods from binary mask."""
        periods = []

        in_period = False
        start_idx = 0

        for i in range(len(stress_mask)):
            if stress_mask[i] and not in_period:
                # Start of new period
                start_idx = i
                in_period = True
            elif not stress_mask[i] and in_period:
                # End of period
                end_idx = i - 1
                period_length = end_idx - start_idx + 1

                if self.min_period_length <= period_length <= self.max_period_length:
                    # Compute severity as mean score in period
                    severity = float(np.mean(severity_scores[start_idx:end_idx+1]))

                    # Normalize severity to 0-1
                    max_severity = np.max(severity_scores)
                    if max_severity > 0:
                        severity = severity / max_severity

                    period = StressPeriod(
                        start_idx=int(start_idx),
                        end_idx=int(end_idx),
                        start_timestamp=int(timestamps[start_idx]),
                        end_timestamp=int(timestamps[end_idx]),
                        stress_type=stress_type,
                        severity=severity,
                        metrics={
                            'length': period_length,
                            'mean_score': float(np.mean(severity_scores[start_idx:end_idx+1])),
                            'max_score': float(np.max(severity_scores[start_idx:end_idx+1])),
                        }
                    )
                    periods.append(period)

                in_period = False

        # Handle case where stress period extends to end
        if in_period:
            end_idx = len(stress_mask) - 1
            period_length = end_idx - start_idx + 1

            if self.min_period_length <= period_length <= self.max_period_length:
                severity = float(np.mean(severity_scores[start_idx:end_idx+1]))
                max_severity = np.max(severity_scores)
                if max_severity > 0:
                    severity = severity / max_severity

                period = StressPeriod(
                    start_idx=int(start_idx),
                    end_idx=int(end_idx),
                    start_timestamp=int(timestamps[start_idx]),
                    end_timestamp=int(timestamps[end_idx]),
                    stress_type=stress_type,
                    severity=severity,
                    metrics={
                        'length': period_length,
                        'mean_score': float(np.mean(severity_scores[start_idx:end_idx+1])),
                        'max_score': float(np.max(severity_scores[start_idx:end_idx+1])),
                    }
                )
                periods.append(period)

        return periods

    def select_stress_periods(
        self,
        lob_tensors: np.ndarray,
        returns: np.ndarray,
        timestamps: np.ndarray,
    ) -> List[StressPeriod]:
        """
        Select all stress periods from LOB data.

        Args:
            lob_tensors: (N, 4, levels, 2) LOB tensor array
            returns: (N,) array of returns
            timestamps: (N,) array of timestamps

        Returns:
            List of StressPeriod objects, sorted by severity
        """
        all_periods = []

        # Detect different types of stress periods
        all_periods.extend(self.detect_high_volatility_periods(returns, timestamps))
        all_periods.extend(self.detect_large_price_moves(returns, timestamps))
        all_periods.extend(self.detect_high_volume_periods(lob_tensors, timestamps))
        all_periods.extend(self.detect_regime_changes(returns, timestamps))

        # Sort by severity (descending)
        all_periods.sort(key=lambda p: p.severity, reverse=True)

        return all_periods


def load_lob_dataset(path: Path) -> Dict[str, np.ndarray]:
    """Load LOB dataset from npz file."""
    data = np.load(path)
    return {key: data[key] for key in data.keys()}


def save_stress_periods(periods: List[StressPeriod], output_path: Path) -> None:
    """Save stress periods to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable format
    periods_dict = [asdict(p) for p in periods]

    with open(output_path, 'w') as f:
        json.dump(periods_dict, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select stress test periods from LOB data.")
    parser.add_argument("--input", type=Path, required=True, help="Path to LOB dataset (.npz)")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    parser.add_argument("--percentile", type=float, default=90.0, help="Percentile threshold (default: 90)")
    parser.add_argument("--min-length", type=int, default=10, help="Minimum period length")
    parser.add_argument("--max-length", type=int, default=100, help="Maximum period length")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load dataset
    print(f"Loading dataset from {args.input}...")
    data = load_lob_dataset(args.input)

    lob_tensors = data['lob']
    returns = data.get('return_5', data.get('return_20'))  # Use available returns
    timestamps = data['timestamps']

    print(f"Dataset shape: {lob_tensors.shape}")
    print(f"Returns shape: {returns.shape}")

    # Select stress periods
    selector = StressPeriodSelector(
        percentile=args.percentile,
        min_period_length=args.min_length,
        max_period_length=args.max_length,
    )

    print(f"Selecting stress periods (percentile={args.percentile})...")
    periods = selector.select_stress_periods(lob_tensors, returns, timestamps)

    # Print summary
    print(f"\nFound {len(periods)} stress periods:")
    for stress_type in ['volatility', 'price_move', 'volume', 'regime_change']:
        count = sum(1 for p in periods if p.stress_type == stress_type)
        print(f"  {stress_type}: {count}")

    # Print top 5 most severe periods
    print(f"\nTop 5 most severe periods:")
    for i, period in enumerate(periods[:5]):
        print(f"  {i+1}. {period.stress_type} (severity={period.severity:.3f}, length={period.metrics['length']})")

    # Save to file
    save_stress_periods(periods, args.output)
    print(f"\nSaved stress periods to {args.output}")


if __name__ == "__main__":
    main()
