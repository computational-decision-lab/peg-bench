"""
Pareto frontier visualization for KAIROS-X evaluation.
Implements visualization from paper Section 7.4.

执行位置: 💻 本地（用户机器）
数据范围: 2025-10-01 to 2025-10-30 (30 days)

Plots risk-return tradeoff curves showing:
- Sharpe ratio vs. volatility
- Return vs. maximum drawdown
- Multiple models on same plot
- Pareto-optimal frontier
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


def load_results(path: Path) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def compute_pareto_frontier(
    points: np.ndarray,
    maximize_both: bool = True,
) -> np.ndarray:
    """
    Compute Pareto frontier from 2D points.

    Args:
        points: (N, 2) array of (x, y) points
        maximize_both: If True, maximize both dimensions; if False, maximize y and minimize x

    Returns:
        (M, 2) array of Pareto-optimal points, sorted by x
    """
    if len(points) == 0:
        return np.array([])

    # Sort by x coordinate
    sorted_indices = np.argsort(points[:, 0])
    sorted_points = points[sorted_indices]

    # Find Pareto frontier
    pareto_indices = []
    current_max_y = -np.inf

    for i, (x, y) in enumerate(sorted_points):
        if maximize_both:
            if y > current_max_y:
                pareto_indices.append(i)
                current_max_y = y
        else:
            # For risk-return: minimize risk (x), maximize return (y)
            if y > current_max_y:
                pareto_indices.append(i)
                current_max_y = y

    pareto_points = sorted_points[pareto_indices]
    return pareto_points


def plot_sharpe_vs_volatility(
    model_results: Dict[str, Dict[str, Any]],
    output_path: Path,
    title: str = "Risk-Return Tradeoff: Sharpe Ratio vs. Volatility",
) -> None:
    """
    Plot Sharpe ratio vs. volatility for multiple models.

    Args:
        model_results: Dictionary of model_name -> results_dict
        output_path: Output path for figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract data points
    points = []
    labels = []
    colors = []

    color_map = {
        'baseline': 'blue',
        'kairos': 'green',
        'kairos_x': 'red',
    }

    for model_name, results in model_results.items():
        if 'sharpe_ratio' in results and 'volatility' in results:
            sharpe = results['sharpe_ratio'].get('point_estimate', 0)
            volatility = results['volatility'].get('point_estimate', 0)

            points.append([volatility, sharpe])
            labels.append(model_name)

            # Assign color
            color = color_map.get(model_name.lower(), 'gray')
            colors.append(color)

    if not points:
        print("No data to plot for Sharpe vs. Volatility")
        return

    points = np.array(points)

    # Plot points
    for i, (point, label, color) in enumerate(zip(points, labels, colors)):
        ax.scatter(point[0], point[1], s=100, c=color, label=label, alpha=0.7, edgecolors='black')

        # Add label
        ax.annotate(
            label,
            (point[0], point[1]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
        )

    # Compute and plot Pareto frontier
    pareto_points = compute_pareto_frontier(points, maximize_both=False)
    if len(pareto_points) > 1:
        ax.plot(
            pareto_points[:, 0],
            pareto_points[:, 1],
            'k--',
            linewidth=2,
            label='Pareto Frontier',
            alpha=0.5,
        )

    ax.set_xlabel('Volatility (Risk)', fontsize=12)
    ax.set_ylabel('Sharpe Ratio (Risk-Adjusted Return)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved Sharpe vs. Volatility plot to {output_path}")
    plt.close()


def plot_return_vs_drawdown(
    model_results: Dict[str, Dict[str, Any]],
    output_path: Path,
    title: str = "Return vs. Risk: Cumulative Return vs. Maximum Drawdown",
) -> None:
    """
    Plot cumulative return vs. maximum drawdown for multiple models.

    Args:
        model_results: Dictionary of model_name -> results_dict
        output_path: Output path for figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract data points
    points = []
    labels = []
    colors = []

    color_map = {
        'baseline': 'blue',
        'kairos': 'green',
        'kairos_x': 'red',
    }

    for model_name, results in model_results.items():
        if 'cumulative_return' in results and 'max_drawdown' in results:
            cum_return = results['cumulative_return'].get('point_estimate', 0)
            max_dd = results['max_drawdown'].get('point_estimate', 0)

            points.append([max_dd, cum_return])
            labels.append(model_name)

            color = color_map.get(model_name.lower(), 'gray')
            colors.append(color)

    if not points:
        print("No data to plot for Return vs. Drawdown")
        return

    points = np.array(points)

    # Plot points
    for i, (point, label, color) in enumerate(zip(points, labels, colors)):
        ax.scatter(point[0], point[1], s=100, c=color, label=label, alpha=0.7, edgecolors='black')

        ax.annotate(
            label,
            (point[0], point[1]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
        )

    # Compute and plot Pareto frontier (minimize drawdown, maximize return)
    pareto_points = compute_pareto_frontier(points, maximize_both=False)
    if len(pareto_points) > 1:
        ax.plot(
            pareto_points[:, 0],
            pareto_points[:, 1],
            'k--',
            linewidth=2,
            label='Pareto Frontier',
            alpha=0.5,
        )

    ax.set_xlabel('Maximum Drawdown (Risk)', fontsize=12)
    ax.set_ylabel('Cumulative Return', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved Return vs. Drawdown plot to {output_path}")
    plt.close()


def plot_multi_objective_pareto(
    model_results: Dict[str, Dict[str, Any]],
    output_path: Path,
    title: str = "Multi-Objective Pareto Frontier",
) -> None:
    """
    Plot 3D Pareto frontier with multiple objectives.

    Args:
        model_results: Dictionary of model_name -> results_dict
        output_path: Output path for figure
        title: Plot title
    """
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract data points
    points = []
    labels = []
    colors = []

    color_map = {
        'baseline': 'blue',
        'kairos': 'green',
        'kairos_x': 'red',
    }

    for model_name, results in model_results.items():
        if all(k in results for k in ['sharpe_ratio', 'cumulative_return', 'max_drawdown']):
            sharpe = results['sharpe_ratio'].get('point_estimate', 0)
            cum_return = results['cumulative_return'].get('point_estimate', 0)
            max_dd = results['max_drawdown'].get('point_estimate', 0)

            points.append([sharpe, cum_return, -max_dd])  # Negative drawdown (higher is better)
            labels.append(model_name)

            color = color_map.get(model_name.lower(), 'gray')
            colors.append(color)

    if not points:
        print("No data to plot for Multi-Objective Pareto")
        return

    points = np.array(points)

    # Plot points
    for i, (point, label, color) in enumerate(zip(points, labels, colors)):
        ax.scatter(
            point[0], point[1], point[2],
            s=100, c=color, label=label,
            alpha=0.7, edgecolors='black'
        )

        ax.text(point[0], point[1], point[2], f'  {label}', fontsize=9)

    ax.set_xlabel('Sharpe Ratio', fontsize=12)
    ax.set_ylabel('Cumulative Return', fontsize=12)
    ax.set_zlabel('Negative Max Drawdown', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved Multi-Objective Pareto po {output_path}")
    plt.close()


def plot_confidence_intervals(
    model_results: Dict[str, Dict[str, Any]],
    output_path: Path,
    metric: str = 'sharpe_ratio',
    title: str = "Model Comparison with 95% Confidence Intervals",
) -> None:
    """
    Plot model comparison with confidence intervals.

    Args:
        model_results: Dictionary of model_name -> results_dict
        output_path: Output path for figure
        metric: Metric to plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract data
    model_names = []
    point_estimates = []
    lower_bounds = []
    upper_bounds = []

    for model_name, results in model_results.items():
        if metric in results:
            result = results[metric]
            model_names.append(model_name)
            point_estimates.append(result.get('point_estimate', 0))
            lower_bounds.append(result.get('lower_bound', 0))
            upper_bounds.append(result.get('upper_bound', 0))

    if not model_names:
        print(f"No data to plot for metric: {metric}")
        return

    # Plot
    x = np.arange(len(model_names))
    errors = [
        [pe - lb for pe, lb in zip(point_estimates, lower_bounds)],
        [ub - pe for pe, ub in zip(point_estimates, upper_bounds)],
    ]

    ax.errorbar(
        x, point_estimates, yerr=errors,
        fmt='o', markersize=8, capsize=5, capthick=2,
        linewidth=2, label='Point Estimate with 95% CI'
    )

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved Confidence Intervals plot to {output_path}")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Pareto frontier for KAIROS-X evaluation")
    parser.add_argument("--results-dir", type=Path, required=True, help="Directory containing result JSON files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for plots")
    parser.add_argument("--models", type=str, nargs='+', default=None, help="Model names to include")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load all result files
    print(f"Loading resu from {args.results_dir}...")
    model_results = {}

    for result_file in args.results_dir.glob("*.json"):
        model_name = result_file.stem
        if args.models is None or model_name in args.models:
            results = load_results(result_file)
            model_results[model_name] = results
            print(f"  Loaded {model_name}")

    if not model_results:
        print("No results found!")
        return

    print(f"\nLoaded {len(model_results)} models")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("\nGenerating plots...")

    # 1. Sharpe vs. Volatility
    plot_sharpe_vs_volatility(
        model_results,
        args.output_dir / "pareto_sharpe_volatility.png"
    )

    # 2. Return vs. Drawdown
    plot_return_vs_drawdown(
        model_results,
        args.output_dir / "pareto_return_drawdown.png"
    )

    # 3. Multi-objective Pareto
    plot_multi_objective_pareto(
        model_results,
        args.output_dir / "pareto_multi_objective.png"
    )

    # 4. Confidence intervals for each metric
    for metric in ['sharpe_ratio', 'cumulative_return', 'directional_accuracy', 'rank_ic']:
        plot_confidence_intervals(
            model_results,
            args.output_dir / f"ci_{metric}.png",
            metric=metric,
            title=f"{metric.replace('_', ' ').title()} Comparison with 95% CI"
        )

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
