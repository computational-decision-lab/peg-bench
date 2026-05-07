"""
Ablation studies for KAIROS-X.
Implements ablation analysis from paper Section 7.4.

执行位置: ☁️ 云端（AutoDL服务器）
数据范围: 2025-10-01 to 2025-10-30 (30 days)

Ablation studies to understand component contributions:
1. Without preference alignment (baseline)
2. Without oracle guidance (random preferences)
3. Without entropy regularization
4. Without risk conditioning
5. Full KAIROS-X

Each ablation trains a separate model variant.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class AblationConfig:
    """Configuration for an ablation experiment."""
    name: str
    description: str
    use_preference_alignment: bool = True
    use_oracle_guidance: bool = True
    use_entropy_regularization: bool = True
    use_risk_conditioning: bool = True

    # Training hyperparameters
    learning_rate: float = 1e-5
    batch_size: int = 32
    num_epochs: int = 10


@dataclass
class AblationResult:
    """Results from an ablation experiment."""
    config_name: str
    description: str

    # Performance metrics
    sharpe_ratio: float
    cumulative_return: float
    max_drawdown: float
    volatility: float
    directional_accuracy: float
    rank_ic: float

    # Training metrics
    training_time: float
    num_parameters: int
    final_loss: float


class AblationStudy:
    """
    Conduct ablation studies for KAIROS-X.

    Tests the contribution of each component:
    1. Preference alignment (DPO)
    2. Oracle guidance (vs random preferences)
    3. Entropy regularization
    4. Risk conditioning
    """

    def __init__(
        self,
        base_model_path: Path,
        dataset_path: Path,
        device: torch.device,
    ):
        self.base_model_path = base_model_path
        self.dataset_path = dataset_path
        self.device = device

    def create_ablation_configs(self) -> List[AblationConfig]:
        """Create configurations for all ablation experiments."""
        configs = []

        # 1. Full model (baseline for comparison)
        configs.append(AblationConfig(
            name="full_kairos_x",
            description="Full KAIROS-X with all components",
            use_preference_alignment=True,
            use_oracle_guidance=True,
            use_entropy_regularization=True,
            use_risk_conditioning=True,
        ))

        # 2. Without preference alignment (standard supervised learning)
        configs.append(AblationConfig(
            name="no_preference_alignment",
            description="Without DPO preference alignment (baseline)",
            use_preference_alignment=False,
            use_oracle_guidance=True,
            use_entropy_regularization=True,
            use_risk_conditioning=True,
        ))

        # 3. Without oracle guidance (random preferences)
        configs.append(AblationConfig(
            name="no_oracle_guidance",
            description="With random preferences instead of oracle",
            use_preference_alignment=True,
            use_oracle_guidance=False,
            use_entropy_regularization=True,
            use_risk_conditioning=True,
        ))

        # 4. Without entropy regularization
        configs.append(AblationConfig(
            name="no_entropy_regularization",
            description="Without entropy regularization in DPO",
            use_preference_alignment=True,
            use_oracle_guidance=True,
            use_entropy_regularization=False,
            use_risk_conditioning=True,
        ))

        # 5. Without risk conditioning
        configs.append(AblationConfig(
            name="no_risk_conditioning",
            description="Without risk-conditioned preferences",
            use_preference_alignment=True,
            use_oracle_guidance=True,
            use_entropy_regularization=True,
            use_risk_conditioning=False,
        ))

        return configs

    def run_ablation(
        self,
        config: AblationConfig,
    ) -> AblationResult:
        """
        Run a single ablation experiment.

        Args:
            config: Ablation configuration

        Returns:
            AblationResult with performance metrics
        """
        print(f"\nRunning ablation: {config.name}")
        print(f"Description: {config.description}")

        # Import necessary modules
        from kronos_scaled.model.transformer import LOBTransformer
        from kronos_scaled.training.train_kairos import train_model, evaluate_model

        # Load base model
        checkpoint = torch.load(self.base_model_path, map_location=self.device)
        model_config = checkpoint.get('config', {})

        # Create model
        model = LOBTransformer(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)

        # Load dataset
        data = np.load(self.dataset_path)

        # Configure training based on ablation settings
        if not config.use_preference_alignment:
            # Standard supervised learning
            print("  Using standard supervised learning (no DPO)")
            # Train with MSE loss
            results = self._train_supervised(model, data, config)

        elif not config.use_oracle_guidance:
            # Use random preferences
            print("  Using random preferences (no oracle)")
            results = self._train_with_random_preferences(model, data, config)

        elif not config.use_entropy_regularization:
            # DPO without entropy regularization
            print("  Using DPO without entropy regularization")
            results = self._train_dpo(model, data, config, entropy_coef=0.0)

        elif not config.use_risk_conditioning:
            # DPO without risk conditioning
            print("  Using DPO without risk conditioning")
            results = self._train_dpo_no_risk(model, data, config)

        else:
            # Full KAIROS-X
            print("  Using full KAIROS-X")
            results = self._train_dpo(model, data, config, entropy_coef=0.01)

        # Evaluate model
        eval_metrics = self._evaluate(model, data)

        # Create result
        result = AblationResult(
            config_name=config.name,
            description=config.description,
            sharpe_ratio=eval_metrics['sharpe_ratio'],
            cumulative_return=eval_metrics['cumulative_return'],
            max_drawdown=eval_metrics['max_drawdown'],
            volatility=eval_metrics['volatility'],
            directional_accuracy=eval_metrics['directional_accuracy'],
            rank_ic=eval_metrics['rank_ic'],
            training_time=results['training_time'],
            num_parameters=results['num_parameters'],
            final_loss=results['final_loss'],
        )

        return result

    def _train_supervised(
        self,
        model: nn.Module,
        data: Dict[str, np.ndarray],
        config: AblationConfig,
    ) -> Dict[str, Any]:
        """Train with standard supervised learning."""
        import time

        start_time = time.time()

        # Simple supervised training (placeholder)
        # In practice, this would use the existing training code
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        criterion = nn.MSELoss()

        # Training loop (simplified)
        model.train()
        for epoch in range(config.num_epochs):
            # Training code here
            pass

        training_time = time.time() - start_time
        num_parameters = sum(p.numel() for p in model.parameters())

        return {
            'training_time': training_time,
            'num_parameters': num_parameters,
            'final_loss': 0.0,  # Placeholder
        }

    def _train_with_random_preferences(
        self,
        model: nn.Module,
        data: Dict[str, np.ndarray],
        config: AblationConfig,
    ) -> Dict[str, Any]:
        """Train with random preferences instead of oracle."""
        import time

        start_time = time.time()

        # Generate random preferences
        # In practice, this would randomly assign winner/loser labels

        # Train with DPO using random preferences
        # (Similar to _train_dpo but with random preferences)

        training_time = time.time() - start_time
        num_parameters = sum(p.numel() for p in model.parameters())

        return {
            'training_time': training_time,
            'num_parameters': num_parameters,
            'final_loss': 0.0,  # Placeholder
        }

    def _train_dpo(
        self,
        model: nn.Module,
        data: Dict[str, np.ndarray],
        config: AblationConfig,
        entropy_coef: float = 0.01,
    ) -> Dict[str, Any]:
        """Train with DPO."""
        import time

        start_time = time.time()

        # DPO training (placeholder)
        # In practice, this would use the DPO training code

        training_time = time.time() - start_time
        num_parameters = sum(p.numel() for p in model.parameters())

        return {
            'training_time': training_time,
            'num_parameters': num_parameters,
            'final_loss': 0.0,  # Placeholder
        }

    def _train_dpo_no_risk(
        self,
        model: nn.Module,
        data: Dict[str, np.ndarray],
        config: AblationConfig,
    ) -> Dict[str, Any]:
        """Train with DPO without risk conditioning."""
        import time

        start_time = time.time()

        # DPO training without risk conditioning
        # Use single risk parameter instead of multiple

        training_time = time.time() - start_time
        num_parameters = sum(p.numel() for p in model.parameters())

        return {
            'training_time': training_time,
            'num_parameters': num_parameters,
            'final_loss': 0.0,  # Placeholder
        }

    def _evaluate(
        self,
        model: nn.Module,
        data: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """Evaluate model and compute metrics."""
        model.eval()

        # Placeholder evaluation
        # In practice, this would use the evaluation code

        return {
            'sharpe_ratio': 0.0,
            'cumulative_return': 0.0,
            'max_drawdown': 0.0,
            'volatility': 0.0,
            'directional_accuracy': 0.0,
            'rank_ic': 0.0,
        }

    def run_all_ablations(self) -> List[AblationResult]:
        """Run all ablation experiments."""
        configs = self.create_ablation_configs()
        results = []

        for config in configs:
            result = self.run_ablation(config)
            results.append(result)

        return results

    def compare_ablations(
        self,
        results: List[AblationResult],
    ) -> Dict[str, Any]:
        """Compare ablation results."""
        # Find full model result
        full_result = None
        for result in results:
            if result.config_name == "full_kairos_x":
                full_result = result
                break

        if full_result is None:
            return {}

        # Compute relative performance
        comparison = {}

        for result in results:
            if result.config_name == "full_kairos_x":
                continue

            relative_performance = {
                'sharpe_ratio': (result.sharpe_ratio - full_result.sharpe_ratio) / (abs(full_result.sharpe_ratio) + 1e-8),
                'cumulative_return': (result.cumulative_return - full_result.cumulative_return) / (abs(full_result.cumulative_return) + 1e-8),
                'directional_accuracy': (result.directional_accuracy - full_result.directional_accuracy) / (abs(full_result.directional_accuracy) + 1e-8),
                'rank_ic': (result.rank_ic - full_result.rank_ic) / (abs(full_result.rank_ic) + 1e-8),
            }

            comparison[result.config_name] = {
                'description': result.description,
                'absolute_metrics': asdict(result),
                'relative_to_full': relative_performance,
            }

        return comparison


def plot_ablation_results(
    results: List[AblationResult],
    output_path: Path,
) -> None:
    """Plot ablation study results."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    metrics = ['sharpe_ratio', 'cumulative_return', 'directional_accuracy', 'rank_ic']
    metric_labels = ['Sharpe Ratio', 'Cumulative Return', 'Directional Accuracy', 'Rank IC']

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]

        names = [r.config_name.replace('_', ' ').title() for r in results]
        values = [getattr(r, metric) for r in results]

        # Bar plot
        x = np.arange(len(names))
        bars = ax.bar(x, values, alpha=0.7, edgecolor='black')

        # Color full model differently
        for i, name in enumerate(names):
            if 'full' in name.lower():
                bars[i].set_color('green')
            else:
                bars[i].set_color('blue')

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f"Ablation Study: {label}", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, v in enumerate(values):
            ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved ablation plot to {output_path}")
    plt.close()


def save_results(
    results: List[AblationResult],
    comparison: Dict[str, Any],
    output_path: Path,
) -> None:
    """Save ablation results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        'results': [asdict(r) for r in results],
        'comparison': comparison,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablation studies for KAIROS-X")
    parser.add_argument("--model", type=Path, required=True, help="Path to base model checkpoint")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset (.npz)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create ablation study
    study = AblationStudy(
        base_model_path=args.model,
        dataset_path=args.dataset,
        device=device,
    )

    # Run all ablations
    print("\nRunning ablation studies...")
    results = study.run_all_ablations()

    # Compare results
    print("\nComparing ablation results...")
    comparison = study.compare_ablations(results)

    # Print results
    print("\nAblation Study Results:")
    for result in results:
        print(f"\n{result.config_name}:")
        print(f"  Description: {result.description}")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.3f}")
        print(f"  Cumulative Return: {result.cumulative_return:.4f}")
        print(f"  Dir Acc: {result.directional_accuracy:.3f}")
        print(f"  Rank IC: {result.rank_ic:.3f}")
        print(f"  Training Time: {result.training_time:.1f}s")

    # Save results
    output_path = args.output_dir / "ablation_results.json"
    save_results(results, comparison, output_path)
    print(f"\nSaved results to {output_path}")

    # Generate plot
    plot_path = args.output_dir / "ablation_comparison.png"
    plot_ablation_results(results, plot_path)

    print("\nAblation study complete!")


if __name__ == "__main__":
    main()
