"""Visualization helpers for the KAIROS-X research workflow."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def plot_code_usage(codes: np.ndarray, output_path: str | Path) -> None:
    """Plot codebook usage histogram."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    unique, counts = np.unique(codes, return_counts=True)
    plt.bar(unique, counts)
    plt.xlabel("Code ID")
    plt.ylabel("Count")
    plt.title("VQ Code Usage Histogram")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_embedding_tsne(latents: np.ndarray, codes: np.ndarray, output_path: str | Path, max_points: int = 2_000) -> None:
    """Plot a t-SNE projection of latent states."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(latents) > max_points:
        indices = np.linspace(0, len(latents) - 1, max_points).astype(int)
        latents = latents[indices]
        codes = codes[indices]
    embedding = TSNE(n_components=2, init="pca", learning_rate="auto").fit_transform(latents)
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=codes, s=10, cmap="tab20")
    plt.colorbar(scatter, label="Code")
    plt.title("VQ Latent t-SNE")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_reconstruction_heatmap(original: np.ndarray, reconstructed: np.ndarray, output_path: str | Path) -> None:
    """Plot original vs reconstruction heatmaps."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, original.shape[0], figsize=(3 * original.shape[0], 4))
    for channel in range(original.shape[0]):
        axes[0, channel].imshow(original[channel], aspect="auto", cmap="coolwarm")
        axes[0, channel].set_title(f"Original C{channel}")
        axes[1, channel].imshow(reconstructed[channel], aspect="auto", cmap="coolwarm")
        axes[1, channel].set_title(f"Recon C{channel}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_training_curves(history: list[dict[str, float]], keys: list[str], output_path: str | Path) -> None:
    """Plot arbitrary scalar histories."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    epochs = [entry.get("epoch", idx + 1) for idx, entry in enumerate(history)]
    for key in keys:
        plt.plot(epochs, [entry.get(key, np.nan) for entry in history], label=key)
    plt.legend()
    plt.xlabel("Epoch")
    plt.title("Training Curves")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_cost_sensitivity(cost_scan: dict[str, dict[str, float]], output_path: str | Path) -> None:
    """Plot Sharpe and turnover across cost regimes."""
    costs = sorted(float(key) for key in cost_scan.keys())
    sharpes = [cost_scan[str(cost)]["sharpe"] for cost in costs]
    turnovers = [cost_scan[str(cost)]["turnover"] for cost in costs]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(costs, sharpes, marker="o", label="Sharpe")
    ax1.set_xlabel("Cost (bps)")
    ax1.set_ylabel("Sharpe")
    ax2 = ax1.twinx()
    ax2.plot(costs, turnovers, marker="s", color="orange", label="Turnover")
    ax2.set_ylabel("Turnover")
    fig.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_aci_diagnostics(scores: list[float], output_path: str | Path) -> None:
    """Plot nonconformity score diagnostics."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(scores)
    plt.title("ACI Nonconformity Scores")
    plt.xlabel("Observation")
    plt.ylabel("Absolute error")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
