"""Trainer for the scale-invariant VQ-VAE representation module."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.datasets import LOBAutoencoderDataset
from src.models.vqvae import ScaleInvariantVQVAE, VQVAEConfig
from src.utils.device import amp_enabled, resolve_torch_device
from src.utils.logging import setup_logging
from src.utils.seed import set_global_seed


@dataclass(slots=True)
class VQVAETrainConfig:
    """Training config for the representation model."""

    dataset_path: str
    output_dir: str
    batch_size: int = 128
    epochs: int = 10
    learning_rate: float = 1e-3
    num_workers: int = 0
    device: str = "cpu"
    amp: bool = False
    seed: int = 7
    max_train_items: int | None = None
    model: VQVAEConfig = field(default_factory=VQVAEConfig)


def _flatten_latents(latents: np.ndarray) -> np.ndarray:
    """Flatten latent tensors into `(N, D)` for clustering/export."""
    return latents.reshape(latents.shape[0], -1)


def _encode_dataset(
    model: ScaleInvariantVQVAE,
    dataset_path: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode an arbitrary dataset with a trained VQ-VAE."""
    dataset = LOBAutoencoderDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    all_codes: list[np.ndarray] = []
    all_latents: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            codes, latents = model.encode_codes(batch)
            all_codes.append(codes.cpu().numpy())
            all_latents.append(latents.cpu().numpy())
    if not all_codes:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(all_codes, axis=0), np.concatenate(all_latents, axis=0)


def _apply_fallback_codebook(latents: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Assign codes by nearest fallback centroid."""
    flat_latents = _flatten_latents(latents).astype(np.float32)
    centers = centers.astype(np.float32)
    diff = flat_latents[:, None, :] - centers[None, :, :]
    distances = np.sum(diff * diff, axis=2)
    return np.argmin(distances, axis=1).astype(np.int64)


def train_vqvae(config: VQVAETrainConfig) -> dict[str, object]:
    """Train the VQ-VAE and export discrete codes/latents for all samples."""
    set_global_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir / "logs", name="kairosx-vqvae")
    device = resolve_torch_device(config.device)

    dataset = LOBAutoencoderDataset(config.dataset_path)
    train_dataset = dataset
    if config.max_train_items is not None and config.max_train_items > 0 and config.max_train_items < len(dataset):
        indices = np.linspace(0, len(dataset) - 1, config.max_train_items, dtype=np.int64).tolist()
        train_dataset = Subset(dataset, indices)
    loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    model = ScaleInvariantVQVAE(config.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled(config.amp, device))

    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        running = {"loss": 0.0, "recon_loss": 0.0, "symmetry_loss": 0.0, "perplexity": 0.0}
        for batch in tqdm(loader, desc=f"VQ-VAE epoch {epoch}"):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled(config.amp, device)):
                outputs = model(batch, with_symmetry=True)
                loss = outputs["loss"]
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running["loss"] += float(outputs["loss"].item())
            running["recon_loss"] += float(outputs["recon_loss"].item())
            running["symmetry_loss"] += float(outputs["symmetry_loss"].item())
            running["perplexity"] += float(outputs["perplexity"].item())

        steps = max(len(loader), 1)
        epoch_metrics = {name: value / steps for name, value in running.items()}
        epoch_metrics["epoch"] = epoch
        history.append(epoch_metrics)
        logger.info("epoch=%d loss=%.5f recon=%.5f sym=%.5f perplexity=%.3f", epoch, epoch_metrics["loss"], epoch_metrics["recon_loss"], epoch_metrics["symmetry_loss"], epoch_metrics["perplexity"])
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": asdict(config.model),
                "epoch": epoch,
                "history": history,
            },
            output_dir / f"vqvae_epoch_{epoch}.pt",
        )

    codes_np, latents_np = _encode_dataset(
        model,
        config.dataset_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=device,
    )
    codes_source = "vq_codebook"
    unique_codes = np.unique(codes_np)
    fallback_codebook_path = None
    min_expected_codes = max(2, min(config.model.num_embeddings // 8, 8))
    if len(unique_codes) < min_expected_codes:
        flat_latents = _flatten_latents(latents_np)
        n_clusters = min(config.model.num_embeddings, len(flat_latents))
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=config.seed, batch_size=min(4096, len(flat_latents)))
        codes_np = kmeans.fit_predict(flat_latents).astype(np.int64)
        codes_source = "kmeans_fallback"
        fallback_codebook_path = output_dir / "fallback_codebook.npz"
        np.savez_compressed(fallback_codebook_path, centers=kmeans.cluster_centers_.astype(np.float32))
        logger.warning("VQ codebook collapsed to %d unique codes; using KMeans fallback with %d clusters", len(unique_codes), n_clusters)
    encoded_path = output_dir / "encoded_tokens.npz"
    np.savez_compressed(encoded_path, codes=codes_np, latents=latents_np)
    (output_dir / "vqvae_history.json").write_text(json.dumps(history, indent=2))
    summary = {
        "checkpoint": str((output_dir / f"vqvae_epoch_{config.epochs}.pt").resolve()),
        "encoded_tokens": str(encoded_path.resolve()),
        "history_path": str((output_dir / "vqvae_history.json").resolve()),
        "codes_source": codes_source,
        "num_unique_codes": int(len(np.unique(codes_np))),
        "fallback_codebook": str(fallback_codebook_path.resolve()) if fallback_codebook_path is not None else None,
    }
    return summary


def export_vqvae_encodings(
    checkpoint_path: str,
    dataset_path: str,
    output_path: str,
    batch_size: int = 128,
    num_workers: int = 0,
    device: str = "cpu",
    fallback_codebook_path: str | None = None,
) -> dict[str, object]:
    """Export codes/latents for a dataset using an existing trained VQ-VAE."""
    resolved_device = resolve_torch_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
    model = ScaleInvariantVQVAE(VQVAEConfig(**checkpoint["config"])).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    native_codes, latents = _encode_dataset(
        model,
        dataset_path,
        batch_size=batch_size,
        num_workers=num_workers,
        device=resolved_device,
    )
    codes = native_codes
    codes_source = "vq_codebook"
    if fallback_codebook_path:
        centers = np.load(fallback_codebook_path)["centers"]
        codes = _apply_fallback_codebook(latents, centers)
        codes_source = "kmeans_fallback"
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, codes=codes.astype(np.int64), latents=latents.astype(np.float32))
    return {
        "encoded_tokens": str(output.resolve()),
        "codes_source": codes_source,
        "num_unique_codes": int(len(np.unique(codes))),
    }


def cli_main() -> None:
    """CLI wrapper."""
    parser = argparse.ArgumentParser(description="Train the KAIROS-X scale-invariant VQ-VAE.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-embeddings", type=int, default=64)
    parser.add_argument("--beta-sym", type=float, default=0.1)
    parser.add_argument("--max-train-items", type=int, default=None)
    args = parser.parse_args()
    train_vqvae(
        VQVAETrainConfig(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            max_train_items=args.max_train_items,
            model=VQVAEConfig(num_embeddings=args.num_embeddings, beta_sym=args.beta_sym),
        )
    )
