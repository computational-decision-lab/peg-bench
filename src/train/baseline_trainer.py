"""Trainer for the Kairos baseline model."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.datasets import DatasetSpec, LOBLatentSequenceDataset, LOBTokenSequenceDataset
from src.eval.metrics import directional_accuracy, rank_ic
from src.models.baseline import KairosBaselineConfig, KairosBaselineModel, baseline_loss
from src.utils.device import amp_enabled, resolve_torch_device
from src.utils.logging import setup_logging
from src.utils.seed import set_global_seed


@dataclass(slots=True)
class BaselineTrainConfig:
    """Training config for the prediction baseline."""

    encoded_path: str
    dataset_path: str
    output_dir: str
    input_mode: str = "codes"
    val_encoded_path: str | None = None
    val_dataset_path: str | None = None
    test_encoded_path: str | None = None
    test_dataset_path: str | None = None
    seq_len: int = 96
    stride: int = 50
    target_key: str = "return_20"
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    val_fraction: float = 0.17
    test_fraction: float = 0.17
    device: str = "cpu"
    amp: bool = False
    seed: int = 7
    model: KairosBaselineConfig = field(default_factory=KairosBaselineConfig)


def _split_indices(length: int, val_fraction: float, test_fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(length)
    val_size = int(length * val_fraction)
    test_size = int(length * test_fraction)
    train_end = max(length - val_size - test_size, 1)
    val_end = min(train_end + val_size, length - test_size if test_size > 0 else length)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def _evaluate(
    model: KairosBaselineModel,
    loader: DataLoader,
    input_mode: str,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    losses: list[float] = []
    with torch.no_grad():
        for inputs, batch_targets in loader:
            inputs = inputs.to(device)
            batch_targets = batch_targets.to(device)
            if input_mode == "codes":
                outputs = model(input_ids=inputs)
            else:
                outputs = model(latents=inputs)
            loss_out = baseline_loss(outputs["return_pred"], batch_targets, model.config.rankic_weight)
            losses.append(float(loss_out["loss"].item()))
            preds.append(outputs["return_pred"].cpu())
            targets.append(batch_targets.cpu())
    pred_np = torch.cat(preds).numpy() if preds else np.zeros(0, dtype=np.float32)
    tgt_np = torch.cat(targets).numpy() if targets else np.zeros(0, dtype=np.float32)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "directional_accuracy": directional_accuracy(pred_np, tgt_np),
        "rank_ic": rank_ic(pred_np, tgt_np),
    }


def _build_sequence_dataset(
    input_mode: str,
    encoded_path: str,
    dataset_path: str,
    spec: DatasetSpec,
) -> LOBTokenSequenceDataset | LOBLatentSequenceDataset:
    """Create the appropriate sequence dataset for baseline training."""
    if input_mode == "codes":
        return LOBTokenSequenceDataset(encoded_path, dataset_path, spec)
    return LOBLatentSequenceDataset(encoded_path, dataset_path, spec)


def train_baseline(config: BaselineTrainConfig) -> dict[str, object]:
    """Train the prediction baseline and save metrics/predictions."""
    set_global_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir / "logs", name="kairosx-baseline")
    device = resolve_torch_device(config.device)

    spec = DatasetSpec(seq_len=config.seq_len, stride=config.stride, target_key=config.target_key)
    train_dataset = _build_sequence_dataset(config.input_mode, config.encoded_path, config.dataset_path, spec)
    if config.val_encoded_path and config.val_dataset_path and config.test_encoded_path and config.test_dataset_path:
        val_dataset = _build_sequence_dataset(config.input_mode, config.val_encoded_path, config.val_dataset_path, spec)
        test_dataset = _build_sequence_dataset(config.input_mode, config.test_encoded_path, config.test_dataset_path, spec)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    else:
        train_idx, val_idx, test_idx = _split_indices(len(train_dataset), config.val_fraction, config.test_fraction)
        train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(Subset(train_dataset, val_idx), batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(Subset(train_dataset, test_idx), batch_size=config.batch_size, shuffle=False)

    model_cfg = config.model
    model_cfg.input_mode = config.input_mode
    model_cfg.seq_len = config.seq_len
    model = KairosBaselineModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled(config.amp, device))

    history: list[dict[str, float]] = []
    best_val_rankic = -1e9
    best_path = output_dir / "best_baseline.pt"

    for epoch in range(1, config.epochs + 1):
        model.train()
        losses: list[float] = []
        for inputs, targets in tqdm(train_loader, desc=f"Baseline epoch {epoch}"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled(config.amp, device)):
                outputs = model(input_ids=inputs if config.input_mode == "codes" else None, latents=inputs if config.input_mode == "latents" else None)
                loss_out = baseline_loss(outputs["return_pred"], targets, model_cfg.rankic_weight)
            scaler.scale(loss_out["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss_out["loss"].item()))

        train_metrics = {"loss": float(np.mean(losses)) if losses else 0.0}
        val_metrics = _evaluate(model, val_loader, config.input_mode, device)
        test_metrics = _evaluate(model, test_loader, config.input_mode, device)
        epoch_metrics = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        history.append(epoch_metrics)
        logger.info("epoch=%d train_loss=%.5f val_rankIC=%.4f val_dir=%.4f test_rankIC=%.4f", epoch, train_metrics["loss"], val_metrics["rank_ic"], val_metrics["directional_accuracy"], test_metrics["rank_ic"])

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": asdict(model_cfg),
            "epoch": epoch,
            "history": history,
        }
        torch.save(checkpoint, output_dir / f"baseline_epoch_{epoch}.pt")
        if val_metrics["rank_ic"] > best_val_rankic:
            best_val_rankic = val_metrics["rank_ic"]
            torch.save(checkpoint, best_path)

    (output_dir / "baseline_history.json").write_text(json.dumps(history, indent=2))
    summary = {
        "best_checkpoint": str(best_path.resolve()),
        "history_path": str((output_dir / "baseline_history.json").resolve()),
        "final_val_rank_ic": best_val_rankic,
    }
    return summary


def cli_main() -> None:
    """CLI wrapper."""
    parser = argparse.ArgumentParser(description="Train the Kairos baseline model.")
    parser.add_argument("--encoded", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--input-mode", type=str, default="codes", choices=["codes", "latents"])
    parser.add_argument("--val-encoded", type=str, default=None)
    parser.add_argument("--val-dataset", type=str, default=None)
    parser.add_argument("--test-encoded", type=str, default=None)
    parser.add_argument("--test-dataset", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--rankic-weight", type=float, default=0.1)
    args = parser.parse_args()
    train_baseline(
        BaselineTrainConfig(
            encoded_path=args.encoded,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            input_mode=args.input_mode,
            val_encoded_path=args.val_encoded,
            val_dataset_path=args.val_dataset,
            test_encoded_path=args.test_encoded,
            test_dataset_path=args.test_dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            device=args.device,
            model=KairosBaselineConfig(input_mode=args.input_mode, seq_len=args.seq_len, rankic_weight=args.rankic_weight),
        )
    )
