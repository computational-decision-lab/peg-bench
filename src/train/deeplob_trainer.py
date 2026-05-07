"""Trainer for the DeepLOB-inspired time-bar baseline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.datasets import DatasetSpec, LOBRawSequenceDataset
from src.eval.metrics import directional_accuracy, rank_ic
from src.models.baseline import baseline_loss
from src.models.deeplob import DeepLOBConfig, DeepLOBTimeBarModel
from src.utils.device import amp_enabled, resolve_torch_device
from src.utils.logging import setup_logging
from src.utils.seed import set_global_seed


@dataclass(slots=True)
class DeepLOBTrainConfig:
    """Training config for the DeepLOB-style baseline."""

    train_dataset_path: str
    val_dataset_path: str
    test_dataset_path: str
    output_dir: str
    seq_len: int = 50
    stride: int = 50
    target_key: str = "return_20"
    batch_size: int = 256
    epochs: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    device: str = "cpu"
    amp: bool = False
    seed: int = 7
    model: DeepLOBConfig = field(default_factory=DeepLOBConfig)


def _evaluate(
    model: DeepLOBTimeBarModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    losses: list[float] = []
    with torch.no_grad():
        for lob, batch_targets in loader:
            lob = lob.to(device)
            batch_targets = batch_targets.to(device)
            outputs = model(lob)
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


def train_deeplob(config: DeepLOBTrainConfig) -> dict[str, object]:
    """Train DeepLOB on raw time-bar windows and save checkpoints."""
    set_global_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir / "logs", name="peg-deeplob")
    device = resolve_torch_device(config.device)

    spec = DatasetSpec(seq_len=config.seq_len, stride=config.stride, target_key=config.target_key)
    train_dataset = LOBRawSequenceDataset(config.train_dataset_path, spec)
    val_dataset = LOBRawSequenceDataset(config.val_dataset_path, spec)
    test_dataset = LOBRawSequenceDataset(config.test_dataset_path, spec)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model_cfg = config.model
    model_cfg.seq_len = config.seq_len
    model = DeepLOBTimeBarModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled(config.amp, device))

    history: list[dict[str, float]] = []
    best_val_rankic = -1e9
    best_path = output_dir / "best_deeplob.pt"

    for epoch in range(1, config.epochs + 1):
        model.train()
        losses: list[float] = []
        for lob, targets in tqdm(train_loader, desc=f"DeepLOB epoch {epoch}"):
            lob = lob.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled(config.amp, device)):
                outputs = model(lob)
                loss_out = baseline_loss(outputs["return_pred"], targets, model_cfg.rankic_weight)
            scaler.scale(loss_out["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss_out["loss"].item()))

        train_metrics = {"loss": float(np.mean(losses)) if losses else 0.0}
        val_metrics = _evaluate(model, val_loader, device)
        test_metrics = _evaluate(model, test_loader, device)
        epoch_metrics = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        history.append(epoch_metrics)
        logger.info(
            "epoch=%d train_loss=%.5f val_rankIC=%.4f val_dir=%.4f test_rankIC=%.4f",
            epoch,
            train_metrics["loss"],
            val_metrics["rank_ic"],
            val_metrics["directional_accuracy"],
            test_metrics["rank_ic"],
        )
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": asdict(model_cfg),
            "epoch": epoch,
            "history": history,
        }
        torch.save(checkpoint, output_dir / f"deeplob_epoch_{epoch}.pt")
        if val_metrics["rank_ic"] > best_val_rankic:
            best_val_rankic = val_metrics["rank_ic"]
            torch.save(checkpoint, best_path)

    (output_dir / "deeplob_history.json").write_text(json.dumps(history, indent=2))
    return {
        "best_checkpoint": str(best_path.resolve()),
        "history_path": str((output_dir / "deeplob_history.json").resolve()),
        "final_val_rank_ic": best_val_rankic,
    }


def predict_deeplob(
    *,
    checkpoint_path: str,
    dataset_path: str,
    seq_len: int,
    stride: int,
    target_key: str,
    device: str,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Run batched inference with a trained DeepLOB checkpoint."""
    torch_device = resolve_torch_device(device)
    payload = torch.load(checkpoint_path, map_location=torch_device)
    model = DeepLOBTimeBarModel(DeepLOBConfig(**payload["config"])).to(torch_device)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.eval()

    spec = DatasetSpec(seq_len=seq_len, stride=stride, target_key=target_key)
    dataset = LOBRawSequenceDataset(dataset_path, spec)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for lob, batch_targets in loader:
            lob = lob.to(torch_device)
            batch_pred = model(lob)["return_pred"].cpu().numpy().astype(np.float32)
            preds.append(batch_pred)
            targets.append(batch_targets.numpy().astype(np.float32))
    predictions = np.concatenate(preds, axis=0) if preds else np.zeros(0, dtype=np.float32)
    truth = np.concatenate(targets, axis=0) if targets else np.zeros(0, dtype=np.float32)
    return predictions, truth
