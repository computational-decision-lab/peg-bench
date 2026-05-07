"""Continuous-action DPO training with reference-policy anchoring."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.models.baseline import KairosBaselineConfig, KairosBaselineModel
from src.models.policy import GaussianActionPolicy, GaussianPolicyConfig, ema_update_reference
from src.preference.dataset import PreferenceTrajectoryDataset
from src.simulation.execution import ExecutionSimulator, SimulatorConfig
from src.utils.device import resolve_torch_device
from src.utils.logging import setup_logging
from src.utils.seed import set_global_seed


@dataclass(slots=True)
class DPOTrainConfig:
    """Config for continuous-action DPO."""

    preferences_path: str
    reference_checkpoint: str
    output_dir: str
    beta: float = 0.1
    learning_rate: float = 3e-4
    batch_size: int = 32
    epochs: int = 10
    entropy_coef: float = 0.01
    covariance_floor: float = 1e-4
    reference_update_tau: float = 0.01
    reference_update_interval: int = 50
    validation_fraction: float = 0.1
    device: str = "cpu"
    seed: int = 7
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)


def _load_reference_policy(config: DPOTrainConfig, device: torch.device) -> tuple[GaussianActionPolicy, GaussianActionPolicy]:
    payload = torch.load(config.reference_checkpoint, map_location=device)
    baseline_cfg = KairosBaselineConfig(**payload["config"])
    reference_backbone = KairosBaselineModel(baseline_cfg).to(device)
    reference_backbone.load_state_dict(payload["model_state_dict"], strict=False)
    reference_policy = GaussianActionPolicy(GaussianPolicyConfig(baseline=baseline_cfg), backbone=reference_backbone).to(device)

    policy_backbone = KairosBaselineModel(baseline_cfg).to(device)
    policy_backbone.load_state_dict(payload["model_state_dict"], strict=False)
    policy = GaussianActionPolicy(GaussianPolicyConfig(baseline=baseline_cfg), backbone=policy_backbone).to(device)
    return policy, reference_policy


class ContinuousDPOTrainer:
    """Train a Gaussian policy from winner/loser preference pairs."""

    def __init__(self, config: DPOTrainConfig) -> None:
        self.config = config
        self.device = resolve_torch_device(config.device)
        self.policy, self.reference = _load_reference_policy(config, self.device)
        for parameter in self.reference.parameters():
            parameter.requires_grad = False
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=config.learning_rate)
        self.simulator = ExecutionSimulator(config.simulator)

    def _sequence_log_prob(
        self,
        policy: GaussianActionPolicy,
        tokens: torch.Tensor,
        actions: torch.Tensor,
        risk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, horizon, seq_len = tokens.shape
        flat_tokens = tokens.reshape(batch_size * horizon, seq_len)
        flat_actions = actions.reshape(batch_size * horizon)
        flat_risk_ids = risk_ids.repeat_interleave(horizon)
        log_prob, mean, std = policy.log_prob(actions=flat_actions, input_ids=flat_tokens, risk_ids=flat_risk_ids)
        return log_prob.reshape(batch_size, horizon).sum(dim=1), mean, std

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the continuous DPO objective."""
        risk_ids = batch["risk_ids"].to(self.device)
        winner_tokens = batch["winner_tokens"].to(self.device)
        loser_tokens = batch["loser_tokens"].to(self.device)
        winner_actions = batch["winner_actions"].to(self.device)
        loser_actions = batch["loser_actions"].to(self.device)

        policy_winner, _, winner_std = self._sequence_log_prob(self.policy, winner_tokens, winner_actions, risk_ids)
        policy_loser, _, loser_std = self._sequence_log_prob(self.policy, loser_tokens, loser_actions, risk_ids)
        with torch.no_grad():
            ref_winner, _, _ = self._sequence_log_prob(self.reference, winner_tokens, winner_actions, risk_ids)
            ref_loser, _, _ = self._sequence_log_prob(self.reference, loser_tokens, loser_actions, risk_ids)
        logits = self.config.beta * ((policy_winner - ref_winner) - (policy_loser - ref_loser))
        dpo_loss = -F.logsigmoid(logits).mean()

        entropy = torch.cat(
            [
                self.policy.entropy(input_ids=winner_tokens.reshape(-1, winner_tokens.shape[-1]), risk_ids=risk_ids.repeat_interleave(winner_tokens.shape[1])),
                self.policy.entropy(input_ids=loser_tokens.reshape(-1, loser_tokens.shape[-1]), risk_ids=risk_ids.repeat_interleave(loser_tokens.shape[1])),
            ]
        ).mean()
        covariance_penalty = F.relu(self.config.covariance_floor - winner_std).mean() + F.relu(self.config.covariance_floor - loser_std).mean()
        loss = dpo_loss - self.config.entropy_coef * entropy + covariance_penalty
        cov_det = float((winner_std.mean() * loser_std.mean()).detach().cpu().item())
        metrics = {
            "loss": float(loss.detach().cpu().item()),
            "dpo_loss": float(dpo_loss.detach().cpu().item()),
            "entropy": float(entropy.detach().cpu().item()),
            "cov_det": cov_det,
            "preference_accuracy": float((logits > 0).float().mean().detach().cpu().item()),
        }
        return loss, metrics

    def _validation_sharpe(self, loader: DataLoader) -> float:
        """Estimate validation Sharpe by replaying policy means on held-out trajectories."""
        self.policy.eval()
        sharpes: list[float] = []
        with torch.no_grad():
            for batch in loader:
                risk_ids = batch["risk_ids"].to(self.device)
                winner_tokens = batch["winner_tokens"].to(self.device)
                winner_lob = batch["winner_lob"].numpy()
                winner_returns = batch["winner_returns"].numpy()
                batch_size, horizon, seq_len = winner_tokens.shape
                flat_tokens = winner_tokens.reshape(batch_size * horizon, seq_len)
                flat_risk_ids = risk_ids.repeat_interleave(horizon)
                actions = self.policy.act(input_ids=flat_tokens, risk_ids=flat_risk_ids, deterministic=True).cpu().numpy()
                actions = actions.reshape(batch_size, horizon)
                for idx in range(batch_size):
                    timestamps = np.arange(horizon, dtype=np.int64)
                    simulated = self.simulator.simulate(
                        trajectory_id=f"val:{idx}",
                        context_id=idx,
                        variant="policy_mean",
                        risk_mode="validation",
                        state_tokens=winner_tokens[idx].cpu().numpy(),
                        lob_states=winner_lob[idx],
                        market_returns=winner_returns[idx],
                        timestamps=timestamps,
                        actions=actions[idx],
                        regime_tag="validation",
                    )
                    sharpes.append(simulated.metrics.sharpe)
        return float(np.mean(sharpes)) if sharpes else 0.0

    def train(self) -> dict[str, object]:
        """Run the full DPO training loop."""
        set_global_seed(self.config.seed)
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logging(output_dir / "logs", name="kairosx-dpo")

        dataset = PreferenceTrajectoryDataset(self.config.preferences_path)
        if len(dataset) == 0:
            raise ValueError("Preference dataset is empty.")
        val_size = max(1, int(len(dataset) * self.config.validation_fraction))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.seed),
        )
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=min(self.config.batch_size, len(val_ds)), shuffle=False)

        history: list[dict[str, float]] = []
        global_step = 0
        best_val_sharpe = -1e9
        best_path = output_dir / "best_dpo.pt"

        for epoch in range(1, self.config.epochs + 1):
            self.policy.train()
            epoch_metrics: list[dict[str, float]] = []
            for batch in tqdm(train_loader, desc=f"DPO epoch {epoch}"):
                self.optimizer.zero_grad(set_to_none=True)
                loss, metrics = self.compute_loss(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()
                global_step += 1
                if global_step % self.config.reference_update_interval == 0:
                    ema_update_reference(self.reference, self.policy, self.config.reference_update_tau)
                epoch_metrics.append(metrics)

            val_sharpe = self._validation_sharpe(val_loader)
            mean_metrics = {
                "epoch": epoch,
                "loss": float(np.mean([m["loss"] for m in epoch_metrics])),
                "dpo_loss": float(np.mean([m["dpo_loss"] for m in epoch_metrics])),
                "entropy": float(np.mean([m["entropy"] for m in epoch_metrics])),
                "cov_det": float(np.mean([m["cov_det"] for m in epoch_metrics])),
                "preference_accuracy": float(np.mean([m["preference_accuracy"] for m in epoch_metrics])),
                "validation_sharpe": val_sharpe,
            }
            history.append(mean_metrics)
            logger.info("epoch=%d loss=%.5f acc=%.4f entropy=%.4f cov_det=%.5f val_sharpe=%.4f", epoch, mean_metrics["loss"], mean_metrics["preference_accuracy"], mean_metrics["entropy"], mean_metrics["cov_det"], val_sharpe)

            checkpoint = {
                "policy_state_dict": self.policy.state_dict(),
                "reference_state_dict": self.reference.state_dict(),
                "config": asdict(self.config),
                "history": history,
            }
            torch.save(checkpoint, output_dir / f"dpo_epoch_{epoch}.pt")
            if val_sharpe > best_val_sharpe:
                best_val_sharpe = val_sharpe
                torch.save(checkpoint, best_path)

        (output_dir / "dpo_history.json").write_text(json.dumps(history, indent=2))
        return {
            "best_checkpoint": str(best_path.resolve()),
            "history_path": str((output_dir / "dpo_history.json").resolve()),
            "best_validation_sharpe": best_val_sharpe,
        }


def cli_main() -> None:
    """CLI wrapper."""
    parser = argparse.ArgumentParser(description="Train the continuous-action DPO policy.")
    parser.add_argument("--preferences", type=str, required=True)
    parser.add_argument("--reference-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    trainer = ContinuousDPOTrainer(
        DPOTrainConfig(
            preferences_path=args.preferences,
            reference_checkpoint=args.reference_checkpoint,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
        )
    )
    trainer.train()
