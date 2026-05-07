"""Kairos-style autoregressive baseline Transformer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(slots=True)
class KairosBaselineConfig:
    """Config for the baseline predictive model."""

    input_mode: str = "codes"
    vocab_size: int = 64
    latent_dim: int = 64
    seq_len: int = 96
    d_model: int = 256
    num_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1
    rankic_weight: float = 0.1


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)


def rank_ic(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Differentiable-ish rank correlation proxy based on centered Pearson ranks."""
    pred_rank = predictions.argsort().argsort().float()
    target_rank = targets.argsort().argsort().float()
    pred_rank = (pred_rank - pred_rank.mean()) / (pred_rank.std() + 1e-8)
    target_rank = (target_rank - target_rank.mean()) / (target_rank.std() + 1e-8)
    return torch.mean(pred_rank * target_rank)


class KairosBaselineModel(nn.Module):
    """Decoder-only style Transformer baseline for code or latent sequences."""

    def __init__(self, config: KairosBaselineConfig) -> None:
        super().__init__()
        self.config = config
        if config.input_mode == "codes":
            self.code_embedding = nn.Embedding(config.vocab_size, config.d_model)
            self.latent_projection = None
        else:
            self.code_embedding = None
            self.latent_projection = nn.Linear(config.latent_dim, config.d_model)

        self.position_embedding = nn.Embedding(config.seq_len, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=4 * config.d_model,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.return_head = nn.Linear(config.d_model, 1)
        self.direction_head = nn.Linear(config.d_model, 1)

    def _embed(self, input_ids: torch.Tensor | None, latents: torch.Tensor | None) -> torch.Tensor:
        if self.config.input_mode == "codes":
            if input_ids is None:
                raise ValueError("input_ids must be provided when input_mode='codes'.")
            hidden = self.code_embedding(input_ids)
        else:
            if latents is None:
                raise ValueError("latents must be provided when input_mode='latents'.")
            hidden = self.latent_projection(latents)
        positions = torch.arange(hidden.size(1), device=hidden.device)
        return hidden + self.position_embedding(positions).unsqueeze(0)

    def forward_features(
        self,
        input_ids: torch.Tensor | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return contextualized hidden states."""
        hidden = self._embed(input_ids, latents)
        mask = _causal_mask(hidden.size(1), hidden.device)
        return self.transformer(hidden, mask=mask)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        latents: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict next-period return and direction from a state sequence."""
        hidden = self.forward_features(input_ids=input_ids, latents=latents)
        pooled = self.norm(hidden[:, -1])
        return {
            "features": pooled,
            "return_pred": self.return_head(pooled).squeeze(-1),
            "direction_logit": self.direction_head(pooled).squeeze(-1),
        }


def baseline_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    rankic_weight: float,
) -> dict[str, torch.Tensor]:
    """Combined MSE + RankIC loss used by the Kairos baseline."""
    mse = nn.functional.mse_loss(predictions, targets)
    rank = rank_ic(predictions, targets)
    total = mse + rankic_weight * (1.0 - rank)
    return {"loss": total, "mse": mse, "rank_ic": rank}
