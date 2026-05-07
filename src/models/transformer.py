"""
Scaled-down Kronos-inspired decoder-only Transformer implemented in PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class KronosConfig:
    vocab_size: int
    max_sequence_length: int = 512
    d_model: int = 256
    n_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    feedforward_multiplier: int = 4


class RotaryPositionalEmbedding(nn.Module):
    """Simple sinusoidal positional embeddings."""

    def __init__(self, dim: int, max_length: int) -> None:
        super().__init__()
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(max_length, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.pe[:, :seq_len]


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)


class KronosTransformer(nn.Module):
    def __init__(self, config: KronosConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = RotaryPositionalEmbedding(config.d_model, config.max_sequence_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.feedforward_multiplier * config.d_model,
            batch_first=True,
            dropout=config.dropout,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward_features(self, tokens: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len = tokens.shape
        if seq_len > self.config.max_sequence_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds configured max_sequence_length {self.config.max_sequence_length}"
            )

        device = tokens.device
        token_embeddings = self.token_embed(tokens)
        positional = self.pos_embed(seq_len).to(device)
        hidden = token_embeddings + positional
        mask = causal_mask(seq_len, device)

        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None

        transformed = self.transformer(hidden, mask=mask, src_key_padding_mask=key_padding_mask)
        return transformed

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(hidden))

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:  # type: ignore[override]
        hidden = self.forward_features(tokens, attention_mask)
        return self.logits(hidden)


def compute_cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute autoregressive cross entropy loss."""
    vocab_size = logits.size(-1)
    return nn.functional.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
