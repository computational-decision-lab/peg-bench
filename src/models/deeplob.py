"""DeepLOB-inspired time-bar baseline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(slots=True)
class DeepLOBConfig:
    """Configuration for the DeepLOB-style sequence model."""

    seq_len: int = 50
    input_dim: int = 80
    stem_channels: int = 64
    branch_channels: int = 32
    lstm_hidden: int = 64
    dropout: float = 0.1
    rankic_weight: float = 0.0


class DeepLOBTimeBarModel(nn.Module):
    """A compact CNN + LSTM model over time-bar LOB windows."""

    def __init__(self, config: DeepLOBConfig) -> None:
        super().__init__()
        self.config = config
        self.stem = nn.Sequential(
            nn.Conv1d(config.input_dim, config.stem_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(config.stem_channels, config.stem_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.branch1 = nn.Conv1d(config.stem_channels, config.branch_channels, kernel_size=1)
        self.branch3 = nn.Conv1d(config.stem_channels, config.branch_channels, kernel_size=3, padding=1)
        self.branch5 = nn.Conv1d(config.stem_channels, config.branch_channels, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(config.dropout)
        self.lstm = nn.LSTM(
            input_size=config.branch_channels * 3,
            hidden_size=config.lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(config.lstm_hidden)
        self.return_head = nn.Linear(config.lstm_hidden, 1)
        self.direction_head = nn.Linear(config.lstm_hidden, 1)

    def forward(self, lob: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict next return from a raw `(B, T, C, D, S)` LOB sequence."""
        batch, seq_len, channels, depth, sides = lob.shape
        features = lob.view(batch, seq_len, channels * depth * sides).transpose(1, 2)
        stem = self.stem(features)
        merged = torch.cat(
            [
                self.branch1(stem),
                self.branch3(stem),
                self.branch5(stem),
            ],
            dim=1,
        )
        merged = self.dropout(merged).transpose(1, 2)
        outputs, _ = self.lstm(merged)
        pooled = self.norm(outputs[:, -1])
        return {
            "features": pooled,
            "return_pred": self.return_head(pooled).squeeze(-1),
            "direction_logit": self.direction_head(pooled).squeeze(-1),
        }
