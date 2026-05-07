"""Scale-invariant VQ-VAE for order-book representation learning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    """EMA-based VQ layer that emits discrete codes and continuous embeddings."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        embedding = torch.randn(embedding_dim, num_embeddings) * 0.02
        self.register_buffer("embedding", embedding)
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", embedding.clone())

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        flat_z = torch.nan_to_num(z.view(-1, self.embedding_dim), nan=0.0, posinf=1e4, neginf=-1e4)
        distances = (
            flat_z.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_z @ self.embedding
            + self.embedding.pow(2).sum(dim=0, keepdim=True)
        )
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type_as(flat_z)
        quantized = encodings @ self.embedding.T
        quantized = quantized.view_as(z)

        if self.training:
            cluster_update = encodings.sum(dim=0)
            self.cluster_size.data.mul_(self.decay).add_(cluster_update, alpha=1 - self.decay)
            dw = flat_z.transpose(0, 1) @ encodings
            self.ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)

            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon)
                * n
            )
            cluster_size = torch.clamp(cluster_size, min=1.0)
            updated_embedding = self.ema_w / cluster_size.unsqueeze(0)
            updated_embedding = torch.nan_to_num(updated_embedding, nan=0.0, posinf=1.0, neginf=-1.0)
            self.embedding.data.copy_(updated_embedding.clamp(-5.0, 5.0))

        commit_loss = self.commitment_cost * torch.mean((z.detach() - quantized) ** 2)
        codebook_loss = torch.mean((z - quantized.detach()) ** 2)
        quantized = z + (quantized - z).detach()
        avg_probs = torch.clamp(torch.mean(encodings.float(), dim=0), min=1e-10)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs))).clamp(max=float(self.num_embeddings))
        codes = encoding_indices.view(z.shape[0], -1)[:, 0]
        return {
            "quantized": quantized,
            "commit_loss": commit_loss,
            "codebook_loss": codebook_loss,
            "perplexity": perplexity,
            "codes": codes,
        }


@dataclass(slots=True)
class VQVAEConfig:
    """Config for the scale-invariant order-book VQ-VAE."""

    input_channels: int = 4
    levels: int = 10
    sides: int = 2
    latent_dim: int = 64
    num_embeddings: int = 64
    hidden_channels: int = 32
    commitment_cost: float = 0.25
    beta_commit: float = 1.0
    beta_sym: float = 0.1
    decay: float = 0.99
    dropout: float = 0.1


def mirror_lob_tensor(x: torch.Tensor) -> torch.Tensor:
    """Apply an order-book symmetry transform by swapping sides and signs."""
    mirrored = torch.flip(x, dims=[-1]).clone()
    mirrored[:, 0] = -mirrored[:, 0]
    if mirrored.shape[1] > 2:
        mirrored[:, 2] = -mirrored[:, 2]
    if mirrored.shape[1] > 3:
        mirrored[:, 3] = -mirrored[:, 3]
    return mirrored


class ScaleInvariantVQVAE(nn.Module):
    """VQ-VAE with an auxiliary symmetry regularizer."""

    def __init__(self, config: VQVAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Conv2d(config.input_channels, config.hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Conv2d(
                config.hidden_channels,
                config.hidden_channels * 2,
                kernel_size=3,
                stride=(2, 1),
                padding=1,
            ),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                config.hidden_channels * 2,
                config.hidden_channels,
                kernel_size=3,
                stride=(2, 1),
                padding=1,
                output_padding=(1, 0),
            ),
            nn.GELU(),
            nn.Conv2d(config.hidden_channels, config.input_channels, kernel_size=3, padding=1),
        )
        self.encoder_out_shape: tuple[int, int, int] | None = None
        self.pre_quant: nn.Linear | None = None
        self.post_quant: nn.Linear | None = None
        self.quantizer = VectorQuantizerEMA(
            num_embeddings=config.num_embeddings,
            embedding_dim=config.latent_dim,
            commitment_cost=config.commitment_cost,
            decay=config.decay,
        )
        self._build_projections()

    def _build_projections(self) -> None:
        with torch.no_grad():
            dummy = torch.zeros(1, self.config.input_channels, self.config.levels, self.config.sides)
            encoded = self.encoder(dummy)
            self.encoder_out_shape = tuple(int(dim) for dim in encoded.shape[1:])
            flat_dim = int(np.prod(self.encoder_out_shape))
        self.pre_quant = nn.Linear(flat_dim, self.config.latent_dim)
        self.post_quant = nn.Linear(self.config.latent_dim, flat_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode raw LOB tensors into continuous latent vectors."""
        hidden = self.encoder(x)
        return torch.tanh(self.pre_quant(hidden.flatten(start_dim=1)))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode quantized latents back into LOB tensors."""
        assert self.encoder_out_shape is not None
        hidden = self.post_quant(latents).view(latents.size(0), *self.encoder_out_shape)
        return self.decoder(hidden)

    def forward(self, x: torch.Tensor, with_symmetry: bool = True) -> dict[str, torch.Tensor]:
        """Forward pass returning reconstruction, discrete codes and all loss terms."""
        z_e = self.encode(x)
        q_out = self.quantizer(z_e)
        reconstruction = self.decode(q_out["quantized"])
        recon_loss = F.mse_loss(reconstruction, x)

        if with_symmetry:
            mirrored = mirror_lob_tensor(x)
            mirrored_latents = self.encode(mirrored)
            symmetry_loss = torch.mean((z_e + mirrored_latents) ** 2)
        else:
            symmetry_loss = torch.tensor(0.0, device=x.device)

        total_loss = (
            recon_loss
            + self.config.beta_commit * (q_out["commit_loss"] + q_out["codebook_loss"])
            + self.config.beta_sym * symmetry_loss
        )
        return {
            "reconstruction": reconstruction,
            "latents": z_e,
            "quantized": q_out["quantized"],
            "codes": q_out["codes"],
            "perplexity": q_out["perplexity"],
            "recon_loss": recon_loss,
            "commit_loss": q_out["commit_loss"],
            "codebook_loss": q_out["codebook_loss"],
            "symmetry_loss": symmetry_loss,
            "loss": total_loss,
        }

    @torch.no_grad()
    def encode_codes(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode inputs into discrete codes and continuous latents."""
        z_e = self.encode(x)
        q_out = self.quantizer(z_e)
        return q_out["codes"], z_e


MicrostructureVQVAE = ScaleInvariantVQVAE
