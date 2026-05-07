"""Continuous Gaussian policy with reference-policy anchoring support."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.baseline import KairosBaselineConfig, KairosBaselineModel


@dataclass(slots=True)
class GaussianPolicyConfig:
    """Config for the continuous action policy."""

    baseline: KairosBaselineConfig
    num_risk_modes: int = 4
    min_std: float = 1e-2
    max_std: float = 0.75
    max_action: float = 1.0


class GaussianActionPolicy(nn.Module):
    """Risk-conditioned Gaussian policy initialized from the baseline backbone."""

    def __init__(self, config: GaussianPolicyConfig, backbone: KairosBaselineModel | None = None) -> None:
        super().__init__()
        self.config = config
        self.backbone = backbone or KairosBaselineModel(config.baseline)
        self.risk_embedding = nn.Embedding(config.num_risk_modes, config.baseline.d_model)
        self.mean_head = nn.Linear(config.baseline.d_model, 1)
        self.std_head = nn.Linear(config.baseline.d_model, 1)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        latents: torch.Tensor | None = None,
        risk_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Gaussian mean and std for continuous delta-position actions."""
        outputs = self.backbone(input_ids=input_ids, latents=latents)
        features = outputs["features"]
        if risk_ids is None:
            risk_ids = torch.zeros(features.size(0), dtype=torch.long, device=features.device)
        features = features + self.risk_embedding(risk_ids)
        mean = torch.tanh(self.mean_head(features)) * self.config.max_action
        std = F.softplus(self.std_head(features)) + self.config.min_std
        std = torch.clamp(std, min=self.config.min_std, max=self.config.max_std)
        return mean.squeeze(-1), std.squeeze(-1)

    def log_prob(
        self,
        actions: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        latents: torch.Tensor | None = None,
        risk_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return log-probability together with mean and std."""
        mean, std = self.forward(input_ids=input_ids, latents=latents, risk_ids=risk_ids)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions)
        return log_prob, mean, std

    def act(
        self,
        input_ids: torch.Tensor | None = None,
        latents: torch.Tensor | None = None,
        risk_ids: torch.Tensor | None = None,
        deterministic: bool = False,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """Sample or deterministically emit actions."""
        mean, std = self.forward(input_ids=input_ids, latents=latents, risk_ids=risk_ids)
        dist = torch.distributions.Normal(mean, std)
        actions = mean if deterministic else dist.rsample()
        return torch.clamp(actions * scale, -self.config.max_action, self.config.max_action)

    def entropy(
        self,
        input_ids: torch.Tensor | None = None,
        latents: torch.Tensor | None = None,
        risk_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return per-sample policy entropy."""
        _, std = self.forward(input_ids=input_ids, latents=latents, risk_ids=risk_ids)
        return 0.5 * torch.log(2 * torch.pi * torch.e * (std**2))


@torch.no_grad()
def ema_update_reference(reference_policy: GaussianActionPolicy, policy: GaussianActionPolicy, tau: float) -> None:
    """EMA update for the reference policy anchor."""
    for ref_param, param in zip(reference_policy.parameters(), policy.parameters()):
        ref_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)
