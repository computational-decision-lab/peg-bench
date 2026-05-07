"""Helpers for selecting the best available Torch device."""

from __future__ import annotations

import torch


def resolve_torch_device(requested: str = "auto") -> torch.device:
    """Resolve a user request into an available torch device."""
    requested = (requested or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def amp_enabled(requested_amp: bool, device: torch.device) -> bool:
    """Only enable AMP on CUDA where it is already used and well-tested here."""
    return bool(requested_amp and device.type == "cuda")
