"""Representation-audit helpers for 64-code VQ summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import math
import numpy as np


def _usage_entropy_from_codes(codes: np.ndarray, codebook_size: int) -> float:
    counts = np.bincount(codes.astype(np.int64).reshape(-1), minlength=codebook_size).astype(np.float64)
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0.0]
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def audit_codebook(
    *,
    codebook_size: int,
    active_codes: int,
    codes_path: str | Path | None = None,
    codes_source: str | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable VQ codebook audit payload."""
    usage_entropy = None
    num_codes_observed = None
    status = "summary_only"
    resolved_path = Path(codes_path) if codes_path else None
    if resolved_path is not None and resolved_path.exists():
        payload = np.load(resolved_path)
        if "codes" in payload:
            codes = np.asarray(payload["codes"], dtype=np.int64)
            usage_entropy = _usage_entropy_from_codes(codes, codebook_size=codebook_size)
            num_codes_observed = int(np.unique(codes).size)
            active_codes = num_codes_observed
            status = "computed"

    active_ratio = float(active_codes) / float(codebook_size) if codebook_size > 0 else 0.0
    collapse_flag = bool(active_codes < 10)
    warning = None
    if collapse_flag:
        warning = "active_codes < 10"
    elif active_ratio < 0.2:
        warning = "active_codes/codebook_size < 0.2"

    max_entropy = float(math.log(codebook_size)) if codebook_size > 0 else 0.0
    return {
        "status": status,
        "codebook_size": int(codebook_size),
        "codes_source": codes_source,
        "codes_path": str(resolved_path) if resolved_path is not None else None,
        "active_codes": int(active_codes),
        "active_ratio": active_ratio,
        "usage_entropy": usage_entropy,
        "max_usage_entropy": max_entropy,
        "collapse_flag": collapse_flag,
        "warning": warning,
        "representation_stability": "unstable" if collapse_flag else "ok",
    }


def audit_from_diagnostic_summary(path: str | Path) -> dict[str, Any]:
    """Compute time/volume representation audits from a diagnostic summary JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    result: dict[str, Any] = {}
    for mode in ("time", "volume"):
        summary = payload.get(mode, {})
        vq_summary = summary.get("vq_summary", {}) if isinstance(summary, dict) else {}
        result[mode] = audit_codebook(
            codebook_size=64,
            active_codes=int(vq_summary.get("num_unique_codes", 0) or 0),
            codes_path=vq_summary.get("train_encoded"),
            codes_source=vq_summary.get("codes_source"),
        )
    return result
