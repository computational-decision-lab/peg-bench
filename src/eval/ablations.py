"""Ablation helpers for summarizing completed experiment artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(slots=True)
class AblationEntry:
    """Single ablation entry used in reports."""

    name: str
    path: str


def summarize_ablation_runs(entries: list[AblationEntry], output_path: str | Path) -> dict[str, object]:
    """Aggregate result JSON files into a single comparison table."""
    summary: dict[str, object] = {"experiments": []}
    for entry in entries:
        payload = json.loads(Path(entry.path).read_text())
        payload["name"] = entry.name
        summary["experiments"].append(payload)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    return summary
