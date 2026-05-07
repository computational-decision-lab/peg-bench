"""Markdown report generation for end-to-end experiments."""

from __future__ import annotations

from pathlib import Path
import json


def generate_markdown_report(
    output_path: str | Path,
    sections: dict[str, object],
    approximations: list[str],
    protocol_targets: list[str] | None = None,
    title: str = "PEG-Bench Report Card",
) -> str:
    """Generate a concise markdown experiment report."""
    lines = [f"# {title}", ""]
    for title, payload in sections.items():
        lines.append(f"## {title}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(payload, indent=2))
        lines.append("```")
        lines.append("")
    if protocol_targets:
        lines.append("## Protocol Targets")
        lines.append("")
        for item in protocol_targets:
            lines.append(f"- {item}")
        lines.append("")
    lines.append("## Engineering Approximations")
    lines.append("")
    for item in approximations:
        lines.append(f"- {item}")
    report = "\n".join(lines)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    return report
