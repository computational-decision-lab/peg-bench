"""YAML-based configuration utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def dataclass_from_dict(cls, data: dict[str, Any]):
    field_names = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    filtered = {key: value for key, value in data.items() if key in field_names}
    return cls(**filtered)


def dataclass_to_dict(instance) -> dict[str, Any]:
    return asdict(instance)
