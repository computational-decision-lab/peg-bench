"""High-level orchestration for raw market data preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.data.bars import BarConfig, build_bar_dataset, save_bar_dataset


@dataclass(slots=True)
class DataPipelineConfig:
    """Config for end-to-end dataset construction."""

    l2_path: str
    trades_path: str | None
    output_path: str
    bar_mode: str = "volume"
    volume_threshold: float = 100_000.0
    interval_ms: int = 1_000
    depth: int = 10
    stride: int = 50
    max_bars: int | None = None


def run_data_pipeline(config: DataPipelineConfig) -> dict[str, object]:
    """Build and persist the configured dataset."""
    bar_config = BarConfig(
        mode=config.bar_mode,
        volume_threshold=config.volume_threshold,
        interval_ms=config.interval_ms,
        depth=config.depth,
        stride=config.stride,
        max_bars=config.max_bars,
    )
    dataset = build_bar_dataset(config.l2_path, config.trades_path, bar_config)
    save_bar_dataset(dataset, config.output_path)
    return {
        "num_bars": int(dataset["lob"].shape[0]),
        "output_path": str(Path(config.output_path).resolve()),
        "bar_mode": config.bar_mode,
    }
