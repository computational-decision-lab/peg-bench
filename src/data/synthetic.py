"""Synthetic raw data generator for the minimal demo and unit tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(slots=True)
class SyntheticMarketConfig:
    """Synthetic market generation config."""

    num_steps: int = 3_000
    base_price: float = 100_000.0
    levels: int = 10
    seed: int = 7


def generate_synthetic_l2_and_trades(
    output_dir: str | Path,
    config: SyntheticMarketConfig | None = None,
) -> tuple[Path, Path]:
    """Create synthetic L2 and trades CSV files for fast smoke tests."""
    cfg = config or SyntheticMarketConfig()
    rng = np.random.default_rng(cfg.seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    price = cfg.base_price
    l2_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    timestamp = 1_700_000_000_000_000

    for step in range(cfg.num_steps):
        price += rng.normal(0.0, 30.0)
        spread = max(0.5, abs(rng.normal(2.0, 0.4)))
        timestamp += int(rng.integers(1_000, 50_000))

        for level in range(cfg.levels):
            bid_price = price - spread - level * 0.5
            ask_price = price + spread + level * 0.5
            bid_amount = float(np.exp(rng.normal(2.2, 0.3)))
            ask_amount = float(np.exp(rng.normal(2.2, 0.3)))
            l2_rows.append(
                {
                    "timestamp": timestamp,
                    "side": "bid",
                    "price": round(bid_price, 4),
                    "amount": round(bid_amount, 6),
                    "is_snapshot": step == 0,
                }
            )
            l2_rows.append(
                {
                    "timestamp": timestamp,
                    "side": "ask",
                    "price": round(ask_price, 4),
                    "amount": round(ask_amount, 6),
                    "is_snapshot": step == 0,
                }
            )

        trade_side = "buy" if rng.random() > 0.5 else "sell"
        trade_rows.append(
            {
                "timestamp": timestamp,
                "side": trade_side,
                "price": round(price, 4),
                "size": round(float(np.exp(rng.normal(0.4, 0.4))), 6),
            }
        )

    l2_path = output_dir / "synthetic_l2.csv"
    trades_path = output_dir / "synthetic_trades.csv"
    pd.DataFrame(l2_rows).to_csv(l2_path, index=False)
    pd.DataFrame(trade_rows).to_csv(trades_path, index=False)
    return l2_path, trades_path
