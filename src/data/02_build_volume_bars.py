"""
Utilities for constructing volume-clock limit order book datasets from Tardis Level-2 CSVs.

The goal is to stream incremental book updates, accumulate executed notional until a threshold
is met, then snapshot the top-N levels on both sides as a tensor suitable for VQ-VAE training.

This implementation is CPU-friendly and works with the Binance Level-2 CSVs already downloaded
under `./tardis`. It approximates executed volume by tracking negative deltas in resting size.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass
class VolumeClockConfig:
    depth: int = 20
    volume_threshold: float = 1_000_000.0  # notional USDT
    chunk_rows: int = 500_000
    levels_per_side: int = 20
    min_lob_levels: int = 5
    max_bars: int | None = None


class OrderBook:
    """Minimal in-memory order book to track bid/ask volumes at each price level."""

    def __init__(self) -> None:
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}

    def _book_for_side(self, side: str) -> dict[float, float]:
        return self.bids if side == "bid" else self.asks

    def update(self, side: str, price: float, amount: float) -> tuple[float, float]:
        """Update book level and return (executed_volume, delta_amount)."""
        book = self._book_for_side(side)
        prev = book.get(price, 0.0)
        delta = amount - prev
        if amount <= 0:
            book.pop(price, None)
        else:
            book[price] = amount

        executed = -delta if delta < 0 else 0.0
        return executed, delta

    def best_bid(self) -> float | None:
        return max(self.bids.keys(), default=None)

    def best_ask(self) -> float | None:
        return min(self.asks.keys(), default=None)

    def top_levels(self, side: str, depth: int) -> list[tuple[float, float]]:
        book = self._book_for_side(side)
        if side == "bid":
            prices = sorted(book.keys(), reverse=True)
        else:
            prices = sorted(book.keys())
        levels: list[tuple[float, float]] = []
        for price in prices[:depth]:
            amount = book[price]
            levels.append((price, amount))
        return levels


def chunk_reader(path: Path, chunk_rows: int) -> Iterator[pd.DataFrame]:
    """Stream the CSV in manageable chunks."""
    cols = [
        "timestamp",
        "is_snapshot",
        "side",
        "price",
        "amount",
    ]
    dtypes = {
        "timestamp": "int64",
        "is_snapshot": "bool",
        "side": "category",
        "price": "float64",
        "amount": "float64",
    }
    for chunk in pd.read_csv(
        path,
        usecols=cols,
        dtype=dtypes,
        chunksize=chunk_rows,
    ):
        yield chunk


def build_volume_bars(csv_path: Path, cfg: VolumeClockConfig) -> dict[str, np.ndarray]:
    book = OrderBook()
    cumulative_notional = 0.0
    executed_bid = 0.0
    executed_ask = 0.0
    ofi_buffer: dict[tuple[str, float], float] = {}

    tensors: list[np.ndarray] = []
    timestamps: list[int] = []
    mids: list[float] = []

    for chunk in chunk_reader(csv_path, cfg.chunk_rows):
        for row in chunk.itertuples(index=False):
            price = float(row.price)
            amount = float(row.amount)
            side = str(row.side)

            executed, delta = book.update(side, price, amount)
            if executed > 0:
                notional = executed * price
                cumulative_notional += notional
                if side == "bid":
                    executed_bid += notional
                else:
                    executed_ask += notional
                key = (side, price)
                ofi_buffer[key] = ofi_buffer.get(key, 0.0) - executed
            else:
                key = (side, price)
                ofi_buffer[key] = ofi_buffer.get(key, 0.0) + delta

            if cumulative_notional < cfg.volume_threshold:
                continue

            best_bid = book.best_bid()
            best_ask = book.best_ask()
            if best_bid is None or best_ask is None:
                continue
            mid = 0.5 * (best_bid + best_ask)

            bids = book.top_levels("bid", cfg.levels_per_side)
            asks = book.top_levels("ask", cfg.levels_per_side)
            if len(bids) < cfg.min_lob_levels or len(asks) < cfg.min_lob_levels:
                continue

            tensor = np.zeros((4, cfg.levels_per_side, 2), dtype=np.float32)
            for idx, levels in enumerate([bids, asks]):
                for lvl, (lvl_price, lvl_amount) in enumerate(levels):
                    rel_price = (lvl_price - mid) / mid
                    log_volume = np.log(max(lvl_amount, 1e-8))
                    key = ("bid" if idx == 0 else "ask", lvl_price)
                    ofi = ofi_buffer.get(key, 0.0)
                    toxicity_den = executed_bid + executed_ask + 1e-8
                    toxicity = (executed_bid - executed_ask) / toxicity_den
                    tensor[0, lvl, idx] = rel_price
                    tensor[1, lvl, idx] = log_volume
                    tensor[2, lvl, idx] = ofi
                    tensor[3, lvl, idx] = toxicity

            tensors.append(tensor)
            timestamps.append(int(row.timestamp))
            mids.append(mid)

            cumulative_notional -= cfg.volume_threshold
            executed_bid = 0.0
            executed_ask = 0.0
            ofi_buffer.clear()

            if cfg.max_bars is not None and len(tensors) >= cfg.max_bars:
                break
        if cfg.max_bars is not None and len(tensors) >= cfg.max_bars:
            break

    tensors_np = np.stack(tensors) if tensors else np.zeros((0, 4, cfg.levels_per_side, 2), dtype=np.float32)
    mids_np = np.asarray(mids, dtype=np.float64)
    ts_np = np.asarray(timestamps, dtype=np.int64)

    # Compute simple forward log-return targets over 5 and 20 bars
    def forward_return(series: np.ndarray, horizon: int) -> np.ndarray:
        if len(series) == 0:
            return np.zeros(0, dtype=np.float32)
        log_prices = np.log(series + 1e-8)
        future = np.roll(log_prices, -horizon)
        future[-horizon:] = log_prices[-horizon:]
        return (future - log_prices).astype(np.float32)

    targets_5 = forward_return(mids_np, 5)
    targets_20 = forward_return(mids_np, 20)

    return {
        "lob": tensors_np,
        "mid_prices": mids_np,
        "timestamps": ts_np,
        "return_5": targets_5,
        "return_20": targets_20,
    }


def save_dataset(data: dict[str, np.ndarray], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build volume-clock LOB dataset from Tardis CSV.")
    parser.add_argument("--input", type=Path, required=True, help="Path to incremental_book_L2 CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz path")
    parser.add_argument("--threshold", type=float, default=1_000_000.0, help="Notional threshold per bar")
    parser.add_argument("--levels", type=int, default=20, help="Number of levels per side")
    parser.add_argument("--chunk", type=int, default=500_000, help="Rows per pandas chunk")
    parser.add_argument("--max-bars", type=int, default=None, help="Optional limit on number of bars for quick tests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = VolumeClockConfig(
        volume_threshold=args.threshold,
        levels_per_side=args.levels,
        chunk_rows=args.chunk,
        max_bars=args.max_bars,
    )
    data = build_volume_bars(args.input, cfg)
    save_dataset(data, args.output)
    print(
        f"Saved {data['lob'].shape[0]} bars "
        f"with tensor shape {data['lob'].shape[1:]} to {args.output}"
    )


if __name__ == "__main__":
    main()
