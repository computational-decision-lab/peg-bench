"""
Construct fixed-interval (time-sampled) LOB tensors from incremental Level-2 CSVs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass
class TimeClockConfig:
    interval_ms: int = 1_000
    levels_per_side: int = 20
    chunk_rows: int = 500_000
    min_lob_levels: int = 5
    max_bars: int | None = None


class SimpleBook:
    def __init__(self) -> None:
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}

    def _book(self, side: str) -> dict[float, float]:
        return self.bids if side == "bid" else self.asks

    def update(self, side: str, price: float, amount: float) -> None:
        book = self._book(side)
        if amount <= 0:
            book.pop(price, None)
        else:
            book[price] = amount

    def best_bid(self) -> float | None:
        return max(self.bids.keys(), default=None)

    def best_ask(self) -> float | None:
        return min(self.asks.keys(), default=None)

    def top(self, side: str, depth: int) -> list[tuple[float, float]]:
        book = self._book(side)
        prices = sorted(book.keys(), reverse=(side == "bid"))
        return [(p, book[p]) for p in prices[:depth]]


def chunk_reader(path: Path, chunk_rows: int) -> Iterator[pd.DataFrame]:
    cols = ["timestamp", "side", "price", "amount"]
    dtypes = {"timestamp": "int64", "side": "category", "price": "float64", "amount": "float64"}
    for chunk in pd.read_csv(path, usecols=cols, dtype=dtypes, chunksize=chunk_rows):
        yield chunk


def build_time_bars(csv_path: Path, cfg: TimeClockConfig) -> dict[str, np.ndarray]:
    book = SimpleBook()
    tensors: list[np.ndarray] = []
    timestamps: list[int] = []
    mids: list[float] = []

    start_ts: int | None = None
    next_cut: int | None = None

    for chunk in chunk_reader(csv_path, cfg.chunk_rows):
        for row in chunk.itertuples(index=False):
            ts = int(row.timestamp)
            price = float(row.price)
            amount = float(row.amount)
            side = str(row.side)
            book.update(side, price, amount)

            if start_ts is None:
                start_ts = ts
                next_cut = ts + cfg.interval_ms * 1_000  # timestamps in microseconds

            while next_cut is not None and ts >= next_cut:
                best_bid = book.best_bid()
                best_ask = book.best_ask()
                if best_bid is None or best_ask is None:
                    next_cut += cfg.interval_ms * 1_000
                    continue
                mid = 0.5 * (best_bid + best_ask)
                bids = book.top("bid", cfg.levels_per_side)
                asks = book.top("ask", cfg.levels_per_side)
                if len(bids) < cfg.min_lob_levels or len(asks) < cfg.min_lob_levels:
                    next_cut += cfg.interval_ms * 1_000
                    continue
                tensor = np.zeros((4, cfg.levels_per_side, 2), dtype=np.float32)
                for idx, levels in enumerate([bids, asks]):
                    for lvl, (lvl_price, lvl_amount) in enumerate(levels):
                        rel_price = (lvl_price - mid) / mid
                        log_volume = np.log(max(lvl_amount, 1e-8))
                        tensor[0, lvl, idx] = rel_price
                        tensor[1, lvl, idx] = log_volume
                        tensor[2, lvl, idx] = lvl_amount
                        tensor[3, lvl, idx] = lvl_price
                tensors.append(tensor)
                timestamps.append(next_cut)
                mids.append(mid)
                next_cut += cfg.interval_ms * 1_000
                if cfg.max_bars is not None and len(tensors) >= cfg.max_bars:
                    break
            if cfg.max_bars is not None and len(tensors) >= cfg.max_bars:
                break
        if cfg.max_bars is not None and len(tensors) >= cfg.max_bars:
            break

    tensors_np = np.stack(tensors) if tensors else np.zeros((0, 4, cfg.levels_per_side, 2), dtype=np.float32)
    mids_np = np.asarray(mids, dtype=np.float64)
    ts_np = np.asarray(timestamps, dtype=np.int64)

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


def save_dataset(payload: dict[str, np.ndarray], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed-interval LOB dataset from Tardis CSV.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-ms", type=int, default=1000)
    parser.add_argument("--levels", type=int, default=20)
    parser.add_argument("--chunk", type=int, default=500_000)
    parser.add_argument("--max-bars", type=int, default=None)
    args = parser.parse_args()

    cfg = TimeClockConfig(
        interval_ms=args.interval_ms,
        levels_per_side=args.levels,
        chunk_rows=args.chunk,
        max_bars=args.max_bars,
    )
    data = build_time_bars(args.input, cfg)
    save_dataset(data, args.output)
    print(f"Saved {data['lob'].shape[0]} time-sampled bars to {args.output}")


if __name__ == "__main__":
    main()
