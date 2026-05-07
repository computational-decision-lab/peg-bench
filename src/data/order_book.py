"""Order book reconstruction utilities for Binance-style L2 and trade feeds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal
import heapq

import pandas as pd


@dataclass(slots=True)
class L2Update:
    """Single incremental book update."""

    timestamp: int
    side: Literal["bid", "ask"]
    price: float
    amount: float
    is_snapshot: bool = False


@dataclass(slots=True)
class TradeEvent:
    """Single trade tick."""

    timestamp: int
    side: Literal["buy", "sell"]
    price: float
    size: float


@dataclass(slots=True)
class OrderBookSnapshot:
    """Top-of-book snapshot view."""

    timestamp: int
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]
    best_bid: float | None
    best_ask: float | None


class OrderBookReconstructor:
    """Minimal in-memory order book with OFI and toxicity tracking."""

    def __init__(self) -> None:
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self._bid_heap: list[float] = []
        self._ask_heap: list[float] = []
        self.last_delta: dict[tuple[str, float], float] = {}
        self.buy_notional: float = 0.0
        self.sell_notional: float = 0.0

    def _book(self, side: Literal["bid", "ask"]) -> dict[float, float]:
        return self.bids if side == "bid" else self.asks

    def apply_l2(self, update: L2Update) -> tuple[float, float]:
        """Apply an L2 update and return `(executed_notional, delta_amount)`."""
        book = self._book(update.side)
        prev = book.get(update.price, 0.0)
        delta = update.amount - prev
        if update.amount <= 0:
            book.pop(update.price, None)
        else:
            if prev <= 0:
                if update.side == "bid":
                    heapq.heappush(self._bid_heap, -update.price)
                else:
                    heapq.heappush(self._ask_heap, update.price)
            book[update.price] = update.amount

        executed_qty = max(0.0, -delta)
        executed_notional = executed_qty * update.price
        self.last_delta[(update.side, update.price)] = delta
        return executed_notional, delta

    def apply_trade(self, trade: TradeEvent) -> float:
        """Track trade flow for toxicity and volume clock construction."""
        notional = trade.price * trade.size
        if trade.side == "buy":
            self.buy_notional += notional
        else:
            self.sell_notional += notional
        return notional

    def reset_flow(self) -> None:
        """Reset cumulative trade-flow statistics after a bar cut."""
        self.buy_notional = 0.0
        self.sell_notional = 0.0
        self.last_delta.clear()

    def top_levels(self, side: Literal["bid", "ask"], depth: int) -> list[tuple[float, float]]:
        """Return top levels for a side using incrementally maintained heaps."""
        book = self._book(side)
        if not book or depth <= 0:
            return []
        heap = self._bid_heap if side == "bid" else self._ask_heap
        restore: list[float] = []
        seen: set[float] = set()
        levels: list[tuple[float, float]] = []
        while heap and len(levels) < depth:
            raw_price = heapq.heappop(heap)
            price = -raw_price if side == "bid" else raw_price
            size = book.get(price)
            if size is None or size <= 0 or price in seen:
                continue
            restore.append(raw_price)
            seen.add(price)
            levels.append((price, size))
        for raw_price in restore:
            heapq.heappush(heap, raw_price)
        return levels

    def best_bid(self) -> float | None:
        levels = self.top_levels("bid", 1)
        return levels[0][0] if levels else None

    def best_ask(self) -> float | None:
        levels = self.top_levels("ask", 1)
        return levels[0][0] if levels else None

    def snapshot(self, timestamp: int, depth: int) -> OrderBookSnapshot:
        """Materialize a top-of-book snapshot."""
        bids = self.top_levels("bid", depth)
        asks = self.top_levels("ask", depth)
        return OrderBookSnapshot(
            timestamp=timestamp,
            bids=bids,
            asks=asks,
            best_bid=bids[0][0] if bids else None,
            best_ask=asks[0][0] if asks else None,
        )

    def visible_volume(self, depth: int) -> float:
        """Compute visible depth across the top levels."""
        volume = 0.0
        for _, size in self.top_levels("bid", depth):
            volume += size
        for _, size in self.top_levels("ask", depth):
            volume += size
        return volume

    def toxicity(self) -> float:
        """Return signed trade-flow imbalance."""
        total = self.buy_notional + self.sell_notional + 1e-8
        return (self.buy_notional - self.sell_notional) / total


OKX_USDT_SWAP_CONTRACT_SIZES: dict[str, float] = {
    "BTC-USDT-SWAP": 0.01,
    "ETH-USDT-SWAP": 0.1,
}


def _size_multiplier_for_path(path: Path) -> float:
    """Normalize venue-specific contract units into coin units."""
    text = path.as_posix().upper()
    if "OKEX-SWAP" not in text and "OKX" not in text:
        return 1.0
    for symbol, contract_size in OKX_USDT_SWAP_CONTRACT_SIZES.items():
        if symbol in text:
            return contract_size
    return 1.0


def _normalize_l2_columns(frame: pd.DataFrame) -> pd.DataFrame:
    column_map = {
        "size": "amount",
        "qty": "amount",
        "quantity": "amount",
        "is_snapshot": "is_snapshot",
    }
    renamed = frame.rename(columns={key: value for key, value in column_map.items() if key in frame.columns})
    if "is_snapshot" not in renamed.columns:
        renamed["is_snapshot"] = False
    required = ["timestamp", "side", "price", "amount", "is_snapshot"]
    missing = [name for name in required if name not in renamed.columns]
    if missing:
        raise ValueError(f"Missing L2 columns: {missing}")
    renamed["side"] = renamed["side"].astype(str).str.lower().replace({"buy": "bid", "sell": "ask"})
    return renamed[required]


def _normalize_trade_columns(frame: pd.DataFrame) -> pd.DataFrame:
    column_map = {
        "amount": "size",
        "qty": "size",
        "quantity": "size",
        "is_buyer_maker": "maker_flag",
    }
    renamed = frame.rename(columns={key: value for key, value in column_map.items() if key in frame.columns})
    if "side" not in renamed.columns:
        if "maker_flag" in renamed.columns:
            renamed["side"] = renamed["maker_flag"].map({True: "sell", False: "buy"})
        else:
            raise ValueError("Trades file must provide `side` or `is_buyer_maker`.")
    required = ["timestamp", "side", "price", "size"]
    missing = [name for name in required if name not in renamed.columns]
    if missing:
        raise ValueError(f"Missing trade columns: {missing}")
    renamed["side"] = renamed["side"].astype(str).str.lower()
    return renamed[required]


def iter_l2_updates(path: str | Path, chunksize: int = 1_000_000) -> Iterator[L2Update]:
    """Stream normalized L2 updates from CSV or parquet."""
    path = Path(path)
    size_multiplier = _size_multiplier_for_path(path)
    required_columns = ["timestamp", "is_snapshot", "side", "price", "amount"]
    dtypes = {
        "timestamp": "int64",
        "is_snapshot": "bool",
        "side": "category",
        "price": "float64",
        "amount": "float64",
    }
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path, columns=[c for c in required_columns if c in required_columns])
        frame = _normalize_l2_columns(frame)
        for row in frame.itertuples(index=False):
            yield L2Update(
                timestamp=int(row.timestamp),
                side=str(row.side),
                price=float(row.price),
                amount=float(row.amount) * size_multiplier,
                is_snapshot=bool(row.is_snapshot),
            )
        return

    with pd.read_csv(
        path,
        chunksize=chunksize,
        usecols=lambda c: c in required_columns,
        dtype=dtypes,
        compression="infer",
        engine="c",
    ) as reader:
        for chunk in reader:
            frame = _normalize_l2_columns(chunk)
            for row in frame.itertuples(index=False):
                yield L2Update(
                    timestamp=int(row.timestamp),
                    side=str(row.side),
                    price=float(row.price),
                    amount=float(row.amount),
                    is_snapshot=bool(row.is_snapshot),
                )


def iter_trade_events(path: str | Path, chunksize: int = 1_000_000) -> Iterator[TradeEvent]:
    """Stream normalized trades from CSV or parquet."""
    path = Path(path)
    size_multiplier = _size_multiplier_for_path(path)
    required_like = {"timestamp", "side", "price", "size", "amount", "qty", "quantity", "is_buyer_maker"}
    dtypes = {
        "timestamp": "int64",
        "side": "category",
        "price": "float64",
        "size": "float64",
        "amount": "float64",
        "qty": "float64",
        "quantity": "float64",
        "is_buyer_maker": "boolean",
    }
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
        frame = _normalize_trade_columns(frame)
        for row in frame.itertuples(index=False):
            yield TradeEvent(
                timestamp=int(row.timestamp),
                side=str(row.side),
                price=float(row.price),
                size=float(row.size) * size_multiplier,
            )
        return

    with pd.read_csv(
        path,
        chunksize=chunksize,
        usecols=lambda c: c in required_like,
        dtype=dtypes,
        compression="infer",
        engine="c",
    ) as reader:
        for chunk in reader:
            frame = _normalize_trade_columns(chunk)
            for row in frame.itertuples(index=False):
                yield TradeEvent(
                    timestamp=int(row.timestamp),
                    side=str(row.side),
                    price=float(row.price),
                    size=float(row.size) * size_multiplier,
                )


def merge_event_streams(
    l2_path: str | Path,
    trades_path: str | Path | None = None,
    chunksize: int = 1_000_000,
) -> Iterator[tuple[str, L2Update | TradeEvent]]:
    """Merge L2 and trade streams in timestamp order."""
    l2_iter = iter_l2_updates(l2_path, chunksize=chunksize)
    trade_iter = iter_trade_events(trades_path, chunksize=chunksize) if trades_path else None

    try:
        current_l2 = next(l2_iter)
    except StopIteration:
        current_l2 = None

    if trade_iter is not None:
        try:
            current_trade = next(trade_iter)
        except StopIteration:
            current_trade = None
    else:
        current_trade = None

    while current_l2 is not None or current_trade is not None:
        if current_trade is None or (
            current_l2 is not None and current_l2.timestamp <= current_trade.timestamp
        ):
            yield "l2", current_l2
            try:
                current_l2 = next(l2_iter)
            except StopIteration:
                current_l2 = None
        else:
            yield "trade", current_trade
            assert trade_iter is not None
            try:
                current_trade = next(trade_iter)
            except StopIteration:
                current_trade = None
