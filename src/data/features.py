"""Feature engineering for reconstructed order book states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.order_book import OrderBookReconstructor, OrderBookSnapshot


@dataclass(slots=True)
class FeatureConfig:
    """Config for feature tensor extraction."""

    depth: int = 10
    channels: int = 4


def compute_snapshot_statistics(
    book: OrderBookReconstructor,
    snapshot: OrderBookSnapshot,
    depth: int,
    trade_notional: float = 0.0,
    trade_count: int = 0,
    event_count: int = 0,
) -> dict[str, float]:
    """Derive benchmark-facing scalar features from a single LOB snapshot."""
    if snapshot.best_bid is None or snapshot.best_ask is None:
        return {
            "spread_bps": 0.0,
            "microprice_offset_bps": 0.0,
            "queue_imbalance": 0.0,
            "top_depth": 0.0,
            "visible_depth": 0.0,
            "toxicity": 0.0,
            "turnover_proxy": float(trade_notional),
            "slippage_proxy_bps": 0.0,
            "trade_notional": float(trade_notional),
            "trade_count": float(trade_count),
            "event_count": float(event_count),
        }

    best_bid = float(snapshot.best_bid)
    best_ask = float(snapshot.best_ask)
    mid = 0.5 * (best_bid + best_ask)
    bid_size = float(snapshot.bids[0][1]) if snapshot.bids else 0.0
    ask_size = float(snapshot.asks[0][1]) if snapshot.asks else 0.0
    total_top = bid_size + ask_size + 1e-8
    queue_imbalance = (bid_size - ask_size) / total_top
    microprice = (best_ask * bid_size + best_bid * ask_size) / total_top
    visible_depth = float(sum(size for _, size in snapshot.bids[:depth]) + sum(size for _, size in snapshot.asks[:depth]))
    top_depth = bid_size + ask_size
    spread_bps = (best_ask - best_bid) / (mid + 1e-8) * 10_000.0
    microprice_offset_bps = np.log((microprice + 1e-8) / (mid + 1e-8)) * 10_000.0
    slippage_proxy_bps = spread_bps + 10_000.0 * trade_notional / (mid * max(visible_depth, 1e-8))
    return {
        "spread_bps": float(spread_bps),
        "microprice_offset_bps": float(microprice_offset_bps),
        "queue_imbalance": float(queue_imbalance),
        "top_depth": float(top_depth),
        "visible_depth": float(visible_depth),
        "toxicity": float(book.toxicity()),
        "turnover_proxy": float(trade_notional),
        "slippage_proxy_bps": float(slippage_proxy_bps),
        "trade_notional": float(trade_notional),
        "trade_count": float(trade_count),
        "event_count": float(event_count),
    }


def snapshot_to_tensor(
    book: OrderBookReconstructor,
    snapshot: OrderBookSnapshot,
    config: FeatureConfig,
) -> np.ndarray:
    """Convert a snapshot into a `(C, depth, 2)` tensor."""
    tensor = np.zeros((config.channels, config.depth, 2), dtype=np.float32)
    if snapshot.best_bid is None or snapshot.best_ask is None:
        return tensor

    mid = 0.5 * (snapshot.best_bid + snapshot.best_ask)
    levels_by_side = [snapshot.bids, snapshot.asks]

    for side_index, levels in enumerate(levels_by_side):
        signed_side = "bid" if side_index == 0 else "ask"
        for level_index, (price, volume) in enumerate(levels[: config.depth]):
            rel_log_price = np.log((price + 1e-8) / (mid + 1e-8))
            log_volume = np.log(max(volume, 1e-8))
            ofi = book.last_delta.get((signed_side, price), 0.0)
            toxicity = book.toxicity()
            if side_index == 1:
                rel_log_price = -rel_log_price
                ofi = -ofi
            tensor[0, level_index, side_index] = rel_log_price
            tensor[1, level_index, side_index] = log_volume
            tensor[2, level_index, side_index] = ofi
            tensor[3, level_index, side_index] = toxicity
    return tensor


def compute_forward_returns(prices: np.ndarray, horizons: list[int]) -> dict[str, np.ndarray]:
    """Compute forward log returns for multiple horizons."""
    outputs: dict[str, np.ndarray] = {}
    log_prices = np.log(prices + 1e-8)
    for horizon in horizons:
        if len(log_prices) == 0:
            outputs[f"return_{horizon}"] = np.zeros(0, dtype=np.float32)
            continue
        future = np.roll(log_prices, -horizon)
        future[-horizon:] = log_prices[-horizon:]
        outputs[f"return_{horizon}"] = (future - log_prices).astype(np.float32)
    return outputs


def derive_trade_volume(tensors: np.ndarray) -> np.ndarray:
    """Approximate visible depth as a volume-like state feature."""
    if len(tensors) == 0:
        return np.zeros(0, dtype=np.float32)
    return np.exp(tensors[:, 1]).sum(axis=(1, 2)).astype(np.float32)
