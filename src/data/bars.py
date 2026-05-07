"""Bar construction for volume-clock and time-clock datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data.features import (
    FeatureConfig,
    compute_forward_returns,
    compute_snapshot_statistics,
    derive_trade_volume,
    snapshot_to_tensor,
)
from src.data.order_book import OrderBookReconstructor, merge_event_streams


@dataclass(slots=True)
class BarConfig:
    """Configuration for bar construction and sample slicing."""

    mode: str = "volume"
    volume_threshold: float = 100_000.0
    interval_ms: int = 1_000
    depth: int = 10
    horizons: tuple[int, ...] = (5, 20, 50)
    stride: int = 50
    max_bars: int | None = None
    chunksize: int = 250_000


def _finalize_dataset(
    tensors: list[np.ndarray],
    timestamps: list[int],
    mids: list[float],
    extras: dict[str, list[float]],
    horizons: tuple[int, ...],
) -> dict[str, np.ndarray]:
    lob = np.stack(tensors) if tensors else np.zeros((0, 4, 10, 2), dtype=np.float32)
    ts = np.asarray(timestamps, dtype=np.int64)
    mid_prices = np.asarray(mids, dtype=np.float64)
    outputs = {
        "lob": lob,
        "timestamps": ts,
        "mid_prices": mid_prices,
        "visible_volume": derive_trade_volume(lob),
    }
    for key, values in extras.items():
        outputs[key] = np.asarray(values, dtype=np.float32)
    benchmark_columns = [
        "spread_bps",
        "microprice_offset_bps",
        "queue_imbalance",
        "top_depth",
        "visible_depth",
        "toxicity_feature",
        "turnover_proxy",
        "slippage_proxy_bps",
        "trade_notional",
        "trade_count",
        "event_count",
    ]
    if benchmark_columns and all(column in outputs for column in benchmark_columns):
        outputs["benchmark_features"] = np.stack(
            [outputs[column] for column in benchmark_columns],
            axis=1,
        ).astype(np.float32)
        outputs["benchmark_feature_names"] = np.asarray(benchmark_columns)
    outputs.update(compute_forward_returns(mid_prices, list(horizons)))
    return outputs


def _empty_extras() -> dict[str, list[float]]:
    return {
        "spread_bps": [],
        "microprice_offset_bps": [],
        "queue_imbalance": [],
        "top_depth": [],
        "visible_depth": [],
        "toxicity_feature": [],
        "turnover_proxy": [],
        "slippage_proxy_bps": [],
        "trade_notional": [],
        "trade_count": [],
        "event_count": [],
    }


def build_volume_clock_bars(
    l2_path: str | Path,
    trades_path: str | Path | None,
    config: BarConfig,
) -> dict[str, np.ndarray]:
    """Build volume-clock bars from merged L2 and trade streams."""
    book = OrderBookReconstructor()
    features = FeatureConfig(depth=config.depth)
    cumulative_notional = 0.0
    tensors: list[np.ndarray] = []
    timestamps: list[int] = []
    mids: list[float] = []
    extras = _empty_extras()
    bar_trade_notional = 0.0
    bar_trade_count = 0
    bar_event_count = 0

    for event_type, event in merge_event_streams(l2_path, trades_path, chunksize=config.chunksize):
        bar_event_count += 1
        if event_type == "l2":
            # Volume-clock bars should advance on observed trade flow rather than
            # all negative book deltas, which also include cancellations/requotes.
            book.apply_l2(event)
            ts = event.timestamp
        else:
            trade_notional = book.apply_trade(event)
            cumulative_notional += trade_notional
            bar_trade_notional += trade_notional
            bar_trade_count += 1
            ts = event.timestamp

        if cumulative_notional < config.volume_threshold:
            continue

        snapshot = book.snapshot(ts, config.depth)
        if snapshot.best_bid is None or snapshot.best_ask is None:
            continue

        tensor = snapshot_to_tensor(book, snapshot, features)
        tensors.append(tensor)
        timestamps.append(ts)
        mids.append(0.5 * (snapshot.best_bid + snapshot.best_ask))
        stats = compute_snapshot_statistics(
            book,
            snapshot,
            config.depth,
            trade_notional=bar_trade_notional,
            trade_count=bar_trade_count,
            event_count=bar_event_count,
        )
        for key, value in stats.items():
            target_key = "toxicity_feature" if key == "toxicity" else key
            extras[target_key].append(float(value))

        cumulative_notional -= config.volume_threshold
        book.reset_flow()
        bar_trade_notional = 0.0
        bar_trade_count = 0
        bar_event_count = 0
        if config.max_bars is not None and len(tensors) >= config.max_bars:
            break

    return _finalize_dataset(tensors, timestamps, mids, extras, config.horizons)


def build_time_bars(
    l2_path: str | Path,
    trades_path: str | Path | None,
    config: BarConfig,
) -> dict[str, np.ndarray]:
    """Build fixed-interval time bars from merged event streams."""
    book = OrderBookReconstructor()
    features = FeatureConfig(depth=config.depth)
    tensors: list[np.ndarray] = []
    timestamps: list[int] = []
    mids: list[float] = []
    extras = _empty_extras()
    next_cut: int | None = None
    bar_trade_notional = 0.0
    bar_trade_count = 0
    bar_event_count = 0

    for event_type, event in merge_event_streams(l2_path, trades_path, chunksize=config.chunksize):
        bar_event_count += 1
        if event_type == "l2":
            executed_notional, _ = book.apply_l2(event)
            bar_trade_notional += executed_notional
            ts = event.timestamp
        else:
            trade_notional = book.apply_trade(event)
            bar_trade_notional += trade_notional
            bar_trade_count += 1
            ts = event.timestamp

        if next_cut is None:
            next_cut = ts + config.interval_ms * 1_000

        while next_cut is not None and ts >= next_cut:
            snapshot = book.snapshot(next_cut, config.depth)
            if snapshot.best_bid is not None and snapshot.best_ask is not None:
                tensor = snapshot_to_tensor(book, snapshot, features)
                tensors.append(tensor)
                timestamps.append(next_cut)
                mids.append(0.5 * (snapshot.best_bid + snapshot.best_ask))
                stats = compute_snapshot_statistics(
                    book,
                    snapshot,
                    config.depth,
                    trade_notional=bar_trade_notional,
                    trade_count=bar_trade_count,
                    event_count=bar_event_count,
                )
                for key, value in stats.items():
                    target_key = "toxicity_feature" if key == "toxicity" else key
                    extras[target_key].append(float(value))
                book.reset_flow()
                bar_trade_notional = 0.0
                bar_trade_count = 0
                bar_event_count = 0
                if config.max_bars is not None and len(tensors) >= config.max_bars:
                    return _finalize_dataset(tensors, timestamps, mids, extras, config.horizons)
            next_cut += config.interval_ms * 1_000

    return _finalize_dataset(tensors, timestamps, mids, extras, config.horizons)


def build_dual_clock_bars(
    l2_path: str | Path,
    trades_path: str | Path | None,
    time_config: BarConfig,
    volume_config: BarConfig,
) -> dict[str, dict[str, np.ndarray]]:
    """Replay the raw event stream once and emit both time and volume bars."""
    if time_config.depth != volume_config.depth:
        raise ValueError("time/volume depth must match for dual-clock preprocessing")
    if tuple(time_config.horizons) != tuple(volume_config.horizons):
        raise ValueError("time/volume horizons must match for dual-clock preprocessing")

    depth = time_config.depth
    horizons = time_config.horizons
    chunksize = min(time_config.chunksize, volume_config.chunksize)

    book_time = OrderBookReconstructor()
    book_vol = OrderBookReconstructor()
    features_time = FeatureConfig(depth=depth)
    features_vol = FeatureConfig(depth=depth)

    time_tensors: list[np.ndarray] = []
    time_timestamps: list[int] = []
    time_mids: list[float] = []
    time_extras = _empty_extras()
    time_next_cut: int | None = None
    time_trade_notional = 0.0
    time_trade_count = 0
    time_event_count = 0

    vol_tensors: list[np.ndarray] = []
    vol_timestamps: list[int] = []
    vol_mids: list[float] = []
    vol_extras = _empty_extras()
    vol_cumulative_notional = 0.0
    vol_trade_notional = 0.0
    vol_trade_count = 0
    vol_event_count = 0

    for event_type, event in merge_event_streams(l2_path, trades_path, chunksize=chunksize):
        time_event_count += 1
        vol_event_count += 1
        if event_type == "l2":
            executed_notional, _ = book_time.apply_l2(event)
            book_vol.apply_l2(event)
            time_trade_notional += executed_notional
            ts = event.timestamp
        else:
            trade_notional_time = book_time.apply_trade(event)
            trade_notional_vol = book_vol.apply_trade(event)
            trade_notional = trade_notional_time if trade_notional_time == trade_notional_vol else trade_notional_time
            time_trade_notional += trade_notional
            time_trade_count += 1
            vol_cumulative_notional += trade_notional
            vol_trade_notional += trade_notional
            vol_trade_count += 1
            ts = event.timestamp

        if time_next_cut is None:
            time_next_cut = ts + time_config.interval_ms * 1_000

        while time_next_cut is not None and ts >= time_next_cut:
            snapshot = book_time.snapshot(time_next_cut, depth)
            if snapshot.best_bid is not None and snapshot.best_ask is not None:
                tensor = snapshot_to_tensor(book_time, snapshot, features_time)
                time_tensors.append(tensor)
                time_timestamps.append(time_next_cut)
                time_mids.append(0.5 * (snapshot.best_bid + snapshot.best_ask))
                stats = compute_snapshot_statistics(
                    book_time,
                    snapshot,
                    depth,
                    trade_notional=time_trade_notional,
                    trade_count=time_trade_count,
                    event_count=time_event_count,
                )
                for key, value in stats.items():
                    target_key = "toxicity_feature" if key == "toxicity" else key
                    time_extras[target_key].append(float(value))
                book_time.reset_flow()
                time_trade_notional = 0.0
                time_trade_count = 0
                time_event_count = 0
                if time_config.max_bars is not None and len(time_tensors) >= time_config.max_bars:
                    time_next_cut = None
                    break
            time_next_cut += time_config.interval_ms * 1_000

        if vol_cumulative_notional >= volume_config.volume_threshold:
            snapshot = book_vol.snapshot(ts, depth)
            if snapshot.best_bid is not None and snapshot.best_ask is not None:
                tensor = snapshot_to_tensor(book_vol, snapshot, features_vol)
                vol_tensors.append(tensor)
                vol_timestamps.append(ts)
                vol_mids.append(0.5 * (snapshot.best_bid + snapshot.best_ask))
                stats = compute_snapshot_statistics(
                    book_vol,
                    snapshot,
                    depth,
                    trade_notional=vol_trade_notional,
                    trade_count=vol_trade_count,
                    event_count=vol_event_count,
                )
                for key, value in stats.items():
                    target_key = "toxicity_feature" if key == "toxicity" else key
                    vol_extras[target_key].append(float(value))
                vol_cumulative_notional -= volume_config.volume_threshold
                book_vol.reset_flow()
                vol_trade_notional = 0.0
                vol_trade_count = 0
                vol_event_count = 0
                if volume_config.max_bars is not None and len(vol_tensors) >= volume_config.max_bars:
                    break

    return {
        "time": _finalize_dataset(time_tensors, time_timestamps, time_mids, time_extras, horizons),
        "volume": _finalize_dataset(vol_tensors, vol_timestamps, vol_mids, vol_extras, horizons),
    }


def build_bar_dataset(
    l2_path: str | Path,
    trades_path: str | Path | None,
    config: BarConfig,
) -> dict[str, np.ndarray]:
    """Build a dataset according to `config.mode`."""
    if config.mode == "volume":
        return build_volume_clock_bars(l2_path, trades_path, config)
    if config.mode == "time":
        return build_time_bars(l2_path, trades_path, config)
    raise ValueError(f"Unsupported bar mode: {config.mode}")


def save_bar_dataset(dataset: dict[str, np.ndarray], output_path: str | Path) -> None:
    """Persist the processed dataset."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **dataset)


def slice_time_indices(length: int, seq_len: int, stride: int) -> np.ndarray:
    """Return non-overlapping or low-overlap sample boundaries."""
    if length <= seq_len:
        return np.zeros(0, dtype=np.int64)
    return np.arange(seq_len, length, stride, dtype=np.int64)
