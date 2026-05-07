"""Adapter for one-day Binance public depth-curve + trades experiments.

This module uses Binance public `bookDepth` snapshots plus real trade ticks as a
research-friendly approximation when full event-level L2 is not publicly
available. It preserves the KAIROS-X pipeline shape while documenting the depth
approximation explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import zipfile

import numpy as np
import pandas as pd
import requests

from src.data.features import compute_forward_returns


@dataclass(slots=True)
class BinancePublicDataConfig:
    """Config for one-day Binance public-data ingestion."""

    symbol: str = "BTCUSDT"
    date: str = "2025-10-01"
    raw_dir: str = "data/raw/binance_one_day"
    levels_per_side: int = 10
    volume_threshold: float = 50_000_000.0


def _download_with_requests(url: str, output_path: Path) -> None:
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(output_path, "wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)


def _download_with_curl(url: str, output_path: Path) -> None:
    subprocess.run(["curl", "-L", "-o", str(output_path), url], check=True)


def _ensure_zip(url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    try:
        _download_with_requests(url, output_path)
    except Exception:
        _download_with_curl(url, output_path)
    return output_path


def _unzip_single(zip_path: Path) -> Path:
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()
        if len(members) != 1:
            raise ValueError(f"Expected a single CSV in {zip_path}, found {members}")
        member = members[0]
        target = zip_path.with_suffix(".csv")
        if target.exists() and target.stat().st_size > 0:
            return target
        archive.extract(member, path=zip_path.parent)
        extracted = zip_path.parent / member
        if extracted != target:
            extracted.rename(target)
        return target


def ensure_one_day_real_files(config: BinancePublicDataConfig) -> dict[str, Path]:
    """Download and unzip one day of public Binance futures trades + bookDepth."""
    raw_dir = Path(config.raw_dir)
    symbol = config.symbol.upper()
    date = config.date

    trades_zip = raw_dir / f"{symbol}-trades-{date}.zip"
    depth_zip = raw_dir / f"{symbol}-bookDepth-{date}.zip"

    trades_url = f"https://data.binance.vision/data/futures/um/daily/trades/{symbol}/{symbol}-trades-{date}.zip"
    depth_url = f"https://data.binance.vision/data/futures/um/daily/bookDepth/{symbol}/{symbol}-bookDepth-{date}.zip"

    _ensure_zip(trades_url, trades_zip)
    _ensure_zip(depth_url, depth_zip)
    trades_csv = _unzip_single(trades_zip)
    depth_csv = _unzip_single(depth_zip)
    return {"trades_csv": trades_csv, "depth_csv": depth_csv}


def _interpolate_side(values: np.ndarray, source_percentages: np.ndarray, target_percentages: np.ndarray) -> np.ndarray:
    return np.interp(target_percentages, source_percentages, values).astype(np.float32)


def _prepare_depth_snapshots(depth_csv: str | Path, levels_per_side: int) -> tuple[pd.DataFrame, np.ndarray]:
    depth_df = pd.read_csv(depth_csv)
    depth_df["timestamp"] = pd.to_datetime(depth_df["timestamp"], utc=True)
    ts_raw = depth_df["timestamp"].astype("int64").to_numpy()
    ts_ms = (ts_raw // 1_000_000).astype(np.int64)
    if ts_ms.max(initial=0) < 10**12:
        ts_ms = ts_ms * 1000
    depth_df["ts_ms"] = ts_ms
    target_percentages = np.linspace(1.0, 5.0, levels_per_side)

    grouped = []
    prev_bid: np.ndarray | None = None
    prev_ask: np.ndarray | None = None
    tensors: list[np.ndarray] = []
    snapshot_rows: list[dict[str, object]] = []

    for ts, group in depth_df.groupby("ts_ms", sort=True):
        bids = group[group["percentage"] < 0].copy()
        asks = group[group["percentage"] > 0].copy()
        bids["abs_pct"] = bids["percentage"].abs()
        asks["abs_pct"] = asks["percentage"].abs()
        bids = bids.sort_values("abs_pct")
        asks = asks.sort_values("abs_pct")

        bid_depth = _interpolate_side(bids["depth"].to_numpy(), bids["abs_pct"].to_numpy(), target_percentages)
        ask_depth = _interpolate_side(asks["depth"].to_numpy(), asks["abs_pct"].to_numpy(), target_percentages)
        bid_notional = _interpolate_side(bids["notional"].to_numpy(), bids["abs_pct"].to_numpy(), target_percentages)
        ask_notional = _interpolate_side(asks["notional"].to_numpy(), asks["abs_pct"].to_numpy(), target_percentages)

        tensor = np.zeros((4, levels_per_side, 2), dtype=np.float32)
        rel_prices = np.log1p(target_percentages / 100.0).astype(np.float32)
        tensor[0, :, 0] = -rel_prices
        tensor[0, :, 1] = rel_prices
        tensor[1, :, 0] = np.log(np.maximum(bid_depth, 1e-8))
        tensor[1, :, 1] = np.log(np.maximum(ask_depth, 1e-8))
        if prev_bid is not None and prev_ask is not None:
            tensor[2, :, 0] = (bid_depth - prev_bid) / np.maximum(prev_bid, 1e-6)
            tensor[2, :, 1] = -((ask_depth - prev_ask) / np.maximum(prev_ask, 1e-6))
        prev_bid = bid_depth
        prev_ask = ask_depth
        tensors.append(tensor)

        snapshot_rows.append(
            {
                "ts_ms": int(ts),
                "timestamp": pd.to_datetime(ts, unit="ms", utc=True),
                "best_bid_proxy": float(bid_notional[0] / max(bid_depth[0], 1e-8)),
                "best_ask_proxy": float(ask_notional[0] / max(ask_depth[0], 1e-8)),
                "visible_depth": float(bid_depth.sum() + ask_depth.sum()),
            }
        )

    snapshot_df = pd.DataFrame(snapshot_rows)
    return snapshot_df, np.stack(tensors) if tensors else np.zeros((0, 4, levels_per_side, 2), dtype=np.float32)


def _aggregate_trades(trades_csv: str | Path, snapshot_ts: np.ndarray) -> pd.DataFrame:
    trades = pd.read_csv(trades_csv, usecols=["price", "qty", "quote_qty", "time", "is_buyer_maker"])
    trades = trades.sort_values("time").reset_index(drop=True)
    interval_idx = np.searchsorted(snapshot_ts, trades["time"].to_numpy(), side="left")
    valid = interval_idx < len(snapshot_ts)
    trades = trades.loc[valid].copy()
    trades["snapshot_idx"] = interval_idx[valid]
    trades["buy_quote"] = np.where(~trades["is_buyer_maker"], trades["quote_qty"], 0.0)
    trades["sell_quote"] = np.where(trades["is_buyer_maker"], trades["quote_qty"], 0.0)
    grouped = trades.groupby("snapshot_idx").agg(
        buy_quote=("buy_quote", "sum"),
        sell_quote=("sell_quote", "sum"),
        trade_quote=("quote_qty", "sum"),
        trade_count=("time", "count"),
    )
    trade_times = trades["time"].to_numpy()
    trade_prices = trades["price"].to_numpy()
    last_trade_idx = np.searchsorted(trade_times, snapshot_ts, side="right") - 1
    last_trade_idx = np.clip(last_trade_idx, 0, len(trade_prices) - 1)
    proxy_price = trade_prices[last_trade_idx]
    base = pd.DataFrame({"snapshot_idx": np.arange(len(snapshot_ts)), "proxy_price": proxy_price})
    base = base.join(grouped, on="snapshot_idx").fillna(0.0)
    total = base["buy_quote"] + base["sell_quote"] + 1e-8
    base["toxicity"] = (base["buy_quote"] - base["sell_quote"]) / total
    return base


def build_time_bars_from_public_depth(
    depth_csv: str | Path,
    trades_csv: str | Path,
    levels_per_side: int = 10,
) -> dict[str, np.ndarray]:
    """Build time bars from public depth snapshots plus aligned trade flow."""
    snapshot_df, tensors = _prepare_depth_snapshots(depth_csv, levels_per_side)
    trade_df = _aggregate_trades(trades_csv, snapshot_df["ts_ms"].to_numpy())
    tensors = tensors.copy()
    toxicity = trade_df["toxicity"].to_numpy(dtype=np.float32)
    tensors[:, 3, :, 0] = toxicity[:, None]
    tensors[:, 3, :, 1] = toxicity[:, None]
    mid_prices = trade_df["proxy_price"].to_numpy(dtype=np.float64)
    payload = {
        "lob": tensors.astype(np.float32),
        "timestamps": snapshot_df["ts_ms"].to_numpy(dtype=np.int64),
        "mid_prices": mid_prices,
        "visible_volume": snapshot_df["visible_depth"].to_numpy(dtype=np.float32),
        "trade_quote_volume": trade_df["trade_quote"].to_numpy(dtype=np.float32),
        "trade_count": trade_df["trade_count"].to_numpy(dtype=np.float32),
        "source": np.array(["binance_public_bookdepth_timebars"]),
    }
    payload.update(compute_forward_returns(mid_prices, [5, 20, 50]))
    return payload


def build_volume_bars_from_public_depth(
    depth_csv: str | Path,
    trades_csv: str | Path,
    levels_per_side: int = 10,
    volume_threshold: float = 50_000_000.0,
) -> dict[str, np.ndarray]:
    """Build volume-clock bars by cutting on trade quote volume and sampling latest depth snapshot."""
    snapshot_df, tensors = _prepare_depth_snapshots(depth_csv, levels_per_side)
    trades = pd.read_csv(trades_csv, usecols=["price", "quote_qty", "time", "is_buyer_maker"])
    trades = trades.sort_values("time").reset_index(drop=True)
    cumulative = trades["quote_qty"].cumsum().to_numpy()
    total_quote = float(cumulative[-1]) if len(cumulative) else 0.0
    if total_quote <= volume_threshold:
        return build_time_bars_from_public_depth(depth_csv, trades_csv, levels_per_side)

    cut_targets = np.arange(volume_threshold, total_quote, volume_threshold)
    cut_indices = np.searchsorted(cumulative, cut_targets)
    cut_indices = np.unique(cut_indices[cut_indices < len(trades)])
    cut_times = trades.iloc[cut_indices]["time"].to_numpy(dtype=np.int64)
    snapshot_idx = np.searchsorted(snapshot_df["ts_ms"].to_numpy(), cut_times, side="right") - 1
    valid = snapshot_idx >= 0
    cut_indices = cut_indices[valid]
    cut_times = cut_times[valid]
    snapshot_idx = snapshot_idx[valid]
    if len(cut_indices) == 0:
        return build_time_bars_from_public_depth(depth_csv, trades_csv, levels_per_side)

    buy_quote_cum = np.where(~trades["is_buyer_maker"], trades["quote_qty"], 0.0).cumsum()
    sell_quote_cum = np.where(trades["is_buyer_maker"], trades["quote_qty"], 0.0).cumsum()
    prev_cut = -1
    toxicity = []
    trade_quotes = []
    mids = []
    selected_tensors = []
    for cut_idx, snap_idx, cut_time in zip(cut_indices, snapshot_idx, cut_times):
        buy_quote = float(buy_quote_cum[cut_idx] - (buy_quote_cum[prev_cut] if prev_cut >= 0 else 0.0))
        sell_quote = float(sell_quote_cum[cut_idx] - (sell_quote_cum[prev_cut] if prev_cut >= 0 else 0.0))
        total = buy_quote + sell_quote + 1e-8
        toxicity.append((buy_quote - sell_quote) / total)
        trade_quotes.append(float(cumulative[cut_idx] - (cumulative[prev_cut] if prev_cut >= 0 else 0.0)))
        mids.append(float(trades.iloc[cut_idx]["price"]))
        selected_tensors.append(tensors[snap_idx].copy())
        prev_cut = int(cut_idx)

    selected = np.stack(selected_tensors).astype(np.float32)
    tox = np.asarray(toxicity, dtype=np.float32)
    selected[:, 3, :, 0] = tox[:, None]
    selected[:, 3, :, 1] = tox[:, None]
    mids_np = np.asarray(mids, dtype=np.float64)
    payload = {
        "lob": selected,
        "timestamps": cut_times.astype(np.int64),
        "mid_prices": mids_np,
        "visible_volume": snapshot_df.iloc[snapshot_idx]["visible_depth"].to_numpy(dtype=np.float32),
        "trade_quote_volume": np.asarray(trade_quotes, dtype=np.float32),
        "trade_count": np.ones(len(cut_times), dtype=np.float32),
        "source": np.array(["binance_public_bookdepth_volumebars"]),
    }
    payload.update(compute_forward_returns(mids_np, [5, 20, 50]))
    return payload


def save_payload(payload: dict[str, np.ndarray], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
