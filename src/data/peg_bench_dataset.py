"""PEG-Bench preprocessing bundle builder for multi-day Tardis-style datasets."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
import json
import math
import os
from pathlib import Path
import re

import numpy as np

from src.data.bars import BarConfig, build_bar_dataset, build_dual_clock_bars, save_bar_dataset


@dataclass(slots=True)
class BenchmarkUniverseSpec:
    """Single asset-venue pair tracked by PEG-Bench."""

    venue: str
    asset: str
    symbol_aliases: tuple[str, ...]
    venue_aliases: tuple[str, ...]


DEFAULT_UNIVERSE: tuple[BenchmarkUniverseSpec, ...] = (
    BenchmarkUniverseSpec(
        venue="binance",
        asset="BTCUSDT",
        symbol_aliases=("btcusdt",),
        venue_aliases=("binance", "binance-futures", "binanceusdm"),
    ),
    BenchmarkUniverseSpec(
        venue="binance",
        asset="ETHUSDT",
        symbol_aliases=("ethusdt",),
        venue_aliases=("binance", "binance-futures", "binanceusdm"),
    ),
    BenchmarkUniverseSpec(
        venue="okx",
        asset="BTCUSDT",
        symbol_aliases=("btc-usdt-swap", "btcusdt"),
        venue_aliases=("okx", "okex"),
    ),
    BenchmarkUniverseSpec(
        venue="okx",
        asset="ETHUSDT",
        symbol_aliases=("eth-usdt-swap", "ethusdt"),
        venue_aliases=("okx", "okex"),
    ),
)


@dataclass(slots=True)
class PEGBenchPreprocessConfig:
    """Configuration for building a PEG-Bench-compatible processed bundle."""

    raw_root: str
    output_root: str
    selection_mode: str = "protocol"
    start_date: str | None = None
    end_date: str | None = None
    train_days: int | None = None
    val_days: int | None = None
    test_days: int | None = None
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    depth: int = 10
    interval_ms: int = 1_000
    volume_threshold: float = 100_000.0
    horizons: tuple[int, ...] = (5, 20, 50)
    bar_modes: tuple[str, ...] = ("time", "volume")
    context_len: int = 96
    max_days: int | None = None
    max_stress_days: int = 12
    min_stress_days: int = 8
    num_workers: int = 1
    batch_size: int = 0
    checkpoint_every: int = 1


@dataclass(slots=True)
class RawDayRecord:
    """Raw-file manifest entry for one asset-venue day."""

    venue: str
    asset: str
    date: str
    l2_path: str | None = None
    trades_path: str | None = None


DATE_RE = re.compile(r"(20\d{2}-\d{2}-\d{2})")
PATH_DATE_RE = re.compile(r"/(20\d{2})/(\d{2})/(\d{2})/")


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _extract_date(path: Path) -> date | None:
    text = path.as_posix()
    direct = DATE_RE.search(text)
    if direct is not None:
        return _parse_date(direct.group(1))
    nested = PATH_DATE_RE.search(text)
    if nested is None:
        return None
    year, month, day = nested.groups()
    return _parse_date(f"{year}-{month}-{day}")


def _daterange(start: date, end: date) -> list[date]:
    values: list[date] = []
    current = start
    while current <= end:
        values.append(current)
        current += timedelta(days=1)
    return values


def _split_count_from_ratio(total_days: int, train_ratio: float, val_ratio: float, test_ratio: float) -> dict[str, int]:
    if total_days <= 0:
        raise ValueError("Cannot build a PEG-Bench split without any discovered dates.")
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("Split ratios must be non-negative.")
    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("At least one split ratio must be positive.")
    normalized = {
        "train": train_ratio / ratio_sum,
        "val": val_ratio / ratio_sum,
        "test": test_ratio / ratio_sum,
    }
    raw = {name: normalized[name] * total_days for name in ("train", "val", "test")}
    counts = {name: int(math.floor(value)) for name, value in raw.items()}
    remainder = total_days - sum(counts.values())
    priorities = sorted(
        ("train", "val", "test"),
        key=lambda name: (raw[name] - counts[name], normalized[name]),
        reverse=True,
    )
    for index in range(remainder):
        counts[priorities[index % len(priorities)]] += 1
    if total_days >= 3:
        for name in ("val", "test"):
            if normalized[name] <= 0 or counts[name] > 0:
                continue
            donor = "train" if counts["train"] > 1 else max(counts, key=counts.get)
            if counts[donor] <= 1:
                continue
            counts[donor] -= 1
            counts[name] += 1
    if counts["train"] <= 0:
        donor = "val" if counts["val"] > counts["test"] else "test"
        if counts[donor] <= 1:
            raise ValueError("Adaptive split left no train days; increase coverage or adjust ratios.")
        counts[donor] -= 1
        counts["train"] += 1
    return counts


def _resolve_split_counts(cfg: PEGBenchPreprocessConfig, total_days: int) -> tuple[dict[str, int], str]:
    explicit = (cfg.train_days, cfg.val_days, cfg.test_days)
    if all(value is not None for value in explicit):
        counts = {
            "train": int(cfg.train_days or 0),
            "val": int(cfg.val_days or 0),
            "test": int(cfg.test_days or 0),
        }
        expected = counts["train"] + counts["val"] + counts["test"]
        if total_days != expected:
            raise ValueError(
                f"Configured split uses {expected} days but discovered {total_days} aligned days. "
                "Adjust start/end or remove train_days/val_days/test_days to enable adaptive ratios."
            )
        return counts, "fixed_days"
    return _split_count_from_ratio(total_days, cfg.train_ratio, cfg.val_ratio, cfg.test_ratio), "adaptive_ratio"


def _build_split_lookup_from_dates(protocol_dates: list[str], split_counts: dict[str, int]) -> dict[str, str]:
    expected = split_counts["train"] + split_counts["val"] + split_counts["test"]
    if len(protocol_dates) != expected:
        raise ValueError(f"Protocol dates contain {len(protocol_dates)} days, expected {expected} from split counts.")
    split_lookup: dict[str, str] = {}
    train_end = split_counts["train"]
    val_end = train_end + split_counts["val"]
    for index, day in enumerate(protocol_dates):
        if index < train_end:
            split_lookup[day] = "train"
        elif index < val_end:
            split_lookup[day] = "val"
        else:
            split_lookup[day] = "test"
    return split_lookup


def _infer_feed(path: Path) -> str | None:
    text = path.as_posix().lower()
    if "incremental_book_l2" in text or "book_l2" in text:
        return "l2"
    if "trades" in text:
        return "trades"
    return None


def _match_universe(path: Path, universe: tuple[BenchmarkUniverseSpec, ...]) -> BenchmarkUniverseSpec | None:
    text = path.as_posix().lower()
    for spec in universe:
        if any(alias in text for alias in spec.venue_aliases) and any(alias in text for alias in spec.symbol_aliases):
            return spec
    return None


def discover_tardis_manifest(
    raw_root: str | Path,
    start_date: str | None,
    end_date: str | None,
    universe: tuple[BenchmarkUniverseSpec, ...] = DEFAULT_UNIVERSE,
) -> list[RawDayRecord]:
    """Scan a raw Tardis directory and collect L2/trade pairs by day."""
    root = Path(raw_root)
    if not root.exists():
        raise FileNotFoundError(f"Raw data root not found: {root}")
    start = _parse_date(start_date) if start_date else None
    end = _parse_date(end_date) if end_date else None
    discovered: dict[tuple[str, str, str], RawDayRecord] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        feed = _infer_feed(path)
        if feed is None:
            continue
        day = _extract_date(path)
        if day is None:
            continue
        if start is not None and day < start:
            continue
        if end is not None and day > end:
            continue
        spec = _match_universe(path, universe)
        if spec is None:
            continue
        key = (spec.venue, spec.asset, day.isoformat())
        record = discovered.setdefault(
            key,
            RawDayRecord(venue=spec.venue, asset=spec.asset, date=day.isoformat()),
        )
        if feed == "l2":
            record.l2_path = str(path.resolve())
        else:
            record.trades_path = str(path.resolve())
    return sorted(discovered.values(), key=lambda item: (item.date, item.venue, item.asset))


def _collect_complete_dates(
    discovered: list[RawDayRecord],
    universe: tuple[BenchmarkUniverseSpec, ...],
) -> dict[tuple[str, str], set[str]]:
    coverage: dict[tuple[str, str], set[str]] = {
        (spec.venue, spec.asset): set()
        for spec in universe
    }
    for record in discovered:
        if record.l2_path is None or record.trades_path is None:
            continue
        coverage.setdefault((record.venue, record.asset), set()).add(record.date)
    return coverage


def _select_protocol_dates(
    discovered: list[RawDayRecord],
    universe: tuple[BenchmarkUniverseSpec, ...],
) -> tuple[list[str], dict[str, object]]:
    coverage = _collect_complete_dates(discovered, universe)
    non_empty_sets = [dates for dates in coverage.values() if dates]
    if not non_empty_sets:
        return [], {
            "alignment_mode": "no_complete_days",
            "coverage_by_pair": {f"{venue}:{asset}": [] for venue, asset in coverage},
        }
    common_dates = sorted(set.intersection(*non_empty_sets)) if len(non_empty_sets) == len(coverage) else []
    if common_dates:
        protocol_dates = common_dates
        alignment_mode = "common_intersection"
    else:
        protocol_dates = sorted(set().union(*non_empty_sets))
        alignment_mode = "union_fallback"
    return protocol_dates, {
        "alignment_mode": alignment_mode,
        "coverage_by_pair": {
            f"{venue}:{asset}": sorted(dates)
            for (venue, asset), dates in sorted(coverage.items())
        },
        "num_pairs_with_complete_days": len(non_empty_sets),
        "num_protocol_days": len(protocol_dates),
    }


def _normalize_selection_mode(value: str | None) -> str:
    mode = (value or "protocol").strip().lower()
    if mode not in {"protocol", "discovered"}:
        raise ValueError(f"Unsupported selection_mode={value!r}; expected 'protocol' or 'discovered'.")
    return mode


def _select_processing_records(
    cfg: PEGBenchPreprocessConfig,
    discovered: list[RawDayRecord],
    universe: tuple[BenchmarkUniverseSpec, ...],
) -> tuple[list[RawDayRecord], list[str], dict[str, object]]:
    mode = _normalize_selection_mode(cfg.selection_mode)
    if mode == "protocol":
        protocol_dates, split_meta = _select_protocol_dates(discovered, universe)
        if protocol_dates:
            selected_days = set(protocol_dates)
            discovered = [record for record in discovered if record.date in selected_days]
        else:
            discovered = []
        return discovered, protocol_dates, split_meta

    complete_records = [record for record in discovered if record.l2_path and record.trades_path]
    selected_dates = sorted({record.date for record in complete_records})
    coverage = _collect_complete_dates(discovered, universe)
    split_meta = {
        "alignment_mode": "discovered_records",
        "coverage_by_pair": {
            f"{venue}:{asset}": sorted(dates)
            for (venue, asset), dates in sorted(coverage.items())
        },
        "num_pairs_with_complete_days": sum(1 for dates in coverage.values() if dates),
        "num_protocol_days": len(selected_dates),
    }
    return complete_records, selected_dates, split_meta


def _load_dataset_summary(dataset_path: Path) -> dict[str, object]:
    payload = np.load(dataset_path, allow_pickle=True)
    mid_prices = payload["mid_prices"].astype(np.float64)
    spread = payload["spread_bps"].astype(np.float64) if "spread_bps" in payload else np.zeros(len(mid_prices))
    top_depth = payload["top_depth"].astype(np.float64) if "top_depth" in payload else np.zeros(len(mid_prices))
    queue_imbalance = (
        payload["queue_imbalance"].astype(np.float64) if "queue_imbalance" in payload else np.zeros(len(mid_prices))
    )
    trade_count = payload["trade_count"].astype(np.float64) if "trade_count" in payload else np.zeros(len(mid_prices))
    returns = np.diff(np.log(mid_prices + 1e-8)) if len(mid_prices) > 1 else np.zeros(0, dtype=np.float64)
    one_minute = 60
    if len(mid_prices) >= one_minute:
        one_minute_returns = np.log(mid_prices[one_minute - 1 :] + 1e-8) - np.log(mid_prices[: 1 - one_minute] + 1e-8)
        jump_threshold = np.percentile(np.abs(one_minute_returns), 95) if len(one_minute_returns) else 0.0
        jump_count = int(np.sum(np.abs(one_minute_returns) >= jump_threshold)) if jump_threshold > 0 else 0
    else:
        one_minute_returns = np.zeros(0, dtype=np.float64)
        jump_count = 0
    median_depth = float(np.median(top_depth)) if len(top_depth) else 0.0
    p05_depth = float(np.percentile(top_depth, 5)) if len(top_depth) else 0.0
    depth_collapse_score = median_depth / max(p05_depth, 1e-8) if median_depth > 0 else 0.0
    return {
        "num_bars": int(len(mid_prices)),
        "timestamp_start": int(payload["timestamps"][0]) if len(mid_prices) else None,
        "timestamp_end": int(payload["timestamps"][-1]) if len(mid_prices) else None,
        "daily_return": float(np.log((mid_prices[-1] + 1e-8) / (mid_prices[0] + 1e-8))) if len(mid_prices) else 0.0,
        "realized_volatility": float(np.std(returns)) if len(returns) else 0.0,
        "max_spread_bps": float(np.max(spread)) if len(spread) else 0.0,
        "median_spread_bps": float(np.median(spread)) if len(spread) else 0.0,
        "imbalance_dispersion": float(np.std(queue_imbalance)) if len(queue_imbalance) else 0.0,
        "trade_arrival_intensity": float(np.mean(trade_count)) if len(trade_count) else 0.0,
        "top_of_book_depth": median_depth,
        "depth_collapse_score": float(depth_collapse_score),
        "jump_count_1m": int(jump_count),
        "benchmark_feature_names": payload["benchmark_feature_names"].tolist()
        if "benchmark_feature_names" in payload
        else [],
    }


def _regime_tag_panel(summaries: list[dict[str, object]]) -> dict[tuple[str, str, str], list[str]]:
    if not summaries:
        return {}
    returns = np.asarray([float(item["daily_return"]) for item in summaries], dtype=np.float64)
    vol = np.asarray([float(item["realized_volatility"]) for item in summaries], dtype=np.float64)
    depth = np.asarray([float(item["top_of_book_depth"]) for item in summaries], dtype=np.float64)
    spread = np.asarray([float(item["max_spread_bps"]) for item in summaries], dtype=np.float64)
    tags: dict[tuple[str, str, str], list[str]] = {}
    q_return_hi = np.percentile(returns, 75)
    q_return_lo = np.percentile(returns, 25)
    q_vol_hi = np.percentile(vol, 75)
    q_depth_lo = np.percentile(depth, 25)
    q_spread_hi = np.percentile(spread, 75)
    for item in summaries:
        key = (str(item["venue"]), str(item["asset"]), str(item["date"]))
        current: list[str] = []
        if float(item["daily_return"]) >= q_return_hi:
            current.append("trend_up")
        if float(item["daily_return"]) <= q_return_lo:
            current.append("trend_down")
        if float(item["realized_volatility"]) >= q_vol_hi:
            current.append("high_vol")
        if float(item["top_of_book_depth"]) <= q_depth_lo or float(item["max_spread_bps"]) >= q_spread_hi:
            current.append("liquidity_stress")
        if not current:
            current.append("neutral")
        tags[key] = current
    return tags


def _stress_day_selection(
    summaries: list[dict[str, object]],
    max_stress_days: int,
    min_stress_days: int,
) -> list[dict[str, object]]:
    if not summaries:
        return []
    fields = [
        ("abs_daily_return", np.asarray([abs(float(item["daily_return"])) for item in summaries], dtype=np.float64)),
        ("realized_volatility", np.asarray([float(item["realized_volatility"]) for item in summaries], dtype=np.float64)),
        ("max_spread_bps", np.asarray([float(item["max_spread_bps"]) for item in summaries], dtype=np.float64)),
        ("depth_collapse_score", np.asarray([float(item["depth_collapse_score"]) for item in summaries], dtype=np.float64)),
        ("jump_count_1m", np.asarray([float(item["jump_count_1m"]) for item in summaries], dtype=np.float64)),
    ]
    thresholds = {name: float(np.percentile(values, 95)) for name, values in fields}
    candidates: list[dict[str, object]] = []
    for index, summary in enumerate(summaries):
        triggers = []
        score = 0.0
        for name, values in fields:
            value = float(values[index])
            threshold = thresholds[name]
            if value >= threshold and threshold > 0:
                triggers.append(name)
            centered = values - np.mean(values)
            scale = np.std(values) + 1e-8
            score += float(centered[index] / scale)
        if triggers:
            payload = dict(summary)
            payload["stress_reasons"] = triggers
            payload["stress_score"] = score
            candidates.append(payload)
    candidates.sort(key=lambda item: (len(item["stress_reasons"]), item["stress_score"]), reverse=True)
    if len(candidates) < min_stress_days:
        ranked = sorted(summaries, key=lambda item: abs(float(item["daily_return"])) + float(item["realized_volatility"]), reverse=True)
        seen = {(item["venue"], item["asset"], item["date"]) for item in candidates}
        for item in ranked:
            key = (item["venue"], item["asset"], item["date"])
            if key in seen:
                continue
            extra = dict(item)
            extra["stress_reasons"] = ["fallback_rank"]
            extra["stress_score"] = abs(float(item["daily_return"])) + float(item["realized_volatility"])
            candidates.append(extra)
            seen.add(key)
            if len(candidates) >= min_stress_days:
                break
    return candidates[:max_stress_days]


def _expected_coverage(cfg: PEGBenchPreprocessConfig, universe: tuple[BenchmarkUniverseSpec, ...]) -> int:
    if cfg.start_date is None or cfg.end_date is None:
        return 0
    return len(_daterange(_parse_date(cfg.start_date), _parse_date(cfg.end_date))) * len(universe)


def _record_id(record: RawDayRecord | dict[str, object]) -> str:
    return f"{record['date'] if isinstance(record, dict) else record.date}/{record['venue'] if isinstance(record, dict) else record.venue}/{record['asset'] if isinstance(record, dict) else record.asset}"


def _chunk_indexed_records(
    indexed_records: list[tuple[int, RawDayRecord]],
    batch_size: int,
) -> list[list[tuple[int, RawDayRecord]]]:
    if batch_size <= 0 or batch_size >= len(indexed_records):
        return [indexed_records]
    return [indexed_records[start : start + batch_size] for start in range(0, len(indexed_records), batch_size)]


def _write_checkpoint(
    output_root: Path,
    cfg: PEGBenchPreprocessConfig,
    processed_indexed: list[tuple[int, dict[str, object]]],
    total_records: int,
    total_batches: int,
    completed_batches: int,
    worker_count: int,
    status: str,
) -> None:
    processed_records = [item for _, item in sorted(processed_indexed, key=lambda pair: pair[0])]
    complete_count = sum(1 for item in processed_records if item.get("status") == "processed")
    incomplete_count = sum(1 for item in processed_records if item.get("status") != "processed")
    latest_ids = [_record_id(item) for item in processed_records[-min(5, len(processed_records)) :]]
    checkpoint = {
        "status": status,
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "config": {
            "raw_root": cfg.raw_root,
            "output_root": cfg.output_root,
            "start_date": cfg.start_date,
            "end_date": cfg.end_date,
            "bar_modes": list(cfg.bar_modes),
            "num_workers": worker_count,
            "batch_size": cfg.batch_size,
            "checkpoint_every": cfg.checkpoint_every,
        },
        "progress": {
            "total_asset_days": total_records,
            "completed_asset_days": len(processed_records),
            "processed_complete": complete_count,
            "processed_incomplete": incomplete_count,
            "remaining_asset_days": max(total_records - len(processed_records), 0),
            "progress_pct": round(100.0 * len(processed_records) / max(total_records, 1), 2),
            "total_batches": total_batches,
            "completed_batches": completed_batches,
        },
        "latest_completed": latest_ids,
        "completed_record_ids": [_record_id(item) for item in processed_records],
    }
    (output_root / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2))


def _process_asset_day(
    record: RawDayRecord,
    split: str,
    output_root: str,
    bar_modes: tuple[str, ...],
    depth: int,
    interval_ms: int,
    volume_threshold: float,
    horizons: tuple[int, ...],
) -> dict[str, object]:
    complete = record.l2_path is not None and record.trades_path is not None
    if not complete:
        return {
            **asdict(record),
            "split": split,
            "status": "incomplete",
        }

    record_root = Path(output_root) / "processed" / record.venue / record.asset / record.date
    record_root.mkdir(parents=True, exist_ok=True)
    processed_paths: dict[str, str] = {}
    summaries: dict[str, dict[str, object]] = {}
    normalized_modes = tuple(dict.fromkeys(bar_modes))
    dataset_paths = {mode: record_root / f"{mode}_bars.npz" for mode in normalized_modes}
    missing_modes = [mode for mode in normalized_modes if not dataset_paths[mode].exists()]

    if set(normalized_modes) == {"time", "volume"} and set(missing_modes) == {"time", "volume"}:
        datasets = build_dual_clock_bars(
            record.l2_path,
            record.trades_path,
            BarConfig(
                mode="time",
                volume_threshold=volume_threshold,
                interval_ms=interval_ms,
                depth=depth,
                horizons=horizons,
            ),
            BarConfig(
                mode="volume",
                volume_threshold=volume_threshold,
                interval_ms=interval_ms,
                depth=depth,
                horizons=horizons,
            ),
        )
        for mode, dataset in datasets.items():
            save_bar_dataset(dataset, dataset_paths[mode])

    for mode in normalized_modes:
        dataset_path = dataset_paths[mode]
        if not dataset_path.exists():
            dataset = build_bar_dataset(
                record.l2_path,
                record.trades_path,
                BarConfig(
                    mode=mode,
                    volume_threshold=volume_threshold,
                    interval_ms=interval_ms,
                    depth=depth,
                    horizons=horizons,
                ),
            )
            save_bar_dataset(dataset, dataset_path)
        processed_paths[mode] = str(dataset_path.resolve())
        summaries[mode] = _load_dataset_summary(dataset_path)
    return {
        **asdict(record),
        "split": split,
        "status": "processed",
        "processed_paths": processed_paths,
        "summaries": summaries,
    }


def build_peg_bench_bundle(
    cfg: PEGBenchPreprocessConfig,
    universe: tuple[BenchmarkUniverseSpec, ...] = DEFAULT_UNIVERSE,
) -> dict[str, object]:
    """Process raw multi-day Tardis feeds into a PEG-Bench-style bundle."""
    output_root = Path(cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    discovered = discover_tardis_manifest(cfg.raw_root, cfg.start_date, cfg.end_date, universe=universe)
    discovered, protocol_dates, split_meta = _select_processing_records(cfg, discovered, universe)
    selection_mode = _normalize_selection_mode(cfg.selection_mode)
    if selection_mode == "discovered":
        split_counts = {"train": len(protocol_dates), "val": 0, "test": 0}
        split_strategy = "selection_mode=discovered"
        split_lookup = {day: "train" for day in protocol_dates}
    else:
        split_counts, split_strategy = _resolve_split_counts(cfg, len(protocol_dates))
        split_lookup = _build_split_lookup_from_dates(protocol_dates, split_counts)
    if cfg.max_days is not None:
        discovered = discovered[: cfg.max_days]
        selected_dates = {record.date for record in discovered}
        split_lookup = {day: split for day, split in split_lookup.items() if day in selected_dates}
        protocol_dates = sorted(split_lookup)
    print(
        f"[PEG-Bench] discovered {len(discovered)} asset-days under {cfg.raw_root} "
        f"for {protocol_dates[0] if protocol_dates else cfg.start_date}..{protocol_dates[-1] if protocol_dates else cfg.end_date}",
        flush=True,
    )

    total_records = len(discovered)
    indexed_records = list(enumerate(discovered, start=1))
    processed_indexed: list[tuple[int, dict[str, object]]] = []
    worker_count = max(1, min(cfg.num_workers, total_records, os.cpu_count() or cfg.num_workers))
    batches = _chunk_indexed_records(indexed_records, cfg.batch_size)
    print(
        f"[PEG-Bench] using {worker_count} workers with {len(batches)} batch(es)"
        + (f" of up to {cfg.batch_size} asset-days" if cfg.batch_size > 0 else ""),
        flush=True,
    )
    _write_checkpoint(
        output_root,
        cfg,
        processed_indexed,
        total_records,
        len(batches),
        0,
        worker_count,
        status="running",
    )
    completed_jobs = 0
    checkpoint_every = max(1, cfg.checkpoint_every)
    for batch_no, batch_records in enumerate(batches, start=1):
        batch_worker_count = max(1, min(worker_count, len(batch_records)))
        batch_start = batch_records[0][0]
        batch_end = batch_records[-1][0]
        print(
            f"[PEG-Bench] batch {batch_no}/{len(batches)} starting "
            f"(records {batch_start}-{batch_end}, size={len(batch_records)}, workers={batch_worker_count})",
            flush=True,
        )
        if batch_worker_count == 1:
            for index, record in batch_records:
                complete = record.l2_path is not None and record.trades_path is not None
                state = "processing" if complete else "incomplete"
                print(f"[PEG-Bench] {index}/{total_records} {state} {record.venue} {record.asset} {record.date}", flush=True)
                item = _process_asset_day(
                    record,
                    split_lookup.get(record.date, "out_of_protocol"),
                    cfg.output_root,
                    cfg.bar_modes,
                    cfg.depth,
                    cfg.interval_ms,
                    cfg.volume_threshold,
                    cfg.horizons,
                )
                if item["status"] == "processed":
                    print(f"[PEG-Bench] {index}/{total_records} completed {record.venue} {record.asset} {record.date}", flush=True)
                processed_indexed.append((index, item))
                completed_jobs += 1
                if completed_jobs % checkpoint_every == 0:
                    _write_checkpoint(
                        output_root,
                        cfg,
                        processed_indexed,
                        total_records,
                        len(batches),
                        batch_no - 1,
                        worker_count,
                        status="running",
                    )
        else:
            with ProcessPoolExecutor(max_workers=batch_worker_count) as executor:
                future_map = {
                    executor.submit(
                        _process_asset_day,
                        record,
                        split_lookup.get(record.date, "out_of_protocol"),
                        cfg.output_root,
                        cfg.bar_modes,
                        cfg.depth,
                        cfg.interval_ms,
                        cfg.volume_threshold,
                        cfg.horizons,
                    ): (index, record)
                    for index, record in batch_records
                }
                for future in as_completed(future_map):
                    index, record = future_map[future]
                    item = future.result()
                    completed_jobs += 1
                    print(
                        f"[PEG-Bench] {completed_jobs}/{total_records} completed "
                        f"{record.venue} {record.asset} {record.date} status={item['status']}",
                        flush=True,
                    )
                    processed_indexed.append((index, item))
                    if completed_jobs % checkpoint_every == 0:
                        _write_checkpoint(
                            output_root,
                            cfg,
                            processed_indexed,
                            total_records,
                            len(batches),
                            batch_no - 1,
                            worker_count,
                            status="running",
                        )
        _write_checkpoint(
            output_root,
            cfg,
            processed_indexed,
            total_records,
            len(batches),
            batch_no,
            worker_count,
            status="running",
        )
        print(f"[PEG-Bench] batch {batch_no}/{len(batches)} finished", flush=True)

    processed_records = [item for _, item in sorted(processed_indexed, key=lambda pair: pair[0])]
    time_summaries: list[dict[str, object]] = []
    complete_count = 0
    incomplete_count = 0
    for item in processed_records:
        if item.get("status") != "processed":
            incomplete_count += 1
            continue
        complete_count += 1
        summaries = item["summaries"]
        time_summary = {
            "venue": item["venue"],
            "asset": item["asset"],
            "date": item["date"],
            "split": item["split"],
            **summaries.get("time", next(iter(summaries.values()))),
        }
        time_summaries.append(time_summary)

    regime_tags = _regime_tag_panel(time_summaries)
    for item in processed_records:
        if item.get("status") != "processed":
            continue
        key = (str(item["venue"]), str(item["asset"]), str(item["date"]))
        item["regime_tags"] = regime_tags.get(key, ["neutral"])
        if "time" in item["summaries"]:
            item["summaries"]["time"]["regime_tags"] = item["regime_tags"]
    stress_days = _stress_day_selection(time_summaries, cfg.max_stress_days, cfg.min_stress_days)

    protocol = {
        "name": "PEG-Bench",
        "date_window": {
            "start": protocol_dates[0] if protocol_dates else cfg.start_date,
            "end": protocol_dates[-1] if protocol_dates else cfg.end_date,
        },
        "split_days": split_counts,
        "split_strategy": split_strategy,
        "alignment_mode": split_meta["alignment_mode"],
        "context_len": cfg.context_len,
        "horizons": list(cfg.horizons),
        "depth": cfg.depth,
        "bar_modes": list(cfg.bar_modes),
        "cost_ladder_bps": [0, 1, 3, 5, 10],
        "realism_ladder": ["R0", "R1", "R2", "R3", "R4"],
        "benchmark_units": [
            "raw_l2_events",
            "reconstructed_top10_lob",
            "time_bars_1s",
            "volume_bars",
            "execution_logs",
        ],
    }
    coverage = {
        "expected_asset_days": len(protocol_dates) * len(universe),
        "discovered_asset_days": len(discovered),
        "complete_asset_days": complete_count,
        "incomplete_asset_days": incomplete_count,
    }
    split_index = {
        day: {"split": split, "index": index}
        for index, (day, split) in enumerate(sorted(split_lookup.items(), key=lambda item: item[0]))
    }
    bundle = {
        "protocol": protocol,
        "coverage": coverage,
        "split_index": split_index,
        "split_metadata": split_meta,
        "stress_days": stress_days,
        "records": processed_records,
    }
    (output_root / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (output_root / "manifest.json").write_text(json.dumps(bundle, indent=2))
    (output_root / "split_index.json").write_text(json.dumps(split_index, indent=2))
    (output_root / "stress_days.json").write_text(json.dumps(stress_days, indent=2))
    _write_checkpoint(
        output_root,
        cfg,
        processed_indexed,
        total_records,
        len(batches),
        len(batches),
        worker_count,
        status="finished",
    )
    print(
        f"[PEG-Bench] finished with {complete_count} complete and {incomplete_count} incomplete asset-days",
        flush=True,
    )
    return bundle
