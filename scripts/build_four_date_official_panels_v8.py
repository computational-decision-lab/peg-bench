from __future__ import annotations

from pathlib import Path
import json
import shutil

import numpy as np

ROOT = Path("/root/prediction-execution-gap")
DATA_ROOT = Path("/root/autodl-tmp/peg_bench_runs")
OUT_ROOT = ROOT / "outputs/final_motivating_slice_rerun_v8/official_four_date"

SPLIT_DATES = {
    "train": ["2025-10-12", "2025-10-17"],
    "val": ["2025-10-26"],
    "test": ["2025-10-27"],
}


def load_daily(mode: str, date: str) -> tuple[Path, dict[str, np.ndarray]]:
    path = DATA_ROOT / f"tardis_90d_{mode}_bundle/processed/binance/BTCUSDT/{date}/{mode}_bars.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    z = np.load(path, allow_pickle=True)
    return path, {k: z[k] for k in z.files}


def build_panel(mode: str, split: str) -> dict[str, object]:
    dates = SPLIT_DATES[split]
    out = OUT_ROOT / mode / "panels" / f"{mode}_{split}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    loaded = [load_daily(mode, d) for d in dates]
    keys = set(loaded[0][1].keys())
    for _, payload in loaded[1:]:
        keys &= set(payload.keys())
    preferred = [
        "lob",
        "timestamps",
        "mid_prices",
        "visible_volume",
        "benchmark_features",
        "benchmark_feature_names",
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
        "return_5",
        "return_20",
        "return_50",
    ]
    ordered = [k for k in preferred if k in keys] + sorted(keys - set(preferred))
    merged: dict[str, np.ndarray] = {}
    lengths: list[int] = []
    offsets: list[int] = []
    segment_keys: list[str] = []
    segment_venues: list[str] = []
    segment_assets: list[str] = []
    segment_dates: list[str] = []
    segment_splits: list[str] = []
    running = 0
    feature_names = None
    source_paths: list[str] = []
    for date, (source_path, payload) in zip(dates, loaded):
        source_paths.append(str(source_path))
        n = int(payload["lob"].shape[0])
        lengths.append(n)
        offsets.append(running)
        running += n
        segment_keys.append(f"binance:BTCUSDT:{date}")
        segment_venues.append("binance")
        segment_assets.append("BTCUSDT")
        segment_dates.append(date)
        segment_splits.append(split)
        for k in ordered:
            if k == "benchmark_feature_names":
                feature_names = payload[k]
                continue
            merged.setdefault(k, [])
            merged[k].append(payload[k])
    final: dict[str, np.ndarray] = {}
    for k, vals in merged.items():
        final[k] = np.concatenate(vals, axis=0)
    if feature_names is not None:
        final["benchmark_feature_names"] = feature_names
    final["segment_lengths"] = np.asarray(lengths, dtype=np.int64)
    final["segment_offsets"] = np.asarray(offsets, dtype=np.int64)
    final["segment_keys"] = np.asarray(segment_keys)
    final["segment_venues"] = np.asarray(segment_venues)
    final["segment_assets"] = np.asarray(segment_assets)
    final["segment_dates"] = np.asarray(segment_dates)
    final["segment_splits"] = np.asarray(segment_splits)
    np.savez_compressed(out, **final)
    summary = {
        "mode": mode,
        "split": split,
        "dates": dates,
        "source_paths": source_paths,
        "output_path": str(out),
        "num_bars": int(final["lob"].shape[0]),
        "segment_lengths": lengths,
        "segment_offsets": offsets,
        "segment_keys": segment_keys,
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    summaries = []
    for mode in ["volume", "time"]:
        for split in ["train", "val", "test"]:
            summaries.append(build_panel(mode, split))
    (OUT_ROOT / "panel_build_summary.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps({"out": str(OUT_ROOT), "panels": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()
