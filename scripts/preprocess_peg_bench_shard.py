#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import traceback
from datetime import UTC, datetime

import yaml

from src.data.peg_bench_dataset import (
    DEFAULT_UNIVERSE,
    PEGBenchPreprocessConfig,
    _build_split_lookup_from_dates,
    _normalize_selection_mode,
    _load_dataset_summary,
    _process_asset_day,
    _record_id,
    _resolve_split_counts,
    _select_processing_records,
    _select_protocol_dates,
    discover_tardis_manifest,
)


TRANSIENT_READ_ERROR_MARKERS = (
    "Integer column has NA values",
    "EOFError",
    "Compressed file ended before the end-of-stream marker was reached",
    "ZstdError",
    "Error tokenizing data",
)
TRANSIENT_FILE_GRACE_SECONDS = 3_600


def _existing_processed(record, split: str, output_root: Path, bar_modes: tuple[str, ...]) -> dict[str, object] | None:
    record_root = output_root / "processed" / record.venue / record.asset / record.date
    processed_paths: dict[str, str] = {}
    summaries: dict[str, dict[str, object]] = {}
    for mode in bar_modes:
        dataset_path = record_root / f"{mode}_bars.npz"
        if not dataset_path.exists():
            return None
        processed_paths[mode] = str(dataset_path.resolve())
        summaries[mode] = _load_dataset_summary(dataset_path)
    return {
        "venue": record.venue,
        "asset": record.asset,
        "date": record.date,
        "l2_path": record.l2_path,
        "trades_path": record.trades_path,
        "split": split,
        "status": "processed",
        "processed_paths": processed_paths,
        "summaries": summaries,
    }


def _load_done(result_path: Path) -> dict[str, dict[str, object]]:
    done: dict[str, dict[str, object]] = {}
    if not result_path.exists():
        return done
    with result_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            done[str(item["record_id"])] = item
    return done


def _write_checkpoint(checkpoint_path: Path, shard_id: int, total: int, done_items: dict[str, dict[str, object]]) -> None:
    values = list(done_items.values())
    processed = sum(1 for item in values if item.get("status") == "processed")
    failed = sum(1 for item in values if item.get("status") == "failed")
    incomplete = sum(1 for item in values if item.get("status") == "incomplete")
    payload = {
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "shard_id": shard_id,
        "total_asset_days": total,
        "completed_asset_days": len(values),
        "processed_complete": processed,
        "processed_failed": failed,
        "processed_incomplete": incomplete,
        "remaining_asset_days": max(total - len(values), 0),
        "latest_completed": [item["record_id"] for item in values[-5:]],
    }
    checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _recently_modified_source(record, grace_seconds: int = TRANSIENT_FILE_GRACE_SECONDS) -> bool:
    now = time.time()
    for value in (record.l2_path, record.trades_path):
        if not value:
            continue
        path = Path(value)
        if path.exists() and now - path.stat().st_mtime <= grace_seconds:
            return True
    return False


def _is_transient_read_error(exc: Exception) -> bool:
    text = repr(exc)
    return any(marker in text for marker in TRANSIENT_READ_ERROR_MARKERS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one shard of PEG-Bench preprocessing sequentially.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()

    payload = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    payload.pop("pipeline_type", None)
    cfg = PEGBenchPreprocessConfig(**payload)
    output_root = Path(cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    discovered = discover_tardis_manifest(cfg.raw_root, cfg.start_date, cfg.end_date, universe=DEFAULT_UNIVERSE)
    discovered, protocol_dates, _ = _select_processing_records(cfg, discovered, DEFAULT_UNIVERSE)
    selection_mode = _normalize_selection_mode(cfg.selection_mode)
    if selection_mode == "discovered":
        split_lookup = {day: "train" for day in protocol_dates}
    else:
        split_counts, _ = _resolve_split_counts(cfg, len(protocol_dates))
        split_lookup = _build_split_lookup_from_dates(protocol_dates, split_counts)
    if cfg.max_days is not None:
        discovered = discovered[: cfg.max_days]

    indexed_records = list(enumerate(discovered, start=1))
    shard_records = [(idx, record) for idx, record in indexed_records if (idx - 1) % args.num_shards == args.shard_id]

    shard_root = output_root / "_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    result_path = shard_root / f"results_shard_{args.shard_id:02d}.jsonl"
    fail_path = shard_root / f"failures_shard_{args.shard_id:02d}.jsonl"
    checkpoint_path = shard_root / f"checkpoint_shard_{args.shard_id:02d}.json"

    done = _load_done(result_path)
    _write_checkpoint(checkpoint_path, args.shard_id, len(shard_records), done)
    print(
        f"[Shard {args.shard_id}/{args.num_shards}] discovered {len(shard_records)} assigned asset-days "
        f"within {cfg.start_date}..{cfg.end_date}",
        flush=True,
    )

    with result_path.open("a", encoding="utf-8") as result_handle, fail_path.open("a", encoding="utf-8") as fail_handle:
        for local_no, (index, record) in enumerate(shard_records, start=1):
            rid = _record_id(record)
            if rid in done and done[rid].get("status") == "processed":
                continue
            split = split_lookup.get(record.date, "train")
            print(
                f"[Shard {args.shard_id}] {local_no}/{len(shard_records)} start {rid}",
                flush=True,
            )
            try:
                item = _existing_processed(record, split, output_root, cfg.bar_modes)
                if item is None:
                    item = _process_asset_day(
                        record,
                        split,
                        cfg.output_root,
                        cfg.bar_modes,
                        cfg.depth,
                        cfg.interval_ms,
                        cfg.volume_threshold,
                        cfg.horizons,
                    )
            except Exception as exc:
                status = "failed"
                if _is_transient_read_error(exc) and _recently_modified_source(record):
                    status = "incomplete"
                item = {
                    "venue": record.venue,
                    "asset": record.asset,
                    "date": record.date,
                    "l2_path": record.l2_path,
                    "trades_path": record.trades_path,
                    "split": split,
                    "status": status,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                if status == "failed":
                    fail_handle.write(json.dumps({"record_id": rid, "index": index, **item}, ensure_ascii=True) + "\n")
                    fail_handle.flush()
                    print(f"[Shard {args.shard_id}] failed {rid}: {exc!r}", flush=True)
                else:
                    print(f"[Shard {args.shard_id}] incomplete {rid}: {exc!r}", flush=True)
            row = {"record_id": rid, "index": index, **item}
            done[rid] = row
            result_handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            result_handle.flush()
            _write_checkpoint(checkpoint_path, args.shard_id, len(shard_records), done)
            print(
                f"[Shard {args.shard_id}] done {rid} status={item.get('status')}",
                flush=True,
            )

    print(f"[Shard {args.shard_id}] finished", flush=True)


if __name__ == "__main__":
    main()
