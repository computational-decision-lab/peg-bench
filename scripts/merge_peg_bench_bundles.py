#!/usr/bin/env python3
"""Merge legacy and remaining PEG-Bench bundles into one final 90d bundle."""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timedelta
from pathlib import Path
import argparse
import json
import os
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.peg_bench_dataset import DEFAULT_UNIVERSE, _regime_tag_panel, _stress_day_selection


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _daterange(start: date, end: date) -> list[str]:
    days: list[str] = []
    cur = start
    while cur <= end:
        days.append(cur.isoformat())
        cur += timedelta(days=1)
    return days


def _split_lookup(days: list[str], train_days: int, val_days: int, test_days: int) -> dict[str, str]:
    expected = train_days + val_days + test_days
    if len(days) != expected:
        raise ValueError(f"Date count {len(days)} != split total {expected}")
    lookup: dict[str, str] = {}
    for idx, day in enumerate(days):
        if idx < train_days:
            lookup[day] = "train"
        elif idx < train_days + val_days:
            lookup[day] = "val"
        else:
            lookup[day] = "test"
    return lookup


def _load_bundle(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _record_key(record: dict[str, object]) -> tuple[str, str, str]:
    return (str(record["venue"]), str(record["asset"]), str(record["date"]))


def _expected_keys(start_date: str, end_date: str) -> set[tuple[str, str, str]]:
    days = _daterange(_parse_date(start_date), _parse_date(end_date))
    return {(spec.venue, spec.asset, day) for spec in DEFAULT_UNIVERSE for day in days}


def _collect_mode_records(bundle: dict[str, object], mode: str) -> dict[tuple[str, str, str], dict[str, object]]:
    out: dict[tuple[str, str, str], dict[str, object]] = {}
    for record in bundle.get("records", []):
        if record.get("status") != "processed":
            continue
        processed = record.get("processed_paths", {})
        if mode not in processed:
            continue
        out[_record_key(record)] = record
    return out


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and Path(os.readlink(dst)).resolve() == src.resolve():
            return
        raise FileExistsError(f"Destination already exists: {dst}")
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge legacy and remaining PEG-Bench bundles.")
    parser.add_argument("--legacy-bundle", required=True)
    parser.add_argument("--remaining-bundle", required=True)
    parser.add_argument("--output-bundle", required=True)
    parser.add_argument("--mode", required=True, choices=["time", "volume"])
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--train-days", type=int, default=60)
    parser.add_argument("--val-days", type=int, default=15)
    parser.add_argument("--test-days", type=int, default=15)
    parser.add_argument("--link-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--max-stress-days", type=int, default=12)
    parser.add_argument("--min-stress-days", type=int, default=8)
    args = parser.parse_args()

    legacy_root = Path(args.legacy_bundle)
    remaining_root = Path(args.remaining_bundle)
    output_root = Path(args.output_bundle)
    output_root.mkdir(parents=True, exist_ok=True)

    legacy_bundle = _load_bundle(legacy_root / "manifest.json")
    remaining_bundle = _load_bundle(remaining_root / "manifest.json")

    legacy = _collect_mode_records(legacy_bundle, args.mode)
    remaining = _collect_mode_records(remaining_bundle, args.mode)

    overlap = sorted(set(legacy) & set(remaining))
    if overlap:
        raise ValueError(f"Legacy and remaining overlap on {len(overlap)} keys, first={overlap[:5]}")

    merged_source: dict[tuple[str, str, str], tuple[str, dict[str, object], Path]] = {}
    for key, record in legacy.items():
        merged_source[key] = ("legacy_30d", record, legacy_root)
    for key, record in remaining.items():
        merged_source[key] = ("remaining_preprocess", record, remaining_root)

    expected_keys = _expected_keys(args.start_date, args.end_date)
    missing = sorted(expected_keys - set(merged_source))
    extra = sorted(set(merged_source) - expected_keys)
    if missing:
        raise ValueError(f"Missing {len(missing)} asset-days, first={missing[:10]}")
    if extra:
        raise ValueError(f"Found {len(extra)} unexpected asset-days, first={extra[:10]}")

    all_days = _daterange(_parse_date(args.start_date), _parse_date(args.end_date))
    split_lookup = _split_lookup(all_days, args.train_days, args.val_days, args.test_days)

    merged_records: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    for venue, asset, day in sorted(merged_source, key=lambda item: (item[2], item[0], item[1])):
        source_name, src_record, src_bundle_root = merged_source[(venue, asset, day)]
        record = deepcopy(src_record)
        src_processed = Path(str(record["processed_paths"][args.mode]))
        dst_processed = output_root / "processed" / venue / asset / day / src_processed.name
        _link_or_copy(src_processed, dst_processed, args.link_mode)
        record["processed_paths"] = {args.mode: str(dst_processed.resolve())}
        record["split"] = split_lookup[day]
        record["source"] = source_name
        record["source_bundle"] = str(src_bundle_root.resolve())
        record["source_processed_path"] = str(src_processed.resolve())
        merged_records.append(record)
        mode_summary = dict(record["summaries"][args.mode])
        summaries.append({
            "venue": venue,
            "asset": asset,
            "date": day,
            "split": record["split"],
            **mode_summary,
        })

    regime_tags = _regime_tag_panel(summaries)
    for record in merged_records:
        key = (str(record["venue"]), str(record["asset"]), str(record["date"]))
        tags = regime_tags.get(key, ["neutral"])
        record["regime_tags"] = tags
        if args.mode in record.get("summaries", {}):
            record["summaries"][args.mode]["regime_tags"] = tags
    updated_summaries = []
    for record in merged_records:
        updated_summaries.append({
            "venue": record["venue"],
            "asset": record["asset"],
            "date": record["date"],
            "split": record["split"],
            **record["summaries"][args.mode],
        })
    stress_days = _stress_day_selection(updated_summaries, args.max_stress_days, args.min_stress_days)

    base_protocol = deepcopy(legacy_bundle.get("protocol", {}))
    protocol = {
        **base_protocol,
        "name": "PEG-Bench",
        "date_window": {"start": args.start_date, "end": args.end_date},
        "split_days": {"train": args.train_days, "val": args.val_days, "test": args.test_days},
        "split_strategy": "fixed_days",
        "alignment_mode": "incremental_merge",
        "bar_modes": [args.mode],
    }
    coverage = {
        "expected_asset_days": len(expected_keys),
        "discovered_asset_days": len(merged_records),
        "complete_asset_days": len(merged_records),
        "incomplete_asset_days": 0,
    }
    split_index = {day: {"split": split_lookup[day], "index": idx} for idx, day in enumerate(all_days)}
    split_metadata = {
        "alignment_mode": "incremental_merge",
        "legacy_bundle": str(legacy_root.resolve()),
        "remaining_bundle": str(remaining_root.resolve()),
        "provenance_counts": {
            "legacy_30d": sum(1 for record in merged_records if record["source"] == "legacy_30d"),
            "remaining_preprocess": sum(1 for record in merged_records if record["source"] == "remaining_preprocess"),
        },
    }
    bundle = {
        "protocol": protocol,
        "coverage": coverage,
        "split_index": split_index,
        "split_metadata": split_metadata,
        "stress_days": stress_days,
        "records": merged_records,
    }
    merge_summary = {
        "mode": args.mode,
        "output_bundle": str(output_root.resolve()),
        "manifest_path": str((output_root / 'manifest.json').resolve()),
        "total_dates": len(all_days),
        "total_asset_days": len(merged_records),
        "duplicate_count": len(overlap),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "provenance_counts": split_metadata["provenance_counts"],
        "link_mode": args.link_mode,
    }

    (output_root / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (output_root / "manifest.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    (output_root / "split_index.json").write_text(json.dumps(split_index, indent=2), encoding="utf-8")
    (output_root / "stress_days.json").write_text(json.dumps(stress_days, indent=2), encoding="utf-8")
    (output_root / "merge_summary.json").write_text(json.dumps(merge_summary, indent=2), encoding="utf-8")
    print(json.dumps(merge_summary, indent=2))


if __name__ == "__main__":
    main()
