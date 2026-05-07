#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from copy import deepcopy

import yaml

from src.data.peg_bench_dataset import (
    DEFAULT_UNIVERSE,
    PEGBenchPreprocessConfig,
    _build_split_lookup_from_dates,
    _record_id,
    _regime_tag_panel,
    _resolve_split_counts,
    _select_protocol_dates,
    _stress_day_selection,
    discover_tardis_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize a sharded PEG-Bench remaining bundle.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    payload = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    payload.pop("pipeline_type", None)
    cfg = PEGBenchPreprocessConfig(**payload)
    output_root = Path(cfg.output_root)
    shard_root = output_root / "_shards"

    discovered = discover_tardis_manifest(cfg.raw_root, cfg.start_date, cfg.end_date, universe=DEFAULT_UNIVERSE)
    protocol_dates, split_meta = _select_protocol_dates(discovered, DEFAULT_UNIVERSE)
    split_counts, split_strategy = _resolve_split_counts(cfg, len(protocol_dates))
    split_lookup = _build_split_lookup_from_dates(protocol_dates, split_counts)
    selected_days = set(protocol_dates)
    discovered = [record for record in discovered if record.date in selected_days]
    indexed_records = list(enumerate(discovered, start=1))
    expected = {_record_id(record): (index, record) for index, record in indexed_records}

    results: dict[str, dict[str, object]] = {}
    for path in sorted(shard_root.glob('results_shard_*.jsonl')):
        with path.open('r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                results[str(row['record_id'])] = row

    missing = sorted(set(expected) - set(results))
    failed = sorted(rid for rid, row in results.items() if row.get('status') != 'processed')
    if missing or failed:
        summary = {
            'status': 'incomplete',
            'missing_count': len(missing),
            'failed_count': len(failed),
            'missing': missing[:50],
            'failed': failed[:50],
        }
        (output_root / 'finalize_status.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary, indent=2))
        raise SystemExit(2)

    processed_records = []
    summaries = []
    for rid, (index, record) in sorted(expected.items(), key=lambda item: item[1][0]):
        row = deepcopy(results[rid])
        row.pop('record_id', None)
        row.pop('index', None)
        row['split'] = split_lookup.get(record.date, row.get('split', 'train'))
        processed_records.append(row)
        summary_mode = 'time' if 'time' in row.get('summaries', {}) else next(iter(row['summaries']))
        summaries.append({
            'venue': row['venue'],
            'asset': row['asset'],
            'date': row['date'],
            'split': row['split'],
            **row['summaries'][summary_mode],
        })

    regime_tags = _regime_tag_panel(summaries)
    for row in processed_records:
        key = (str(row['venue']), str(row['asset']), str(row['date']))
        tags = regime_tags.get(key, ['neutral'])
        row['regime_tags'] = tags
        for mode in row.get('summaries', {}):
            row['summaries'][mode]['regime_tags'] = tags
    updated_summaries = []
    for row in processed_records:
        summary_mode = 'time' if 'time' in row.get('summaries', {}) else next(iter(row['summaries']))
        updated_summaries.append({
            'venue': row['venue'],
            'asset': row['asset'],
            'date': row['date'],
            'split': row['split'],
            **row['summaries'][summary_mode],
        })
    stress_days = _stress_day_selection(updated_summaries, cfg.max_stress_days, cfg.min_stress_days)

    protocol = {
        'name': 'PEG-Bench',
        'date_window': {
            'start': protocol_dates[0] if protocol_dates else cfg.start_date,
            'end': protocol_dates[-1] if protocol_dates else cfg.end_date,
        },
        'split_days': split_counts,
        'split_strategy': split_strategy,
        'alignment_mode': split_meta['alignment_mode'],
        'context_len': cfg.context_len,
        'horizons': list(cfg.horizons),
        'depth': cfg.depth,
        'bar_modes': list(cfg.bar_modes),
        'cost_ladder_bps': [0, 1, 3, 5, 10],
        'realism_ladder': ['R0', 'R1', 'R2', 'R3', 'R4'],
        'benchmark_units': [
            'raw_l2_events',
            'reconstructed_top10_lob',
            'time_bars_1s',
            'volume_bars',
            'execution_logs',
        ],
    }
    coverage = {
        'expected_asset_days': len(protocol_dates) * len(DEFAULT_UNIVERSE),
        'discovered_asset_days': len(discovered),
        'complete_asset_days': len(processed_records),
        'incomplete_asset_days': 0,
    }
    split_index = {
        day: {'split': split, 'index': index}
        for index, (day, split) in enumerate(sorted(split_lookup.items(), key=lambda item: item[0]))
    }
    split_metadata = {
        **split_meta,
        'builder': 'sharded_runner',
    }
    bundle = {
        'protocol': protocol,
        'coverage': coverage,
        'split_index': split_index,
        'split_metadata': split_metadata,
        'stress_days': stress_days,
        'records': processed_records,
    }
    (output_root / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    (output_root / 'manifest.json').write_text(json.dumps(bundle, indent=2), encoding='utf-8')
    (output_root / 'split_index.json').write_text(json.dumps(split_index, indent=2), encoding='utf-8')
    (output_root / 'stress_days.json').write_text(json.dumps(stress_days, indent=2), encoding='utf-8')
    (output_root / 'finalize_status.json').write_text(json.dumps({'status': 'finished', 'records': len(processed_records)}, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'finished', 'records': len(processed_records)}, indent=2))


if __name__ == '__main__':
    main()
