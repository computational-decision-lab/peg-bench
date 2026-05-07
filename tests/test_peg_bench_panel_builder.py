from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from peg_bench.data.panel_builder import build_split_panel


def _write_day(path: Path, base: int) -> None:
    length = 6
    np.savez_compressed(
        path,
        lob=np.full((length, 4, 10, 2), float(base), dtype=np.float32),
        timestamps=np.arange(base, base + length, dtype=np.int64),
        mid_prices=np.linspace(100.0, 101.0, length, dtype=np.float64),
        return_20=np.linspace(0.0, 0.5, length, dtype=np.float32),
        benchmark_feature_names=np.asarray(["spread_bps"]),
    )


class PEGBenchPanelBuilderTest(unittest.TestCase):
    def test_build_split_panel_uses_split_plan_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            processed = root / "processed"
            day1 = processed / "binance" / "BTCUSDT" / "2025-09-28" / "time_bars.npz"
            day2 = processed / "binance" / "BTCUSDT" / "2025-09-29" / "time_bars.npz"
            day3 = processed / "binance" / "BTCUSDT" / "2025-09-30" / "time_bars.npz"
            for path in (day1, day2, day3):
                path.parent.mkdir(parents=True, exist_ok=True)
            _write_day(day1, 0)
            _write_day(day2, 10)
            _write_day(day3, 20)

            bundle = {
                "records": [
                    {
                        "venue": "binance",
                        "asset": "BTCUSDT",
                        "date": "2025-09-28",
                        "split": "train",
                        "status": "processed",
                        "processed_paths": {"time": str(day1)},
                    },
                    {
                        "venue": "binance",
                        "asset": "BTCUSDT",
                        "date": "2025-09-29",
                        "split": "train",
                        "status": "processed",
                        "processed_paths": {"time": str(day2)},
                    },
                    {
                        "venue": "binance",
                        "asset": "BTCUSDT",
                        "date": "2025-09-30",
                        "split": "train",
                        "status": "processed",
                        "processed_paths": {"time": str(day3)},
                    },
                ]
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(bundle, indent=2))

            split_plan_path = root / "split_plan.csv"
            with split_plan_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "exchange",
                        "symbol",
                        "train_start",
                        "train_end",
                        "val_start",
                        "val_end",
                        "test_start",
                        "test_end",
                        "train_days",
                        "val_days",
                        "test_days",
                        "notes",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "exchange": "binance",
                        "symbol": "BTCUSDT",
                        "train_start": "2025-09-28",
                        "train_end": "2025-09-28",
                        "val_start": "2025-09-29",
                        "val_end": "2025-09-29",
                        "test_start": "2025-09-30",
                        "test_end": "2025-09-30",
                        "train_days": 1,
                        "val_days": 1,
                        "test_days": 1,
                        "notes": "chronological_split",
                    }
                )

            panel_path = root / "time_test.npz"
            summary = build_split_panel(
                manifest_path=manifest_path,
                split="test",
                mode="time",
                output_path=panel_path,
                split_plan_path=split_plan_path,
            )
            payload = np.load(panel_path, allow_pickle=True)

            self.assertEqual(summary["num_records"], 1)
            self.assertEqual(summary["split_source"], "split_plan_override")
            self.assertEqual(payload["lob"].shape[0], 6)
            self.assertEqual(payload["segment_keys"].tolist(), ["binance:BTCUSDT:2025-09-30"])
            self.assertEqual(payload["segment_dates"].tolist(), ["2025-09-30"])


if __name__ == "__main__":
    unittest.main()
