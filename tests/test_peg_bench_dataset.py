from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data.peg_bench_dataset import PEGBenchPreprocessConfig, build_peg_bench_bundle
from src.data.synthetic import SyntheticMarketConfig, generate_synthetic_l2_and_trades


class PEGBenchDatasetTest(unittest.TestCase):
    def test_build_bundle_from_synthetic_raw_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            l2_path, trades_path = generate_synthetic_l2_and_trades(root / "seed", SyntheticMarketConfig(num_steps=300))
            raw_root = root / "raw"
            raw_root.mkdir(parents=True, exist_ok=True)
            l2_target = raw_root / "binance-futures_BTCUSDT_incremental_book_L2_2025-09-28.csv"
            trades_target = raw_root / "binance-futures_BTCUSDT_trades_2025-09-28.csv"
            shutil.copy2(l2_path, l2_target)
            shutil.copy2(trades_path, trades_target)

            bundle = build_peg_bench_bundle(
                PEGBenchPreprocessConfig(
                    raw_root=str(raw_root),
                    output_root=str(root / "bundle"),
                    start_date="2025-09-28",
                    end_date="2025-09-28",
                    train_days=1,
                    val_days=0,
                    test_days=0,
                    depth=10,
                    volume_threshold=500.0,
                )
            )

            self.assertEqual(bundle["coverage"]["complete_asset_days"], 1)
            self.assertEqual(len(bundle["records"]), 1)
            record = bundle["records"][0]
            self.assertEqual(record["split"], "train")
            self.assertEqual(record["status"], "processed")
            time_path = Path(record["processed_paths"]["time"])
            payload = np.load(time_path, allow_pickle=True)
            self.assertIn("benchmark_features", payload)
            self.assertIn("benchmark_feature_names", payload)
            self.assertGreater(payload["lob"].shape[0], 0)


if __name__ == "__main__":
    unittest.main()
