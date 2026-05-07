from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from peg_bench.data.window_index import build_panel_window_index
from src.eval.benchmark_baselines import build_last_lob_features, build_token_frequency_features


class BenchmarkBaselineFeatureTests(unittest.TestCase):
    def test_feature_builders_follow_window_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            panel_path = root / "panel.npz"
            index_path = root / "index.csv"
            encoded_path = root / "encoded.npz"

            lob = np.arange(12 * 4 * 2 * 2, dtype=np.float32).reshape(12, 4, 2, 2)
            timestamps = np.arange(12, dtype=np.int64)
            returns = np.linspace(-0.1, 0.1, 12, dtype=np.float32)
            segment_lengths = np.asarray([6, 6], dtype=np.int64)
            segment_offsets = np.asarray([0, 6], dtype=np.int64)
            segment_keys = np.asarray(["binance:BTCUSDT:2025-10-01", "okx:ETHUSDT:2025-10-02"])
            segment_venues = np.asarray(["binance", "okx"])
            segment_assets = np.asarray(["BTCUSDT", "ETHUSDT"])
            segment_dates = np.asarray(["2025-10-01", "2025-10-02"])
            segment_splits = np.asarray(["train", "train"])
            np.savez_compressed(
                panel_path,
                lob=lob,
                timestamps=timestamps,
                return_20=returns,
                segment_lengths=segment_lengths,
                segment_offsets=segment_offsets,
                segment_keys=segment_keys,
                segment_venues=segment_venues,
                segment_assets=segment_assets,
                segment_dates=segment_dates,
                segment_splits=segment_splits,
            )
            build_panel_window_index(panel_path, index_path, seq_len=4, stride=2, target_key="return_20")
            np.savez_compressed(encoded_path, codes=np.asarray([0, 1, 1, 2, 2, 3, 3, 2, 1, 0, 0, 1], dtype=np.int64))

            token_view = build_token_frequency_features(
                panel_path=panel_path,
                index_path=index_path,
                encoded_path=encoded_path,
                vocab_size=4,
                target_key="return_20",
            )
            lob_view = build_last_lob_features(
                panel_path=panel_path,
                index_path=index_path,
                target_key="return_20",
            )

            with index_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(token_view["features"].shape[0], len(rows))
            self.assertEqual(token_view["features"].shape[1], 4)
            self.assertEqual(lob_view["features"].shape[0], len(rows))
            self.assertEqual(lob_view["features"].shape[1], 16)

            first_counts = token_view["features"][0]
            np.testing.assert_allclose(first_counts, np.asarray([0.25, 0.5, 0.25, 0.0], dtype=np.float32))
            last_index = int(rows[0]["window_end_index_exclusive"]) - 1
            np.testing.assert_allclose(lob_view["features"][0], lob[last_index].reshape(-1))


if __name__ == "__main__":
    unittest.main()
