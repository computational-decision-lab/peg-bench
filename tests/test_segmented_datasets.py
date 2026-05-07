from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data.datasets import DatasetSpec, LOBTokenSequenceDataset


class SegmentedDatasetTest(unittest.TestCase):
    def test_token_dataset_does_not_cross_segment_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "tokens.npz"
            data_path = root / "panel.npz"

            codes = np.arange(10, dtype=np.int64)
            targets = np.linspace(0.0, 0.9, 10, dtype=np.float32)
            segment_lengths = np.asarray([5, 5], dtype=np.int64)

            np.savez_compressed(token_path, codes=codes, latents=np.zeros((10, 2), dtype=np.float32))
            np.savez_compressed(data_path, return_20=targets, segment_lengths=segment_lengths)

            dataset = LOBTokenSequenceDataset(token_path, data_path, DatasetSpec(seq_len=3, stride=1, target_key="return_20"))

            self.assertEqual(len(dataset), 4)
            first_tokens, first_target = dataset[0]
            third_tokens, third_target = dataset[2]

            np.testing.assert_array_equal(first_tokens.numpy(), np.asarray([0, 1, 2], dtype=np.int64))
            np.testing.assert_array_equal(third_tokens.numpy(), np.asarray([5, 6, 7], dtype=np.int64))
            self.assertAlmostEqual(float(first_target.item()), 0.3, places=6)
            self.assertAlmostEqual(float(third_target.item()), 0.8, places=6)


if __name__ == "__main__":
    unittest.main()
