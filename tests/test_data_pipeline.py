from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data.pipeline import DataPipelineConfig, run_data_pipeline
from src.data.synthetic import SyntheticMarketConfig, generate_synthetic_l2_and_trades


class DataPipelineTest(unittest.TestCase):
    def test_synthetic_pipeline_builds_bars(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            l2_path, trades_path = generate_synthetic_l2_and_trades(root, SyntheticMarketConfig(num_steps=120))
            output = root / "bars.npz"
            result = run_data_pipeline(
                DataPipelineConfig(
                    l2_path=str(l2_path),
                    trades_path=str(trades_path),
                    output_path=str(output),
                    volume_threshold=2_000.0,
                    max_bars=32,
                )
            )
            payload = np.load(output)
            self.assertGreater(result["num_bars"], 0)
            self.assertEqual(payload["lob"].shape[1:], (4, 10, 2))


if __name__ == "__main__":
    unittest.main()
