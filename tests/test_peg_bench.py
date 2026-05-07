from __future__ import annotations

import unittest

import numpy as np

from src.eval.peg_bench import run_realism_ladder


class PEGBenchLadderTest(unittest.TestCase):
    def test_realism_ladder_outputs_expected_sections(self) -> None:
        predictions = np.linspace(-0.02, 0.03, 24).astype(np.float32)
        targets = np.linspace(-0.01, 0.02, 24).astype(np.float32)
        state_tokens = np.zeros((24, 8), dtype=np.int64)
        lob_states = np.zeros((24, 4, 10, 2), dtype=np.float32)
        lob_states[:, 1, :, :] = np.log(2.0)
        market_returns = np.sin(np.linspace(0, 3.14, 24)).astype(np.float32) * 0.001
        timestamps = np.arange(24, dtype=np.int64)
        actions = np.tanh(predictions).astype(np.float32)

        summary = run_realism_ladder(
            predictions=predictions,
            targets=targets,
            state_tokens=state_tokens,
            lob_states=lob_states,
            market_returns=market_returns,
            timestamps=timestamps,
            actions=actions,
        )

        self.assertIn("realism_ladder", summary)
        self.assertIn("report_card", summary)
        self.assertIn("cost_scan", summary)
        self.assertIn("R0", summary["realism_ladder"])
        self.assertIn("R4", summary["realism_ladder"])
        self.assertIn("3.0", summary["cost_scan"])
        self.assertIn("prediction", summary["report_card"])
        self.assertIn("execution", summary["report_card"])


if __name__ == "__main__":
    unittest.main()
