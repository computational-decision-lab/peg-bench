from __future__ import annotations

import unittest

from src.eval.gap_metrics import GapInput, compute_gap_report


class GapMetricsTest(unittest.TestCase):
    def test_gap_report_with_multi_method_panel(self) -> None:
        panel = {
            "kairos": GapInput(offline_score=0.32, full_realism_utility=-98.6, regime_utilities=[-40.0, -98.6, -55.0]),
            "lightgbm": GapInput(offline_score=0.26, full_realism_utility=-112.3, regime_utilities=[-60.0, -112.3, -70.0]),
            "policy": GapInput(offline_score=0.10, full_realism_utility=-20.0, regime_utilities=[-10.0, -20.0, -12.0]),
        }
        report = compute_gap_report(panel, k=1)
        self.assertEqual(report["status"], "computed")
        self.assertIn("metric_gap", report)
        self.assertIn("ranking_gap", report)
        self.assertIn("reversal_rate_at_k", report)
        self.assertIn("fragility_gap", report)

    def test_gap_report_marks_single_method_as_protocol_target(self) -> None:
        panel = {"kairos": GapInput(offline_score=0.32, full_realism_utility=-98.6, regime_utilities=[-98.6])}
        report = compute_gap_report(panel, k=1)
        self.assertEqual(report["status"], "protocol_target")
        self.assertIsNone(report["metric_gap"])


if __name__ == "__main__":
    unittest.main()
