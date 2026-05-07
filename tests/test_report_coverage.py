from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.eval.benchmark_evaluator import BenchmarkMethodSpec
from src.eval.report_coverage import summarize_report_coverage


def _write_report(path: Path, date: str) -> None:
    path.write_text(json.dumps({"date": date, "report_card": {"prediction": {}, "execution": {}, "robustness": {}}}))


class ReportCoverageTest(unittest.TestCase):
    def test_report_coverage_detects_missing_and_extra_days(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline_day1 = root / "baseline_2025-10-24.json"
            baseline_day2 = root / "baseline_2025-10-25.json"
            aligned_day1 = root / "aligned_2025-10-28.json"
            _write_report(baseline_day1, "2025-10-24")
            _write_report(baseline_day2, "2025-10-25")
            _write_report(aligned_day1, "2025-10-28")

            summary = summarize_report_coverage(
                method_specs=[
                    BenchmarkMethodSpec(name="baseline", reports=(str(baseline_day1), str(baseline_day2))),
                    BenchmarkMethodSpec(name="aligned", reports=(str(aligned_day1),)),
                ],
                split_lookup={
                    "2025-10-24": "test",
                    "2025-10-25": "test",
                    "2025-10-26": "test",
                    "2025-10-27": "test",
                },
                evaluation_split="test",
            )

            baseline = summary["methods"]["baseline"]
            aligned = summary["methods"]["aligned"]
            self.assertEqual(baseline["missing_dates"], ["2025-10-26", "2025-10-27"])
            self.assertEqual(aligned["missing_dates"], ["2025-10-24", "2025-10-25", "2025-10-26", "2025-10-27"])
            self.assertEqual(aligned["extra_dates"], ["2025-10-28"])
            self.assertEqual(len(summary["rows"]), 9)


if __name__ == "__main__":
    unittest.main()
