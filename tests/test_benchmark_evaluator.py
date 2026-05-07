from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.eval.benchmark_evaluator import (
    BenchmarkEvaluatorConfig,
    BenchmarkMethodSpec,
    evaluate_benchmark_methods,
    load_method_specs_from_json,
)


def _write_report(path: Path, rank_ic: float, sharpe: float, date: str) -> None:
    payload = {
        "date": date,
        "report_card": {
            "prediction": {
                "rank_ic": rank_ic,
                "directional_accuracy": 0.55 + rank_ic,
                "mse": 0.01,
                "mae": 0.02,
            },
            "execution": {
                "anchor_cost_bps": 3.0,
                "post_cost_sharpe": sharpe,
                "max_drawdown": 0.12,
                "turnover": 0.4,
                "calmar": 1.5,
                "net_return": 0.25,
                "cvar95": 0.08,
                "average_holding_period": 3.0,
                "cost_ladder": {
                    "0.0": {"sharpe": sharpe + 0.2, "max_drawdown": 0.10, "turnover": 0.4, "calmar": 1.8, "net_pnl": 0.30},
                    "3.0": {"sharpe": sharpe, "max_drawdown": 0.12, "turnover": 0.4, "calmar": 1.5, "net_pnl": 0.25},
                },
            },
            "robustness": {
                "worst_regime_sharpe": sharpe - 0.4,
                "stress_day_survival": 1.0,
                "regime_utilities": [sharpe - 0.4, sharpe - 0.2, sharpe],
            },
        },
    }
    path.write_text(json.dumps(payload, indent=2))


class BenchmarkEvaluatorTest(unittest.TestCase):
    def test_evaluator_aggregates_method_panel(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline_day1 = root / "baseline_2025-12-12.json"
            baseline_day2 = root / "baseline_2025-12-13.json"
            aligned_day1 = root / "aligned_2025-12-12.json"
            aligned_day2 = root / "aligned_2025-12-13.json"
            _write_report(baseline_day1, rank_ic=0.05, sharpe=0.8, date="2025-12-12")
            _write_report(baseline_day2, rank_ic=0.06, sharpe=0.9, date="2025-12-13")
            _write_report(aligned_day1, rank_ic=0.04, sharpe=1.1, date="2025-12-12")
            _write_report(aligned_day2, rank_ic=0.05, sharpe=1.2, date="2025-12-13")

            result = evaluate_benchmark_methods(
                [
                    BenchmarkMethodSpec(name="baseline", reports=(str(baseline_day1), str(baseline_day2))),
                    BenchmarkMethodSpec(name="aligned", reports=(str(aligned_day1), str(aligned_day2))),
                ],
                split_lookup={"2025-12-12": "test", "2025-12-13": "test"},
                config=BenchmarkEvaluatorConfig(
                    output_path=str(root / "benchmark_report.json"),
                    markdown_path=str(root / "benchmark_report.md"),
                    evaluation_split="test",
                    bootstrap_samples=50,
                ),
            )

            self.assertEqual(result["gap"]["status"], "computed")
            self.assertEqual(result["leaderboard"][0]["method"], "aligned")
            self.assertIn("baseline", result["methods"])
            self.assertIn("aligned", result["methods"])
            self.assertTrue((root / "benchmark_report.json").exists())
            self.assertTrue((root / "benchmark_report.md").exists())

    def test_evaluator_tracks_missing_dates_and_loads_split_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline_day1 = root / "baseline_2025-12-12.json"
            baseline_day2 = root / "baseline_2025-12-13.json"
            aligned_day1 = root / "aligned_2025-12-12.json"
            _write_report(baseline_day1, rank_ic=0.05, sharpe=0.8, date="2025-12-12")
            _write_report(baseline_day2, rank_ic=0.06, sharpe=0.9, date="2025-12-13")
            _write_report(aligned_day1, rank_ic=0.04, sharpe=1.1, date="2025-12-12")

            split_plan = root / "split_plan.csv"
            split_plan.write_text(
                "\n".join(
                    [
                        "exchange,symbol,train_start,train_end,val_start,val_end,test_start,test_end,train_days,val_days,test_days,notes",
                        "binance,BTCUSDT,2025-12-10,2025-12-11,2025-12-12,2025-12-12,2025-12-13,2025-12-13,2,1,1,chronological_split",
                    ]
                )
            )

            spec_path = root / "spec.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "split_plan_path": str(split_plan),
                        "methods": [
                            {"name": "baseline", "reports": [str(baseline_day1), str(baseline_day2)]},
                            {"name": "aligned", "reports": [str(aligned_day1)]},
                        ],
                    },
                    indent=2,
                )
            )

            method_specs, split_lookup, _ = load_method_specs_from_json(spec_path)
            self.assertEqual(split_lookup["2025-12-12"], "val")
            self.assertEqual(split_lookup["2025-12-13"], "test")

            result = evaluate_benchmark_methods(
                method_specs,
                split_lookup=split_lookup,
                config=BenchmarkEvaluatorConfig(
                    output_path=str(root / "benchmark_report_test.json"),
                    markdown_path=str(root / "benchmark_report_test.md"),
                    evaluation_split="test",
                    bootstrap_samples=20,
                ),
            )

            baseline_audit = result["methods"]["baseline"]["auditability"]
            aligned_audit = result["methods"]["aligned"]["auditability"]
            self.assertEqual(baseline_audit["expected_num_days"], 1)
            self.assertEqual(baseline_audit["observed_num_days"], 1)
            self.assertEqual(baseline_audit["missing_dates"], [])
            self.assertEqual(aligned_audit["expected_num_days"], 1)
            self.assertEqual(aligned_audit["observed_num_days"], 0)
            self.assertEqual(aligned_audit["missing_dates"], ["2025-12-13"])
            self.assertEqual(result["auditability"]["expected_dates_in_split"], ["2025-12-13"])


if __name__ == "__main__":
    unittest.main()
