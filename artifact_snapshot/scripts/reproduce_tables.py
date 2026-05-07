#!/usr/bin/env python3
from pathlib import Path
import json

BUNDLE_ROOT = Path("/root/prediction-execution-gap/outputs/final_acceptance_v3")
TABLE_ROOT = Path("/root/prediction-execution-gap/paper_tables/generated_20260503")

TABLE_MAP = {
    "Table 2": {
        "inputs": [
            "/root/prediction-execution-gap/outputs/final_90d/r0_leaderboard.csv",
            "/root/prediction-execution-gap/outputs/final_90d/r4_leaderboard.csv",
        ],
        "outputs": [
            str(TABLE_ROOT / "table_02_r0_r4_leaders.csv"),
            str(TABLE_ROOT / "table_02_r0_r4_leaders.md"),
            str(TABLE_ROOT / "table_02_r0_r4_leaders.tex"),
        ],
        "note": "Table 2 uses the clean-active R4 universe, not the full official RR@1 protocol universe.",
    },
    "Table 3": {
        "inputs": [
            "/root/prediction-execution-gap/outputs/final_90d/gap_metrics.json",
            "/root/prediction-execution-gap/src/eval/gap_metrics.py",
        ],
        "outputs": [
            str(TABLE_ROOT / "table_03_gap_metrics.csv"),
            str(TABLE_ROOT / "table_03_gap_metrics.md"),
            str(TABLE_ROOT / "table_03_gap_metrics.tex"),
        ],
        "note": "Table 3 reports current implemented RR@1 with deterministic stable sorting and k=1.",
    },
    "Table 4": {
        "inputs": [
            "/root/prediction-execution-gap/outputs/final_90d/report_card_90d.csv",
            "/root/prediction-execution-gap/outputs/final_90d/ogpa_results_summary.csv",
            "/root/autodl-tmp/peg_bench_runs/final_90d_missing_panel_runs/wrapper_audit_volume_h20_v2fast/wrapper_audit_results.csv",
        ],
        "outputs": [
            str(TABLE_ROOT / "table_04_execution_diagnostics.md"),
            str(TABLE_ROOT / "table_04_execution_diagnostics.tex"),
        ],
        "note": "Table 4 is diagnostic-only and must not be framed as a formal-method win table.",
    },
}

def main() -> None:
    print(json.dumps({
        "bundle_root": str(BUNDLE_ROOT),
        "table_root": str(TABLE_ROOT),
        "tables": TABLE_MAP,
    }, indent=2))

if __name__ == "__main__":
    main()
