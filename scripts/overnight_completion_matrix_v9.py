from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/root/prediction-execution-gap")
OUT = ROOT / "outputs/final_motivating_slice_rerun_v9"


def exists(path: str) -> bool:
    return (ROOT / path).exists()


ROWS = [
    {
        "item": "Tables 6/7/14/15 motivating slice",
        "status": "completed_protocol_different",
        "best_artifact": "outputs/final_motivating_slice_rerun_v9",
        "evidence": "v9 four-date rerun; DeepLOB rows upgraded to official DeepLOB rerun",
        "remaining_blocker": "original normalized motivating-slice artifact/action path/tokenizer/checkpoints unavailable",
    },
    {
        "item": "DeepLOB motivating-slice row",
        "status": "completed_protocol_different",
        "best_artifact": "outputs/final_motivating_slice_rerun_v9/official_four_date",
        "evidence": "official DeepLOB time h5/h50 predictions and summaries",
        "remaining_blocker": "not exact historical reproduction",
    },
    {
        "item": "Ridge/HistGBM/LightGBM/Kairos motivating-slice rows",
        "status": "completed_protocol_different",
        "best_artifact": "outputs/final_motivating_slice_rerun_v8/official_four_date",
        "evidence": "official-lite volume h5/h50 predictions and summaries",
        "remaining_blocker": "lite settings; original full checkpoints unavailable",
    },
    {
        "item": "Cost ladder 0/1/3/5/10 bps",
        "status": "completed_existing_artifact" if exists("outputs/final_experiment_completion_v5/cost_ladder_panel_results.csv") else "missing",
        "best_artifact": "outputs/final_experiment_completion_v5/cost_ladder_panel_results.csv",
        "evidence": "v5 six-panel cost ladder for core methods",
        "remaining_blocker": "not exhaustive over target-only method zoo",
    },
    {
        "item": "Wrapper matrix",
        "status": "completed_existing_artifact" if exists("outputs/final_experiment_completion_v5/wrapper_matrix.csv") else "missing",
        "best_artifact": "outputs/final_experiment_completion_v5/wrapper_matrix.csv",
        "evidence": "v5 wrapper matrix and best params",
        "remaining_blocker": "wrapper benefits remain panel-specific",
    },
    {
        "item": "Day-block bootstrap CI",
        "status": "completed_existing_artifact" if exists("outputs/final_experiment_completion_v5/day_block_bootstrap_ci.csv") else "missing",
        "best_artifact": "outputs/final_experiment_completion_v5/day_block_bootstrap_ci.csv",
        "evidence": "v5 day-resample bootstrap CI",
        "remaining_blocker": "not full raw action-path bootstrap",
    },
    {
        "item": "Per-asset/per-venue report cards",
        "status": "absorbed_legacy_or_schema_caveated" if exists("outputs/final_missing_completion_v7/per_asset_venue_report_cards.csv") else "missing",
        "best_artifact": "outputs/final_missing_completion_v7/per_asset_venue_report_cards.csv",
        "evidence": "v7 absorbed report-card artifacts",
        "remaining_blocker": "schema-caveated; not definitive v5 schema-complete evidence",
    },
    {
        "item": "Raw reconstruction / feed audit",
        "status": "absorbed_existing_audit_artifact" if exists("outputs/final_missing_completion_v7/reconstruction_audit.csv") else "partial_or_missing",
        "best_artifact": "outputs/final_missing_completion_v7/reconstruction_audit.csv",
        "evidence": "v7 reconstruction/shard/checksum artifacts",
        "remaining_blocker": "not a full raw reconstruction rerun unless paper scope accepts absorbed artifacts",
    },
    {
        "item": "HftBacktest",
        "status": "audit_only" if exists("outputs/final_experiment_completion_v5/hftbacktest_subset_audit.csv") else "missing",
        "best_artifact": "outputs/final_experiment_completion_v5/hftbacktest_subset_audit.csv",
        "evidence": "subset audit available",
        "remaining_blocker": "not leaderboard-integrated",
    },
    {
        "item": "TABL/Kronos/iTransformer/TimesFM/PPO/SAC",
        "status": "target_only",
        "best_artifact": "outputs/final_experiment_completion_v5/method_family_runnability_audit.csv",
        "evidence": "method-family runnability/status audit",
        "remaining_blocker": "not completed ranked baselines",
    },
    {
        "item": "OG-PA / DPO formal-method rows",
        "status": "diagnostic_failed_or_no_trade",
        "best_artifact": "outputs/final_experiment_completion_v5/final_experiment_completion_matrix.csv",
        "evidence": "bug-fixed diagnostics remain failed/no-trade",
        "remaining_blocker": "no positive method-win evidence",
    },
]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "overnight_completion_matrix.csv", ROWS)
    counts: dict[str, int] = {}
    for row in ROWS:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    (OUT / "overnight_completion_counts.json").write_text(json.dumps(counts, indent=2))
    lines = ["# Overnight Completion Matrix v9", ""]
    for status, count in sorted(counts.items()):
        lines.append(f"- {status}: `{count}`")
    lines.append("")
    lines.append("## Remaining Hard Blockers")
    for row in ROWS:
        if row["remaining_blocker"]:
            lines.append(f"- {row['item']}: {row['remaining_blocker']}")
    (OUT / "overnight_completion_summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "completion_matrix_written", "out": str(OUT), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
