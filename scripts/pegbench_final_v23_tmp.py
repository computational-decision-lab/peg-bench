
import csv
from pathlib import Path
from datetime import datetime
root=Path("outputs/pegbench_24h_unattended_v21")
def rows(path):
    p=root/path
    if not p.exists() or p.stat().st_size==0:
        return 0
    with p.open(newline='', encoding='utf-8', errors='ignore') as f:
        return max(sum(1 for _ in f)-1,0)
items=[]
items.append(("cost_ladder_0_1_3_5_10bps", "completed", rows("cost_ladder_full_from_daily_reports.csv"), "cost_ladder_full_from_daily_reports.csv", "full cost ladder rows present; use completed CSV"))
items.append(("bootstrap_ci_action_day", "completed", rows("action_path_metric_bootstrap_ci.csv"), "action_path_metric_bootstrap_ci.csv", "action/day bootstrap metric rows present; also check gap/day-block companion files"))
items.append(("wrapper_matrix", "completed", rows("wrapper_matrix_completed.csv"), "wrapper_matrix_completed.csv", "raw vs wrapped matrix present"))
items.append(("per_asset_per_venue", "supplemental_completed", rows("per_asset_venue/per_asset_venue_breakdown.csv"), "per_asset_venue/per_asset_venue_breakdown.csv", "fast/supplemental grouped breakdown, not canonical raw-L2 grouped leaderboard"))
items.append(("target_baselines_gpu_v23", "lite_completed", rows("target_baseline_gpu_accelerated_summary_v23.csv"), "target_baseline_gpu_accelerated_summary_v23.csv", "TABL-lite and iTransformer-lite x 6 panels completed on 6 GPUs; not official repo rows"))
items.append(("target_baselines_existing", "completed_or_lite_mix", rows("target_baseline_summary_updated.csv"), "target_baseline_summary_updated.csv", "existing frozen target rows including ElasticNet/lite/smoke where available"))
items.append(("hftbacktest_fixed_action", "audit_only_completed", rows("hftbacktest_fixed_action_sensitivity.csv"), "hftbacktest_fixed_action_sensitivity.csv", "fixed-action sensitivity audit only; not raw-L2 leaderboard"))
items.append(("tables_6_7_14_15_motivating_slice", "rerun_collected_needs_value_level_pdf_verification", sum(rows(f"p0_numeric_source_recovery/motivating_slice_rerun_now/collected/table{i}.csv") for i in [6,7,14,15]), "p0_numeric_source_recovery/motivating_slice_rerun_now/collected/table{6,7,14,15}.csv", "CSV/MD/TEX collected; do not treat old PDF numbers as reproduced until value-level diff is checked"))
items.append(("table5_numeric_claims", "source_candidate_found", rows("p0_pdf_fill_decisions.csv"), "p0_pdf_fill_decisions.csv", "candidate source found; use p0 decisions and pdf fill values"))
items.append(("ogpa_dpo_numeric_rows", "diagnostic_source_found_not_promoted", rows("p1_sidecar/ogpa_dpo/ogpa_dpo_numeric_evidence_flat.csv"), "p1_sidecar/ogpa_dpo/ogpa_dpo_numeric_evidence_flat.csv", "source candidates found; reduced reruns mixed/timeout; do not claim win or clean official row unless separately promoted"))
items.append(("official_tabl_itransformer", "blocked_official_lite_available", rows("p1_sidecar/official_repo_resolution/official_repo_resolution.csv"), "p1_sidecar/official_repo_resolution/official_repo_resolution.csv", "official normalized row not produced; lite rows available"))
items.append(("timesfm_kronos", "blocked_dependency_or_adapter", rows("p1_sidecar/timesfm_kronos/timesfm_kronos_dependency_resolution.csv"), "p1_sidecar/timesfm_kronos/timesfm_kronos_dependency_resolution.csv", "TimesFM/lingvo and Kronos official/full unresolved after attempts"))
items.append(("ppo_sac_full", "blocked_entrypoint_or_adapter", rows("p1_sidecar/ppo_sac/ppo_sac_dependency_and_adapter_attempt.csv"), "p1_sidecar/ppo_sac/ppo_sac_resolution_report.md", "full normalized PPO/SAC row not produced"))
items.append(("raw_l2_hftbacktest", "blocked_audit_only_available", 0, "p1_sidecar/hft_raw_l2/hftbacktest_raw_l2_attempt_report.md", "raw-L2 HftBacktest replay not completed; fixed-action audit available"))
items.append(("raw_reconstruction_seed_stability", "partial_completed", rows("p1_sidecar/raw_reconstruction_seed_bootstrap/standardized_input_sha256_manifest.csv"), "p1_sidecar/raw_reconstruction_seed_bootstrap/", "checksum/manifest and reduced seed/action bootstrap evidence present, not full raw reconstruction audit"))
out=root/"final_all_scope_completion_report_v23.md"
lines=["# PEG-Bench All-Scope Completion Report v23", "", f"- generated_at: {datetime.now().isoformat(timespec='seconds')}", "- interpretation: all paper-critical runnable items completed; impossible/unresolved official/full items are blocker/caveat, not completed claims.", "", "## Bottom Line", "", "- For paper safety, remaining unsupported PDF/main.tex numeric claims must be either filled from listed source files or downgraded/removed.", "- The reviewer-requested core experiments are present: cost ladder, bootstrap, wrapper matrix, supplemental per-asset/venue, target-lite baselines, and fixed-action HftBacktest audit.", "- Official/full rows that still lack normalized outputs must not be claimed as completed.", "", "## Status Table", "", "| item | status | rows | source | note |", "|---|---:|---:|---|---|"]
for item,status,n,src,note in items:
    lines.append(f"| {item} | {status} | {n} | `{src}` | {note} |")
lines += ["", "## ETA Meaning", "", "- Done-now scope: runnable paper-critical completion plus blocker/caveat packaging is in final sync stage.", "- Not-short scope: turning every official/full P1 item into official normalized rows would require separate engineering/reproduction work and is estimated at 0.5-2+ days with no success guarantee for dependency/adapter-blocked items.", "", "## Must-Caveat Items", "", "- Tables 6/7/14/15: use collected rerun files only after value-level PDF diff; otherwise mark historical/status/caveat, not reproduced evidence.", "- OG-PA/DPO: diagnostic/source-candidate evidence only unless a clean normalized R0/R4 row is explicitly promoted.", "- official TABL/iTransformer/TimesFM/Kronos/PPO/SAC: not completed official benchmark rows in this run.", "- HftBacktest raw-L2: not completed; fixed-action sensitivity is audit-only.", "- per-asset/per-venue: supplemental fast report, not canonical raw-L2 grouped leaderboard.", ""]
out.write_text("\n".join(lines), encoding="utf-8")
csvout=root/"all_scope_status_v23.csv"
with csvout.open('w', newline='', encoding='utf-8') as f:
    w=csv.writer(f)
    w.writerow(['item','status','rows','source','note'])
    w.writerows(items)
print(out)
print(csvout)
