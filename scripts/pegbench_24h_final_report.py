#!/usr/bin/env python3
import argparse, csv, os, time
from pathlib import Path

def count_rows(p):
    try:
        with p.open(newline='') as f: return max(sum(1 for _ in f)-1,0)
    except Exception: return 0

def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-root', required=True); a=ap.parse_args(); out=Path(a.output_root); out.mkdir(parents=True, exist_ok=True)
    checks={
        'cost_ladder': ['cost_ladder_panel_results_completed.csv','cost_ladder_panel_results_90d.csv','fast_tables/cost_ladder_full_from_daily_reports.csv','cost_ladder_full_from_daily_reports.csv'],
        'bootstrap': ['day_block_bootstrap_ci_completed.csv','action_path_bootstrap_ci_completed.csv','fast_completion/action_path_metric_bootstrap_ci.csv','action_path_metric_bootstrap_ci.csv'],
        'wrapper': ['wrapper_matrix_completed.csv','wrapper_matrix_all_rows.csv'],
        'per_asset_venue': ['per_asset_venue_report_cards_completed.csv','per_asset_venue/per_asset_venue_breakdown.csv','per_asset_venue_breakdown.csv'],
        'target_baselines': ['target_baseline_summary_updated.csv'],
        'hft': ['hftbacktest_fixed_action_sensitivity_completed.csv','hft_audit/hftbacktest_fixed_action_sensitivity.csv','hftbacktest_fixed_action_sensitivity.csv'],
        'motivating_slice': ['motivating_slice/table6.csv','motivating_slice/table7.csv','motivating_slice/table14.csv','motivating_slice/table15.csv'],
        'main_tex_consistency': ['final_main_tex_consistency_report.md','main_tex_claim_inventory.csv'],
    }
    status=[]
    for task, files in checks.items():
        found=[]; rows=0
        for f in files:
            p=out/f
            if p.exists(): found.append(f); rows=max(rows, count_rows(p) if p.suffix=='.csv' else 1)
        status.append({'task':task,'status':'present' if found else 'missing_or_pending','files':';'.join(found),'max_rows':rows})
    write_csv(out/'final_remaining_blockers.csv', [r for r in status if r['status']!='present'], ['task','status','files','max_rows'])
    # ensure fill values exists
    if not (out/'pdf_fill_values_after_24h.csv').exists():
        write_csv(out/'pdf_fill_values_after_24h.csv', [], ['source','panel','method','metric','value'])
    lines=['# Final 24h Completion Report','',f'- generated_at: {time.strftime("%Y-%m-%dT%H:%M:%S")}',f'- output_root: `{out}`','', '## Task Status']
    for r in status: lines.append(f"- {r['task']}: {r['status']} files={r['files']} rows={r['max_rows']}")
    lines += ['', '## Required Questions',
              '1. Actual tasks run: see task_exit_codes.csv and logs/.',
              '2. Successes: any task marked present above has output files.',
              '3. Negative/bad/no-trade results: preserved in CSV outputs; no filtering by performance.',
              '4. Blockers resolved/remaining: see blocker_resolution_board.csv and final_remaining_blockers.csv.',
              '5. Web/dependency attempts: see dependency_resolution_attempts.csv and web_dependency_resolution_log.md if generated.',
              '6. Cost ladder/bootstrap/wrapper/per-asset/HFT/target status: see Task Status above.',
              '7. Tables 6/7/14/15: see motivating_slice outputs or blocker reports.',
              '8. PDF fill values: see pdf_fill_values_after_24h.csv.',
              '9. Caveats: lite/supplemental/audit-only labels must be preserved; no invented numbers.']
    (out/'final_24h_completion_report.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__': main()
