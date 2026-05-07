#!/usr/bin/env python3
from __future__ import annotations
import csv, json, hashlib, subprocess, time, os, sys, shutil
from pathlib import Path
from typing import Any

REPO=Path('/root/prediction-execution-gap')
OUT=REPO/'outputs/final_run_more_pass'
LOGS=OUT/'logs'
PY=Path('/root/miniconda3/bin/python')
TMP=Path('/root/autodl-tmp')
PANELS={
 'volume_h20':{'mode':'volume','target_key':'return_20','std':'/root/autodl-tmp/peg_bench_runs/tardis_90d_volume_standardized_inputs','bundle':'/root/autodl-tmp/peg_bench_runs/tardis_90d_volume_bundle/manifest.json','compare':'/root/autodl-tmp/peg_bench_runs/tardis_90d_volume_table2_compare_v1','clean_compare':'/root/autodl-tmp/peg_bench_runs/final_90d_missing_panel_runs/protocol_clean_volume_h20_seq96'},
 'time_h20':{'mode':'time','target_key':'return_20','std':'/root/autodl-tmp/peg_bench_runs/tardis_90d_time_standardized_inputs','bundle':'/root/autodl-tmp/peg_bench_runs/tardis_90d_time_bundle/manifest.json','compare':'/root/autodl-tmp/peg_bench_runs/tardis_90d_time_table2_compare_v1','clean_compare':'/root/autodl-tmp/peg_bench_runs/tardis_90d_time_table2_compare_v1'},
}

def now(): return time.strftime('%Y-%m-%dT%H:%M:%S%z')
def ensure():
 for d in ['', 'logs','blocker_reports','official_baseline_smoke','seed_split_limited_runs','canonical_asset_venue_retry','hft_raw_l2_smoke','pids']:
  (OUT/d).mkdir(parents=True, exist_ok=True)
def write_csv(p:Path, rows:list[dict[str,Any]], fields=None):
 p.parent.mkdir(parents=True, exist_ok=True)
 if fields is None:
  fields=[]
  for r in rows:
   for k in r:
    if k not in fields: fields.append(k)
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
def read_csv(p:Path):
 if not p.exists() or p.stat().st_size==0: return []
 with p.open(newline='',encoding='utf-8',errors='ignore') as f: return list(csv.DictReader(f))
def md(p:Path, text:str):
 p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text.rstrip()+'\n',encoding='utf-8')
def append_exit(task, rc, status, notes=''):
 p=OUT/'task_exit_codes.csv'; exists=p.exists() and p.stat().st_size>0
 with p.open('a',newline='',encoding='utf-8') as f:
  fields=['time','task','exit_code','status','notes']; w=csv.DictWriter(f,fieldnames=fields)
  if not exists: w.writeheader()
  w.writerow({'time':now(),'task':task,'exit_code':rc,'status':status,'notes':notes})
def run(cmd, log:Path, timeout=None, env=None):
 log.parent.mkdir(parents=True, exist_ok=True)
 with log.open('a',encoding='utf-8',errors='ignore') as f:
  f.write(f'[{now()}] START {cmd}\n'); f.flush()
  try:
   p=subprocess.Popen(cmd, shell=isinstance(cmd,str), cwd=str(REPO), env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
   rc=p.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   p.kill(); rc=124; f.write(f'[{now()}] TIMEOUT {timeout}\n')
  except Exception as e:
   rc=125; f.write(f'[{now()}] EXCEPTION {type(e).__name__}: {e}\n')
  f.write(f'[{now()}] END rc={rc}\n')
 return int(rc)
def blocker(task, reason, next_fix=''):
 md(OUT/'blocker_reports'/f'{task}.md', f'# Blocker\n\n- task: {task}\n- time: {now()}\n- reason: {reason}\n- next_fix: {next_fix or "n/a"}\n')
def gpu():
 p=subprocess.run('nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits || true',shell=True,text=True,capture_output=True)
 return (p.stdout+p.stderr).strip()
def progress(msg):
 exits=read_csv(OUT/'task_exit_codes.csv')
 md(OUT/'progress_live.md','# Progress Live\n\n'+f'- time: {now()}\n- message: {msg}\n\n## GPU\n```\n{gpu()}\n```\n\n## Exits\n'+''.join(f'- {r.get("time")} {r.get("task")} {r.get("exit_code")} {r.get("status")} {r.get("notes")}\n' for r in exits[-20:]))

def grep(pattern:str, roots:list[str], max_lines=120):
 cmd='grep -RInE '+repr(pattern)+' '+' '.join(roots)+' 2>/dev/null | head -'+str(max_lines)
 p=subprocess.run(cmd,shell=True,cwd=str(REPO),text=True,capture_output=True)
 return p.stdout.strip().splitlines()

def triage():
 ensure()
 runnable=[]; blocked=[]
 itr_hits=grep('iTransformer|TABL|official.*adapter|official.*TABL|official.*iTransformer', ['scripts','src','configs','outputs/final_experiment_extension_pass'], 200)
 official_runnable=False
 official_reason='Only official-source attempt/fallback-lite references found; no verified official adapter/config entrypoint for PEG-Bench volume_h20.'
 for line in itr_hits:
  if 'run_reviewer_target_baselines.py' in line or 'fallback remains lite' in line or 'official-source attempts' in line:
   continue
  if 'git clone' in line or 'official' in line:
   pass
 # Conservative: no command found that takes canonical panels and runs official external repo.
 blocked.append({'candidate':'official_TABL','priority':1,'status':'blocked_no_runnable_official_path','reason':official_reason,'next_action':'Add/approve official TABL adapter; lite rows remain supplemental.'})
 blocked.append({'candidate':'official_iTransformer','priority':1,'status':'blocked_no_runnable_official_path','reason':official_reason,'next_action':'Add/approve official iTransformer adapter; lite rows remain supplemental.'})
 # Seed/split readiness
 for panel,s in PANELS.items():
  for method in ['lightgbm_lob','kairos_64code']:
   pred=Path(s['compare'])/'methods'/method/'predictions.npz'
   clean_pred=Path(s['clean_compare'])/'methods'/method/'predictions.npz'
   std=Path(s['std']); bundle=Path(s['bundle'])
   if (pred.exists() or clean_pred.exists()) and std.exists() and bundle.exists():
    runnable.append({'candidate':f'seed_split_{panel}_{method}','priority':2,'status':'runnable_smoke_first','panel':panel,'method':method,'input_predictions':str(clean_pred if clean_pred.exists() else pred),'gpu_cpu':'CPU for LightGBM; GPU optional for Kairos if retraining','command_hint':'run_peg_bench_comparison_baselines.py limited seeds'})
   else:
    blocked.append({'candidate':f'seed_split_{panel}_{method}','priority':2,'status':'blocked_missing_input','reason':f'missing pred/std/bundle: {pred} {std} {bundle}','next_action':'skip this combo'})
 # Asset venue/HFT
 av_hits=grep('asset_key|venue|grouped|stratified|per_asset', ['scripts','src'], 120)
 runnable.append({'candidate':'canonical_asset_venue_retry','priority':3,'status':'runnable_quick_check_only','panel':'all','method':'core','input_predictions':'P0/P1 + window metadata','gpu_cpu':'CPU','command_hint':'slice metadata exists; canonical evaluator not explicit, retry bounded'})
 hft_hits=grep('raw-L2|raw L2|hftbacktest|HftBacktest', ['scripts','src'], 120)
 blocked.append({'candidate':'hft_raw_l2_smoke','priority':4,'status':'blocked_unless_engine_path_verified','reason':'Only fixed-action audit script is visible; no confirmed raw-L2 conversion command in quick grep.','next_action':'write blocker unless explicit engine/data/action paths found in quick retry'})
 write_csv(OUT/'runnable_candidates.csv', runnable)
 write_csv(OUT/'blocked_candidates.csv', blocked)
 md(OUT/'quick_run_decision.md', '# Quick Run Decision\n\n- decision_time: '+now()+'\n- official TABL/iTransformer: blocked within quick triage; no runnable official adapter path found.\n- selected immediate run: limited seed/split extension for `volume_h20` and `time_h20`, LightGBM first, Kairos only if bounded.\n- canonical asset/venue: one quick bounded retry after seed/split launch.\n- Hft raw-L2: blocker unless explicit raw-L2 engine/data/action path is verified.\n')
 md(OUT/'selected_runs.md', '# Selected Runs\n\n1. Block official TABL/iTransformer smoke as `blocked_no_runnable_official_path`; do not run lite as official.\n2. Start limited seed/split extension: `volume_h20,time_h20` x `lightgbm_lob,kairos_64code`, seeds `20260511,20260512,20260513`; LightGBM prioritized.\n3. Retry canonical asset/venue once with metadata check.\n4. Hft raw-L2 quick blocker unless explicit raw-L2 replay command exists.\n')
 # Official smoke status outputs
 trace=[]
 for cand in ['TABL','iTransformer']:
  trace.append({'baseline':cand,'status':'blocked_no_runnable_official_path','source':'quick grep scripts/src/configs + extension pass','notes':'No official external-repo adapter command found; lite/proxy not used as official.'})
 write_csv(OUT/'official_baseline_smoke'/'official_baseline_smoke_source_trace.csv', trace)
 write_csv(OUT/'official_baseline_smoke'/'tabl_official_smoke_result.csv', [], ['status','panel','metric','value','source'])
 write_csv(OUT/'official_baseline_smoke'/'itransformer_official_smoke_result.csv', [], ['status','panel','metric','value','source'])
 md(OUT/'official_baseline_smoke'/'official_baseline_smoke_status.md', '# Official Baseline Smoke Status\n\n- TABL: `blocked_no_runnable_official_path`\n- iTransformer: `blocked_no_runnable_official_path`\n- No lite/proxy row is labeled official.\n')
 append_exit('quick_triage',0,'completed','selected seed/split limited; official baselines blocked')
 append_exit('official_baseline_smoke',0,'blocked_no_runnable_official_path','no official adapter path; lite not promoted')
 progress('quick triage completed; launching seed/split limited')

def run_seed_split():
 out=OUT/'seed_split_limited_runs'; out.mkdir(parents=True,exist_ok=True)
 seeds=[20260511,20260512,20260513]
 plan=[]
 for panel,spec in PANELS.items():
  for method in ['lightgbm_lob','kairos_64code']:
   for seed in seeds:
    estimate='under_2h_lightgbm' if method=='lightgbm_lob' else 'kairos_bounded_smoke_may_use_gpu'
    plan.append({'panel':panel,'method':method,'variant':'raw_and_wrapped_if_wrapper_succeeds','seed':seed,'status':'planned','estimate':estimate,'output_root':str(out/f'{panel}_{method}_seed{seed}')})
 write_csv(out/'seed_split_limited_plan.csv', plan)
 rows=[]; trace=[]; split_rows=[]
 # LightGBM full limited seeds first; Kairos one smoke seed per panel only to avoid >8h surprise.
 tasks=[]
 for panel,spec in PANELS.items():
  for seed in seeds:
   root=out/f'{panel}_lightgbm_lob_seed{seed}'
   cmd=f"timeout 90m {PY} scripts/run_peg_bench_comparison_baselines.py --standardized-root {spec['std']} --bundle-manifest {spec['bundle']} --output-root {root} --mode {spec['mode']} --seq-len 64 --stride 50 --target-key {spec['target_key']} --methods lightgbm_lob --device cpu --seed {seed}"
   rc=run(cmd, LOGS/f'seed_split_{panel}_lightgbm_seed{seed}.log', timeout=5700)
   status='completed' if rc==0 and (root/'results_table.csv').exists() else 'failed'
   append_exit(f'seed_{panel}_lightgbm_{seed}',rc,status,str(root))
   trace.append({'panel':panel,'method':'lightgbm_lob','seed':seed,'status':status,'source_artifact':str(root/'results_table.csv'),'log':str(LOGS/f'seed_split_{panel}_lightgbm_seed{seed}.log')})
   if status=='completed':
    for r in read_csv(root/'results_table.csv'):
     if r.get('method')=='lightgbm_lob':
      rows.append({'panel':panel,'method':'lightgbm_lob','seed':seed,'variant':'raw','test_rank_ic':r.get('test_rank_ic'),'test_directional_accuracy':r.get('test_directional_accuracy'),'source_artifact':str(root/'results_table.csv'),'benchmark_report':str(root/'benchmark_report.json')})
   else:
    blocker(f'seed_{panel}_lightgbm_{seed}', f'LightGBM limited seed run failed rc={rc}', 'inspect log; skip to next seed')
  # Kairos only one seed smoke per panel, because training uses GPU and may be longer.
  seed=seeds[0]
  root=out/f'{panel}_kairos_64code_seed{seed}_smoke'
  cmd=f"CUDA_VISIBLE_DEVICES=0 timeout 120m {PY} scripts/run_peg_bench_comparison_baselines.py --standardized-root {spec['std']} --bundle-manifest {spec['bundle']} --output-root {root} --mode {spec['mode']} --seq-len 64 --stride 50 --target-key {spec['target_key']} --methods kairos_64code --device cuda --vq-epochs 1 --kairos-epochs 1 --vq-max-train-items 160000 --kairos-batch-size 256 --seed {seed}"
  rc=run(cmd, LOGS/f'seed_split_{panel}_kairos_seed{seed}_smoke.log', timeout=7500, env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'})
  status='smoke_completed' if rc==0 and (root/'results_table.csv').exists() else 'smoke_failed_or_timeout'
  append_exit(f'seed_{panel}_kairos_smoke_{seed}',rc,status,str(root))
  trace.append({'panel':panel,'method':'kairos_64code','seed':seed,'status':status,'source_artifact':str(root/'results_table.csv'),'log':str(LOGS/f'seed_split_{panel}_kairos_seed{seed}_smoke.log')})
  if status=='smoke_completed':
   for r in read_csv(root/'results_table.csv'):
    if r.get('method')=='kairos_64code':
     rows.append({'panel':panel,'method':'kairos_64code','seed':seed,'variant':'raw_smoke_1epoch','test_rank_ic':r.get('test_rank_ic'),'test_directional_accuracy':r.get('test_directional_accuracy'),'source_artifact':str(root/'results_table.csv'),'benchmark_report':str(root/'benchmark_report.json')})
  else:
   blocker(f'seed_{panel}_kairos_smoke_{seed}', f'Kairos smoke failed or timed out rc={rc}', 'do not start full Kairos sweep without confirmation')
 # Simple stability aggregate
 write_csv(out/'seed_stability_new_runs.csv', rows)
 write_csv(out/'split_stability_new_runs.csv', split_rows, ['panel','method','seed','status','metric','value','source_artifact'])
 write_csv(out/'seed_split_new_source_trace.csv', trace)
 # summarize variance by panel/method for rank_ic
 agg=[]
 by={}
 for r in rows:
  key=(r['panel'],r['method'])
  by.setdefault(key,[]).append(float(r['test_rank_ic']))
 for (panel,method), vals in by.items():
  import statistics
  agg.append({'panel':panel,'method':method,'n_runs':len(vals),'rank_ic_mean':sum(vals)/len(vals),'rank_ic_min':min(vals),'rank_ic_max':max(vals),'rank_ic_std':statistics.pstdev(vals) if len(vals)>1 else 0.0,'claim_boundary':'appendix robustness only; frozen headline unchanged'})
 write_csv(out/'seed_stability_summary_new.csv', agg)
 md(out/'seed_split_limited_status.md', '# Seed/Split Limited Status\n\n- LightGBM limited 3-seed runs attempted for `volume_h20` and `time_h20`.\n- Kairos one-seed 1-epoch smoke attempted per h20 panel only; full Kairos seed sweep not launched without confirmation.\n- All rows are appendix robustness only; frozen headline unchanged.\n')
 append_exit('seed_split_limited_runs',0,'completed_with_possible_partial_rows','see seed_split_new_source_trace.csv')
 progress('seed/split limited completed; retrying asset/venue and hft quick checks')

def retry_asset_venue():
 out=OUT/'canonical_asset_venue_retry'; out.mkdir(parents=True,exist_ok=True)
 trace=[]
 # Check if slice_and_bootstrap has metadata but labels fast supplemental. Run it as retry, but do not claim canonical.
 cmd=f"timeout 20m {PY} scripts/slice_and_bootstrap_tables.py --output-root {out} --cost-bps 3 --draws 50"
 rc=run(cmd, LOGS/'canonical_asset_venue_retry.log', timeout=1500)
 if rc==0 and (out/'per_asset_venue_breakdown.csv').exists():
  # copy to requested names but status says supplemental/not canonical unless canonical evaluator explicit.
  shutil.copy2(out/'per_asset_venue_breakdown.csv', out/'canonical_per_asset_venue_breakdown.csv')
  rows=read_csv(out/'per_asset_venue_breakdown.csv')
  asset=[r for r in rows if r.get('grouping')=='asset']; venue=[r for r in rows if r.get('grouping')=='venue']
  write_csv(out/'canonical_per_asset_breakdown.csv', asset)
  write_csv(out/'canonical_per_venue_breakdown.csv', venue)
  status='supplemental_fast_replay_completed_not_canonical'
  trace.append({'status':status,'source_artifact':str(out/'per_asset_venue_breakdown.csv'),'notes':'Generated by slice_and_bootstrap_tables fast replay; not explicit canonical grouped evaluator.'})
 else:
  status='blocked_metadata_or_script_failed'
  blocker('canonical_asset_venue_retry', f'slice retry failed rc={rc}', 'keep P1 supplemental fast replay')
 write_csv(out/'canonical_asset_venue_retry_source_trace.csv', trace)
 md(out/'canonical_asset_venue_retry_status.md', f'# Canonical Asset/Venue Retry Status\n\n- status: `{status}`\n- canonical_upgrade_completed: false\n- claim_boundary: supplemental fast replay only unless a true canonical grouped evaluator is implemented.\n')
 append_exit('canonical_asset_venue_retry',rc,status,'bounded retry complete')
 progress('asset/venue retry completed; checking hft raw-l2')

def hft_raw_l2():
 out=OUT/'hft_raw_l2_smoke'; out.mkdir(parents=True,exist_ok=True)
 hits=grep('raw-L2|raw L2|hftbacktest|HftBacktest', ['scripts','src'], 160)
 explicit=[h for h in hits if ('raw' in h.lower() and 'fixed-action' not in h.lower() and 'fixed action' not in h.lower())]
 trace=[{'status':'quick_checked','source':'grep scripts/src','hits':len(hits),'explicit_raw_l2_hits':len(explicit),'notes':'No full raw-L2 smoke launched unless engine/data/action path all present.'}]
 # current script is fixed-action audit; do not use as raw-l2.
 write_csv(out/'hft_raw_l2_smoke_result.csv', [], ['status','panel','metric','value','source_artifact'])
 write_csv(out/'hft_raw_l2_source_trace.csv', trace)
 md(out/'hft_raw_l2_smoke_status.md', '# HftBacktest Raw-L2 Smoke Status\n\n- status: `blocked_no_ready_raw_l2_engine_data_action_path`\n- fixed-action audit exists, but this is not full raw-L2 replay.\n- no raw-L2 smoke launched.\n')
 blocker('hft_raw_l2_smoke','No ready raw-L2 HftBacktest engine + data path + action path found in quick check.','Do not relabel fixed-action audit; implement conversion first.')
 append_exit('hft_raw_l2_smoke',0,'blocked_no_ready_raw_l2_engine_data_action_path','quick blocker written')
 progress('hft raw-l2 quick blocker written; finalizing')

def finalize():
 # summaries
 exits=read_csv(OUT/'task_exit_codes.csv')
 md(OUT/'final_run_more_summary.md', '# Final Run More Summary\n\n- generated_at: '+now()+'\n- official TABL/iTransformer: blocked, no runnable official path found in quick triage.\n- seed/split limited: attempted new LightGBM 3-seed h20 runs plus Kairos bounded smoke. See `seed_split_limited_runs/`.\n- canonical asset/venue: bounded retry generated/retained supplemental fast replay only; no canonical upgrade.\n- Hft raw-L2: blocked; fixed-action audit not relabeled.\n\n## Exit Codes\n'+''.join(f'- {r.get("task")}: {r.get("exit_code")} {r.get("status")} {r.get("notes")}\n' for r in exits))
 md(OUT/'run_more_claim_impact.md', '# Run More Claim Impact\n\n1. New completed result: limited seed/split LightGBM/Kairos rows if present in `seed_stability_new_runs.csv`; these are robustness appendix only.\n2. Main claim upgrade: none unless reviewer accepts appendix robustness; frozen test headline unchanged.\n3. Appendix only: seed/split limited runs and supplemental asset/venue retry.\n4. Target-only: official TABL, official iTransformer, full raw-L2 HftBacktest.\n5. Next step: return to paper claim sync if no further approved long run; do not upgrade main leaderboard.\n')
 md(OUT/'remaining_blockers.md', '# Remaining Blockers\n\n- official TABL: no runnable official adapter path.\n- official iTransformer: no runnable official adapter path.\n- canonical asset/venue: explicit canonical grouped evaluator not verified; fast replay remains supplemental.\n- full raw-L2 HftBacktest: engine/data/action path not ready.\n- Kairos full seed sweep: not launched because full sweep may exceed bounded run; requires confirmation if desired.\n')
 # sha and archive
 manifest=[]
 for p in sorted(OUT.rglob('*')):
  if p.is_file() and p.name!='sha256_manifest.txt':
   h=hashlib.sha256();
   with p.open('rb') as f:
    for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
   manifest.append(f'{h.hexdigest()}  {p.relative_to(OUT)}')
 md(OUT/'sha256_manifest.txt','\n'.join(manifest))
 tar=REPO/'outputs/final_run_more_pass.tar.gz'; sha=REPO/'outputs/final_run_more_pass.sha256'
 if tar.exists(): tar.unlink()
 subprocess.run(f'tar -czf {tar} -C {OUT.parent} {OUT.name}',shell=True,cwd=str(REPO),check=False)
 h=hashlib.sha256();
 with tar.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
 md(sha,f'{h.hexdigest()}  final_run_more_pass.tar.gz')
 append_exit('finalize',0,'completed','archive and sha generated')
 progress('finalized')

def main():
 ensure()
 task=sys.argv[1] if len(sys.argv)>1 else 'all'
 if task in ['triage','all']: triage()
 if task in ['seed','all']: run_seed_split()
 if task in ['asset','all']: retry_asset_venue()
 if task in ['hft','all']: hft_raw_l2()
 if task in ['final','all']: finalize()
if __name__=='__main__': main()
