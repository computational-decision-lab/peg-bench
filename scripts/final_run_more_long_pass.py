#!/usr/bin/env python3
from __future__ import annotations
import csv, json, hashlib, subprocess, time, os, sys, shutil
from pathlib import Path
from typing import Any
REPO=Path('/root/prediction-execution-gap')
OUT=REPO/'outputs/final_run_more_long_pass'
LOGS=OUT/'logs'
PY=Path('/root/miniconda3/bin/python')
PANELS={
 'time_h20':{'mode':'time','target_key':'return_20','std':'/root/autodl-tmp/peg_bench_runs/tardis_90d_time_standardized_inputs','bundle':'/root/autodl-tmp/peg_bench_runs/tardis_90d_time_bundle/manifest.json'},
 'volume_h20':{'mode':'volume','target_key':'return_20','std':'/root/autodl-tmp/peg_bench_runs/tardis_90d_volume_standardized_inputs','bundle':'/root/autodl-tmp/peg_bench_runs/tardis_90d_volume_bundle/manifest.json'},
}
def now(): return time.strftime('%Y-%m-%dT%H:%M:%S%z')
def ensure():
 for d in ['','logs','pids','kairos_seed_long','asset_venue_retry_small','blocker_reports']:
  (OUT/d).mkdir(parents=True,exist_ok=True)
def write_csv(p, rows, fields=None):
 p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
 if fields is None:
  fields=[]
  for r in rows:
   for k in r:
    if k not in fields: fields.append(k)
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
def read_csv(p):
 p=Path(p)
 if not p.exists() or p.stat().st_size==0: return []
 with p.open(newline='',encoding='utf-8',errors='ignore') as f: return list(csv.DictReader(f))
def md(p,text): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(text.rstrip()+'\n',encoding='utf-8')
def append_exit(task,rc,status,notes=''):
 p=OUT/'task_exit_codes.csv'; exists=p.exists() and p.stat().st_size>0
 with p.open('a',newline='',encoding='utf-8') as f:
  fields=['time','task','exit_code','status','notes']; w=csv.DictWriter(f,fieldnames=fields)
  if not exists: w.writeheader()
  w.writerow({'time':now(),'task':task,'exit_code':rc,'status':status,'notes':notes})
def run(cmd,log,timeout=None,env=None):
 log=Path(log); log.parent.mkdir(parents=True,exist_ok=True)
 with log.open('a',encoding='utf-8',errors='ignore') as f:
  f.write(f'[{now()}] START {cmd}\n'); f.flush()
  try:
   p=subprocess.Popen(cmd,shell=True,cwd=str(REPO),env=env,stdout=f,stderr=subprocess.STDOUT,text=True)
   rc=p.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   p.kill(); rc=124; f.write(f'[{now()}] TIMEOUT {timeout}\n')
  except Exception as e:
   rc=125; f.write(f'[{now()}] EXCEPTION {type(e).__name__}: {e}\n')
  f.write(f'[{now()}] END rc={rc}\n')
 return rc
def gpu():
 p=subprocess.run('nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits || true',shell=True,text=True,capture_output=True)
 return (p.stdout+p.stderr).strip()
def progress(msg):
 exits=read_csv(OUT/'task_exit_codes.csv')
 md(OUT/'progress_live.md','# Progress Live\n\n'+f'- time: {now()}\n- message: {msg}\n\n```\n{gpu()}\n```\n\n'+''.join(f'- {r.get("time")} {r.get("task")} {r.get("exit_code")} {r.get("status")} {r.get("notes")}\n' for r in exits[-30:]))
def blocker(task,reason,next_fix):
 md(OUT/'blocker_reports'/f'{task}.md',f'# Blocker\n\n- task: {task}\n- time: {now()}\n- reason: {reason}\n- next_fix: {next_fix}\n')
def kairos_long():
 ensure(); seeds=[20260511,20260512,20260513]; rows=[]; trace=[]
 md(OUT/'selected_long_runs.md','# Selected 10h Long Runs\n\n1. Retry `time_h20` Kairos with `OPENBLAS_NUM_THREADS=64` and smaller batch/max-train to avoid previous segfault.\n2. If time allows, run `volume_h20` Kairos extra seeds under same safer env.\n3. Run asset/venue retry with tiny draws separately; still supplemental unless canonical evaluator completes.\n')
 base_env={**os.environ,'OPENBLAS_NUM_THREADS':'64','OMP_NUM_THREADS':'32','MKL_NUM_THREADS':'32','NUMEXPR_NUM_THREADS':'32','CUDA_VISIBLE_DEVICES':'0'}
 for panel in ['time_h20','volume_h20']:
  spec=PANELS[panel]
  for seed in seeds:
   root=OUT/'kairos_seed_long'/f'{panel}_kairos_64code_seed{seed}_safe'
   cmd=f"timeout 4h {PY} scripts/run_peg_bench_comparison_baselines.py --standardized-root {spec['std']} --bundle-manifest {spec['bundle']} --output-root {root} --mode {spec['mode']} --seq-len 64 --stride 50 --target-key {spec['target_key']} --methods kairos_64code --device cuda --vq-epochs 1 --kairos-epochs 2 --vq-max-train-items 120000 --vq-batch-size 256 --kairos-batch-size 128 --seed {seed}"
   rc=run(cmd, LOGS/f'kairos_{panel}_seed{seed}_safe.log', timeout=4*3600+300, env=base_env)
   status='completed' if rc==0 and (root/'results_table.csv').exists() else 'failed_or_timeout'
   append_exit(f'kairos_{panel}_{seed}_safe',rc,status,str(root))
   trace.append({'panel':panel,'method':'kairos_64code','seed':seed,'status':status,'source_artifact':str(root/'results_table.csv'),'benchmark_report':str(root/'benchmark_report.json'),'log':str(LOGS/f'kairos_{panel}_seed{seed}_safe.log')})
   if status=='completed':
    for r in read_csv(root/'results_table.csv'):
     if r.get('method')=='kairos_64code':
      rows.append({'panel':panel,'method':'kairos_64code','seed':seed,'variant':'safe_1vq_2kairos_epoch','test_rank_ic':r.get('test_rank_ic'),'test_directional_accuracy':r.get('test_directional_accuracy'),'source_artifact':str(root/'results_table.csv'),'benchmark_report':str(root/'benchmark_report.json')})
   else:
    blocker(f'kairos_{panel}_{seed}_safe',f'Kairos safe long run failed rc={rc}', 'If repeated rc=139, stop Kairos sweep; use completed LightGBM robustness only.')
   write_csv(OUT/'kairos_seed_long'/'kairos_seed_long_rows.csv',rows)
   write_csv(OUT/'kairos_seed_long'/'kairos_seed_long_source_trace.csv',trace)
   progress(f'completed attempt {panel} seed {seed} status {status}')
 write_csv(OUT/'kairos_seed_long'/'kairos_seed_long_rows.csv',rows)
 write_csv(OUT/'kairos_seed_long'/'kairos_seed_long_source_trace.csv',trace)
 append_exit('kairos_seed_long',0,'completed_with_possible_partial_rows','see kairos_seed_long_rows.csv')
def asset_small():
 out=OUT/'asset_venue_retry_small'; out.mkdir(exist_ok=True)
 cmd=f"timeout 90m {PY} scripts/slice_and_bootstrap_tables.py --output-root {out} --cost-bps 3 --draws 5"
 rc=run(cmd, LOGS/'asset_venue_retry_small.log', timeout=5700)
 status='supplemental_completed_not_canonical' if rc==0 and (out/'per_asset_venue_breakdown.csv').exists() else 'failed_or_timeout'
 if status.startswith('supplemental'):
  rows=read_csv(out/'per_asset_venue_breakdown.csv')
  write_csv(out/'canonical_per_asset_breakdown.csv',[r for r in rows if r.get('grouping')=='asset'])
  write_csv(out/'canonical_per_venue_breakdown.csv',[r for r in rows if r.get('grouping')=='venue'])
  shutil.copy2(out/'per_asset_venue_breakdown.csv', out/'canonical_per_asset_venue_breakdown.csv')
 else:
  blocker('asset_venue_retry_small',f'asset small retry failed rc={rc}','Keep existing P1 supplemental; do not claim canonical.')
 write_csv(out/'canonical_asset_venue_retry_source_trace.csv',[{'status':status,'source_artifact':str(out/'per_asset_venue_breakdown.csv'),'log':str(LOGS/'asset_venue_retry_small.log'),'claim_boundary':'supplemental only, not canonical'}])
 md(out/'canonical_asset_venue_retry_status.md',f'# Asset/Venue Small Retry\n\n- status: `{status}`\n- canonical_upgrade_completed: false\n- claim_boundary: supplemental only.\n')
 append_exit('asset_venue_retry_small',rc,status,'bounded smaller retry')
 progress('asset small retry done')
def finalize():
 exits=read_csv(OUT/'task_exit_codes.csv')
 md(OUT/'final_run_more_long_summary.md','# Final Run More Long Summary\n\n- generated_at: '+now()+'\n- purpose: long autonomous follow-up after run_more pass.\n\n## Exits\n'+''.join(f'- {r.get("task")}: {r.get("exit_code")} {r.get("status")} {r.get("notes")}\n' for r in exits))
 md(OUT/'run_more_long_claim_impact.md','# Claim Impact\n\n- Any Kairos safe seed rows are appendix robustness only unless promoted after paper claim sync.\n- Asset/venue retry remains supplemental unless explicit canonical evaluator is implemented.\n- No main frozen headline changes.\n')
 manifest=[]
 for p in sorted(OUT.rglob('*')):
  if p.is_file() and p.name!='sha256_manifest.txt':
   h=hashlib.sha256()
   with p.open('rb') as f:
    for c in iter(lambda:f.read(1024*1024), b''): h.update(c)
   manifest.append(f'{h.hexdigest()}  {p.relative_to(OUT)}')
 md(OUT/'sha256_manifest.txt','\n'.join(manifest))
 tar=REPO/'outputs/final_run_more_long_pass.tar.gz'; sha=REPO/'outputs/final_run_more_long_pass.sha256'
 subprocess.run(f'tar -czf {tar} -C {OUT.parent} {OUT.name}',shell=True,cwd=str(REPO))
 h=hashlib.sha256()
 with tar.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024), b''): h.update(c)
 md(sha,f'{h.hexdigest()}  final_run_more_long_pass.tar.gz')
 append_exit('finalize',0,'completed','archive generated')
 progress('finalized')
def main():
 ensure(); task=sys.argv[1] if len(sys.argv)>1 else 'all'
 if task in ['all','asset']: asset_small()
 if task in ['all','kairos']: kairos_long()
 if task in ['all','final']: finalize()
if __name__=='__main__': main()
