#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json, os, subprocess, time, shutil, sys
from pathlib import Path
REPO=Path('/root/prediction-execution-gap')
OUT=REPO/'outputs/final_run_more_gpu_parallel_pass'
LOGS=OUT/'logs'
PY=Path('/root/miniconda3/bin/python')
PANELS={
 'time_h20':{'mode':'time','target_key':'return_20','std':'/root/autodl-tmp/peg_bench_runs/tardis_90d_time_standardized_inputs','bundle':'/root/autodl-tmp/peg_bench_runs/tardis_90d_time_bundle/manifest.json'},
 'volume_h20':{'mode':'volume','target_key':'return_20','std':'/root/autodl-tmp/peg_bench_runs/tardis_90d_volume_standardized_inputs','bundle':'/root/autodl-tmp/peg_bench_runs/tardis_90d_volume_bundle/manifest.json'},
}
JOBS=[
 ('time_h20','kairos_64code',20260521,0),
 ('time_h20','kairos_64code',20260522,1),
 ('time_h20','kairos_64code',20260523,2),
 ('volume_h20','kairos_64code',20260521,3),
 ('volume_h20','kairos_64code',20260522,4),
 ('volume_h20','kairos_64code',20260523,5),
]
def now(): return time.strftime('%Y-%m-%dT%H:%M:%S%z')
def ensure():
 for d in ['','logs','pids','jobs','blocker_reports']:
  (OUT/d).mkdir(parents=True, exist_ok=True)
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
def append_exit(row):
 p=OUT/'task_exit_codes.csv'; exists=p.exists() and p.stat().st_size>0
 with p.open('a',newline='',encoding='utf-8') as f:
  fields=['time','task','panel','method','seed','gpu','exit_code','status','output_root','log','notes']
  w=csv.DictWriter(f,fieldnames=fields)
  if not exists: w.writeheader()
  w.writerow(row)
def gpu_snapshot():
 p=subprocess.run('nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits || true',shell=True,text=True,capture_output=True)
 return (p.stdout+p.stderr).strip()
def progress(msg):
 exits=read_csv(OUT/'task_exit_codes.csv')
 md(OUT/'progress_live.md',f'# Progress Live\n\n- time: {now()}\n- message: {msg}\n\n## GPU\n```\n{gpu_snapshot()}\n```\n\n## Latest exits\n'+''.join(f'- gpu{r.get("gpu")} {r.get("panel")} {r.get("seed")} {r.get("exit_code")} {r.get("status")}\n' for r in exits[-20:]))
def job_cmd(panel, seed):
 spec=PANELS[panel]
 root=OUT/'jobs'/f'{panel}_kairos_seed{seed}'
 cmd=f"{PY} scripts/run_peg_bench_comparison_baselines.py --standardized-root {spec['std']} --bundle-manifest {spec['bundle']} --output-root {root} --mode {spec['mode']} --seq-len 64 --stride 50 --target-key {spec['target_key']} --methods kairos_64code --device cuda --vq-epochs 1 --kairos-epochs 2 --vq-max-train-items 120000 --vq-batch-size 256 --kairos-batch-size 128 --seed {seed}"
 return cmd, root
def launch():
 ensure()
 md(OUT/'parallel_plan.md','# 6-GPU Parallel Kairos Run\n\nRuns six independent h20 Kairos safe jobs in parallel, one per GPU. These are appendix robustness rows only; frozen headline unchanged.\n')
 plan=[]
 for panel,method,seed,gpu in JOBS:
  cmd, root=job_cmd(panel, seed)
  log=LOGS/f'gpu{gpu}_{panel}_{method}_seed{seed}.log'
  env=os.environ.copy(); env.update({'CUDA_VISIBLE_DEVICES':str(gpu),'OPENBLAS_NUM_THREADS':'64','OMP_NUM_THREADS':'32','MKL_NUM_THREADS':'32','NUMEXPR_NUM_THREADS':'32'})
  with log.open('w',encoding='utf-8') as f:
   f.write(f'[{now()}] START gpu={gpu} panel={panel} seed={seed}\nCMD {cmd}\n')
  p=subprocess.Popen('timeout 10h '+cmd, shell=True, cwd=str(REPO), env=env, stdout=log.open('a',encoding='utf-8'), stderr=subprocess.STDOUT, text=True)
  (OUT/'pids'/f'gpu{gpu}_{panel}_seed{seed}.pid').write_text(str(p.pid)+'\n')
  plan.append({'panel':panel,'method':method,'seed':seed,'gpu':gpu,'pid':p.pid,'output_root':str(root),'log':str(log),'status':'launched'})
 write_csv(OUT/'parallel_run_manifest.csv',plan)
 progress('launched 6 Kairos jobs')
 md(OUT/'monitor_command.md', 'Use: `tail -f outputs/final_run_more_gpu_parallel_pass/logs/gpu0_time_h20_kairos_64code_seed20260521.log` and `nvidia-smi`.')
def monitor_once(final=False):
 ensure(); manifest=read_csv(OUT/'parallel_run_manifest.csv')
 rows=[]; trace=[]; still=[]
 for m in manifest:
  pid=m.get('pid'); panel=m.get('panel'); seed=m.get('seed'); gpu=m.get('gpu'); root=Path(m.get('output_root','')); log=m.get('log','')
  alive=False
  if pid:
   alive=subprocess.run(f'ps -p {pid} -o pid= >/dev/null 2>&1',shell=True).returncode==0
  if alive:
   still.append(m); continue
  # determine rc from timeout not easy after parent gone; use artifact presence and tail
  status='completed' if (root/'results_table.csv').exists() else 'failed_or_not_finished'
  rc=0 if status=='completed' else 1
  # avoid duplicate exits
  existing={(r.get('panel'),r.get('seed')) for r in read_csv(OUT/'task_exit_codes.csv')}
  if (panel,seed) not in existing:
   append_exit({'time':now(),'task':'kairos_parallel_job','panel':panel,'method':'kairos_64code','seed':seed,'gpu':gpu,'exit_code':rc,'status':status,'output_root':str(root),'log':log,'notes':'parallel 6GPU job'})
  if status=='completed':
   for r in read_csv(root/'results_table.csv'):
    if r.get('method')=='kairos_64code':
     rows.append({'panel':panel,'method':'kairos_64code','seed':seed,'gpu':gpu,'variant':'parallel_safe_1vq_2kairos_epoch','test_rank_ic':r.get('test_rank_ic'),'test_directional_accuracy':r.get('test_directional_accuracy'),'source_artifact':str(root/'results_table.csv'),'benchmark_report':str(root/'benchmark_report.json'),'log':log,'claim_boundary':'appendix robustness only'})
  else:
   md(OUT/'blocker_reports'/f'gpu{gpu}_{panel}_seed{seed}.md',f'# Blocker\n\n- panel: {panel}\n- seed: {seed}\n- gpu: {gpu}\n- status: {status}\n- log: {log}\n')
  trace.append({'panel':panel,'method':'kairos_64code','seed':seed,'gpu':gpu,'status':status,'source_artifact':str(root/'results_table.csv'),'benchmark_report':str(root/'benchmark_report.json'),'log':log})
 # merge rows with existing
 old=read_csv(OUT/'kairos_parallel_rows.csv')
 key={(r.get('panel'),r.get('seed')) for r in old}
 for r in rows:
  if (r.get('panel'),r.get('seed')) not in key: old.append(r)
 write_csv(OUT/'kairos_parallel_rows.csv',old)
 oldt=read_csv(OUT/'kairos_parallel_source_trace.csv')
 keyt={(r.get('panel'),r.get('seed')) for r in oldt}
 for r in trace:
  if (r.get('panel'),r.get('seed')) not in keyt: oldt.append(r)
 write_csv(OUT/'kairos_parallel_source_trace.csv',oldt)
 progress(f'monitor: {len(still)} jobs still running')
 if final or not still:
  finalize()
 return len(still)
def finalize():
 exits=read_csv(OUT/'task_exit_codes.csv')
 rows=read_csv(OUT/'kairos_parallel_rows.csv')
 md(OUT/'final_gpu_parallel_summary.md','# Final GPU Parallel Summary\n\n- generated_at: '+now()+f'\n- completed_rows: {len(rows)}\n- note: appendix robustness only; frozen headline unchanged.\n\n## Exits\n'+''.join(f'- gpu{r.get("gpu")} {r.get("panel")} seed {r.get("seed")}: {r.get("status")}\n' for r in exits))
 md(OUT/'run_more_gpu_claim_impact.md','# Claim Impact\n\n- Completed Kairos parallel rows can be used only as appendix robustness/smoke evidence.\n- Do not upgrade main leaderboard or official baseline status.\n- If all six complete, rerun paper claim sync for appendix tables only.\n')
 manifest=[]
 for p in sorted(OUT.rglob('*')):
  if p.is_file() and p.name!='sha256_manifest.txt':
   h=hashlib.sha256()
   with p.open('rb') as f:
    for c in iter(lambda:f.read(1024*1024), b''): h.update(c)
   manifest.append(f'{h.hexdigest()}  {p.relative_to(OUT)}')
 md(OUT/'sha256_manifest.txt','\n'.join(manifest))
 tar=REPO/'outputs/final_run_more_gpu_parallel_pass.tar.gz'; sha=REPO/'outputs/final_run_more_gpu_parallel_pass.sha256'
 subprocess.run(f'tar -czf {tar} -C {OUT.parent} {OUT.name}',shell=True,cwd=str(REPO))
 h=hashlib.sha256()
 with tar.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024), b''): h.update(c)
 md(sha,f'{h.hexdigest()}  final_run_more_gpu_parallel_pass.tar.gz')
if __name__=='__main__':
 ensure(); task=sys.argv[1] if len(sys.argv)>1 else 'launch'
 if task=='launch': launch()
 elif task=='monitor': monitor_once(False)
 elif task=='final': monitor_once(True)
