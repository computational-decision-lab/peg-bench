#!/usr/bin/env python3
import csv, json, os, re, shutil, subprocess, tarfile, time
from pathlib import Path

ROOT=Path('/root/prediction-execution-gap')
OUT=ROOT/'outputs/pegbench_24h_unattended_v21'
P0=OUT/'p0_numeric_source_recovery'
P0.mkdir(parents=True, exist_ok=True)

TARGET_PATTERNS=[
    ('table5','Table 5 / table5 motivating-slice neighborhood'),
    ('table6','Table 6 motivating-slice'),
    ('table7','Table 7 motivating-slice'),
    ('table14','Table 14 tokenizer/stride audit'),
    ('table15','Table 15 cross-metric audit'),
    ('ogpa','OG-PA numeric rows'),
    ('og-pa','OG-PA numeric rows'),
    ('dpo','DPO numeric rows'),
]
NUM_RE=re.compile(r'[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?', re.I)

def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def run(cmd, timeout=3600, cwd=ROOT):
    try:
        p=subprocess.run(cmd, shell=True, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        return p.returncode, p.stdout
    except subprocess.TimeoutExpired as e:
        return 124, 'TIMEOUT '+str(e)

def safe_copy(src, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        dest=dest_dir/src.name
        if src.is_file(): shutil.copy2(src,dest)
        return str(dest)
    except Exception as e:
        return 'COPY_FAIL '+str(e)

def scan_claims():
    paths=[]
    for base in [ROOT, ROOT/'paper', ROOT/'sections', ROOT/'tables', ROOT/'outputs']:
        if base.exists():
            for ext in ['*.tex','*.md','*.csv','*.json']:
                paths.extend(base.rglob(ext))
    rows=[]
    for p in sorted(set(paths)):
        if any(part.startswith('.git') for part in p.parts): continue
        try: text=p.read_text(errors='ignore')
        except Exception: continue
        low=text.lower()
        if not any(k in low for k,_ in TARGET_PATTERNS): continue
        for i,line in enumerate(text.splitlines(),1):
            l=line.lower()
            matched=[name for name,_ in TARGET_PATTERNS if name in l]
            if matched and (NUM_RE.search(line) or any(x in l for x in ['table','og-pa','dpo','source','missing','blocker','diagnostic'])):
                rows.append({'file':str(p),'line':i,'matched_terms':';'.join(matched),'has_numeric':'yes' if NUM_RE.search(line) else 'no','text':line.strip()[:800]})
    write_csv(P0/'p0_claim_inventory.csv', rows, ['file','line','matched_terms','has_numeric','text'])
    return rows

def collect_sources():
    rows=[]
    copy_dir=P0/'source_candidates'
    search_roots=[ROOT/'outputs', Path('/root/autodl-tmp/peg_bench_runs')]
    names=['table5','table6','table7','table14','table15','ogpa','og-pa','dpo','formal_method','motivating_slice']
    for sr in search_roots:
        if not sr.exists(): continue
        for p in sr.rglob('*'):
            if not p.is_file(): continue
            s=str(p).lower()
            if any(n in s for n in names) and p.suffix.lower() in ['.csv','.md','.tex','.json','.tgz','.tar','.gz','.npz','.log']:
                dest=safe_copy(p, copy_dir)
                rows.append({'source_path':str(p),'kind':'candidate','copied_to':dest,'size':p.stat().st_size})
    write_csv(P0/'p0_source_candidates.csv', rows, ['source_path','kind','copied_to','size'])
    return rows

def rerun_motivating():
    run_dir=P0/'motivating_slice_rerun_now'
    run_dir.mkdir(parents=True, exist_ok=True)
    attempts=[]
    cmds=[
        f'python scripts/rerun_motivating_slice_tables_6_7_14_15.py --output-root {run_dir}',
        'python scripts/rerun_motivating_slice_tables_6_7_14_15.py',
        f'python scripts/rerun_motivating_slice_official_lite_v8.py --output-root {run_dir}',
        'python scripts/rerun_motivating_slice_official_lite_v8.py',
    ]
    for cmd in cmds:
        rc,out=run(cmd, timeout=7200)
        log=run_dir/(re.sub(r'[^A-Za-z0-9]+','_',cmd)[:80]+'.log')
        log.write_text(out)
        attempts.append({'cmd':cmd,'exit_code':rc,'log':str(log)})
        # collect new outputs after every attempt
        for base in [ROOT/'outputs/final_motivating_slice_rerun_v8', ROOT/'outputs/final_motivating_slice_rerun_v9', run_dir]:
            if base.exists():
                for f in base.glob('table*.*'):
                    safe_copy(f, run_dir/'collected')
        if any((run_dir/'collected').glob('table6*')) and any((run_dir/'collected').glob('table7*')):
            pass
    write_csv(P0/'motivating_slice_rerun_attempts.csv', attempts, ['cmd','exit_code','log'])
    return attempts

def ogpa_source_check():
    rows=[]
    candidates=list((ROOT/'outputs').rglob('*ogpa*'))+list((ROOT/'outputs').rglob('*dpo*'))+list((Path('/root/autodl-tmp/peg_bench_runs')).rglob('*ogpa*'))+list((Path('/root/autodl-tmp/peg_bench_runs')).rglob('*dpo*'))
    copy_dir=P0/'ogpa_dpo_candidates'
    for p in sorted(set(candidates)):
        if p.is_file() and p.suffix.lower() in ['.csv','.json','.md','.log','.npz','.pt']:
            dest=safe_copy(p, copy_dir)
            rows.append({'source_path':str(p),'copied_to':dest,'size':p.stat().st_size})
    write_csv(P0/'ogpa_dpo_source_candidates.csv', rows, ['source_path','copied_to','size'])
    # Try to consolidate numeric csv/json candidates into one raw evidence table.
    ev=[]
    for r in rows:
        p=Path(r['copied_to'])
        if not p.exists(): continue
        if p.suffix=='.csv':
            try:
                with p.open(newline='') as f:
                    for row in csv.DictReader(f):
                        ev.append({'source':str(p),'row':json.dumps(row)[:1200]})
            except Exception: pass
        elif p.suffix=='.json':
            try:
                obj=json.loads(p.read_text())
                ev.append({'source':str(p),'row':json.dumps(obj)[:1200]})
            except Exception: pass
    write_csv(P0/'ogpa_dpo_numeric_evidence_flat.csv', ev, ['source','row'])
    return rows

def decisions(claims, sources):
    rows=[]
    source_text='\n'.join(r['source_path'].lower() for r in sources)
    for target in ['table5','table6','table7','table14','table15','ogpa','dpo']:
        c=[r for r in claims if target in r['matched_terms'].lower()]
        has_num=any(r['has_numeric']=='yes' for r in c)
        has_src=target in source_text or (target=='ogpa' and 'ogpa' in source_text) or (target=='dpo' and 'dpo' in source_text)
        decision='verified_or_candidate_source_found' if has_src else 'source_missing_must_caveat_or_remove_numeric_claim'
        if target in ['table6','table7','table14','table15'] and (P0/'motivating_slice_rerun_now/collected').exists():
            if list((P0/'motivating_slice_rerun_now/collected').glob(f'{target}*')): decision='rerun_output_collected_verify_values_before_pdf_fill'
        rows.append({'target':target,'numeric_claim_present_in_scan':'yes' if has_num else 'no','source_candidate_found':'yes' if has_src else 'no','decision':decision,'claim_rows':len(c)})
    write_csv(P0/'p0_pdf_fill_decisions.csv', rows, ['target','numeric_claim_present_in_scan','source_candidate_found','decision','claim_rows'])
    return rows

def main():
    (P0/'STARTED').write_text(time.strftime('%Y-%m-%dT%H:%M:%S'))
    claims=scan_claims()
    sources=collect_sources()
    attempts=rerun_motivating()
    ogpa=ogpa_source_check()
    dec=decisions(claims,sources+ogpa)
    report=['# P0 Numeric Source Recovery Report','',f'- generated_at: {time.strftime("%Y-%m-%dT%H:%M:%S")}',f'- claim_rows: {len(claims)}',f'- source_candidates: {len(sources)}',f'- ogpa_dpo_candidates: {len(ogpa)}',f'- motivating_attempts: {len(attempts)}','','## Decisions']
    for d in dec:
        report.append(f"- {d['target']}: {d['decision']} (numeric_claim={d['numeric_claim_present_in_scan']}, source_candidate={d['source_candidate_found']}, claim_rows={d['claim_rows']})")
    report += ['', '## Policy', 'If a target remains `source_missing_must_caveat_or_remove_numeric_claim`, do not treat its numeric values as verified reproduced evidence. Either rerun successfully, add the source artifact, or remove/caveat the PDF/main.tex numeric claim.']
    (P0/'p0_numeric_source_recovery_report.md').write_text('\n'.join(report)+'\n')
    # copy key outputs up one level for easy sync
    for f in ['p0_claim_inventory.csv','p0_source_candidates.csv','p0_pdf_fill_decisions.csv','p0_numeric_source_recovery_report.md','ogpa_dpo_source_candidates.csv','ogpa_dpo_numeric_evidence_flat.csv','motivating_slice_rerun_attempts.csv']:
        src=P0/f
        if src.exists(): shutil.copy2(src, OUT/f)

if __name__=='__main__': main()
