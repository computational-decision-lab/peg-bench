#!/usr/bin/env python3
import argparse, csv, json, re, shutil, subprocess, os, time
from pathlib import Path
from collections import defaultdict

def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)

def read_csv(path):
    if not path.exists(): return []
    with path.open(newline='') as f: return list(csv.DictReader(f))

def fmt(x):
    try: return f'{float(x):.6g}'
    except Exception: return '' if x is None else str(x)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-root', required=True)
    args=ap.parse_args(); out=Path(args.output_root); out.mkdir(parents=True, exist_ok=True)
    files=[Path('main.tex'), *Path('sections').glob('*.tex'), *Path('tables').glob('*.tex')]
    claim_rows=[]; labels=[]
    for p in files:
        if not p.exists(): continue
        text=p.read_text(errors='ignore')
        for i,line in enumerate(text.splitlines(),1):
            if re.search(r'[-+]?[0-9]+(?:\.[0-9]+)?', line) and any(k in line.lower() for k in ['sharpe','rankic','rr@','bps','row','wrapper','stress','table','figure','cost','bootstrap','elasticnet','tabl','itransformer','hft','og-pa','ppo','sac','kronos','timesfm']):
                claim_rows.append({'file':str(p),'line':i,'text':line.strip()[:500]})
        for m in re.finditer(r'\\label\{((?:fig|tab):[^}]+)\}', text):
            labels.append({'label':m.group(1),'file':str(p),'line':text[:m.start()].count('\n')+1})
    write_csv(out/'main_tex_claim_inventory.csv', claim_rows, ['file','line','text'])
    write_csv(out/'figure_table_label_inventory.csv', labels, ['label','file','line'])
    # Summarize canonical reports if available under benchmark_reports or old run roots.
    report_paths=list((out/'benchmark_reports').glob('*.json'))
    if not report_paths:
        report_paths=list(Path('/root/autodl-tmp/peg_bench_runs').glob('**/benchmark_report.json'))[:100]
    value_rows=[]
    for rp in report_paths:
        try: d=json.loads(rp.read_text())
        except Exception: continue
        panel='unknown'
        s=str(rp)
        for p in ['time_h5','time_h20','time_h50','volume_h5','volume_h20','volume_h50']:
            if p in s: panel=p
        if panel=='unknown':
            if 'time_h5' in s: panel='time_h5'
            elif 'time_h50' in s: panel='time_h50'
            elif 'time_table2' in s or 'time_table' in s: panel='time_h20'
            elif 'volume_h5' in s: panel='volume_h5'
            elif 'volume_h50' in s: panel='volume_h50'
            elif 'volume_h20' in s or 'volume_table' in s: panel='volume_h20'
        gap=d.get('gap',{})
        for m,md in d.get('methods',{}).items():
            pred=md.get('prediction',{}); exe=md.get('execution',{}); rob=md.get('robustness',{})
            value_rows.append({'source':str(rp),'panel':panel,'method':m,'rank_ic':fmt(pred.get('rank_ic',{}).get('mean')),'directional_accuracy':fmt(pred.get('directional_accuracy',{}).get('mean')),'r4_sharpe':fmt(exe.get('post_cost_sharpe',{}).get('mean')),'max_drawdown':fmt(exe.get('max_drawdown',{}).get('mean')),'turnover':fmt(exe.get('turnover',{}).get('mean')),'worst_regime_sharpe':fmt(rob.get('worst_regime_sharpe',{}).get('mean')),'stress_survival':fmt(rob.get('stress_day_survival',{}).get('mean')),'metric_gap':fmt(gap.get('metric_gap')),'ranking_gap':fmt(gap.get('ranking_gap')),'rr1':fmt(gap.get('reversal_rate_at_k'))})
    write_csv(out/'pdf_fill_values_after_24h.csv', value_rows, ['source','panel','method','rank_ic','directional_accuracy','r4_sharpe','max_drawdown','turnover','worst_regime_sharpe','stress_survival','metric_gap','ranking_gap','rr1'])
    # Diff placeholder with unsupported flags.
    diff_rows=[]
    for row in claim_rows:
        text=row['text'].lower()
        status='needs_manual_source_check'
        if any(k in text for k in ['target','future','protocol','frozen','license','release','artifact']): status='scope_or_protocol_text'
        if any(k in text for k in ['og-pa','dpo']) and re.search(r'-?\d+\.\d+', text): status='numeric_ogpa_requires_source_artifact'
        diff_rows.append({'file':row['file'],'line':row['line'],'claim_text':row['text'],'status':status,'source_hint':'benchmark_reports/csv inventory generated in this output'})
    write_csv(out/'table_figure_value_diff.csv', diff_rows, ['file','line','claim_text','status','source_hint'])
    gaps=[]
    if not (out/'target_baseline_summary_updated.csv').exists(): gaps.append({'gap':'target_baseline_summary_updated.csv missing','severity':'retry_pending','recommended_action':'run target baseline merge after panel jobs'})
    if not (out/'hftbacktest_fixed_action_sensitivity_completed.csv').exists() and not (out/'hft_audit/hftbacktest_fixed_action_sensitivity.csv').exists(): gaps.append({'gap':'HFT audit completed file missing','severity':'retry_pending','recommended_action':'rerun hft_execution_audit.py'})
    write_csv(out/'remaining_data_gaps.csv', gaps, ['gap','severity','recommended_action'])
    (out/'patched_files_manifest.csv').write_text('file,action\n')
    md=['# Main TeX Consistency Report','',f'- generated_at: {time.strftime("%Y-%m-%dT%H:%M:%S")}',f'- numeric_claim_rows: {len(claim_rows)}',f'- labels: {len(labels)}',f'- benchmark_value_rows: {len(value_rows)}','','This is an automated result-file inventory. Rows marked `numeric_ogpa_requires_source_artifact` should not be treated as independently verified until their source CSV/JSON is copied into this output.']
    (out/'final_main_tex_consistency_report.md').write_text('\n'.join(md)+'\n')
if __name__=='__main__': main()
