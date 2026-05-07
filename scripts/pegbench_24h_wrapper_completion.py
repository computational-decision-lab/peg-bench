#!/usr/bin/env python3
import argparse, csv, shutil
from pathlib import Path

def read_csv(p):
    if not p.exists(): return []
    with p.open(newline='') as f: return list(csv.DictReader(f))
def write_csv(p, rows, fields):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-root', required=True); a=ap.parse_args(); out=Path(a.output_root)
    candidates=[out/'wrapper_matrix_all_rows.csv', out/'copied_evidence/wrapper_matrix_from_completed_cells.csv', out/'copied_evidence/wrapper_matrix.csv']
    rows=[]
    for c in candidates:
        rows=read_csv(c)
        if rows: break
    if not rows:
        rows=[{'status':'missing','note':'no wrapper matrix source found'}]; fields=['status','note']
    else:
        fields=list(rows[0].keys())
    write_csv(out/'wrapper_matrix_completed.csv', rows, fields)
    # delta summary from existing if possible.
    srcs=[out/'wrapper_raw_vs_best_summary.csv', out/'copied_evidence/wrapper_delta_summary.csv']
    drows=[]
    for s in srcs:
        drows=read_csv(s)
        if drows: break
    if drows: write_csv(out/'wrapper_delta_summary_completed.csv', drows, list(drows[0].keys()))
    activity=[]
    for r in rows:
        if 'panel' in r and 'method' in r and 'test_turnover_3bps' in r:
            try: to=float(r.get('test_turnover_3bps') or 0)
            except Exception: to=0
            activity.append({'panel':r.get('panel',''),'method':r.get('method',''),'variant':r.get('variant',''),'turnover':r.get('test_turnover_3bps',''),'zero_action_flag':'yes' if abs(to)<=1e-12 else 'no'})
    if activity: write_csv(out/'wrapper_action_activity_completed.csv', activity, list(activity[0].keys()))
    (out/'wrapper_matrix_completion_report.md').write_text(f'# Wrapper Matrix Completion\n\nrows: {len(rows)}\nsource_candidates: {", ".join(str(x) for x in candidates)}\n')
if __name__=='__main__': main()
