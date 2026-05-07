#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--target-root', required=True); ap.add_argument('--output', required=True)
    a=ap.parse_args(); root=Path(a.target_root); out=Path(a.output); rows=[]
    for p in sorted(root.glob('**/results_table.csv')):
        with p.open(newline='') as f:
            for r in csv.DictReader(f):
                r['source_file']=str(p); rows.append(r)
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    if not fields: fields=['status','note']; rows=[{'status':'missing','note':'no target results_table.csv found'}]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    (out.with_suffix('.md')).write_text(f'# Target Baseline Merge\n\nrows: {len(rows)}\nroot: `{root}`\n')
if __name__=='__main__': main()
