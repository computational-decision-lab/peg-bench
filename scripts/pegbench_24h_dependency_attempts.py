#!/usr/bin/env python3
import argparse, subprocess, time, csv, sys, os
from pathlib import Path
packages=[('stable_baselines3','stable-baselines3'),('gymnasium','gymnasium'),('hftbacktest','hftbacktest'),('lightgbm','lightgbm'),('timesfm','timesfm'),('kronos','kronos'),('torch','torch')]
def run(cmd, timeout=1800):
    try:
        p=subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        return p.returncode, p.stdout[-4000:]
    except subprocess.TimeoutExpired as e:
        return 124, 'TIMEOUT '+str(e)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-root', required=True); a=ap.parse_args(); out=Path(a.output_root); out.mkdir(parents=True, exist_ok=True)
    rows=[]; md=['# Dependency Resolution Attempts','']
    for mod,pkg in packages:
        rc,txt=run(f'python - <<PY\ntry:\n import {mod}\n print("IMPORT_OK", getattr({mod}, "__version__", "no_version"))\nexcept Exception as e:\n print("IMPORT_FAIL", repr(e))\n raise SystemExit(1)\nPY', timeout=120)
        status='import_ok' if rc==0 else 'import_failed'
        fix='none'
        if rc!=0 and pkg not in ['kronos']:
            fix=f'timeout 30m pip install {pkg}'
            rc2,txt2=run(f'timeout 30m python -m pip install {pkg}', timeout=1900)
            rc3,txt3=run(f'python - <<PY\ntry:\n import {mod}\n print("IMPORT_OK_AFTER", getattr({mod}, "__version__", "no_version"))\nexcept Exception as e:\n print("IMPORT_FAIL_AFTER", repr(e))\n raise SystemExit(1)\nPY', timeout=120)
            txt += '\nINSTALL:\n'+txt2+'\nRECHECK:\n'+txt3
            status='resolved_after_install' if rc3==0 else 'failed_after_install_attempt'
        rows.append({'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S'),'module':mod,'package':pkg,'status':status,'attempted_fix':fix,'log_file':f'dep_{mod}.log'})
        (out/f'dep_{mod}.log').write_text(txt)
        md.append(f'## {mod}\n- status: {status}\n- attempted_fix: {fix}\n')
    with (out/'web_sources_checked.csv').open('a', newline='') as f:
        w=csv.writer(f); w.writerow([time.strftime('%Y-%m-%dT%H:%M:%S'),'dependency_attempts','PyPI/package import checks','see dependency_resolution_attempts.csv'])
    with (out/'dependency_resolution_attempts.csv').open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    (out/'web_dependency_resolution_log.md').write_text('\n'.join(md)+'\n')
    (out/'timesfm_resolution_report.md').write_text('See dependency_resolution_attempts.csv for timesfm import/install status.\n')
    (out/'kronos_resolution_report.md').write_text('Kronos package/import attempt recorded. If no PyPI package, official repo adapter remains pending.\n')
    (out/'ppo_sac_resolution_report.md').write_text('stable_baselines3/gymnasium dependency attempts recorded. Training adapter may still be needed.\n')
if __name__=='__main__': main()
