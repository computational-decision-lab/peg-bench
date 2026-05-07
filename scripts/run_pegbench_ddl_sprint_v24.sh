#!/usr/bin/env bash
set +e
cd /root/prediction-execution-gap || exit 0
OUT="outputs/pegbench_ddl_sprint_v24"
TMP="/root/autodl-tmp/pegbench_ddl_sprint_v24"
mkdir -p "$OUT" "$OUT/logs" "$TMP"
TASKS="$OUT/task_exit_codes.csv"
ACTIVE="$OUT/active_jobs.csv"
STATUS="$OUT/status.md"
[ -f "$TASKS" ] || echo "timestamp,name,gpu,exit_code,duration_sec,log,notes" > "$TASKS"
write_status(){
  {
    echo "# PEG-Bench DDL Sprint v24"
    echo
    echo "- updated_at: $(date -Is)"
    echo "- output_root: $OUT"
    echo "- tmp_root: $TMP"
    echo
    echo "## Active Jobs"
    if [ -f "$ACTIVE" ]; then
      while IFS=, read -r name gpu pid log notes; do
        [ "$name" = "name" ] && continue
        if ps -p "$pid" >/dev/null 2>&1; then echo "- $name gpu=$gpu pid=$pid running log=$log notes=$notes"; else echo "- $name gpu=$gpu pid=$pid finished log=$log notes=$notes"; fi
      done < "$ACTIVE"
    fi
    echo
    echo "## GPU"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true
    echo
    echo "## Recent Results"
    tail -n 40 "$TASKS" 2>/dev/null || true
  } > "$STATUS"
}
run_bg(){
  local name="$1" gpu="$2" limit="$3" cmd="$4" notes="$5"
  local log="$OUT/logs/${name}.log"
  (
    local start end rc dur
    start=$(date +%s)
    echo "### DDL_TASK $name START $(date -Is) GPU=$gpu LIMIT=$limit NOTES=$notes"
    echo "### CMD: $cmd"
    if [ "$gpu" = "cpu" ]; then timeout --preserve-status "$limit" bash -lc "$cmd"; else CUDA_VISIBLE_DEVICES="$gpu" timeout --preserve-status "$limit" bash -lc "$cmd"; fi
    rc=$?; end=$(date +%s); dur=$((end-start))
    echo "### DDL_TASK $name END $(date -Is) RC=$rc DUR=$dur"
    echo "$(date -Is),$name,$gpu,$rc,$dur,$log,$notes" >> "$TASKS"
  ) > "$log" 2>&1 &
  echo "$name,$gpu,$!,$log,$notes" >> "$ACTIVE"
}
: > "$ACTIVE"; echo "name,gpu,pid,log,notes" > "$ACTIVE"
# GPU 0: P0 motivating slice official-lite v9 time path; if it starts from scratch, cap to 4h.
run_bg "p0_motivating_time_official_lite_v9" 0 4h "/root/miniconda3/bin/python scripts/overnight_time_official_lite_v9.py; mkdir -p $OUT/motivating_time_v9; cp -a outputs/final_motivating_slice_rerun_v9_time_official_lite/* $OUT/motivating_time_v9/ 2>/dev/null" "P0 Tables 6/7/14/15 improve protocol evidence; may remain supplemental if protocol differs"
# GPU 1: P0/P1 DPO clean reduced retry using less context to finish and write clean summary.
run_bg "p0_dpo_volume_h20_clean_small" 1 4h "/root/miniconda3/bin/python scripts/run_peg_bench_dpo_stage.py --standardized-root /root/autodl-tmp/peg_bench_runs/tardis_90d_volume_standardized_inputs --compare-root /root/autodl-tmp/peg_bench_runs/tardis_90d_volume_table2_compare_v1 --output-root $TMP/dpo_volume_h20_clean_small --mode volume --seq-len 64 --stride 64 --horizon 64 --target-key return_20 --max-contexts 32 --dpo-epochs 1 --dpo-batch-size 8 --device cuda --preference-mode scalar --preference-template balanced --preference-gaps 0.0 --preference-aci-options false --backtest-progress-every 100000 --seed 20260511; rc=\$?; mkdir -p $OUT/dpo_volume_h20_clean_small; find $TMP/dpo_volume_h20_clean_small -maxdepth 3 -type f \( -name '*.json' -o -name '*.csv' -o -name '*.md' -o -name '*.log' \) -exec cp --parents {} $OUT/dpo_volume_h20_clean_small/ \; 2>/dev/null; exit \$rc" "P0 OG-PA/DPO diagnostic to possible clean reduced row"
# GPU 2: official iTransformer repo adapter attempt on volume_h20/time_h20 via repo inventory/minimal runner.
run_bg "p1_itransformer_official_adapter" 2 5h "python - <<'PY'
from pathlib import Path
import subprocess, json, os, sys, csv, shutil
out=Path('outputs/pegbench_ddl_sprint_v24/itransformer_official'); out.mkdir(parents=True, exist_ok=True)
repo=Path('outputs/pegbench_24h_unattended_v21/p1_sidecar/official_repo_resolution/iTransformer')
(out/'attempt_report.md').write_text('# iTransformer Official Adapter Attempt\n\n')
with (out/'attempt_report.md').open('a') as f:
    f.write(f'- repo_exists: {repo.exists()}\n')
    if repo.exists():
        f.write('- files:\n')
        for p in list(repo.rglob('*.py'))[:80]: f.write(f'  - {p}\n')
# run existing lite as fallback not official, copy v23 summary rows for comparison
subprocess.run('/root/miniconda3/bin/python scripts/run_reviewer_target_baselines.py --panel volume_h20 --output-root outputs/pegbench_ddl_sprint_v24/itransformer_official/fallback_lite --methods itransformer_lite --epochs 3 --batch-size 256 --max-train-windows 240000 --device cuda', shell=True)
PY" "P1 official iTransformer best effort; fallback remains lite"
# GPU 3: TABL official search/install attempt plus stronger TABL-lite representative.
run_bg "p1_tabl_official_or_lite_stronger" 3 5h "python - <<'PY'
from pathlib import Path
import subprocess, csv, os
out=Path('outputs/pegbench_ddl_sprint_v24/tabl_official'); out.mkdir(parents=True, exist_ok=True)
queries=['Temporal Attention Augmented Bilinear Network TABL github','TABL LOB official implementation']
(out/'tabl_resolution_report.md').write_text('# TABL Official Resolution\n\nNo local official TABL repo was present at sprint start. Running stronger benchmark-native TABL-lite representative rows while recording official-search need.\n')
subprocess.run('/root/miniconda3/bin/python scripts/run_reviewer_target_baselines.py --panel volume_h20 --output-root outputs/pegbench_ddl_sprint_v24/tabl_official/fallback_lite --methods tabl_lite --epochs 4 --batch-size 256 --max-train-windows 240000 --device cuda', shell=True)
PY" "P1 official TABL best effort; fallback remains lite"
# GPU 4: PPO/SAC minimal full-row attempt using existing env search or create simple action baseline proxy; bounded.
run_bg "p1_ppo_sac_full_attempt" 4 5h "python - <<'PY'
from pathlib import Path
import subprocess, importlib.util, json, sys
out=Path('outputs/pegbench_ddl_sprint_v24/ppo_sac'); out.mkdir(parents=True, exist_ok=True)
mods=['stable_baselines3','gymnasium','gym']
lines=['# PPO/SAC Full Attempt\n']
for m in mods:
    spec=importlib.util.find_spec(m)
    lines.append(f'- {m}: {bool(spec)}\n')
if importlib.util.find_spec('stable_baselines3') is None:
    lines.append('- installing stable-baselines3/gymnasium bounded 30m\n')
    subprocess.run('timeout 30m /root/miniconda3/bin/pip install stable-baselines3 gymnasium', shell=True)
for m in mods:
    lines.append(f'- after {m}: {bool(importlib.util.find_spec(m))}\n')
lines.append('\nNo normalized PPO/SAC row is promoted unless a PEG Gym adapter and R0/R4 evaluator output are produced.\n')
(out/'ppo_sac_resolution_report.md').write_text(''.join(lines))
PY" "P1 PPO/SAC dependency+adapter attempt"
# GPU 5/CPU: Hft raw-L2 conversion attempt and TimesFM/Kronos dependency retry in parallel CPU-ish but assigned lane.
run_bg "p1_hft_raw_l2_and_timesfm_kronos" 5 5h "python - <<'PY'
from pathlib import Path
import subprocess, importlib.util, json, os
out=Path('outputs/pegbench_ddl_sprint_v24/hft_timesfm_kronos'); out.mkdir(parents=True, exist_ok=True)
lines=['# HFT Raw-L2 + TimesFM/Kronos Sprint\n']
# HFT fixed audit rerun and raw candidate inventory
subprocess.run('/root/miniconda3/bin/python scripts/hft_execution_audit.py --output-root outputs/pegbench_ddl_sprint_v24/hft_timesfm_kronos/hft_fixed 2>&1 | tee outputs/pegbench_ddl_sprint_v24/hft_timesfm_kronos/hft_fixed.log', shell=True)
lines.append('- hft fixed audit rerun attempted; raw-L2 promotion requires converted replay output.\n')
# dependencies bounded
for pkg,mod in [('timesfm','timesfm'),('lingvo','lingvo'),('kronos','kronos')]:
    before=importlib.util.find_spec(mod) is not None
    lines.append(f'- {mod} before: {before}\n')
    if not before:
        subprocess.run(f'timeout 30m /root/miniconda3/bin/pip install {pkg}', shell=True)
    after=importlib.util.find_spec(mod) is not None
    lines.append(f'- {mod} after: {after}\n')
(out/'resolution_report.md').write_text(''.join(lines))
PY" "P1 HFT raw-L2/TimesFM/Kronos bounded retry"
write_status
# monitor max 5h, then finalize sprint report and compact copy.
END=$(( $(date +%s) + 5*3600 ))
while [ $(date +%s) -lt $END ]; do
  alive=0
  while IFS=, read -r name gpu pid log notes; do [ "$name" = "name" ] && continue; ps -p "$pid" >/dev/null 2>&1 && alive=$((alive+1)); done < "$ACTIVE"
  write_status
  [ "$alive" -eq 0 ] && break
  sleep 120
done
# kill overstays after deadline chunk
while IFS=, read -r name gpu pid log notes; do [ "$name" = "name" ] && continue; if ps -p "$pid" >/dev/null 2>&1; then pkill -P "$pid" 2>/dev/null || true; kill "$pid" 2>/dev/null || true; echo "$(date -Is),$name,$gpu,KILLED_AFTER_CHUNK,0,$log,$notes" >> "$TASKS"; fi; done < "$ACTIVE"
python - <<'PY'
from pathlib import Path
from datetime import datetime
root=Path('outputs/pegbench_ddl_sprint_v24')
lines=['# DDL Sprint v24 Report','',f'- generated_at: {datetime.now().isoformat(timespec="seconds")}', '- purpose: 10-14h DDL sprint for remaining paper/P1 items.', '', '## Task Results', '']
if (root/'task_exit_codes.csv').exists(): lines.append((root/'task_exit_codes.csv').read_text())
lines += ['', '## Interpretation Rules', '', '- Promote only rows with normalized evaluator outputs.', '- Lite/proxy/diagnostic outputs must stay labeled.', '- If Tables 6/7/14/15 remain protocol_different, remove or caveat old PDF numbers.', '- If OG-PA/DPO lacks clean normalized row, keep diagnostic only.']
(root/'ddl_sprint_v24_report.md').write_text('\n'.join(lines))
PY
write_status
