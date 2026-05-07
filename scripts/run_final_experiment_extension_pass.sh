#!/usr/bin/env bash
set +euo pipefail

REPO="/root/prediction-execution-gap"
OUT="$REPO/outputs/final_experiment_extension_pass"
LOGS="$OUT/logs"
PY="/root/miniconda3/bin/python"
HELPER="$REPO/scripts/final_experiment_extension_pass.py"
mkdir -p "$LOGS" "$OUT/blocker_reports" "$OUT/pids"
cd "$REPO" || exit 2

write_progress_loop() {
  while true; do
    {
      echo "# Progress Live"
      echo "- time: $(date -Is)"
      echo "- runner_pid: $$"
      echo ""
      echo "## GPU Snapshot"
      echo '```'
      nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true
      echo '```'
      echo ""
      echo "## Tail task exit codes"
      tail -n 20 "$OUT/task_exit_codes.csv" 2>/dev/null || true
    } > "$OUT/progress_live.md"
    sleep 1800
  done
}
write_progress_loop &
PROGRESS_PID=$!
echo "progress,$PROGRESS_PID,$(date -Is)" >> "$OUT/pids/task_pids.csv"

run_task() {
  local task="$1"
  local timeout_s="$2"
  local log="$LOGS/${task}.log"
  echo "[$(date -Is)] START $task" | tee -a "$log"
  timeout "$timeout_s" "$PY" "$HELPER" --task "$task" --out "$OUT" >> "$log" 2>&1
  local rc=$?
  echo "[$(date -Is)] END $task rc=$rc" | tee -a "$log"
  if [ "$rc" -ne 0 ]; then
    mkdir -p "$OUT/blocker_reports"
    cat > "$OUT/blocker_reports/${task}_runner_blocker.md" <<EOF
# Runner Blocker

- task: $task
- time: $(date -Is)
- exit_code: $rc
- last_error: see logs/${task}.log
- next_fix: inspect log and rerun task only if safe
EOF
    if [ -f "$OUT/task_exit_codes.csv" ]; then
      echo "$(date -Is),$task,$rc,runner_failed,see logs/${task}.log" >> "$OUT/task_exit_codes.csv"
    fi
  fi
  return 0
}

# Required preflight artifacts first.
run_task preflight 900

# E1/E2 highest priority. These are conservative feasibility + normalization tasks;
# they do not claim GPU training unless a verified official entrypoint exists.
run_task e1_tabl 2700 &
PID_E1=$!
echo "e1_tabl,$PID_E1,$(date -Is)" >> "$OUT/pids/task_pids.csv"
run_task e2_itransformer 2700 &
PID_E2=$!
echo "e2_itransformer,$PID_E2,$(date -Is)" >> "$OUT/pids/task_pids.csv"
wait "$PID_E1" "$PID_E2"

# Feasibility/sidecar tasks; each isolated and non-blocking.
run_task e3_timesfm_kronos 2700
run_task e4_ppo_sac 2700
run_task e5_asset_venue 2700
run_task e6_seed_split 1800
run_task final_reports 1800

kill "$PROGRESS_PID" 2>/dev/null || true
"$PY" "$HELPER" --task final_reports --out "$OUT" >> "$LOGS/final_reports_second_pass.log" 2>&1 || true
exit 0
