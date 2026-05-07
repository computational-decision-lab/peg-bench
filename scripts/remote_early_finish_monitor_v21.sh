#!/usr/bin/env bash
set +e

cd /root/prediction-execution-gap || exit 0

OUT="outputs/pegbench_24h_unattended_v21"
LOGDIR="$OUT/early_finish"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/early_finish_monitor.log"
STATUS="$LOGDIR/early_finish_status.md"

MAX_WAIT_SECONDS=$((4 * 60 * 60))
START=$(date +%s)

log() {
  echo "[$(date -Is)] $*" | tee -a "$LOG"
}

active_compute() {
  ps -ef | grep -E 'run_pegbench_24h_gpu_accelerator_v22.sh|run_peg_bench_dpo_stage.py|run_reviewer_target_baselines.py|slice_and_bootstrap_tables.py|rerun_motivating_slice_official_lite_v8.py' | grep -v grep
}

write_status() {
  {
    echo "# Early Finish Monitor"
    echo
    echo "- updated_at: $(date -Is)"
    echo "- max_wait_seconds: $MAX_WAIT_SECONDS"
    echo "- elapsed_seconds: $(($(date +%s)-START))"
    echo "- policy: finish as soon as current meaningful compute is done; do not run merely to fill 24h"
    echo
    echo "## Active Compute"
    active_compute | sed 's/^/- /' || true
    echo
    echo "## Accelerator"
    cat "$OUT/gpu_accelerator_v22/accelerator_status.md" 2>/dev/null || true
  } > "$STATUS"
}

log "started early finish monitor"

# Stop repeat dispatchers immediately so no new rounds are launched while we
# wait for already-running useful compute to finish.
log "stopping repeat dispatchers; preserving already-running compute children"
pkill -f 'run_pegbench_24h_supervisor.sh' 2>/dev/null || true
pkill -f 'run_pegbench_24h_worker.sh' 2>/dev/null || true
pkill -f 'run_pegbench_24h_p0_worker.sh' 2>/dev/null || true
pkill -f 'run_pegbench_24h_p1_sidecar_supervisor.sh' 2>/dev/null || true
pkill -f 'pegbench_24h_p1_sidecar_worker.py' 2>/dev/null || true

while true; do
  write_status
  now=$(date +%s)
  elapsed=$((now-START))
  if ! active_compute >/dev/null 2>&1; then
    log "no active compute remains"
    break
  fi
  if [ "$elapsed" -ge "$MAX_WAIT_SECONDS" ]; then
    log "max wait reached; proceeding to finalization with remaining active compute caveated"
    break
  fi
  sleep 120
done

# Do not kill active experiment processes unless the max wait was reached.
if active_compute >/dev/null 2>&1; then
  log "active compute still present after max wait; terminating remaining repeated/accelerator compute"
  pkill -f 'run_pegbench_24h_gpu_accelerator_v22.sh' 2>/dev/null || true
  pkill -f 'run_peg_bench_dpo_stage.py' 2>/dev/null || true
  pkill -f 'run_reviewer_target_baselines.py' 2>/dev/null || true
  pkill -f 'slice_and_bootstrap_tables.py' 2>/dev/null || true
  pkill -f 'rerun_motivating_slice_official_lite_v8.py' 2>/dev/null || true
fi

log "generating final reports"
bash scripts/run_pegbench_24h_final_report.sh >> "$LOG" 2>&1 || true
bash scripts/run_pegbench_24h_sync_bundle.sh >> "$LOG" 2>&1 || true

{
  echo "# Early Completion"
  echo
  echo "- completed_at: $(date -Is)"
  echo "- policy: user requested early completion, not mandatory 24h runtime"
  echo "- result_root: $OUT"
  echo "- compact_local_sync: handled by local results-only sync script"
  echo
  echo "## Remaining Caveats"
  echo "- Official TABL/iTransformer/TimesFM/Kronos/PPO/SAC rows remain caveated unless normalized rows are present."
  echo "- Full raw-L2 HftBacktest remains caveated unless raw-L2 replay output is present."
  echo "- Tables 6/7/14/15 should use collected/rerun artifacts only after value-level PDF fill verification."
  echo "- OG-PA/DPO reduced reruns are diagnostic unless clean normalized R0/R4 rows are promoted by final verification."
} > "$OUT/early_completion_report.md"

touch "$OUT/early_completion_marker"
write_status
log "early finish complete"
