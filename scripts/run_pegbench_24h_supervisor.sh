#!/usr/bin/env bash
set +e
cd /root/prediction-execution-gap || exit 2
OUT="outputs/pegbench_24h_unattended_v21"
LOGDIR="$OUT/logs"
mkdir -p "$OUT" "$LOGDIR" "$OUT/task_logs" "$OUT/status_snapshots"
START_TS_FILE="$OUT/supervisor_start_ts"
if [ -f "$START_TS_FILE" ]; then
  START_TS=$(cat "$START_TS_FILE")
else
  START_TS=$(date +%s)
  echo "$START_TS" > "$START_TS_FILE"
fi
END_TS=$((START_TS + 24*60*60))
if [ ! -f "$OUT/supervisor_restart_log.csv" ]; then
  echo "timestamp,event,exit_code,detail" > "$OUT/supervisor_restart_log.csv"
fi
cat > "$OUT/supervisor_status.md" <<STATUS
# PEG-Bench 24h Supervisor

- start_ts: $START_TS
- end_ts: $END_TS
- started_at: $(date -Is)
- host: $(hostname)
STATUS
while [ $(date +%s) -lt $END_TS ]; do
  NOW=$(date -Is)
  echo "SUPERVISOR HEARTBEAT $NOW" >> "$LOGDIR/heartbeat.log"
  if [ -f "$OUT/final_success_marker" ]; then
    timeout --preserve-status 20m bash scripts/run_pegbench_24h_final_verify.sh >> "$LOGDIR/final_verify.log" 2>&1
    VERIFY_RC=$?
    echo "$NOW,final_verify,$VERIFY_RC,success_marker_present" >> "$OUT/supervisor_restart_log.csv"
    if [ "$VERIFY_RC" = "0" ]; then
      echo "completed_before_24h $(date -Is)" >> "$OUT/supervisor_status.md"
      break
    fi
  fi
  timeout --preserve-status 4h bash scripts/run_pegbench_24h_worker.sh >> "$LOGDIR/worker_stdout.log" 2>> "$LOGDIR/worker_stderr.log"
  RC=$?
  echo "$(date -Is),worker_exit,$RC,supervisor_restart_or_continue" >> "$OUT/supervisor_restart_log.csv"
  bash scripts/run_pegbench_24h_sync_bundle.sh >> "$LOGDIR/sync_bundle.log" 2>&1
  if [ $(date +%s) -lt $END_TS ]; then sleep 60; fi
done
bash scripts/run_pegbench_24h_final_report.sh >> "$LOGDIR/final_report.log" 2>&1
bash scripts/run_pegbench_24h_sync_bundle.sh >> "$LOGDIR/final_sync_bundle.log" 2>&1
echo "supervisor_finished $(date -Is)" >> "$OUT/supervisor_status.md"
