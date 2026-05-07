#!/usr/bin/env bash
set +e

cd /root/prediction-execution-gap || exit 2
OUT="outputs/pegbench_24h_unattended_v21"
LOGDIR="$OUT/logs"
mkdir -p "$OUT/p1_sidecar" "$LOGDIR"

START_TS_FILE="$OUT/supervisor_start_ts"
if [ -f "$START_TS_FILE" ]; then
  START_TS="$(cat "$START_TS_FILE")"
else
  START_TS="$(date +%s)"
fi
END_TS=$((START_TS + 24*60*60))
STATUS="$OUT/p1_sidecar/p1_sidecar_supervisor_status.md"
RESTART="$OUT/p1_sidecar/p1_sidecar_restart_log.csv"
if [ ! -f "$RESTART" ]; then
  echo "timestamp,event,exit_code,detail" > "$RESTART"
fi

cat > "$STATUS" <<STATUS
# P1 Sidecar Supervisor

- start_ts: $START_TS
- end_ts: $END_TS
- started_at: $(date -Is)
- purpose: auxiliary P0/P1 rerun, source recovery, dependency resolution, and blocker logging
STATUS

while [ "$(date +%s)" -lt "$END_TS" ]; do
  echo "P1_SIDECAR_SUPERVISOR_HEARTBEAT $(date -Is)" >> "$LOGDIR/heartbeat.log"
  timeout --preserve-status 4h /root/miniconda3/bin/python scripts/pegbench_24h_p1_sidecar_worker.py \
    --output-root "$OUT" --rounds 1 >> "$LOGDIR/p1_sidecar_stdout.log" 2>> "$LOGDIR/p1_sidecar_stderr.log"
  rc=$?
  echo "$(date -Is),worker_exit,$rc,continue_until_24h_or_full_completion" >> "$RESTART"
  rsync -a "$OUT/p1_sidecar/" "$OUT/p1_sidecar_latest/" >> "$LOGDIR/p1_sidecar_sync.log" 2>&1
  if [ "$(date +%s)" -lt "$END_TS" ]; then
    sleep 300
  fi
done

echo "p1_sidecar_supervisor_finished $(date -Is)" >> "$STATUS"
