#!/usr/bin/env bash
set +e
cd /root/prediction-execution-gap || exit 0
OUT="outputs/pegbench_24h_unattended_v21"
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR"
START_TS=$(cat "$OUT/supervisor_start_ts" 2>/dev/null || date +%s)
END_TS=$((START_TS + 24*60*60))
while [ $(date +%s) -lt $END_TS ]; do
  {
    echo "DAEMON_5MIN_HEARTBEAT $(date -Is)"
    echo "status_board=$(cat $OUT/run_status_board.md 2>/dev/null | tr "\n" ";" | cut -c1-1000)"
    echo "procs=$(ps -ef | grep -E "pegbench_24h|run_reviewer_target|slice_and_bootstrap|hft_execution|rerun_motivating" | grep -v grep | tr "\n" ";" | cut -c1-2000)"
    echo "disk=$(df -h /root/autodl-tmp 2>/dev/null | tail -1 | tr -s " ")"
    echo "gpu=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | tr "\n" ";")"
    echo "---"
  } >> "$LOGDIR/heartbeat.log"
  sleep 300
done
