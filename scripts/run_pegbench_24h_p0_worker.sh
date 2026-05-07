#!/usr/bin/env bash
set +e
cd /root/prediction-execution-gap || exit 0
OUT="outputs/pegbench_24h_unattended_v21"
mkdir -p "$OUT/task_logs"
START=$(cat "$OUT/supervisor_start_ts" 2>/dev/null || date +%s)
END=$((START + 24*60*60))
round=0
while [ $(date +%s) -lt $END ]; do
  round=$((round+1))
  echo "P0_HEARTBEAT round=$round $(date -Is)" >> "$OUT/logs/heartbeat.log"
  timeout --preserve-status 3h python scripts/pegbench_24h_p0_numeric_source_recovery.py >> "$OUT/task_logs/p0_numeric_source_recovery_round${round}.log" 2>&1
  rc=$?
  echo "$(date -Is),p0_numeric_source_recovery_round${round},$rc" >> "$OUT/p0_task_exit_codes.csv"
  bash scripts/run_pegbench_24h_sync_bundle.sh >> "$OUT/logs/p0_sync.log" 2>&1
  # If all P0 decisions are no longer source-missing, sleep longer but continue checking.
  if [ -f "$OUT/p0_pdf_fill_decisions.csv" ] && ! grep -q "source_missing_must_caveat_or_remove_numeric_claim" "$OUT/p0_pdf_fill_decisions.csv"; then
    echo "P0_ALL_HAVE_DECISION_OR_SOURCE $(date -Is)" >> "$OUT/logs/heartbeat.log"
    sleep 1800
  else
    sleep 900
  fi
done
