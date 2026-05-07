#!/usr/bin/env bash
set +e
cd /root/prediction-execution-gap || exit 2
mkdir -p scripts outputs/pegbench_24h_unattended_v21 logs
OUT="/root/prediction-execution-gap/outputs/pegbench_24h_unattended_v21"
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR" "$OUT/task_logs" "$OUT/copied_evidence" "$OUT/generated_figures" "$OUT/paper_tables" "$OUT/status_snapshots"
cat > scripts/run_pegbench_24h_supervisor.sh <<'EOF'
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
EOF
chmod +x scripts/run_pegbench_24h_supervisor.sh

cat > scripts/run_pegbench_24h_worker.sh <<'EOF'
#!/usr/bin/env bash
set +e
cd /root/prediction-execution-gap || exit 2
OUT="outputs/pegbench_24h_unattended_v21"
LOGDIR="$OUT/logs"
TASKLOG="$OUT/task_logs"
mkdir -p "$OUT" "$LOGDIR" "$TASKLOG" "$OUT/copied_evidence" "$OUT/generated_figures" "$OUT/paper_tables" "$OUT/status_snapshots" "$OUT/blockers" "$OUT/target_baselines"
if [ ! -f "$OUT/task_exit_codes.csv" ]; then echo "timestamp,round,task,exit_code,duration_sec,output" > "$OUT/task_exit_codes.csv"; fi
if [ ! -f "$OUT/blocker_resolution_board.csv" ]; then echo "blocker_id,task,failure_type,exact_error,attempted_fix_1,result_1,attempted_fix_2,result_2,attempted_fix_3,result_3,web_search_attempted,final_status,output_file,next_retry_time" > "$OUT/blocker_resolution_board.csv"; fi
if [ ! -f "$OUT/web_sources_checked.csv" ]; then echo "timestamp,task,query_or_url,result" > "$OUT/web_sources_checked.csv"; fi
if [ ! -f "$OUT/lock_resolution_log.md" ]; then echo "# Lock Resolution Log" > "$OUT/lock_resolution_log.md"; fi
if [ ! -f "$OUT/blocker_resolution_log.md" ]; then echo "# Blocker Resolution Log" > "$OUT/blocker_resolution_log.md"; fi
if [ ! -f "$OUT/web_dependency_resolution_log.md" ]; then echo "# Web Dependency Resolution Log" > "$OUT/web_dependency_resolution_log.md"; fi
round_file="$OUT/worker_round"
ROUND=1
if [ -f "$round_file" ]; then ROUND=$(( $(cat "$round_file") + 1 )); fi
echo "$ROUND" > "$round_file"
heartbeat() {
  local task="$1"
  {
    echo "HEARTBEAT $(date -Is)"
    echo "current_task=$task"
    echo "round=$ROUND"
    echo "active_pid=$$"
    echo "disk_usage=$(df -h /root/autodl-tmp 2>/dev/null | tail -1 | tr -s ' ')"
    echo "memory_usage=$(free -h 2>/dev/null | tr '\n' ';')"
    echo "gpu=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | tr '\n' ';')"
    echo "blockers=$(($(wc -l < "$OUT/blocker_resolution_board.csv" 2>/dev/null || echo 1)-1))"
    echo "---"
  } >> "$LOGDIR/heartbeat.log"
  cat > "$OUT/run_status_board.md" <<STATUS
# Run Status Board

- timestamp: $(date -Is)
- round: $ROUND
- current_task: $task
- host: $(hostname)
- task_exit_rows: $(wc -l < "$OUT/task_exit_codes.csv" 2>/dev/null)
- blocker_rows: $(wc -l < "$OUT/blocker_resolution_board.csv" 2>/dev/null)
- output_dir: $OUT
STATUS
}
record_task() {
  local task="$1" rc="$2" start="$3" output="$4"
  local end=$(date +%s)
  echo "$(date -Is),$ROUND,$task,$rc,$((end-start)),$output" >> "$OUT/task_exit_codes.csv"
  if [ "$rc" != "0" ]; then
    local bid="${task}_round${ROUND}_$(date +%s)"
    echo "$bid,$task,task_nonzero_exit,exit_code_$rc,inspect_log,$output,retry_or_alternate,next_round,continue_other_tasks,done,no,failed_pending_retry,$output,$(date -Is)" >> "$OUT/blocker_resolution_board.csv"
    echo "## $bid" >> "$OUT/blocker_resolution_log.md"
    echo "- task: $task" >> "$OUT/blocker_resolution_log.md"
    echo "- exit_code: $rc" >> "$OUT/blocker_resolution_log.md"
    echo "- log: $output" >> "$OUT/blocker_resolution_log.md"
  fi
}
run_task() {
  local task="$1" timeout_s="$2" cmd="$3"
  heartbeat "$task"
  local start=$(date +%s)
  local log="$TASKLOG/${ROUND}_${task}.log"
  echo "### TASK $task START $(date -Is)" >> "$log"
  timeout --preserve-status "$timeout_s" bash -lc "$cmd" >> "$log" 2>&1
  local rc=$?
  echo "### TASK $task END $(date -Is) rc=$rc" >> "$log"
  record_task "$task" "$rc" "$start" "$log"
  heartbeat "after_$task"
  return 0
}
resolve_locks() {
  heartbeat "resolve_locks"
  find . /root/autodl-tmp/peg_bench_runs -name '*.lock' -o -name 'run.lock' -o -name 'output.lock' -o -name 'cache.lock' 2>/dev/null | while read -r lock; do
    [ -e "$lock" ] || continue
    mtime=$(stat -c %Y "$lock" 2>/dev/null || echo 0)
    age=$(( $(date +%s) - mtime ))
    pid=$(grep -Eo '[0-9]+' "$lock" 2>/dev/null | head -1)
    echo "- $(date -Is) lock=$lock age=$age pid=$pid" >> "$OUT/lock_resolution_log.md"
    if [ -n "$pid" ]; then
      ps -p "$pid" >/dev/null 2>&1
      alive=$?
      if [ "$alive" != "0" ]; then mv "$lock" "$lock.stale.$(date +%s)"; echo "  moved stale pid lock" >> "$OUT/lock_resolution_log.md"; fi
    elif [ "$age" -gt 1800 ]; then
      mv "$lock" "$lock.stale.$(date +%s)"
      echo "  moved stale mtime lock" >> "$OUT/lock_resolution_log.md"
    fi
  done
}
copy_existing_evidence() {
  mkdir -p "$OUT/copied_evidence" "$OUT/benchmark_reports"
  cp -f outputs/final_acceptance_v3/cost_ladder_*_90d.csv "$OUT/copied_evidence/" 2>/dev/null
  cp -f outputs/final_acceptance_v3/table3_gap_day_bootstrap_summary.csv "$OUT/copied_evidence/" 2>/dev/null
  cp -f outputs/final_acceptance_v3/panel_bootstrap_table18.csv "$OUT/copied_evidence/" 2>/dev/null
  cp -f outputs/final_acceptance_v3/wrapper_matrix*.csv outputs/final_acceptance_v3/wrapper_delta_summary.csv outputs/final_acceptance_v3/wrapper_audit_final.csv "$OUT/copied_evidence/" 2>/dev/null
  find /root/autodl-tmp/peg_bench_runs -path '*benchmark_report.json' | grep -E 'tardis_90d_(time|volume)|protocol_clean_volume_h20' | head -80 > "$OUT/benchmark_report_paths.txt"
  while read -r p; do [ -f "$p" ] && cp -f "$p" "$OUT/benchmark_reports/$(echo "$p" | sed 's#[/:]#_#g').json"; done < "$OUT/benchmark_report_paths.txt"
}
run_task A_main_tex_consistency 45m "python scripts/pegbench_24h_generate_consistency.py --output-root $OUT"
resolve_locks
run_task B_fast_cost_ladder 45m "python scripts/fast_reviewer_tables.py --output-root $OUT/fast_tables && cp -f $OUT/fast_tables/*.csv $OUT/ 2>/dev/null"
run_task C_fast_completion_bootstrap 90m "python scripts/reviewer_fast_completion.py --output-root $OUT/fast_completion --draws 1000 && cp -f $OUT/fast_completion/*.csv $OUT/ 2>/dev/null"
run_task D_slice_bootstrap_per_asset 90m "python scripts/slice_and_bootstrap_tables.py --output-root $OUT/per_asset_venue --draws 1000 && cp -f $OUT/per_asset_venue/*.csv $OUT/ 2>/dev/null"
run_task E_hft_audit 90m "python scripts/hft_execution_audit.py --output-root $OUT/hft_audit && cp -f $OUT/hft_audit/*.csv $OUT/ 2>/dev/null"
run_task H_motivating_slice_rerun 90m "python scripts/rerun_motivating_slice_tables_6_7_14_15.py --output-root $OUT/motivating_slice 2>/dev/null || python scripts/rerun_motivating_slice_tables_6_7_14_15.py; cp -f outputs/final_motivating_slice_rerun_v8/table*.csv outputs/final_motivating_slice_rerun_v8/table*.md outputs/final_motivating_slice_rerun_v8/table*.tex $OUT/motivating_slice/ 2>/dev/null"
# Target baselines, round-robin, one panel per task so failures do not block.
for panel in time_h5 time_h20 time_h50 volume_h5 volume_h20 volume_h50; do
  run_task "F_target_elasticnet_$panel" 90m "CUDA_VISIBLE_DEVICES=1 python scripts/run_reviewer_target_baselines.py --panel $panel --output-root $OUT/target_baselines/$panel --methods elasticnet_tokens --epochs 1 --max-train-windows 120000 --device cuda || python scripts/run_reviewer_target_baselines.py --panel $panel --output-root $OUT/target_baselines/$panel --methods elasticnet_tokens --epochs 1 --max-train-windows 120000 --device cpu"
done
for panel in time_h5 time_h20 time_h50 volume_h5 volume_h20 volume_h50; do
  gpu=$(( (ROUND + ${#panel}) % 6 ))
  run_task "F_target_lite_$panel" 3h "CUDA_VISIBLE_DEVICES=$gpu python scripts/run_reviewer_target_baselines.py --panel $panel --output-root $OUT/target_baselines/$panel --methods tabl_lite,itransformer_lite --epochs 2 --batch-size 256 --max-train-windows 200000 --device cuda || python scripts/run_reviewer_target_baselines.py --panel $panel --output-root $OUT/target_baselines/$panel --methods tabl_lite,itransformer_lite --epochs 1 --batch-size 128 --max-train-windows 80000 --device cpu"
done
run_task F_merge_target_baselines 30m "find $OUT/target_baselines -name results_table.csv -print0 | xargs -0 -I{} sh -c 'head -1 {} | sed -n 1p; break' > $OUT/target_baseline_summary_updated.csv.tmp 2>/dev/null; python scripts/pegbench_24h_merge_targets.py --target-root $OUT/target_baselines --output $OUT/target_baseline_summary_updated.csv"
run_task F_dependency_attempts 2h "python scripts/pegbench_24h_dependency_attempts.py --output-root $OUT"
run_task G_wrapper_completion 2h "python scripts/pegbench_24h_wrapper_completion.py --output-root $OUT"
run_task Z_generate_figures 45m "python scripts/generate_result_figures.py --release artifact_snapshot/release --out-dir $OUT/generated_figures 2>/dev/null || python scripts/generate_result_figures.py"
run_task Z_final_report 30m "python scripts/pegbench_24h_generate_consistency.py --output-root $OUT; bash scripts/run_pegbench_24h_final_report.sh; bash scripts/run_pegbench_24h_sync_bundle.sh"
copy_existing_evidence
bash scripts/run_pegbench_24h_sync_bundle.sh >> "$LOGDIR/worker_sync_bundle.log" 2>&1
exit 0
EOF
chmod +x scripts/run_pegbench_24h_worker.sh

cat > scripts/run_pegbench_24h_sync_bundle.sh <<'EOF'
#!/usr/bin/env bash
set +e
cd /root/prediction-execution-gap || exit 0
OUT="outputs/pegbench_24h_unattended_v21"
mkdir -p "$OUT"
find "$OUT" -type f ! -name sha256_manifest.txt -print0 | xargs -0 sha256sum > "$OUT/sha256_manifest.txt" 2>/dev/null
( cd outputs && tar -czf pegbench_24h_unattended_v21_latest.tar.gz pegbench_24h_unattended_v21 ) 2>/dev/null
EOF
chmod +x scripts/run_pegbench_24h_sync_bundle.sh

cat > scripts/run_pegbench_24h_final_verify.sh <<'EOF'
#!/usr/bin/env bash
set +e
OUT="outputs/pegbench_24h_unattended_v21"
missing=0
for f in final_24h_completion_report.md final_remaining_blockers.csv pdf_fill_values_after_24h.csv sha256_manifest.txt; do
  [ -s "$OUT/$f" ] || missing=$((missing+1))
done
exit $missing
EOF
chmod +x scripts/run_pegbench_24h_final_verify.sh

cat > scripts/run_pegbench_24h_final_report.sh <<'EOF'
#!/usr/bin/env bash
set +e
cd /root/prediction-execution-gap || exit 0
OUT="outputs/pegbench_24h_unattended_v21"
mkdir -p "$OUT"
python scripts/pegbench_24h_final_report.py --output-root "$OUT" >> "$OUT/logs/final_report_py.log" 2>&1
find "$OUT" -type f ! -name sha256_manifest.txt -print0 | xargs -0 sha256sum > "$OUT/sha256_manifest.txt" 2>/dev/null
EOF
chmod +x scripts/run_pegbench_24h_final_report.sh
