#!/usr/bin/env bash
set +e

cd /root/prediction-execution-gap || exit 0

MAIN_OUT="outputs/pegbench_24h_unattended_v21"
ACCEL_OUT="$MAIN_OUT/gpu_accelerator_v22"
LOGDIR="$ACCEL_OUT/logs"
mkdir -p "$ACCEL_OUT" "$LOGDIR" "$ACCEL_OUT/dpo" "$ACCEL_OUT/target_baselines" "$ACCEL_OUT/per_asset_venue_fast" "$ACCEL_OUT/motivating_slice_fast"

STATUS="$ACCEL_OUT/accelerator_status.md"
TASKS="$ACCEL_OUT/accelerator_task_exit_codes.csv"
ACTIVE="$ACCEL_OUT/active_jobs.csv"

if [ ! -f "$TASKS" ]; then
  echo "timestamp,wave,name,gpu,exit_code,duration_sec,log" > "$TASKS"
fi

write_status() {
  local wave="$1"
  {
    echo "# PEG-Bench GPU Accelerator v22"
    echo
    echo "- updated_at: $(date -Is)"
    echo "- wave: $wave"
    echo "- note: supplemental accelerator; main 24h supervisor remains authoritative"
    echo
    echo "## Active Jobs"
    if [ -f "$ACTIVE" ]; then
      while IFS=, read -r name gpu pid log; do
        [ "$name" = "name" ] && continue
        if ps -p "$pid" >/dev/null 2>&1; then
          echo "- $name gpu=$gpu pid=$pid running log=$log"
        else
          echo "- $name gpu=$gpu pid=$pid finished log=$log"
        fi
      done < "$ACTIVE"
    fi
    echo
    echo "## GPU Snapshot"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true
    echo
    echo "## Recent Task Results"
    tail -n 30 "$TASKS" 2>/dev/null || true
  } > "$STATUS"
}

run_bg() {
  local wave="$1"
  local name="$2"
  local gpu="$3"
  local limit="$4"
  local cmd="$5"
  local log="$LOGDIR/${wave}_${name}.log"
  local marker="$ACCEL_OUT/${wave}_${name}.done"

  if [ -f "$marker" ]; then
    echo "$(date -Is),$wave,$name,$gpu,SKIP_DONE,0,$log" >> "$TASKS"
    return 0
  fi

  (
    local start end dur rc
    start=$(date +%s)
    echo "### GPU_ACCEL_TASK $wave/$name START $(date -Is) GPU=$gpu LIMIT=$limit"
    echo "### CMD: $cmd"
    if [ "$gpu" = "cpu" ]; then
      timeout --preserve-status "$limit" bash -lc "$cmd"
    else
      CUDA_VISIBLE_DEVICES="$gpu" timeout --preserve-status "$limit" bash -lc "$cmd"
    fi
    rc=$?
    end=$(date +%s)
    dur=$((end-start))
    echo "### GPU_ACCEL_TASK $wave/$name END $(date -Is) RC=$rc DUR=$dur"
    echo "$(date -Is),$wave,$name,$gpu,$rc,$dur,$log" >> "$TASKS"
    touch "$marker"
  ) > "$log" 2>&1 &
  local pid=$!
  echo "$name,$gpu,$pid,$log" >> "$ACTIVE"
}

wait_wave() {
  local wave="$1"
  while true; do
    local alive=0
    if [ -f "$ACTIVE" ]; then
      while IFS=, read -r name gpu pid log; do
        [ "$name" = "name" ] && continue
        if ps -p "$pid" >/dev/null 2>&1; then
          alive=$((alive+1))
        fi
      done < "$ACTIVE"
    fi
    write_status "$wave"
    [ "$alive" -eq 0 ] && break
    sleep 60
  done
}

start_wave() {
  : > "$ACTIVE"
  echo "name,gpu,pid,log" > "$ACTIVE"
}

cat > "$STATUS" <<EOF
# PEG-Bench GPU Accelerator v22

- started_at: $(date -Is)
- output: $ACCEL_OUT
- goal: keep idle RTX6000 GPUs busy with independent reduced reruns and supplemental fills
EOF

# Wave 1: fill GPU2-5 immediately with independent DPO reduced reruns.
# GPU0 is normally occupied by the main/P1 DPO loop; GPU1 is often used by the main target-baseline loop.
start_wave
run_bg wave1 dpo_time_h20_seed20260506 2 95m "/root/miniconda3/bin/python scripts/run_peg_bench_dpo_stage.py --standardized-root /root/autodl-tmp/peg_bench_runs/tardis_90d_time_standardized_inputs --compare-root /root/autodl-tmp/peg_bench_runs/tardis_90d_time_table2_compare_v1 --output-root $ACCEL_OUT/dpo/time_h20_seed20260506 --mode time --seq-len 64 --stride 64 --horizon 64 --target-key return_20 --max-contexts 64 --dpo-epochs 1 --dpo-batch-size 16 --device cuda --preference-mode scalar --preference-template balanced --preference-gaps 0.0 --preference-aci-options false --backtest-progress-every 50000 --seed 20260506"
run_bg wave1 dpo_volume_h5_seed20260507 3 95m "/root/miniconda3/bin/python scripts/run_peg_bench_dpo_stage.py --standardized-root /root/autodl-tmp/peg_bench_runs/tardis_90d_volume_standardized_inputs --compare-root /root/autodl-tmp/peg_bench_runs/tardis_90d_volume_h5_table2_compare_v1 --output-root $ACCEL_OUT/dpo/volume_h5_seed20260507 --mode volume --seq-len 64 --stride 64 --horizon 64 --target-key return_5 --max-contexts 64 --dpo-epochs 1 --dpo-batch-size 16 --device cuda --preference-mode scalar --preference-template balanced --preference-gaps 0.0 --preference-aci-options false --backtest-progress-every 50000 --seed 20260507"
run_bg wave1 dpo_time_h5_seed20260508 4 95m "/root/miniconda3/bin/python scripts/run_peg_bench_dpo_stage.py --standardized-root /root/autodl-tmp/peg_bench_runs/tardis_90d_time_standardized_inputs --compare-root /root/autodl-tmp/peg_bench_runs/tardis_90d_time_h5_table2_compare_v1 --output-root $ACCEL_OUT/dpo/time_h5_seed20260508 --mode time --seq-len 64 --stride 64 --horizon 64 --target-key return_5 --max-contexts 64 --dpo-epochs 1 --dpo-batch-size 16 --device cuda --preference-mode scalar --preference-template balanced --preference-gaps 0.0 --preference-aci-options false --backtest-progress-every 50000 --seed 20260508"
run_bg wave1 dpo_volume_h50_seed20260509 5 95m "/root/miniconda3/bin/python scripts/run_peg_bench_dpo_stage.py --standardized-root /root/autodl-tmp/peg_bench_runs/tardis_90d_volume_standardized_inputs --compare-root /root/autodl-tmp/peg_bench_runs/tardis_90d_volume_h50_table2_compare_v1 --output-root $ACCEL_OUT/dpo/volume_h50_seed20260509 --mode volume --seq-len 64 --stride 64 --horizon 64 --target-key return_50 --max-contexts 64 --dpo-epochs 1 --dpo-batch-size 16 --device cuda --preference-mode scalar --preference-template balanced --preference-gaps 0.0 --preference-aci-options false --backtest-progress-every 50000 --seed 20260509"
run_bg wave1 per_asset_venue_draws200 cpu 45m "python scripts/slice_and_bootstrap_tables.py --output-root $ACCEL_OUT/per_asset_venue_fast --draws 200; cp -f $ACCEL_OUT/per_asset_venue_fast/*.csv $MAIN_OUT/ 2>/dev/null"
run_bg wave1 motivating_slice_v8_fast 1 70m "python scripts/rerun_motivating_slice_official_lite_v8.py --output-root $ACCEL_OUT/motivating_slice_fast; mkdir -p $MAIN_OUT/motivating_slice; cp -f $ACCEL_OUT/motivating_slice_fast/table*.csv $ACCEL_OUT/motivating_slice_fast/table*.md $ACCEL_OUT/motivating_slice_fast/table*.tex $MAIN_OUT/motivating_slice/ 2>/dev/null"
wait_wave wave1

# Wave 2: use all six cards for target-lite reruns in independent dirs.
start_wave
panels=(time_h5 time_h20 time_h50 volume_h5 volume_h20 volume_h50)
gpus=(0 1 2 3 4 5)
i=0
for panel in "${panels[@]}"; do
  gpu="${gpus[$i]}"
  i=$((i+1))
  run_bg wave2 "target_lite_${panel}" "$gpu" 2h "/root/miniconda3/bin/python scripts/run_reviewer_target_baselines.py --panel $panel --output-root $ACCEL_OUT/target_baselines/$panel --methods tabl_lite,itransformer_lite --epochs 2 --batch-size 256 --max-train-windows 200000 --device cuda || /root/miniconda3/bin/python scripts/run_reviewer_target_baselines.py --panel $panel --output-root $ACCEL_OUT/target_baselines/$panel --methods tabl_lite,itransformer_lite --epochs 1 --batch-size 128 --max-train-windows 80000 --device cpu"
done
wait_wave wave2

# Final non-fatal merge/sync.
/root/miniconda3/bin/python scripts/pegbench_24h_merge_targets.py --output-root "$MAIN_OUT" >> "$LOGDIR/merge_targets.log" 2>&1 || true
bash scripts/run_pegbench_24h_final_report.sh >> "$LOGDIR/final_report.log" 2>&1 || true
bash scripts/run_pegbench_24h_sync_bundle.sh >> "$LOGDIR/sync_bundle.log" 2>&1 || true

{
  echo "# GPU Accelerator Completion"
  echo
  echo "- completed_at: $(date -Is)"
  echo "- task_log: $TASKS"
  echo "- note: all outputs are supplemental/accelerated unless promoted by final verification."
} > "$ACCEL_OUT/gpu_accelerator_completion_report.md"
write_status completed
