#!/usr/bin/env bash
set +e
cd /root/prediction-execution-gap || exit 0
MAIN_OUT="outputs/pegbench_24h_unattended_v21"
ACCEL_OUT="$MAIN_OUT/gpu_accelerator_v21"
LOGDIR="$ACCEL_OUT/logs"
mkdir -p "$ACCEL_OUT" "$LOGDIR" "$ACCEL_OUT/dpo" "$ACCEL_OUT/target_baselines" "$ACCEL_OUT/per_asset_venue_fast" "$ACCEL_OUT/motivating_slice_fast"
STATUS="$ACCEL_OUT/accelerator_status.md"
TASKS="$ACCEL_OUT/accelerator_task_exit_codes.csv"
if [ ! -f "$TASKS" ]; then echo "timestamp,name,gpu,exit_code,duration_sec,log" > "$TASKS"; fi
cat > "$STATUS" <<EOF
# PEG-Bench GPU Accelerator v21

- started_at: $(date -Is)
- purpose: use idle RTX6000 GPUs for parallel reduced reruns without stopping the main supervisor
- output: $ACCEL_OUT
EOF
run_bg() {
  name="$1"
  gpu="$2"
  limit="$3"
  cmd="$4"
  log="$LOGDIR/${name}.log"
  marker="$ACCEL_OUT/${name}.done"
  if [ -f "$marker" ]; then
    echo "$(date -Is),$name,$gpu,SKIP_DONE,0,$log" >> "$TASKS"
    return 0
  fi
  (
    start=$(date +%s)
    echo "### GPU_ACCEL_TASK $name START $(date -Is) GPU=$gpu LIMIT=$limit"
    echo "### CMD: $cmd"
    if [ "$gpu" = "cpu" ]; then
      timeout --preserve-status "$limit" bash -lc "$cmd"
    else
      CUDA_VISIBLE_DEVICES="$gpu" timeout --preserve-status "$limit" bash -lc "$cmd"
    fi
    rc=$?
    end=$(date +%s)
    dur=$((end-start))
    echo "### GPU_ACCEL_TASK $name END $(date -Is) RC=$rc DUR=$dur"
    echo "$(date -Is),$name,$gpu,$rc,$dur,$log" >> "$TASKS"
    touch "$marker"
  ) > "$log" 2>&1 &
  pid=$!
  echo "$name,$gpu,$pid,$log" >> "$ACCEL_OUT/active_jobs.csv"
}

: > "$ACCEL_OUT/active_jobs.csv"
echo "name,gpu,pid,log" > "$ACCEL_OUT/active_jobs.csv"

# Wave 1: use the idle GPUs for independent reduced DPO reruns. These write to separate dirs and do not interfere with the main P1 sidecar.
run_bg dpo_time_h20_seed20260506 2 90m "/root/miniconda3/bin/python scripts/run_peg_bench_dpo_stage.py --standardized-root /root/autodl-tmp/peg_bench_runs/tardis_90d_time_standardized_inputs --compare-root /root/autodl-tmp/peg_bench_runs/tardis_90d_time_table2_compare_v1 --output-root $ACCEL_OUT/dpo/time_h20_seed20260506 --mode time --seq-len 64 --stride 64 --horizon 64 --target-key return_20 --max-contexts 64 --dpo-epochs 1 --dpo-batch-size 16 --device cuda --preference-mode scalar --preference-template balanced --preference-gaps 0.0 --preference-aci-options false --backtest-progress-every 50000 --seed 20260506"
run_bg dpo_volume_h5_seed20260507 3 90m "/root/miniconda3/bin/python scripts/run_peg_bench_dpo_stage.py --standardized-root /root/autodl-tmp/peg_bench_runs/tardis_90d_volume_standardized_inputs --compare-root /root/autodl-tmp/peg_bench_runs/tardis_90d_volume_h5_table2_compare_v1 --output-root $ACCEL_OUT/dpo/volume_h5_seed20260507 --mode volume --seq-len 64 --stride 64 --horizon 64 --target-key return_5 --max-contexts 64 --dpo-epochs 1 --dpo-batch-size 16 --device cuda --preference-mode scalar --preference-template balanced --preference-gaps 0.0 --preference-aci-options false --backtest-progress-every 50000 --seed 20260507"
run_bg dpo_time_h5_seed20260508 4 90m "/root/miniconda3/bin/python scripts/run_peg_bench_dpo_stage.py --standardized-root /root/autodl-tmp/peg_bench_runs/tardis_90d_time_standardized_inputs --compare-root /root/autodl-tmp/peg_bench_runs/tardis_90d_time_h5_table2_compare_v1 --output-root $ACCEL_OUT/dpo/time_h5_seed20260508 --mode time --seq-len 64 --stride 64 --horizon 64 --target-key return_5 --max-contexts 64 --dpo-epochs 1 --dpo-batch-size 16 --device cuda --preference-mode scalar --preference-template balanced --preference-gaps 0.0 --preference-aci-options false --backtest-progress-every 50000 --seed 20260508"
run_bg dpo_volume_h50_seed20260509 5 90m "/root/miniconda3/bin/python scripts/run_peg_bench_dpo_stage.py --standardized-root /root/autodl-tmp/peg_bench_runs/tardis_90d_volume_standardized_inputs --compare-root /root/autodl-tmp/peg_bench_runs/tardis_90d_volume_h50_table2_compare_v1 --output-root $ACCEL_OUT/dpo/volume_h50_seed20260509 --mode volume --seq-len 64 --stride 64 --horizon 64 --target-key return_50 --max-contexts 64 --dpo-epochs 1 --dpo-batch-size 16 --device cuda --preference-mode scalar --preference-template balanced --preference-gaps 0.0 --preference-aci-options false --backtest-progress-every 50000 --seed 20260509"

# CPU-side quick reduced alternatives so full 1000-draw bottlenecks do not block having a usable supplemental artifact.
run_bg per_asset_venue_draws200 cpu 45m "python scripts/slice_and_bootstrap_tables.py --output-root $ACCEL_OUT/per_asset_venue_fast --draws 200; cp -f $ACCEL_OUT/per_asset_venue_fast/*.csv $MAIN_OUT/ 2>/dev/null"
run_bg motivating_slice_v8_fast 1 60m "python scripts/rerun_motivating_slice_official_lite_v8.py --output-root $ACCEL_OUT/motivating_slice_fast; cp -f $ACCEL_OUT/motivating_slice_fast/table*.csv $ACCEL_OUT/motivating_slice_fast/table*.md $ACCEL_OUT/motivating_slice_fast/table*.tex $MAIN_OUT/motivating_slice/ 2>/dev/null"

# Monitor wave 1.
while true; do
  alive=0
  now=$(date -Is)
  {
    echo "# PEG-Bench GPU Accelerator v21"
    echo
    echo "- updated_at: $now"
    echo "- active_jobs:"
    while IFS=, read -r name gpu pid log; do
      [ "$name" = "name" ] && continue
      if ps -p "$pid" >/dev/null 2>&1; then
        alive=$((alive+1))
        echo "  - $name gpu=$gpu pid=$pid running log=$log"
      else
        echo "  - $name gpu=$gpu pid=$pid finished log=$log"
      fi
    done < "$ACCEL_OUT/active_jobs.csv"
    echo
    echo "## GPU snapshot"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true
  } > "$STATUS"
  [ "$alive" -eq 0 ] && break
  sleep 60
done

# Wave 2: target-baseline lite reruns on the now-free GPUs, separate output dirs for comparison/fill if useful.
: > "$ACCEL_OUT/active_jobs_wave2.csv"
echo "name,gpu,pid,log" > "$ACCEL_OUT/active_jobs_wave2.csv"
ACTIVE_FILE="$ACCEL_OUT/active_jobs.csv"
mv "$ACTIVE_FILE" "$ACCEL_OUT/active_jobs_wave1.csv" 2>/dev/null
ln -sf active_jobs_wave2.csv "$ACTIVE_FILE" 2>/dev/null || true
panels=(time_h5 time_h20 time_h50 volume_h5 volume_h20 volume_h50)
gpus=(0 1 2 3 4 5)
i=0
for panel in "${panels[@]}"; do
  gpu="${gpus[$i]}"
  i=$((i+1))
  run_bg "target_lite_${panel}" "$gpu" 2h "/root/miniconda3/bin/python scripts/run_reviewer_target_baselines.py --panel $panel --output-root $ACCEL_OUT/target_baselines/$panel --methods tabl_lite,itransformer_lite --epochs 2 --batch-size 256 --max-train-windows 200000 --device cuda || /root/miniconda3/bin/python scripts/run_reviewer_target_baselines.py --panel $panel --output-root $ACCEL_OUT/target_baselines/$panel --methods tabl_lite,itransformer_lite --epochs 1 --batch-size 128 --max-train-windows 80000 --device cpu"
done

while true; do
  alive=0
  now=$(date -Is)
  {
    echo "# PEG-Bench GPU Accelerator v21"
    echo
    echo "- updated_at: $now"
    echo "- wave: target_lite"
    echo "- active_jobs:"
    while IFS=, read -r name gpu pid log; do
      [ "$name" = "name" ] && continue
      if ps -p "$pid" >/dev/null 2>&1; then
        alive=$((alive+1))
        echo "  - $name gpu=$gpu pid=$pid running log=$log"
      else
        echo "  - $name gpu=$gpu pid=$pid finished log=$log"
      fi
    done < "$ACCEL_OUT/active_jobs.csv"
    echo
    echo "## GPU snapshot"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true
  } > "$STATUS"
  [ "$alive" -eq 0 ] && break
  sleep 60
done

# Merge/summarize if available; keep this non-fatal.
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
