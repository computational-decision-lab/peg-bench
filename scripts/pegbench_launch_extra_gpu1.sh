#!/usr/bin/env bash
set +e
cd /root/prediction-execution-gap || exit 0
MAIN_OUT="outputs/pegbench_24h_unattended_v21"
ACCEL_OUT="$MAIN_OUT/gpu_accelerator_v22"
LOGDIR="$ACCEL_OUT/logs"
mkdir -p "$ACCEL_OUT/dpo/time_h50_seed20260510" "$LOGDIR"
TASKS="$ACCEL_OUT/accelerator_task_exit_codes.csv"
ACTIVE="$ACCEL_OUT/active_jobs.csv"
name="dpo_time_h50_seed20260510_extra_gpu1"
log="$LOGDIR/wave1_extra_${name}.log"
marker="$ACCEL_OUT/wave1_extra_${name}.done"
if [ -f "$marker" ]; then
  echo "EXTRA_ALREADY_DONE"
  exit 0
fi
if ps -ef | grep "$name" | grep -v grep >/dev/null; then
  echo "EXTRA_ALREADY_RUNNING"
  ps -ef | grep "$name" | grep -v grep
  exit 0
fi
(
  start=$(date +%s)
  echo "### GPU_ACCEL_EXTRA $name START $(date -Is) GPU=1"
  CUDA_VISIBLE_DEVICES=1 timeout --preserve-status 95m /root/miniconda3/bin/python scripts/run_peg_bench_dpo_stage.py \
    --standardized-root /root/autodl-tmp/peg_bench_runs/tardis_90d_time_standardized_inputs \
    --compare-root /root/autodl-tmp/peg_bench_runs/tardis_90d_time_h50_table2_compare_v1 \
    --output-root "$ACCEL_OUT/dpo/time_h50_seed20260510" \
    --mode time --seq-len 64 --stride 64 --horizon 64 --target-key return_50 \
    --max-contexts 64 --dpo-epochs 1 --dpo-batch-size 16 --device cuda \
    --preference-mode scalar --preference-template balanced --preference-gaps 0.0 \
    --preference-aci-options false --backtest-progress-every 50000 --seed 20260510
  rc=$?
  dur=$(($(date +%s)-start))
  echo "### GPU_ACCEL_EXTRA $name END $(date -Is) RC=$rc DUR=$dur"
  echo "$(date -Is),wave1_extra,$name,1,$rc,$dur,$log" >> "$TASKS"
  touch "$marker"
) > "$log" 2>&1 &
pid=$!
echo "$name,1,$pid,$log" >> "$ACTIVE"
echo "EXTRA_STARTED=$pid"
