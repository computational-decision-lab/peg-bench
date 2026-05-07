#!/usr/bin/env bash
set +e
cd /root/prediction-execution-gap || exit 0
OUT="outputs/pegbench_24h_unattended_v21"
LOGDIR="$OUT/task_logs"
mkdir -p "$OUT/target_baselines_parallel" "$LOGDIR"
echo "parallel_gpu_jobs_start,$(date -Is)" >> "$OUT/parallel_gpu_jobs_status.csv"
PANELS=(time_h5 time_h20 time_h50 volume_h5 volume_h20 volume_h50)
GPUS=(0 1 2 3 4 5)
for i in ${!PANELS[@]}; do
  panel=${PANELS[$i]}; gpu=${GPUS[$i]}
  (
    set +e
    echo "START panel=$panel gpu=$gpu $(date -Is)"
    timeout --preserve-status 20h bash -lc "CUDA_VISIBLE_DEVICES=$gpu python scripts/run_reviewer_target_baselines.py --panel $panel --output-root $OUT/target_baselines_parallel/$panel --methods elasticnet_tokens,tabl_lite,itransformer_lite --epochs 3 --batch-size 256 --max-train-windows 250000 --device cuda"
    rc=$?
    echo "END panel=$panel gpu=$gpu rc=$rc $(date -Is)"
    if [ "$rc" != "0" ]; then
      echo "$panel cuda_failed rc=$rc; retry cpu/light $(date -Is)"
      timeout --preserve-status 6h bash -lc "python scripts/run_reviewer_target_baselines.py --panel $panel --output-root $OUT/target_baselines_parallel/$panel --methods elasticnet_tokens,tabl_lite,itransformer_lite --epochs 1 --batch-size 128 --max-train-windows 80000 --device cpu"
      echo "CPU_RETRY panel=$panel rc=$? $(date -Is)"
    fi
  ) > "$LOGDIR/parallel_target_${panel}.log" 2>&1 &
  echo "$panel,$gpu,$!,started,$(date -Is)" >> "$OUT/parallel_gpu_jobs_status.csv"
done
wait
echo "parallel_gpu_jobs_done,$(date -Is)" >> "$OUT/parallel_gpu_jobs_status.csv"
python scripts/pegbench_24h_merge_targets.py --target-root "$OUT/target_baselines_parallel" --output "$OUT/target_baseline_summary_parallel.csv" >> "$LOGDIR/parallel_target_merge.log" 2>&1
