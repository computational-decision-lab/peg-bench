#!/usr/bin/env bash
set -euo pipefail
while [ ! -f "/root/autodl-tmp/peg_bench_runs/tardis_30d_volume_dpo_multiseed_fair_v2/multiseed_summary.json" ]; do
  sleep 300
done
cd /root/prediction-execution-gap
/root/miniconda3/bin/python scripts/finalize_30d_multiseed.py   --time-multiseed-root "/root/autodl-tmp/peg_bench_runs/tardis_30d_time_dpo_multiseed_fair_v2"   --volume-multiseed-root "/root/autodl-tmp/peg_bench_runs/tardis_30d_volume_dpo_multiseed_fair_v2"   --output-dir "/root/prediction-execution-gap/outputs/final_30d_multiseed_fair"
