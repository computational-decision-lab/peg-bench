#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODE="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

run_mode() {
  local mode="$1"
  local config=""
  local bundle_root=""
  local standardized_root=""
  local compare_root=""
  local methods=""

  case "${mode}" in
    time)
      config="configs/peg_bench_tardis_90d_time.yaml"
      bundle_root="/root/autodl-tmp/peg_bench_runs/tardis_90d_time_bundle"
      standardized_root="/root/autodl-tmp/peg_bench_runs/tardis_90d_time_standardized_inputs"
      compare_root="/root/autodl-tmp/peg_bench_runs/tardis_90d_time_table2_compare_v1"
      methods="ridge_tokens,histgbm_tokens,lightgbm_lob,kairos_64code,deeplob_timebars"
      ;;
    volume)
      config="configs/peg_bench_tardis_90d_volume.yaml"
      bundle_root="/root/autodl-tmp/peg_bench_runs/tardis_90d_volume_bundle"
      standardized_root="/root/autodl-tmp/peg_bench_runs/tardis_90d_volume_standardized_inputs"
      compare_root="/root/autodl-tmp/peg_bench_runs/tardis_90d_volume_table2_compare_v1"
      methods="ridge_tokens,histgbm_tokens,lightgbm_lob,kairos_64code"
      ;;
    *)
      echo "Unsupported mode: ${mode}" >&2
      return 1
      ;;
  esac

  echo "[$(date '+%F %T')] mode=${mode} start"

  if [[ ! -f "${bundle_root}/manifest.json" ]]; then
    echo "[$(date '+%F %T')] mode=${mode} preprocess"
    "${PYTHON_BIN}" scripts/preprocess.py --config "${config}"
  else
    echo "[$(date '+%F %T')] mode=${mode} preprocess skipped: ${bundle_root}/manifest.json exists"
  fi

  mkdir -p "${standardized_root}/panels" "${standardized_root}/manifests"
  if [[ ! -f "${standardized_root}/panels/${mode}_panel_summary.json" ]]; then
    echo "[$(date '+%F %T')] mode=${mode} build_panels"
    "${PYTHON_BIN}" scripts/build_peg_bench_panels.py \
      --manifest "${bundle_root}/manifest.json" \
      --mode "${mode}" \
      --output-root "${standardized_root}/panels"
  else
    echo "[$(date '+%F %T')] mode=${mode} build_panels skipped: summary exists"
  fi

  if [[ ! -f "${compare_root}/results_table.json" ]]; then
    echo "[$(date '+%F %T')] mode=${mode} run_comparison_baselines"
    "${PYTHON_BIN}" scripts/run_peg_bench_comparison_baselines.py \
      --standardized-root "${standardized_root}" \
      --bundle-manifest "${bundle_root}/manifest.json" \
      --output-root "${compare_root}" \
      --mode "${mode}" \
      --methods "${methods}" \
      --device cuda
  else
    echo "[$(date '+%F %T')] mode=${mode} comparison skipped: ${compare_root}/results_table.json exists"
  fi

  echo "[$(date '+%F %T')] mode=${mode} done"
}

case "${MODE}" in
  all)
    run_mode time
    run_mode volume
    ;;
  time|volume)
    run_mode "${MODE}"
    ;;
  *)
    echo "Usage: $0 [time|volume|all]" >&2
    exit 1
    ;;
esac
