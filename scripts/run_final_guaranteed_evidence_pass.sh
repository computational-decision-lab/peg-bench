#!/usr/bin/env bash
set +e

REPO="/root/prediction-execution-gap"
OUT="${REPO}/outputs/final_guaranteed_evidence_pass"
PY="/root/miniconda3/bin/python"
SCRIPT="${REPO}/scripts/final_guaranteed_evidence_pass.py"
TAR_PATH="${REPO}/outputs/final_guaranteed_evidence_pass.tar.gz"
SHA_PATH="${REPO}/outputs/final_guaranteed_evidence_pass.sha256"

mkdir -p "${OUT}/logs" "${OUT}/pids" "${OUT}/blocker_reports"

EXIT_CSV="${OUT}/task_exit_codes.csv"
PID_CSV="${OUT}/pids/task_pids.csv"
STATUS_MD="${OUT}/running_status.md"

echo "task,start_time,end_time,exit_code,log_path" > "${EXIT_CSV}"
echo "task,pid,start_time,log_path" > "${PID_CSV}"

write_status() {
  local task="$1"
  local state="$2"
  {
    echo "# Running Status"
    echo
    echo "- updated_at: $(date -Is)"
    echo "- current_task: ${task}"
    echo "- state: ${state}"
    echo "- output_dir: ${OUT}"
    echo
    echo "## Exit Codes"
    if [ -f "${EXIT_CSV}" ]; then
      cat "${EXIT_CSV}"
    fi
  } > "${STATUS_MD}"
}

write_runner_blocker() {
  local task="$1"
  local code="$2"
  local log_path="$3"
  local blocker="${OUT}/blocker_reports/${task}_runner_failure.md"
  {
    echo "# Runner-Level Blocker"
    echo
    echo "- task: ${task}"
    echo "- time: $(date -Is)"
    echo "- exit_code: ${code}"
    echo "- log_path: ${log_path}"
    echo
    echo "## Handling"
    echo "- This task failed or timed out under the 45 minute cap."
    echo "- The runner continued to the next task rather than stopping the full evidence pass."
    echo "- See the task log and any task-specific blocker report for source/path details."
  } > "${blocker}"
}

run_task() {
  local task="$1"
  local log_path="${OUT}/logs/${task}.log"
  local start_time
  local end_time
  local code
  start_time="$(date -Is)"
  write_status "${task}" "running"
  (
    cd "${REPO}" || exit 90
    timeout --preserve-status 45m "${PY}" "${SCRIPT}" --out "${OUT}" --task "${task}"
  ) > "${log_path}" 2>&1 &
  local pid=$!
  echo "${task},${pid},${start_time},${log_path}" >> "${PID_CSV}"
  wait "${pid}"
  code=$?
  end_time="$(date -Is)"
  echo "${task},${start_time},${end_time},${code},${log_path}" >> "${EXIT_CSV}"
  if [ "${code}" -ne 0 ]; then
    write_runner_blocker "${task}" "${code}" "${log_path}"
  fi
  write_status "${task}" "finished_exit_${code}"
  return 0
}

write_status "preflight" "starting"
run_task "preflight"

run_task "p0_1_main_results"
run_task "p0_2_gap_metrics"
run_task "p0_3_cost_ladder"
run_task "p0_4_bootstrap_ci"
run_task "p0_5_wrapper_matrix"
run_task "p0_6_formal_boundary"

run_task "p1_7_per_asset_venue"
run_task "p1_8_hft_audit"
run_task "p1_9_stability"

run_task "final_reports"

write_status "packaging" "running"
cd "${REPO}" || exit 91
if [ -f "${OUT}/sha256_manifest.txt" ]; then
  :
else
  find "${OUT}" -type f -print0 | sort -z | xargs -0 sha256sum > "${OUT}/sha256_manifest.txt"
fi
tar -czf "${TAR_PATH}" -C "${REPO}/outputs" "final_guaranteed_evidence_pass"
sha256sum "${TAR_PATH}" > "${SHA_PATH}"
write_status "packaging" "finished"

exit 0
