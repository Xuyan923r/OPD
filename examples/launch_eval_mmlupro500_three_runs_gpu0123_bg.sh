#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/idfsdata/yexuyan/OPD"
SCRIPT_PATH="${ROOT_DIR}/scripts/eval_mmlupro500_all_ckpts_three_runs_gpu0123.sh"
LOG_DIR="${ROOT_DIR}/runlogs"
LOG_FILE="${LOG_DIR}/eval_mmlupro500_three_runs_gpu0123_conc128_$(date -u +%Y%m%d_%H%M%S).log"

mkdir -p "${LOG_DIR}"

if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "Script not found: ${SCRIPT_PATH}" >&2
  exit 1
fi

echo "Launching MMLU-Pro-500 all-checkpoint eval for three run roots in background..."
echo "Script: ${SCRIPT_PATH}"
echo "Log: ${LOG_FILE}"

nohup env MAX_NUM_SEQS=128 bash "${SCRIPT_PATH}" > "${LOG_FILE}" 2>&1 &
PID=$!

echo "Started. PID=${PID}"
echo "Tail logs with:"
echo "  tail -f ${LOG_FILE}"
