#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/idfsdata/yexuyan/OPD"
SCRIPT_PATH="${ROOT_DIR}/scripts/eval_all_iter_ckpts_gpqa_diamond_gpu0123_conc256.sh"
LOG_DIR="${ROOT_DIR}/runlogs"
LOG_FILE="${LOG_DIR}/eval_gpqa_diamond_all_iter_gpu0123_conc256_$(date -u +%Y%m%d_%H%M%S).log"

mkdir -p "${LOG_DIR}"

if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "Script not found: ${SCRIPT_PATH}" >&2
  exit 1
fi

echo "Launching GPQA-Diamond all-ckpt evaluation in background..."
echo "Script: ${SCRIPT_PATH}"
echo "Log: ${LOG_FILE}"

nohup bash "${SCRIPT_PATH}" > "${LOG_FILE}" 2>&1 &
PID=$!

echo "Started. PID=${PID}"
echo "Tail logs with:"
echo "  tail -f ${LOG_FILE}"
