#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/idfsdata/yexuyan/OPD"
SCRIPT_PATH="${ROOT_DIR}/scripts/convert_all_iter_ckpts_opsd_answer_to_hf.sh"
LOG_DIR="${ROOT_DIR}/runlogs"
LOG_FILE="${LOG_DIR}/convert_opsd_answer_ckpts_to_hf_$(date -u +%Y%m%d_%H%M%S).log"

mkdir -p "${LOG_DIR}"

if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "Script not found: ${SCRIPT_PATH}" >&2
  exit 1
fi

echo "Launching checkpoint format conversion in background..."
echo "Script: ${SCRIPT_PATH}"
echo "Log: ${LOG_FILE}"

nohup bash "${SCRIPT_PATH}" > "${LOG_FILE}" 2>&1 &
PID=$!

echo "Started. PID=${PID}"
echo "Tail logs with:"
echo "  tail -f ${LOG_FILE}"
