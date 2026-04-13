#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/idfsdata/yexuyan/OPD"
PYTHON_BIN="${PYTHON_BIN:-/idfsdata/yexuyan/conda_envs/verl/bin/python}"

MODEL_PATH="${MODEL_PATH:-/idfsdata/yexuyan/OPD/models/Qwen3-1.7B}"
DATASET_PATH="${DATASET_PATH:-/idfsdata/yexuyan/OPD/MMLU-Pro-full-opsd-answer-only.jsonl}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"

TP_SIZE="${TP_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-42}"

RUN_ID="mmlupro_full_vllm_tp${TP_SIZE}_gpu${CUDA_DEVICES//,/}_$(date -u +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/runlogs"
OUT_DIR="${ROOT_DIR}/evaluation"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"
OUTPUT_FILE="${OUT_DIR}/${RUN_ID}_outputs.json"
SUMMARY_FILE="${OUT_DIR}/${RUN_ID}_summary.json"

mkdir -p "${LOG_DIR}" "${OUT_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model path not found: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Dataset not found: ${DATASET_PATH}" >&2
  echo "Try downloading first (network required):" >&2
  echo "  ${PYTHON_BIN} ${ROOT_DIR}/scripts/download_mmlupro_full.py --output ${DATASET_PATH}" >&2
  exit 1
fi

echo "Launching background MMLU-Pro eval..."
echo "RUN_ID: ${RUN_ID}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_DEVICES}"
echo "Model: ${MODEL_PATH}"
echo "Dataset: ${DATASET_PATH}"
echo "Log: ${LOG_FILE}"

nohup env \
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/eval_mmlupro_vllm.py" \
    --model_path "${MODEL_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --output_file "${OUTPUT_FILE}" \
    --summary_file "${SUMMARY_FILE}" \
    --tensor_parallel_size "${TP_SIZE}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --max_num_seqs "${MAX_NUM_SEQS}" \
    --batch_size "${BATCH_SIZE}" \
    --max_tokens "${MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --seed "${SEED}" \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "Started. PID=${PID}"
echo "Tail logs with:"
echo "  tail -f ${LOG_FILE}"
