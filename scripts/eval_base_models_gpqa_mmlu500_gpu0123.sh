#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/idfsdata/yexuyan/OPD"
PYTHON_EVAL="/idfsdata/yexuyan/conda_envs/verl/bin/python"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

FINAL_RESULTS_JSONL="${FINAL_RESULTS_JSONL:-/idfsdata/yexuyan/OPD/FinalResults-GR.jsonl}"
OUT_DIR="${OUT_DIR:-/idfsdata/yexuyan/OPD/evaluation/base_model_eval_results}"

CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-4}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
RESCUE_MAX_NEW_TOKENS="${RESCUE_MAX_NEW_TOKENS:-4}"

GPQA_DATASET="${GPQA_DATASET:-/idfsdata/yexuyan/OPD/evaluation/opsd_import/data/gpqa_diamond.jsonl}"
MMLU500_DATASET="${MMLU500_DATASET:-/idfsdata/yexuyan/OPD/MMLU-Pro-test-500.jsonl}"

MODELS=(
  "/idfsdata/yexuyan/OPD/models/Qwen3-1.7B"
  "/idfsdata/yexuyan/OPD/models/Qwen3-8B"
)

mkdir -p "${OUT_DIR}" "$(dirname "${FINAL_RESULTS_JSONL}")" "/idfsdata/yexuyan/OPD/.cache/flashinfer"

if [[ ! -f "${GPQA_DATASET}" ]]; then
  echo "GPQA dataset not found: ${GPQA_DATASET}" >&2
  exit 1
fi
if [[ ! -f "${MMLU500_DATASET}" ]]; then
  echo "MMLU-Pro-500 dataset not found: ${MMLU500_DATASET}" >&2
  exit 1
fi

sanitize_name() {
  local raw="$1"
  echo "${raw}" | sed 's/[^A-Za-z0-9._-]/_/g'
}

append_result() {
  local dataset_name="$1"
  local model_path="$2"
  local output_json="$3"
  local status="$4"
  local utc_now="$5"
  local acc_str="$6"
  "${PYTHON_EVAL}" - "${FINAL_RESULTS_JSONL}" "${dataset_name}" "${model_path}" "${output_json}" "${status}" "${utc_now}" "${acc_str}" <<'PY'
import json
import sys

out_path, dataset_name, model_path, output_json, status, utc_now, acc_str = sys.argv[1:]
record = {
    "dataset": dataset_name,
    "ckpt": "base_model",
    "model": model_path,
    "accuracy": None if acc_str == "None" else float(acc_str),
    "result_file": output_json,
    "status": status,
    "time_utc": utc_now,
}
with open(out_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
print(record)
PY
}

for model_path in "${MODELS[@]}"; do
  if [[ ! -d "${model_path}" ]]; then
    echo "Model not found, skip: ${model_path}"
    continue
  fi

  model_tag="$(sanitize_name "$(basename "${model_path}")")"

  gpqa_output="${OUT_DIR}/gpqa_diamond_base_${model_tag}.json"
  mmlu_output="${OUT_DIR}/mmlu_pro_test500_base_${model_tag}.json"

  echo "============================================================"
  echo "Evaluating GPQA-Diamond | model=${model_path}"
  echo "output=${gpqa_output}"
  echo "============================================================"
  utc_now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if OPD_AUTOCONFIG_REGISTER_EXIST_OK=1 HOME="/idfsdata/yexuyan/OPD" XDG_CACHE_HOME="/idfsdata/yexuyan/OPD/.cache" CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
    "${PYTHON_EVAL}" "${ROOT_DIR}/evaluation/opsd_import/eval/evaluate_gpqa_diamond.py" \
    --model_path "${model_path}" \
    --dataset_path "${GPQA_DATASET}" \
    --output_path "${gpqa_output}" \
    --tensor_parallel_size "${TP_SIZE}" \
    --gpu_memory_utilization "${GPU_MEM_UTIL}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --max_num_seqs "${MAX_NUM_SEQS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --rescue_max_new_tokens "${RESCUE_MAX_NEW_TOKENS}"; then
    gpqa_acc="$("${PYTHON_EVAL}" - "${gpqa_output}" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    payload = json.load(f)
acc = float(payload.get("metrics", {}).get("accuracy", 0.0)) * 100.0
print(f"{acc:.2f}")
PY
)"
    append_result "gpqa_diamond" "${model_path}" "${gpqa_output}" "ok" "${utc_now}" "${gpqa_acc}"
  else
    append_result "gpqa_diamond" "${model_path}" "${gpqa_output}" "failed" "${utc_now}" "None"
  fi

  echo "============================================================"
  echo "Evaluating MMLU-Pro-500 | model=${model_path}"
  echo "output=${mmlu_output}"
  echo "============================================================"
  utc_now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if OPD_AUTOCONFIG_REGISTER_EXIST_OK=1 HOME="/idfsdata/yexuyan/OPD" XDG_CACHE_HOME="/idfsdata/yexuyan/OPD/.cache" CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
    "${PYTHON_EVAL}" "${ROOT_DIR}/evaluation/opsd_import/eval/evaluate_mmlu_pro.py" \
    --model_path "${model_path}" \
    --dataset_path "${MMLU500_DATASET}" \
    --output_path "${mmlu_output}" \
    --tensor_parallel_size "${TP_SIZE}" \
    --gpu_memory_utilization "${GPU_MEM_UTIL}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --max_num_seqs "${MAX_NUM_SEQS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --rescue_max_new_tokens "${RESCUE_MAX_NEW_TOKENS}"; then
    mmlu_acc="$("${PYTHON_EVAL}" - "${mmlu_output}" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    payload = json.load(f)
acc = float(payload.get("metrics", {}).get("accuracy", 0.0)) * 100.0
print(f"{acc:.2f}")
PY
)"
    append_result "mmlupro_test500" "${model_path}" "${mmlu_output}" "ok" "${utc_now}" "${mmlu_acc}"
  else
    append_result "mmlupro_test500" "${model_path}" "${mmlu_output}" "failed" "${utc_now}" "None"
  fi
done

echo "Base model evaluations finished."
