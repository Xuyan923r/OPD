#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/idfsdata/yexuyan/OPD"
PYTHON_CONVERT="/idfsdata/yexuyan/conda_envs/slime/bin/python"
PYTHON_EVAL="/idfsdata/yexuyan/conda_envs/verl/bin/python"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

DATASET_PATH="${DATASET_PATH:-/idfsdata/yexuyan/OPD/MMLU-Pro-test-500.jsonl}"
ORIGIN_HF_DIR="${ORIGIN_HF_DIR:-/idfsdata/yexuyan/OPD/models/Qwen3-1.7B}"
FINAL_RESULTS_JSONL="${FINAL_RESULTS_JSONL:-/idfsdata/yexuyan/OPD/FinalResults-GR.jsonl}"
CACHE_HOME="${CACHE_HOME:-/idfsdata/yexuyan/OPD/.cache}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"

TP_SIZE="${TP_SIZE:-4}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
RESCUE_MAX_NEW_TOKENS="${RESCUE_MAX_NEW_TOKENS:-4}"

RUN_ROOTS=(
  "/idfsdata/yexuyan/OPD/runs/OPD-8Bvia1.7B-Science"
  "/idfsdata/yexuyan/OPD/runs/OPSD-8Bvia1.7B-Science-Answer"
  "/idfsdata/yexuyan/OPD/runs/8Bvia1.7B-64*1"
)

mkdir -p "$(dirname "${FINAL_RESULTS_JSONL}")" "${CACHE_HOME}/flashinfer"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Dataset not found: ${DATASET_PATH}" >&2
  exit 1
fi
if [[ ! -d "${ORIGIN_HF_DIR}" ]]; then
  echo "Origin HF model dir not found: ${ORIGIN_HF_DIR}" >&2
  exit 1
fi

sanitize_name() {
  local raw="$1"
  echo "${raw}" | sed 's/[^A-Za-z0-9._-]/_/g'
}

for RUN_ROOT in "${RUN_ROOTS[@]}"; do
  CKPT_ROOT="${RUN_ROOT}/checkpoints"
  HF_ROOT="${RUN_ROOT}/hf_ckpts"
  RUN_TAG="$(sanitize_name "$(basename "${RUN_ROOT}")")"
  RESULTS_DIR="${ROOT_DIR}/evaluation/mmlu_pro_test500_ckpt_results_${RUN_TAG}"

  mkdir -p "${HF_ROOT}" "${RESULTS_DIR}"

  if [[ ! -d "${CKPT_ROOT}" ]]; then
    echo "[${RUN_TAG}] Checkpoint root not found: ${CKPT_ROOT}, skip."
    continue
  fi

  mapfile -t CKPTS < <(find "${CKPT_ROOT}" -maxdepth 1 -mindepth 1 -type d -name "iter_*" | sort)
  if [[ ${#CKPTS[@]} -eq 0 ]]; then
    echo "[${RUN_TAG}] No iter_* checkpoints found under ${CKPT_ROOT}, skip."
    continue
  fi

  echo "################################################################"
  echo "[${RUN_TAG}] Found ${#CKPTS[@]} checkpoints"
  echo "[${RUN_TAG}] Results dir: ${RESULTS_DIR}"
  echo "################################################################"

  for ckpt in "${CKPTS[@]}"; do
    ckpt_name="$(basename "${ckpt}")"
    hf_dir="${HF_ROOT}/${ckpt_name}"
    output_json="${RESULTS_DIR}/${ckpt_name}.json"
    utc_now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    echo "============================================================"
    echo "[${utc_now}] [${RUN_TAG}] Processing ${ckpt_name}"
    echo "Megatron ckpt: ${ckpt}"
    echo "HF dir: ${hf_dir}"
    echo "Result file: ${output_json}"
    echo "============================================================"

    if [[ ! -f "${hf_dir}/config.json" ]]; then
      echo "[${RUN_TAG}/${ckpt_name}] Converting Megatron distcp -> HF..."
      HOME="/idfsdata/yexuyan/OPD" XDG_CACHE_HOME="${CACHE_HOME}" CUDA_VISIBLE_DEVICES="" \
        "${PYTHON_CONVERT}" "${ROOT_DIR}/tools/convert_torch_dist_to_hf_bridge.py" \
        --input-dir "${ckpt}" \
        --output-dir "${hf_dir}" \
        --origin-hf-dir "${ORIGIN_HF_DIR}" \
        --force
    else
      echo "[${RUN_TAG}/${ckpt_name}] HF checkpoint exists, skip conversion."
    fi

    echo "[${RUN_TAG}/${ckpt_name}] Running MMLU-Pro-500 evaluation on GPUs ${CUDA_DEVICES} with max_num_seqs=${MAX_NUM_SEQS}..."
    if OPD_AUTOCONFIG_REGISTER_EXIST_OK=1 HOME="/idfsdata/yexuyan/OPD" XDG_CACHE_HOME="${CACHE_HOME}" CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
      "${PYTHON_EVAL}" "${ROOT_DIR}/evaluation/opsd_import/eval/evaluate_mmlu_pro.py" \
      --model_path "${hf_dir}" \
      --dataset_path "${DATASET_PATH}" \
      --output_path "${output_json}" \
      --tensor_parallel_size "${TP_SIZE}" \
      --gpu_memory_utilization "${GPU_MEM_UTIL}" \
      --max_model_len "${MAX_MODEL_LEN}" \
      --max_num_seqs "${MAX_NUM_SEQS}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --rescue_max_new_tokens "${RESCUE_MAX_NEW_TOKENS}"; then

      acc="$("${PYTHON_EVAL}" - "${output_json}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    payload = json.load(f)
acc = float(payload.get("metrics", {}).get("accuracy", 0.0)) * 100.0
print(f"{acc:.2f}")
PY
)"

      utc_now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      "${PYTHON_EVAL}" - "${FINAL_RESULTS_JSONL}" "${RUN_TAG}" "${ckpt_name}" "${acc}" "${output_json}" "${hf_dir}" "${utc_now}" <<'PY'
import json
import sys

out_path, run_tag, ckpt_name, acc_str, output_json, model_path, utc_now = sys.argv[1:]
record = {
    "dataset": "mmlupro_test500",
    "run": run_tag,
    "ckpt": ckpt_name,
    "model": model_path,
    "accuracy": float(acc_str),
    "result_file": output_json,
    "status": "ok",
    "time_utc": utc_now,
}
with open(out_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
print(f"[append] {record}")
PY
    else
      utc_now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      "${PYTHON_EVAL}" - "${FINAL_RESULTS_JSONL}" "${RUN_TAG}" "${ckpt_name}" "${output_json}" "${hf_dir}" "${utc_now}" <<'PY'
import json
import sys

out_path, run_tag, ckpt_name, output_json, model_path, utc_now = sys.argv[1:]
record = {
    "dataset": "mmlupro_test500",
    "run": run_tag,
    "ckpt": ckpt_name,
    "model": model_path,
    "accuracy": None,
    "result_file": output_json,
    "status": "failed",
    "time_utc": utc_now,
}
with open(out_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
print(f"[append-failed] {record}")
PY
      echo "[${RUN_TAG}/${ckpt_name}] Evaluation failed, continue next checkpoint."
    fi
  done
done

echo "All run roots processed."
