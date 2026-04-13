#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/idfsdata/yexuyan/OPD"
PYTHON_CONVERT="/idfsdata/yexuyan/conda_envs/slime/bin/python"
PYTHON_EVAL="/idfsdata/yexuyan/conda_envs/verl/bin/python"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

CKPT_ROOT="${CKPT_ROOT:-/idfsdata/yexuyan/OPD/runs/OPD-8Bvia1.7B-Science/checkpoints}"
HF_ROOT="${HF_ROOT:-/idfsdata/yexuyan/OPD/runs/OPD-8Bvia1.7B-Science/hf_ckpts}"
ORIGIN_HF_DIR="${ORIGIN_HF_DIR:-/idfsdata/yexuyan/OPD/models/Qwen3-1.7B}"
DATASET_PATH="${DATASET_PATH:-/idfsdata/yexuyan/OPD/evaluation/opsd_import/data/gpqa_diamond.jsonl}"
RESULTS_DIR="${RESULTS_DIR:-/idfsdata/yexuyan/OPD/evaluation/gpqa_diamond_ckpt_results_OPD-8Bvia1.7B-Science}"
FINAL_RESULTS_JSONL="${FINAL_RESULTS_JSONL:-/idfsdata/yexuyan/OPD/FinalResults-GR.jsonl}"

TP_SIZE="${TP_SIZE:-4}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
RESCUE_MAX_NEW_TOKENS="${RESCUE_MAX_NEW_TOKENS:-4}"
CACHE_HOME="/idfsdata/yexuyan/OPD/.cache"

mkdir -p "${HF_ROOT}" "${RESULTS_DIR}" "$(dirname "${FINAL_RESULTS_JSONL}")" "${CACHE_HOME}/flashinfer"

if [[ ! -d "${CKPT_ROOT}" ]]; then
  echo "Checkpoint root not found: ${CKPT_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${ORIGIN_HF_DIR}" ]]; then
  echo "Origin HF model dir not found: ${ORIGIN_HF_DIR}" >&2
  exit 1
fi
if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Dataset not found: ${DATASET_PATH}" >&2
  exit 1
fi

mapfile -t CKPTS < <(find "${CKPT_ROOT}" -maxdepth 1 -mindepth 1 -type d -name "iter_*" | sort)
if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "No iter_* checkpoints found in ${CKPT_ROOT}" >&2
  exit 1
fi

echo "Found ${#CKPTS[@]} checkpoints under ${CKPT_ROOT}"
echo "Results json dir: ${RESULTS_DIR}"
echo "Final jsonl: ${FINAL_RESULTS_JSONL}"

for ckpt in "${CKPTS[@]}"; do
  ckpt_name="$(basename "${ckpt}")"
  hf_dir="${HF_ROOT}/${ckpt_name}"
  output_json="${RESULTS_DIR}/${ckpt_name}.json"
  utc_now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  echo "============================================================"
  echo "[${utc_now}] Processing ${ckpt_name}"
  echo "Megatron ckpt: ${ckpt}"
  echo "HF dir: ${hf_dir}"
  echo "Result file: ${output_json}"
  echo "============================================================"

  if [[ ! -f "${hf_dir}/config.json" ]]; then
    echo "[${ckpt_name}] Converting Megatron distcp -> HF..."
    HOME="/idfsdata/yexuyan/OPD" XDG_CACHE_HOME="${CACHE_HOME}" \
      "${PYTHON_CONVERT}" "${ROOT_DIR}/tools/convert_torch_dist_to_hf_bridge.py" \
      --input-dir "${ckpt}" \
      --output-dir "${hf_dir}" \
      --origin-hf-dir "${ORIGIN_HF_DIR}" \
      --force
  else
    echo "[${ckpt_name}] HF checkpoint exists, skip conversion."
  fi

  echo "[${ckpt_name}] Running GPQA-Diamond evaluation on GPUs 0,1,2,3 with max_num_seqs=${MAX_NUM_SEQS}..."
  if OPD_AUTOCONFIG_REGISTER_EXIST_OK=1 HOME="/idfsdata/yexuyan/OPD" XDG_CACHE_HOME="${CACHE_HOME}" CUDA_VISIBLE_DEVICES=0,1,2,3 \
    "${PYTHON_EVAL}" "${ROOT_DIR}/evaluation/opsd_import/eval/evaluate_gpqa_diamond.py" \
    --model_path "${hf_dir}" \
    --dataset_path "${DATASET_PATH}" \
    --output_path "${output_json}" \
    --tensor_parallel_size "${TP_SIZE}" \
    --gpu_memory_utilization "${GPU_MEM_UTIL}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --max_num_seqs "${MAX_NUM_SEQS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --rescue_max_new_tokens "${RESCUE_MAX_NEW_TOKENS}"; then

    read -r acc extraction rescued fallback < <("${PYTHON_EVAL}" - "${output_json}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    payload = json.load(f)
metrics = payload.get("metrics", {})
acc = float(metrics.get("accuracy", 0.0)) * 100.0
extraction_rate = float(metrics.get("extraction_rate", 0.0))
rescue_rate = float(metrics.get("rescue_rate", 0.0))
hard_fallback_rate = float(metrics.get("hard_fallback_rate", 0.0))
print(f"{acc:.2f} {extraction_rate:.6f} {rescue_rate:.6f} {hard_fallback_rate:.6f}")
PY
)

    utc_now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    "${PYTHON_EVAL}" - "${FINAL_RESULTS_JSONL}" "${ckpt_name}" "${acc}" "${output_json}" "${hf_dir}" "${utc_now}" "${extraction}" "${rescued}" "${fallback}" <<'PY'
import json
import sys

(
    out_path,
    ckpt_name,
    acc_str,
    output_json,
    model_path,
    utc_now,
    extraction_rate,
    rescue_rate,
    hard_fallback_rate,
) = sys.argv[1:]

record = {
    "dataset": "gpqa_diamond",
    "ckpt": ckpt_name,
    "model": model_path,
    "accuracy": float(acc_str),
    "result_file": output_json,
    "extraction_rate": float(extraction_rate),
    "rescue_rate": float(rescue_rate),
    "hard_fallback_rate": float(hard_fallback_rate),
    "status": "ok",
    "time_utc": utc_now,
}
with open(out_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
print(f"[append] {record}")
PY
  else
    utc_now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    "${PYTHON_EVAL}" - "${FINAL_RESULTS_JSONL}" "${ckpt_name}" "${output_json}" "${hf_dir}" "${utc_now}" <<'PY'
import json
import sys

out_path, ckpt_name, output_json, model_path, utc_now = sys.argv[1:]
record = {
    "dataset": "gpqa_diamond",
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
    echo "[${ckpt_name}] Evaluation failed, continue next checkpoint."
  fi
done

echo "All checkpoints processed."
