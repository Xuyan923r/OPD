#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/idfsdata/yexuyan/OPD"
PYTHON_CONVERT="/idfsdata/yexuyan/conda_envs/slime/bin/python"

CKPT_ROOT="${CKPT_ROOT:-/idfsdata/yexuyan/OPD/runs/OPSD-8Bvia1.7B-Science-Answer/checkpoints}"
HF_ROOT="${HF_ROOT:-/idfsdata/yexuyan/OPD/runs/OPSD-8Bvia1.7B-Science-Answer/hf_ckpts}"
ORIGIN_HF_DIR="${ORIGIN_HF_DIR:-/idfsdata/yexuyan/OPD/models/Qwen3-1.7B}"
CACHE_HOME="${CACHE_HOME:-/idfsdata/yexuyan/OPD/.cache}"

mkdir -p "${HF_ROOT}" "${CACHE_HOME}/flashinfer"

if [[ ! -d "${CKPT_ROOT}" ]]; then
  echo "Checkpoint root not found: ${CKPT_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${ORIGIN_HF_DIR}" ]]; then
  echo "Origin HF model dir not found: ${ORIGIN_HF_DIR}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON_CONVERT}" ]]; then
  echo "Python converter not executable: ${PYTHON_CONVERT}" >&2
  exit 1
fi

mapfile -t CKPTS < <(find "${CKPT_ROOT}" -maxdepth 1 -mindepth 1 -type d -name "iter_*" | sort)
if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "No iter_* checkpoints found in ${CKPT_ROOT}" >&2
  exit 1
fi

echo "Found ${#CKPTS[@]} checkpoints under ${CKPT_ROOT}"
echo "Target HF root: ${HF_ROOT}"

ok_count=0
skip_count=0
fail_count=0

for ckpt in "${CKPTS[@]}"; do
  ckpt_name="$(basename "${ckpt}")"
  hf_dir="${HF_ROOT}/${ckpt_name}"
  utc_now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  echo "============================================================"
  echo "[${utc_now}] Processing ${ckpt_name}"
  echo "Megatron ckpt: ${ckpt}"
  echo "HF target: ${hf_dir}"
  echo "============================================================"

  if [[ -f "${hf_dir}/config.json" ]]; then
    echo "[${ckpt_name}] HF checkpoint already exists, skip."
    skip_count=$((skip_count + 1))
    continue
  fi

  if HOME="/idfsdata/yexuyan/OPD" XDG_CACHE_HOME="${CACHE_HOME}" CUDA_VISIBLE_DEVICES="" \
    "${PYTHON_CONVERT}" "${ROOT_DIR}/tools/convert_torch_dist_to_hf_bridge.py" \
    --input-dir "${ckpt}" \
    --output-dir "${hf_dir}" \
    --origin-hf-dir "${ORIGIN_HF_DIR}" \
    --force; then
    echo "[${ckpt_name}] Conversion done."
    ok_count=$((ok_count + 1))
  else
    echo "[${ckpt_name}] Conversion failed."
    fail_count=$((fail_count + 1))
  fi
done

echo "============================================================"
echo "Conversion finished."
echo "ok=${ok_count} skip=${skip_count} failed=${fail_count}"
echo "HF root: ${HF_ROOT}"
echo "============================================================"

if (( fail_count > 0 )); then
  exit 1
fi
