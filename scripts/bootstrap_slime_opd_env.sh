#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="${CONDA_SH:-/home/yexuyan/miniconda3/etc/profile.d/conda.sh}"
SOURCE_ENV_PATH="${SOURCE_ENV_PATH:-/idfsdata/yexuyan/conda_envs/slime}"
TARGET_ENV_PATH="${TARGET_ENV_PATH:-/idfsdata/yexuyan/conda_envs/slime-opd}"
SGLANG_SRC="${SGLANG_SRC:-/idfsdata/yexuyan/slime_deps/sglang}"
MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/idfsdata/yexuyan/slime_deps/Megatron-LM}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
INSTALL_FROM_NETWORK="${INSTALL_FROM_NETWORK:-0}"

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    echo "${label} not found: ${path}" >&2
    exit 1
  fi
}

require_path "${CONDA_SH}" "Conda activation script"
require_path "${SGLANG_SRC}" "Local SGLang source tree"
require_path "${MEGATRON_LM_PATH}" "Local Megatron-LM source tree"
require_path "${ROOT_DIR}" "Repository root"

set +u
source "${CONDA_SH}"
set -u

if [[ ! -d "${TARGET_ENV_PATH}" ]]; then
  if [[ -d "${SOURCE_ENV_PATH}" ]]; then
    echo "Cloning existing slime env:"
    echo "  source: ${SOURCE_ENV_PATH}"
    echo "  target: ${TARGET_ENV_PATH}"
    conda create -y --clone "${SOURCE_ENV_PATH}" --prefix "${TARGET_ENV_PATH}"
  else
    echo "Source env not found, creating a fresh env at ${TARGET_ENV_PATH}"
    conda create -y --prefix "${TARGET_ENV_PATH}" "python=${PYTHON_VERSION}" pip
  fi
fi

set +u
conda activate "${TARGET_ENV_PATH}"
set -u
export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"

python -m pip install --upgrade pip setuptools wheel packaging

if [[ "${INSTALL_FROM_NETWORK}" == "1" ]]; then
  python -m pip install cuda-python==13.1.0
  python -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu129
  python -m pip install cmake ninja
  python -m pip install -e "${SGLANG_SRC}/python[all]"
  MAX_JOBS="${MAX_JOBS:-64}" python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
  python -m pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps
  python -m pip install --no-build-isolation "transformer_engine[pytorch]==2.10.0"
  python -m pip install flash-linear-attention==0.4.0
  NVCC_APPEND_FLAGS="--threads 4" python -m pip install --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4
  python -m pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b --no-cache-dir --force-reinstall
  python -m pip install git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl --no-build-isolation
  python -m pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation
  python -m pip install nvidia-cudnn-cu12==9.16.0.29 "numpy<2"
else
  echo "INSTALL_FROM_NETWORK=0, skipping heavyweight package downloads."
  echo "This mode assumes ${TARGET_ENV_PATH} already contains the compiled slime stack."
fi

python -m pip install -e "${SGLANG_SRC}/python"
python -m pip install -e "${MEGATRON_LM_PATH}"
python -m pip install -e "${ROOT_DIR}"

python "${ROOT_DIR}/scripts/check_slime_env.py" \
  --repo-root "${ROOT_DIR}" \
  --sglang-src "${SGLANG_SRC}" \
  --megatron-src "${MEGATRON_LM_PATH}" \
  --teacher-model "${ROOT_DIR}/models/Qwen3-14B" \
  --student-model "${ROOT_DIR}/models/Qwen3-4B" \
  --raw-data "${ROOT_DIR}/data/science_with_answer.jsonl"

echo "slime-opd environment is ready at ${TARGET_ENV_PATH}"
