#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/idfsdata/yexuyan/OPD"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/idfsdata/yexuyan/conda_envs/slime}"
CONDA_SH="${CONDA_SH:-/home/yexuyan/miniconda3/etc/profile.d/conda.sh}"
SGLANG_SRC="${SGLANG_SRC:-/idfsdata/yexuyan/slime_deps/sglang}"
MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/idfsdata/yexuyan/slime_deps/Megatron-LM}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
HOST_CC="${HOST_CC:-/usr/bin/gcc-11}"
HOST_CXX="${HOST_CXX:-/usr/bin/g++-11}"
HOST_CUDAHOSTCXX="${HOST_CUDAHOSTCXX:-/usr/bin/g++-11}"

RUN_INNER=0
PREFLIGHT_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --inner)
      RUN_INNER=1
      ;;
    --preflight)
      PREFLIGHT_ONLY=1
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
  shift
done

TEACHER_MODEL="${TEACHER_MODEL:-/idfsdata/yexuyan/OPD/models/Qwen3-8B}"
STUDENT_MODEL="${STUDENT_MODEL:-/idfsdata/yexuyan/OPD/models/Qwen3-1.7B}"
RAW_DATA="${RAW_DATA:-/idfsdata/yexuyan/OPD/data/science_with_answer.jsonl}"
PREPARED_DATA="${PREPARED_DATA:-/idfsdata/yexuyan/OPD/data/science_with_answer_opd_prompt_only_boxed.jsonl}"
EVAL_CONFIG="${EVAL_CONFIG:-/idfsdata/yexuyan/OPD/examples/mmlu_pro_dev40_eval.yaml}"

TEACHER_GPUS="${TEACHER_GPUS:-0}"
STUDENT_GPUS="${STUDENT_GPUS:-1,2,3}"
TEACHER_PORT="${TEACHER_PORT:-31081}"
RAY_PORT="${RAY_PORT:-6381}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8266}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/idfsdata/yexuyan/}"

NUM_EPOCH="${NUM_EPOCH:-1}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-4096}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-6000}"
TRAIN_SEQ_LENGTH="${TRAIN_SEQ_LENGTH:-8192}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-6144}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-64}"
TRAIN_MEMORY_MARGIN_BYTES="${TRAIN_MEMORY_MARGIN_BYTES:-1073741824}"
TRAIN_PYTORCH_CUDA_ALLOC_CONF="${TRAIN_PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64,garbage_collection_threshold:0.6}"
ENABLE_ACTIVATION_RECOMPUTE="${ENABLE_ACTIVATION_RECOMPUTE:-1}"
RECOMPUTE_GRANULARITY="${RECOMPUTE_GRANULARITY:-full}"
RECOMPUTE_METHOD="${RECOMPUTE_METHOD:-uniform}"
RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-1}"
ENABLE_LOSS_RECOMPUTE="${ENABLE_LOSS_RECOMPUTE:-1}"
TEACHER_MEM_FRACTION="${TEACHER_MEM_FRACTION:-0.85}"
STUDENT_SGLANG_MEM_FRACTION="${STUDENT_SGLANG_MEM_FRACTION:-0.30}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-12}"
WANDB_PROJECT="${WANDB_PROJECT:-OPSD}"
DEFAULT_APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'
APPLY_CHAT_TEMPLATE_KWARGS="${APPLY_CHAT_TEMPLATE_KWARGS:-$DEFAULT_APPLY_CHAT_TEMPLATE_KWARGS}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
MAX_CHECKPOINTS="${MAX_CHECKPOINTS:-10}"
ENABLE_SEQUENCE_PARALLEL="${ENABLE_SEQUENCE_PARALLEL:-auto}"
QKV_FORMAT="${QKV_FORMAT:-auto}"
USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-auto}"
USE_COLOCATE="${USE_COLOCATE:-0}"
TRAIN_ACTOR_GPUS_PER_NODE="${TRAIN_ACTOR_GPUS_PER_NODE:-2}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-1}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
RAY_NUM_GPUS="${RAY_NUM_GPUS:-3}"
TEACHER_TP="${TEACHER_TP:-1}"

RUN_ID="${RUN_ID:-opd_qwen3_14b_to_4b_science_$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/runlogs}"

if [[ "${RUN_INNER}" != "1" ]]; then
  if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
    exec env \
      RUN_ID="${RUN_ID}" \
      RUN_ROOT="${RUN_ROOT}" \
      LOG_ROOT="${LOG_ROOT}" \
      CONDA_ENV_PATH="${CONDA_ENV_PATH}" \
      CONDA_SH="${CONDA_SH}" \
      SGLANG_SRC="${SGLANG_SRC}" \
      MEGATRON_LM_PATH="${MEGATRON_LM_PATH}" \
      TEACHER_MODEL="${TEACHER_MODEL}" \
      STUDENT_MODEL="${STUDENT_MODEL}" \
      RAW_DATA="${RAW_DATA}" \
      PREPARED_DATA="${PREPARED_DATA}" \
      EVAL_CONFIG="${EVAL_CONFIG}" \
      TEACHER_GPUS="${TEACHER_GPUS}" \
      STUDENT_GPUS="${STUDENT_GPUS}" \
      TEACHER_PORT="${TEACHER_PORT}" \
      RAY_PORT="${RAY_PORT}" \
      RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT}" \
      RAY_TEMP_DIR="${RAY_TEMP_DIR}" \
      NUM_EPOCH="${NUM_EPOCH}" \
      ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE}" \
      N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT}" \
      GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}" \
      ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN}" \
      ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN}" \
      TRAIN_SEQ_LENGTH="${TRAIN_SEQ_LENGTH}" \
      MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU}" \
      MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE}" \
      LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE}" \
      TRAIN_MEMORY_MARGIN_BYTES="${TRAIN_MEMORY_MARGIN_BYTES}" \
      TRAIN_PYTORCH_CUDA_ALLOC_CONF="${TRAIN_PYTORCH_CUDA_ALLOC_CONF}" \
      ENABLE_ACTIVATION_RECOMPUTE="${ENABLE_ACTIVATION_RECOMPUTE}" \
      RECOMPUTE_GRANULARITY="${RECOMPUTE_GRANULARITY}" \
      RECOMPUTE_METHOD="${RECOMPUTE_METHOD}" \
      RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS}" \
      ENABLE_LOSS_RECOMPUTE="${ENABLE_LOSS_RECOMPUTE}" \
      TEACHER_MEM_FRACTION="${TEACHER_MEM_FRACTION}" \
      STUDENT_SGLANG_MEM_FRACTION="${STUDENT_SGLANG_MEM_FRACTION}" \
      SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS}" \
      WANDB_PROJECT="${WANDB_PROJECT}" \
      APPLY_CHAT_TEMPLATE_KWARGS="${APPLY_CHAT_TEMPLATE_KWARGS}" \
      EVAL_INTERVAL="${EVAL_INTERVAL}" \
      SAVE_INTERVAL="${SAVE_INTERVAL}" \
      MAX_CHECKPOINTS="${MAX_CHECKPOINTS}" \
      QKV_FORMAT="${QKV_FORMAT}" \
      USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE}" \
      bash "$0" --inner --preflight
  fi
  mkdir -p "${LOG_ROOT}" "${RUN_ROOT}"
  LAUNCH_LOG="${LOG_ROOT}/${RUN_ID}.log"
  nohup env \
    RUN_ID="${RUN_ID}" \
    RUN_ROOT="${RUN_ROOT}" \
    LOG_ROOT="${LOG_ROOT}" \
    CONDA_ENV_PATH="${CONDA_ENV_PATH}" \
    CONDA_SH="${CONDA_SH}" \
    SGLANG_SRC="${SGLANG_SRC}" \
    MEGATRON_LM_PATH="${MEGATRON_LM_PATH}" \
    TEACHER_MODEL="${TEACHER_MODEL}" \
    STUDENT_MODEL="${STUDENT_MODEL}" \
    RAW_DATA="${RAW_DATA}" \
    PREPARED_DATA="${PREPARED_DATA}" \
    EVAL_CONFIG="${EVAL_CONFIG}" \
    TEACHER_GPUS="${TEACHER_GPUS}" \
    STUDENT_GPUS="${STUDENT_GPUS}" \
    TEACHER_PORT="${TEACHER_PORT}" \
    RAY_PORT="${RAY_PORT}" \
    RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT}" \
    RAY_TEMP_DIR="${RAY_TEMP_DIR}" \
    NUM_EPOCH="${NUM_EPOCH}" \
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE}" \
    N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT}" \
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}" \
    ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN}" \
    ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN}" \
    TRAIN_SEQ_LENGTH="${TRAIN_SEQ_LENGTH}" \
    MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU}" \
    MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE}" \
    LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE}" \
    TRAIN_MEMORY_MARGIN_BYTES="${TRAIN_MEMORY_MARGIN_BYTES}" \
    TRAIN_PYTORCH_CUDA_ALLOC_CONF="${TRAIN_PYTORCH_CUDA_ALLOC_CONF}" \
    ENABLE_ACTIVATION_RECOMPUTE="${ENABLE_ACTIVATION_RECOMPUTE}" \
    RECOMPUTE_GRANULARITY="${RECOMPUTE_GRANULARITY}" \
    RECOMPUTE_METHOD="${RECOMPUTE_METHOD}" \
    RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS}" \
    ENABLE_LOSS_RECOMPUTE="${ENABLE_LOSS_RECOMPUTE}" \
    TEACHER_MEM_FRACTION="${TEACHER_MEM_FRACTION}" \
    STUDENT_SGLANG_MEM_FRACTION="${STUDENT_SGLANG_MEM_FRACTION}" \
    SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS}" \
    WANDB_PROJECT="${WANDB_PROJECT}" \
    APPLY_CHAT_TEMPLATE_KWARGS="${APPLY_CHAT_TEMPLATE_KWARGS}" \
    EVAL_INTERVAL="${EVAL_INTERVAL}" \
    SAVE_INTERVAL="${SAVE_INTERVAL}" \
    MAX_CHECKPOINTS="${MAX_CHECKPOINTS}" \
    QKV_FORMAT="${QKV_FORMAT}" \
    USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE}" \
    bash "$0" --inner >"${LAUNCH_LOG}" 2>&1 &
  JOB_PID=$!
  echo "Started background OPD job."
  echo "PID: ${JOB_PID}"
  echo "Main log: ${LAUNCH_LOG}"
  exit 0
fi

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${RUN_ROOT}/pids" "${RUN_ROOT}/checkpoints"
MAIN_LOG="${LOG_ROOT}/${RUN_ID}.log"
TEACHER_LOG="${LOG_ROOT}/${RUN_ID}_teacher.log"
TRAJECTORY_DIR="${RUN_ROOT}/rollout_trajectories"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Conda activation script not found: ${CONDA_SH}" >&2
  exit 1
fi

set +u
source "${CONDA_SH}"
conda activate "${CONDA_ENV_PATH}"
set -u

if [[ -n "${HOST_CC:-}" ]]; then
  export CC="${HOST_CC}"
fi
if [[ -n "${HOST_CXX:-}" ]]; then
  export CXX="${HOST_CXX}"
fi
if [[ -n "${HOST_CUDAHOSTCXX:-}" ]]; then
  export CUDAHOSTCXX="${HOST_CUDAHOSTCXX}"
fi
if [[ -z "${CUDAHOSTCXX:-}" && -n "${CXX:-}" ]]; then
  # Let nvcc reuse the known-good host compiler when SGLang JIT kernels build.
  export CUDAHOSTCXX="${CXX}"
fi

export PYTHONUNBUFFERED=1
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

echo "Compiler override:"
echo "  CC=${CC:-}"
echo "  CXX=${CXX:-}"
echo "  CUDAHOSTCXX=${CUDAHOSTCXX:-}"

CUDA_LIB64=""
if [[ -d "${CUDA_HOME}/lib64" ]]; then
  CUDA_LIB64="${CUDA_HOME}/lib64"
elif [[ -d "${CUDA_HOME}/targets/x86_64-linux/lib" ]]; then
  CUDA_LIB64="${CUDA_HOME}/targets/x86_64-linux/lib"
fi

if [[ -n "${CUDA_LIB64}" ]]; then
  export CUDA_HOME
  export LD_LIBRARY_PATH="${CUDA_LIB64}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export LIBRARY_PATH="${CUDA_LIB64}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
fi

echo "CUDA toolchain:"
echo "  CUDA_HOME=${CUDA_HOME}"
echo "  CUDA_LIB64=${CUDA_LIB64}"
echo "  LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
echo "  LIBRARY_PATH=${LIBRARY_PATH:-}"

HAS_TRANSFORMER_ENGINE=0
if python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("transformer_engine") is not None else 1)
PY
then
  HAS_TRANSFORMER_ENGINE=1
fi

if [[ "${ENABLE_SEQUENCE_PARALLEL}" == "auto" ]]; then
  if [[ "${HAS_TRANSFORMER_ENGINE}" == "1" ]]; then
    ENABLE_SEQUENCE_PARALLEL=1
  else
    ENABLE_SEQUENCE_PARALLEL=0
  fi
fi

echo "Sequence parallel: ${ENABLE_SEQUENCE_PARALLEL}"

if [[ "${QKV_FORMAT}" == "auto" ]]; then
  if [[ "${HAS_TRANSFORMER_ENGINE}" == "1" ]]; then
    QKV_FORMAT="thd"
  else
    QKV_FORMAT="bshd"
  fi
fi

if [[ "${QKV_FORMAT}" == "thd" && "${HAS_TRANSFORMER_ENGINE}" != "1" ]]; then
  echo "Warning: QKV_FORMAT=thd requires transformer_engine in this environment. Falling back to bshd." >&2
  QKV_FORMAT="bshd"
  if [[ "${USE_COLOCATE}" == "1" ]]; then
  TRAIN_ARGS+=(--colocate)
fi

if [[ "${USE_DYNAMIC_BATCH_SIZE}" == "1" ]]; then
    echo "Warning: USE_DYNAMIC_BATCH_SIZE=1 is incompatible with bshd. Falling back to fixed micro-batching." >&2
    USE_DYNAMIC_BATCH_SIZE=0
  fi
fi

if [[ "${USE_DYNAMIC_BATCH_SIZE}" == "auto" ]]; then
  if [[ "${QKV_FORMAT}" == "thd" ]]; then
    USE_DYNAMIC_BATCH_SIZE=1
  else
    USE_DYNAMIC_BATCH_SIZE=0
  fi
fi

# Megatron colocate/offload mode always uses TorchMemorySaver, and
# TorchMemorySaver currently cannot run with expandable_segments enabled.
if [[ "${TRAIN_PYTORCH_CUDA_ALLOC_CONF}" == *"expandable_segments"* ]]; then
  echo "Warning: TRAIN_PYTORCH_CUDA_ALLOC_CONF=${TRAIN_PYTORCH_CUDA_ALLOC_CONF} is incompatible with TorchMemorySaver in colocate mode. Disabling TRAIN_PYTORCH_CUDA_ALLOC_CONF for this run."
  TRAIN_PYTORCH_CUDA_ALLOC_CONF=""
fi

echo "QKV format: ${QKV_FORMAT}"
echo "Dynamic batch size: ${USE_DYNAMIC_BATCH_SIZE}"
if [[ "${USE_DYNAMIC_BATCH_SIZE}" != "1" ]]; then
  echo "Micro batch size: ${MICRO_BATCH_SIZE}"
fi
echo "Train seq length: ${TRAIN_SEQ_LENGTH}"
echo "Log probs chunk size: ${LOG_PROBS_CHUNK_SIZE}"
echo "Train allocator: ${TRAIN_PYTORCH_CUDA_ALLOC_CONF:-<disabled>}"
echo "SGLang max running requests: ${SGLANG_MAX_RUNNING_REQUESTS}"
echo "Ray temp/log dir: ${RAY_TEMP_DIR}"

if (( TRAIN_SEQ_LENGTH <= ROLLOUT_MAX_RESPONSE_LEN )); then
  echo "TRAIN_SEQ_LENGTH (${TRAIN_SEQ_LENGTH}) must be greater than ROLLOUT_MAX_RESPONSE_LEN (${ROLLOUT_MAX_RESPONSE_LEN}) to leave room for prompt tokens." >&2
  exit 1
fi

python "${ROOT_DIR}/scripts/check_slime_env.py" \
  --repo-root "${ROOT_DIR}" \
  --sglang-src "${SGLANG_SRC}" \
  --megatron-src "${MEGATRON_LM_PATH}" \
  --teacher-model "${TEACHER_MODEL}" \
  --student-model "${STUDENT_MODEL}" \
  --raw-data "${RAW_DATA}"

if [[ ! -d "${TEACHER_MODEL}" ]]; then
  echo "Teacher model not found: ${TEACHER_MODEL}" >&2
  exit 1
fi
if [[ ! -d "${STUDENT_MODEL}" ]]; then
  echo "Student model not found: ${STUDENT_MODEL}" >&2
  exit 1
fi
if [[ ! -f "${RAW_DATA}" ]]; then
  echo "Raw data not found: ${RAW_DATA}" >&2
  exit 1
fi

STARTED_RAY=0
TEACHER_PID=""
PRUNER_PID=""

cleanup() {
  local exit_code=$?
  if [[ -n "${TEACHER_PID}" ]] && kill -0 "${TEACHER_PID}" 2>/dev/null; then
    kill "${TEACHER_PID}" 2>/dev/null || true
    wait "${TEACHER_PID}" 2>/dev/null || true
  fi
  if [[ -n "${PRUNER_PID}" ]] && kill -0 "${PRUNER_PID}" 2>/dev/null; then
    kill "${PRUNER_PID}" 2>/dev/null || true
    wait "${PRUNER_PID}" 2>/dev/null || true
  fi
  if [[ "${STARTED_RAY}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${STUDENT_GPUS}" ray stop --force >/dev/null 2>&1 || true
  fi
  exit "${exit_code}"
}
trap cleanup EXIT INT TERM

RAW_DATA="${RAW_DATA}" PREPARED_DATA="${PREPARED_DATA}" python - <<'PY'
import json
import os
from pathlib import Path

raw_path = Path(os.environ["RAW_DATA"])
prepared_path = Path(os.environ["PREPARED_DATA"])

prepared_path.parent.mkdir(parents=True, exist_ok=True)
count = 0
with raw_path.open("r", encoding="utf-8") as src, prepared_path.open("w", encoding="utf-8") as dst:
    for line in src:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        messages = item.get("messages") or []
        user_message = None
        for message in messages:
            if message.get("role") == "user" and message.get("content"):
                user_message = message.get("content")
                break
        answer = item.get("answer")
        if not user_message or answer is None:
            continue
        format_instruction = (
            "\n\nRespond briefly. The last line must be exactly in the format: Final Answer: \\boxed{A}\n"
            "Replace A with the single correct option letter. Do not put any other text inside \\boxed{}."
        )
        cleaned = {
            "messages": [{"role": "user", "content": user_message + format_instruction}],
            "answer": answer,
            "source": item.get("source"),
        }
        dst.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
        count += 1

if count == 0:
    raise SystemExit("Prepared dataset is empty.")
print(f"Prepared {count} prompt-only OPD samples at {prepared_path}")
PY

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  echo "Preflight check passed."
  echo "Prepared data: ${PREPARED_DATA}"
  echo "Eval config: ${EVAL_CONFIG}"
  echo "QKV format: ${QKV_FORMAT}"
  echo "Dynamic batch size: ${USE_DYNAMIC_BATCH_SIZE}"
  if [[ "${USE_DYNAMIC_BATCH_SIZE}" != "1" ]]; then
    echo "Micro batch size: ${MICRO_BATCH_SIZE}"
  fi
  echo "Train seq length: ${TRAIN_SEQ_LENGTH}"
  echo "Log probs chunk size: ${LOG_PROBS_CHUNK_SIZE}"
  echo "Train allocator: ${TRAIN_PYTORCH_CUDA_ALLOC_CONF:-<disabled>}"
  echo "SGLang max running requests: ${SGLANG_MAX_RUNNING_REQUESTS}"
  exit 0
fi

echo "Run ID: ${RUN_ID}"
echo "Teacher GPUs: ${TEACHER_GPUS}"
echo "Student GPUs: ${STUDENT_GPUS}"
echo "Prepared data: ${PREPARED_DATA}"
echo "Eval config: ${EVAL_CONFIG}"
echo "QKV format: ${QKV_FORMAT}"
echo "Dynamic batch size: ${USE_DYNAMIC_BATCH_SIZE}"
if [[ "${USE_DYNAMIC_BATCH_SIZE}" != "1" ]]; then
  echo "Micro batch size: ${MICRO_BATCH_SIZE}"
fi
echo "Train seq length: ${TRAIN_SEQ_LENGTH}"
echo "Log probs chunk size: ${LOG_PROBS_CHUNK_SIZE}"
echo "Train allocator: ${TRAIN_PYTORCH_CUDA_ALLOC_CONF:-<disabled>}"
echo "SGLang max running requests: ${SGLANG_MAX_RUNNING_REQUESTS}"
echo "Run root: ${RUN_ROOT}"
echo "Teacher log: ${TEACHER_LOG}"
echo "Main log: ${MAIN_LOG}"
echo "Trajectory dir: ${TRAJECTORY_DIR}"

checkpoint_pruner() {
  local ckpt_dir="$1"
  local keep_count="$2"
  mkdir -p "${ckpt_dir}"
  while true; do
    mapfile -t ckpts < <(find "${ckpt_dir}" -maxdepth 1 -mindepth 1 -type d -name 'iter_*' | sort)
    local total="${#ckpts[@]}"
    if (( total > keep_count )); then
      local remove_count=$((total - keep_count))
      for ((i=0; i<remove_count; i++)); do
        rm -rf "${ckpts[$i]}"
      done
    fi
    sleep 30
  done
}

checkpoint_pruner "${RUN_ROOT}/checkpoints" "${MAX_CHECKPOINTS}" &
PRUNER_PID=$!
echo "${PRUNER_PID}" > "${RUN_ROOT}/pids/checkpoint_pruner.pid"

CUDA_VISIBLE_DEVICES="${TEACHER_GPUS}" \
python -m sglang.launch_server \
  --model-path "${TEACHER_MODEL}" \
  --host 127.0.0.1 \
  --port "${TEACHER_PORT}" \
  --tp "${TEACHER_TP}" \
  --mem-fraction-static "${TEACHER_MEM_FRACTION}" \
  >"${TEACHER_LOG}" 2>&1 &
TEACHER_PID=$!
echo "${TEACHER_PID}" > "${RUN_ROOT}/pids/teacher.pid"
echo "Started teacher server, pid=${TEACHER_PID}"

for _ in $(seq 1 120); do
  if ! kill -0 "${TEACHER_PID}" 2>/dev/null; then
    echo "Teacher server exited early. Check ${TEACHER_LOG}" >&2
    exit 1
  fi
  if curl -sf "http://127.0.0.1:${TEACHER_PORT}/health_generate" >/dev/null; then
    echo "Teacher server is healthy on port ${TEACHER_PORT}"
    break
  fi
  sleep 5
done

if ! curl -sf "http://127.0.0.1:${TEACHER_PORT}/health_generate" >/dev/null; then
  echo "Teacher server did not become healthy in time. Check ${TEACHER_LOG}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${STUDENT_GPUS}" \
ray start \
  --head \
  --node-ip-address 127.0.0.1 \
  --port "${RAY_PORT}" \
  --num-gpus "${RAY_NUM_GPUS}" \
  --disable-usage-stats \
  --dashboard-host 127.0.0.1 \
  --dashboard-port "${RAY_DASHBOARD_PORT}" \
  --temp-dir "${RAY_TEMP_DIR}"
STARTED_RAY=1

MODEL_ARGS=(
  --swiglu
  --num-layers 28
  --hidden-size 2048
  --ffn-hidden-size 6144
  --num-attention-heads 16
  --group-query-attention
  --num-query-groups 8
  --use-rotary-position-embeddings
  --disable-bias-linear
  --normalization RMSNorm
  --norm-epsilon 1e-6
  --rotary-base 1000000
  --vocab-size 151936
  --kv-channels 128
  --qk-layernorm
  --seq-length "${TRAIN_SEQ_LENGTH}"
)

if [[ -n "${TRAIN_PYTORCH_CUDA_ALLOC_CONF}" ]]; then
  printf -v TRAIN_ENV_VARS_JSON '{"PYTORCH_CUDA_ALLOC_CONF":"%s"}' "${TRAIN_PYTORCH_CUDA_ALLOC_CONF}"
else
  TRAIN_ENV_VARS_JSON='{}'
fi

TRAIN_ARGS=(
  --actor-num-nodes 1
  --actor-num-gpus-per-node "${TRAIN_ACTOR_GPUS_PER_NODE}"
  --num-gpus-per-node "${RAY_NUM_GPUS}"
  --hf-checkpoint "${STUDENT_MODEL}"
  --ref-load "${STUDENT_MODEL}"
  --megatron-to-hf-mode bridge
  --save "${RUN_ROOT}/checkpoints"
  --save-interval "${SAVE_INTERVAL}"
  --save-rollout-trajectories-dir "${TRAJECTORY_DIR}"
  --prompt-data "${PREPARED_DATA}"
  --input-key messages
  --label-key answer
  --apply-chat-template
  --apply-chat-template-kwargs "${APPLY_CHAT_TEMPLATE_KWARGS}"
  --rollout-shuffle
  --rm-type math
  --reward-key accuracy
  --eval-reward-key accuracy
  --custom-rm-path slime.rollout.on_policy_distillation.reward_func
  --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
  --rm-url "http://127.0.0.1:${TEACHER_PORT}/generate"
  --eval-interval "${EVAL_INTERVAL}"
  --eval-config "${EVAL_CONFIG}"
  --n-samples-per-eval-prompt 1
  --eval-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --eval-top-k 1
  --num-epoch "${NUM_EPOCH}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-temperature 0.8
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --advantage-estimator grpo
  --train-env-vars "${TRAIN_ENV_VARS_JSON}"
  --train-memory-margin-bytes "${TRAIN_MEMORY_MARGIN_BYTES}"
  --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}"
  --use-opd
  --opd-type sglang
  --opd-kl-coef 1.0
  --use-kl-loss
  --kl-loss-coef 0.00
  --kl-loss-type low_var_kl
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
  # Megatron only accepts `adam`, but its optimizer config uses decoupled weight decay by default,
  # which is AdamW semantics.
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --tensor-model-parallel-size 2
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --qkv-format "${QKV_FORMAT}"
  --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-mem-fraction-static "${STUDENT_SGLANG_MEM_FRACTION}"
  --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}"
  --sglang-enable-metrics
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --attention-backend flash
)

if [[ "${USE_DYNAMIC_BATCH_SIZE}" == "1" ]]; then
  TRAIN_ARGS+=(
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  )
else
  TRAIN_ARGS+=(
    --micro-batch-size "${MICRO_BATCH_SIZE}"
  )
fi

if [[ "${ENABLE_ACTIVATION_RECOMPUTE}" == "1" ]]; then
  TRAIN_ARGS+=(
    --recompute-granularity "${RECOMPUTE_GRANULARITY}"
    --recompute-method "${RECOMPUTE_METHOD}"
    --recompute-num-layers "${RECOMPUTE_NUM_LAYERS}"
  )
fi

if [[ "${ENABLE_LOSS_RECOMPUTE}" == "1" ]]; then
  TRAIN_ARGS+=(--recompute-loss-function)
fi

if [[ "${ENABLE_SEQUENCE_PARALLEL}" == "1" ]]; then
  TRAIN_ARGS+=(--sequence-parallel)
fi

TRAIN_ARGS+=(
  --use-wandb
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${RUN_ID}"
  --wandb-always-use-train-step
  --disable-wandb-random-suffix
)

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  TRAIN_ARGS+=(
    --wandb-key "${WANDB_API_KEY}"
  )
fi

RUNTIME_ENV_JSON="$(ROOT_DIR="${ROOT_DIR}" MEGATRON_LM_PATH="${MEGATRON_LM_PATH}" CUDA_HOME="${CUDA_HOME}" CUDA_LIB64="${CUDA_LIB64}" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" LIBRARY_PATH="${LIBRARY_PATH:-}" python - <<'PY'
import json
import os

root_dir = os.environ["ROOT_DIR"]
megatron_lm_path = os.environ["MEGATRON_LM_PATH"]
cuda_home = os.environ.get("CUDA_HOME", "")
cuda_lib64 = os.environ.get("CUDA_LIB64", "")
ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
library_path = os.environ.get("LIBRARY_PATH", "")

print(
    json.dumps(
        {
            "env_vars": {
                "PYTHONPATH": f"{megatron_lm_path}:{root_dir}",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "CUDA_HOME": cuda_home,
                "LD_LIBRARY_PATH": ld_library_path,
                "LIBRARY_PATH": library_path,
                "no_proxy": "127.0.0.1,localhost",
                "NO_PROXY": "127.0.0.1,localhost",
            }
        }
    )
)
PY
)"

echo "Submitting Ray job..."
CUDA_VISIBLE_DEVICES="${STUDENT_GPUS}" \
ray job submit \
  --address "http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
  --runtime-env-json "${RUNTIME_ENV_JSON}" \
  -- python "${ROOT_DIR}/train.py" "${MODEL_ARGS[@]}" "${TRAIN_ARGS[@]}"
