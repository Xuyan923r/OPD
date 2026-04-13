#!/usr/bin/env bash
# nohup bash /idfsdata/yexuyan/OPD/examples/launch_grpo_qwen3_1p7b_rlvr_science_boxed_16x8_gpu0123_bg.sh > /idfsdata/yexuyan/OPD/runlogs/grpo_1p7b_rlvr_16x8_gpu4567_launcher_$(date -u +%Y%m%d_%H%M%S).log 2>&1 &

set -euo pipefail

ROOT_DIR="/idfsdata/yexuyan/OPD"
cd "${ROOT_DIR}"

CONDA_SH="/home/yexuyan/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV_PATH="/idfsdata/yexuyan/conda_envs/slime"
SGLANG_SRC="/idfsdata/yexuyan/slime_deps/sglang"
MEGATRON_LM_PATH="/idfsdata/yexuyan/slime_deps/Megatron-LM"
CUDA_HOME="/usr/local/cuda"
HOST_CC="/usr/bin/gcc-11"
HOST_CXX="/usr/bin/g++-11"
HOST_CUDAHOSTCXX="/usr/bin/g++-11"

MODEL_PATH="/idfsdata/yexuyan/OPD/models/Qwen3-1.7B"
RAW_DATA="/idfsdata/yexuyan/OPD/data/science_with_answer.jsonl"
PREPARED_DATA="/idfsdata/yexuyan/OPD/data/science_with_answer_grpo_rlvr_prompt_only_boxed.jsonl"
EVAL_CONFIG="/idfsdata/yexuyan/OPD/examples/mmlu_pro_dev40_eval.yaml"

STUDENT_GPUS="4,5,6,7"
RAY_PORT="6384"
RAY_DASHBOARD_PORT="8269"
RAY_TEMP_DIR="/tmp/ray_g4567_rlvr"
RAY_DASHBOARD_AGENT_LISTEN_PORT="52780"
RAY_DASHBOARD_AGENT_GRPC_PORT="52781"
RAY_RUNTIME_ENV_AGENT_PORT="52782"
RAY_METRICS_EXPORT_PORT="52783"
CURRENT_USER="$(whoami)"

NUM_EPOCH="${NUM_EPOCH:-1}"
NUM_ROLLOUT="${NUM_ROLLOUT:-205}"
ROLLOUT_BATCH_SIZE="16"
N_SAMPLES_PER_PROMPT="8"
GLOBAL_BATCH_SIZE="128"
ROLLOUT_MAX_PROMPT_LEN="4096"
ROLLOUT_MAX_RESPONSE_LEN="6000"
ROLLOUT_TEMPERATURE="1.0"
ROLLOUT_TOP_P="0.95"
TRAIN_SEQ_LENGTH="8192"
MICRO_BATCH_SIZE="1"
LOG_PROBS_CHUNK_SIZE="64"
TRAIN_MEMORY_MARGIN_BYTES="1073741824"
TRAIN_PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64,garbage_collection_threshold:0.6"
RECOMPUTE_GRANULARITY="full"
RECOMPUTE_METHOD="uniform"
RECOMPUTE_NUM_LAYERS="1"
STUDENT_SGLANG_MEM_FRACTION="0.45"
SGLANG_MAX_RUNNING_REQUESTS="32"
WANDB_PROJECT="OPSD"
APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'
EVAL_INTERVAL="5"
SAVE_INTERVAL="20"
MAX_CHECKPOINTS="${MAX_CHECKPOINTS:-0}"
TRAIN_ACTOR_GPUS_PER_NODE="2"
ROLLOUT_NUM_GPUS="2"
ROLLOUT_NUM_GPUS_PER_ENGINE="1"
RAY_NUM_GPUS="4"
QKV_FORMAT="bshd"

RUN_ID="grpo_qwen3_1p7b_rlvr_science_boxed_16x8_gpu4567_$(date -u +%Y%m%d_%H%M%S)"
RUN_ROOT="${ROOT_DIR}/runs/${RUN_ID}"
LOG_ROOT="${ROOT_DIR}/runlogs"
MAIN_LOG="${LOG_ROOT}/${RUN_ID}.log"
LAUNCHER_LOG="${LOG_ROOT}/${RUN_ID}_launcher.log"
TRAJECTORY_DIR="${RUN_ROOT}/rollout_trajectories"

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${RUN_ROOT}/pids" "${RUN_ROOT}/checkpoints" "${RAY_TEMP_DIR}"

set +u
source "${CONDA_SH}"
conda activate "${CONDA_ENV_PATH}"
set -u

export CC="${HOST_CC}"
export CXX="${HOST_CXX}"
export CUDAHOSTCXX="${HOST_CUDAHOSTCXX}"
export PYTHONUNBUFFERED=1
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

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

kill_own_pid() {
  local pid="$1"
  if [[ -z "${pid}" ]]; then
    return 0
  fi
  if ! kill -0 "${pid}" 2>/dev/null; then
    return 0
  fi
  if [[ "$(ps -p "${pid}" -o user= 2>/dev/null | tr -d ' ')" != "${CURRENT_USER}" ]]; then
    return 0
  fi
  kill -TERM "${pid}" 2>/dev/null || true
  sleep 0.2
  kill -KILL "${pid}" 2>/dev/null || true
}

kill_own_gpu_processes() {
  local g pid pids
  for g in "$@"; do
    pids=$(nvidia-smi -i "${g}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)
    [[ -z "${pids}" ]] && continue
    for pid in ${pids}; do
      echo "Cleaning GPU ${g} process pid=${pid} (owner=${CURRENT_USER})"
      kill_own_pid "${pid}"
    done
  done
}

kill_own_port_listener() {
  local port="$1"
  local pid
  while read -r pid; do
    [[ -z "${pid}" ]] && continue
    echo "Cleaning port ${port} listener pid=${pid} (owner=${CURRENT_USER})"
    kill_own_pid "${pid}"
  done < <(lsof -t -iTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true)
}

echo "Cleaning only ${CURRENT_USER}'s processes on GPUs 4,5,6,7 and ports ${RAY_PORT}/${RAY_DASHBOARD_PORT}/${RAY_DASHBOARD_AGENT_LISTEN_PORT}/${RAY_DASHBOARD_AGENT_GRPC_PORT}/${RAY_RUNTIME_ENV_AGENT_PORT}/${RAY_METRICS_EXPORT_PORT}..."
kill_own_gpu_processes 4 5 6 7
kill_own_port_listener "${RAY_PORT}"
kill_own_port_listener "${RAY_DASHBOARD_PORT}"
kill_own_port_listener "${RAY_DASHBOARD_AGENT_LISTEN_PORT}"
kill_own_port_listener "${RAY_DASHBOARD_AGENT_GRPC_PORT}"
kill_own_port_listener "${RAY_RUNTIME_ENV_AGENT_PORT}"
kill_own_port_listener "${RAY_METRICS_EXPORT_PORT}"
sleep 2

python "${ROOT_DIR}/scripts/check_slime_env.py" \
  --repo-root "${ROOT_DIR}" \
  --sglang-src "${SGLANG_SRC}" \
  --megatron-src "${MEGATRON_LM_PATH}" \
  --teacher-model "${MODEL_PATH}" \
  --student-model "${MODEL_PATH}" \
  --raw-data "${RAW_DATA}"

RAW_DATA="${RAW_DATA}" PREPARED_DATA="${PREPARED_DATA}" python - <<'PY'
import json
import os
import re
from pathlib import Path

raw_path = Path(os.environ["RAW_DATA"])
prepared_path = Path(os.environ["PREPARED_DATA"])
prepared_path.parent.mkdir(parents=True, exist_ok=True)

format_instruction = (
    "\n\nThink through the problem carefully and write out your detailed reasoning. "
    "The very last non-empty line must be exactly in the format: Final Answer: \\boxed{A}\n"
    "Replace A with the single correct option letter only. Do not put any extra text inside \\boxed{}."
)

def normalize_answer(answer: object) -> str | None:
    text = str(answer).strip().upper()
    if len(text) == 1 and text.isalpha():
        return text
    boxed_match = re.search(r"\\BOXED\{\s*([A-Z])\s*\}", text)
    if boxed_match:
        return boxed_match.group(1)
    letter_match = re.search(r"\b([A-Z])\b", text)
    if letter_match:
        return letter_match.group(1)
    return None

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
        answer = normalize_answer(item.get("answer"))
        if not user_message or answer is None:
            continue
        cleaned = {
            "messages": [{"role": "user", "content": user_message + format_instruction}],
            "answer": answer,
            "source": item.get("source"),
        }
        dst.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
        count += 1
if count == 0:
    raise SystemExit("Prepared dataset is empty.")
print(f"Prepared {count} RLVR prompt-only samples at {prepared_path}")
PY

PRUNER_PID=""
STARTED_RAY=0

cleanup() {
  local exit_code=$?
  if [[ -n "${PRUNER_PID}" ]] && kill -0 "${PRUNER_PID}" 2>/dev/null; then
    kill "${PRUNER_PID}" 2>/dev/null || true
    wait "${PRUNER_PID}" 2>/dev/null || true
  fi
  if [[ "${STARTED_RAY}" == "1" ]]; then
    kill_own_port_listener "${RAY_DASHBOARD_PORT}"
    kill_own_port_listener "${RAY_PORT}"
    kill_own_port_listener "${RAY_DASHBOARD_AGENT_LISTEN_PORT}"
    kill_own_port_listener "${RAY_DASHBOARD_AGENT_GRPC_PORT}"
    kill_own_port_listener "${RAY_RUNTIME_ENV_AGENT_PORT}"
    kill_own_port_listener "${RAY_METRICS_EXPORT_PORT}"
    kill_own_gpu_processes 4 5 6 7
  fi
  exit "${exit_code}"
}
trap cleanup EXIT INT TERM

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

if (( MAX_CHECKPOINTS > 0 )); then
  checkpoint_pruner "${RUN_ROOT}/checkpoints" "${MAX_CHECKPOINTS}" > /dev/null 2>&1 &
  PRUNER_PID=$!
  echo "${PRUNER_PID}" > "${RUN_ROOT}/pids/checkpoint_pruner.pid"
else
  echo "Checkpoint pruning disabled (MAX_CHECKPOINTS=${MAX_CHECKPOINTS})."
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
  --dashboard-agent-listen-port "${RAY_DASHBOARD_AGENT_LISTEN_PORT}" \
  --dashboard-agent-grpc-port "${RAY_DASHBOARD_AGENT_GRPC_PORT}" \
  --runtime-env-agent-port "${RAY_RUNTIME_ENV_AGENT_PORT}" \
  --metrics-export-port "${RAY_METRICS_EXPORT_PORT}" \
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

printf -v TRAIN_ENV_VARS_JSON '{"PYTORCH_CUDA_ALLOC_CONF":"%s"}' "${TRAIN_PYTORCH_CUDA_ALLOC_CONF}"

TRAIN_ARGS=(
  --actor-num-nodes 1
  --actor-num-gpus-per-node "${TRAIN_ACTOR_GPUS_PER_NODE}"
  --num-gpus-per-node "${RAY_NUM_GPUS}"
  --hf-checkpoint "${MODEL_PATH}"
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
  --rm-type mmlu_pro
  --reward-key accuracy
  --eval-reward-key accuracy
  --custom-rm-path slime.rollout.rlvr_student_only.reward_func
  --custom-reward-post-process-path slime.rollout.rlvr_student_only.post_process_rewards
  --eval-interval "${EVAL_INTERVAL}"
  --eval-config "${EVAL_CONFIG}"
  --n-samples-per-eval-prompt 1
  --eval-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --eval-top-k 1
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE}"
  --rollout-top-p "${ROLLOUT_TOP_P}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --advantage-estimator grpo
  --train-env-vars "${TRAIN_ENV_VARS_JSON}"
  --train-memory-margin-bytes "${TRAIN_MEMORY_MARGIN_BYTES}"
  --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}"
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
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
  --micro-batch-size "${MICRO_BATCH_SIZE}"
  --recompute-granularity "${RECOMPUTE_GRANULARITY}"
  --recompute-method "${RECOMPUTE_METHOD}"
  --recompute-num-layers "${RECOMPUTE_NUM_LAYERS}"
  --recompute-loss-function
  --use-wandb
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${RUN_ID}"
  --wandb-always-use-train-step
  --disable-wandb-random-suffix
)

if [[ -n "${NUM_ROLLOUT}" ]]; then
  TRAIN_ARGS+=(--num-rollout "${NUM_ROLLOUT}")
else
  TRAIN_ARGS+=(--num-epoch "${NUM_EPOCH}")
fi

RUNTIME_ENV_JSON="$(ROOT_DIR="${ROOT_DIR}" MEGATRON_LM_PATH="${MEGATRON_LM_PATH}" CUDA_HOME="${CUDA_HOME}" CUDA_LIB64="${CUDA_LIB64}" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" LIBRARY_PATH="${LIBRARY_PATH:-}" HOST_CC="${HOST_CC}" HOST_CXX="${HOST_CXX}" HOST_CUDAHOSTCXX="${HOST_CUDAHOSTCXX}" python - <<'PY'
import json
import os
print(json.dumps({
    "env_vars": {
        "PYTHONPATH": f"{os.environ['MEGATRON_LM_PATH']}:{os.environ['ROOT_DIR']}",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "CUDA_HOME": os.environ.get("CUDA_HOME", ""),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
        "LIBRARY_PATH": os.environ.get("LIBRARY_PATH", ""),
        "CC": os.environ.get("HOST_CC", ""),
        "CXX": os.environ.get("HOST_CXX", ""),
        "CUDAHOSTCXX": os.environ.get("HOST_CUDAHOSTCXX", ""),
        "no_proxy": "127.0.0.1,localhost",
        "NO_PROXY": "127.0.0.1,localhost",
    }
}))
PY
)"

echo "Run ID: ${RUN_ID}"
echo "Run root: ${RUN_ROOT}"
echo "Prepared RLVR data: ${PREPARED_DATA}"
echo "Student GPUs: ${STUDENT_GPUS}"
echo "Trajectory dir: ${TRAJECTORY_DIR}"
echo "Submitting Ray job..."

CUDA_VISIBLE_DEVICES="${STUDENT_GPUS}" \
ray job submit \
  --address "http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
  --runtime-env-json "${RUNTIME_ENV_JSON}" \
  -- python "${ROOT_DIR}/train.py" "${MODEL_ARGS[@]}" "${TRAIN_ARGS[@]}" \
  2>&1 | tee "${MAIN_LOG}"
