#!/usr/bin/env bash
# nohup bash /idfsdata/yexuyan/OPD/examples/launch_opd_qwen3_8b_to_1p7b_science_boxed_64x1_gpu4567_flat.sh > /idfsdata/yexuyan/OPD/runlogs/flat64x1_launcher_$(date -u +%Y%m%d_%H%M%S).log 2>&1 &

set -euo pipefail

# --- 基础路径配置 ---
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

TEACHER_MODEL="/idfsdata/yexuyan/OPD/models/Qwen3-8B"
STUDENT_MODEL="/idfsdata/yexuyan/OPD/models/Qwen3-1.7B"
RAW_DATA="/idfsdata/yexuyan/OPD/data/science_with_answer.jsonl"
PREPARED_DATA="/idfsdata/yexuyan/OPD/data/science_with_answer_opd_prompt_only_boxed.jsonl"
EVAL_CONFIG="/idfsdata/yexuyan/OPD/examples/mmlu_pro_dev40_eval.yaml"

# --- 硬件与端口配置 (GPU 4,5,6,7) ---
TEACHER_GPUS="4"
STUDENT_GPUS="5,6,7"
TEACHER_PORT="31081"
RAY_PORT="6381"
RAY_DASHBOARD_PORT="8266"
RAY_TEMP_DIR="/idfsdata/yexuyan/opd_ray_tmp_4567_64x1"

# --- 训练超参数 ---
NUM_EPOCH="1"
ROLLOUT_BATCH_SIZE="64"
N_SAMPLES_PER_PROMPT="1"
GLOBAL_BATCH_SIZE="64"
ROLLOUT_MAX_PROMPT_LEN="4096"
ROLLOUT_MAX_RESPONSE_LEN="6000"
ROLLOUT_TEMPERATURE="1.0"
ROLLOUT_TOP_P="0.95"
TRAIN_SEQ_LENGTH="8192"
MAX_TOKENS_PER_GPU="6144"
MICRO_BATCH_SIZE="1"
LOG_PROBS_CHUNK_SIZE="64"
TRAIN_MEMORY_MARGIN_BYTES="1073741824"
TRAIN_PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64,garbage_collection_threshold:0.6"
ENABLE_ACTIVATION_RECOMPUTE="1"
RECOMPUTE_GRANULARITY="full"
RECOMPUTE_METHOD="uniform"
RECOMPUTE_NUM_LAYERS="1"
ENABLE_LOSS_RECOMPUTE="1"
TEACHER_MEM_FRACTION="0.85"
STUDENT_SGLANG_MEM_FRACTION="0.30"
SGLANG_MAX_RUNNING_REQUESTS="12"
WANDB_PROJECT="OPSD"
APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'
EVAL_INTERVAL="5"
SAVE_INTERVAL="20"
MAX_CHECKPOINTS="10"
TRAIN_ACTOR_GPUS_PER_NODE="2"
ROLLOUT_NUM_GPUS="1"
ROLLOUT_NUM_GPUS_PER_ENGINE="1"
RAY_NUM_GPUS="3"
TEACHER_TP="1"
ENABLE_SEQUENCE_PARALLEL="0"
QKV_FORMAT="bshd"
USE_DYNAMIC_BATCH_SIZE="0"

# --- 日志与 ID ---
RUN_ID="opd_qwen3_8b_to_1p7b_science_boxed_64x1_gpu4567_flat_$(date -u +%Y%m%d_%H%M%S)"
RUN_ROOT="${ROOT_DIR}/runs/${RUN_ID}"
LOG_ROOT="${ROOT_DIR}/runlogs"
MAIN_LOG="${LOG_ROOT}/${RUN_ID}.log"
TEACHER_LOG="${LOG_ROOT}/${RUN_ID}_teacher.log"
TRAJECTORY_DIR="${RUN_ROOT}/rollout_trajectories"

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${RUN_ROOT}/pids" "${RUN_ROOT}/checkpoints"

# --- 环境激活 ---
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

# ==========================================
# 安全清理逻辑：仅杀掉 GPU 4, 5, 6, 7 上属于 yexuyan 的进程
# ==========================================
echo "Cleaning up processes for user $(whoami) on GPUs 4, 5, 6, 7..."
for g in 4 5 6 7; do
  pids=$(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)
  if [[ -z "$pids" ]]; then continue; fi
  for pid in $pids; do
    if ps -p "$pid" -o user= | grep -q "$(whoami)"; then
      echo "Killing own process $pid on GPU $g"
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
done

pkill -u "$(whoami)" -9 -f 'sglang.launch_server' || true
pkill -u "$(whoami)" -9 -f 'ray::' || true
pkill -u "$(whoami)" -9 -f 'raylet' || true
pkill -u "$(whoami)" -9 -f 'gcs_server' || true
pkill -u "$(whoami)" -9 -f 'dashboard.py' || true
pkill -u "$(whoami)" -9 -f 'train.py' || true
pkill -u "$(whoami)" -9 -f "${TEACHER_MODEL}" || true
pkill -u "$(whoami)" -9 -f "${STUDENT_MODEL}" || true

ray stop --force || true
sleep 3
# ==========================================

# --- 环境自检 ---
python "${ROOT_DIR}/scripts/check_slime_env.py" \
  --repo-root "${ROOT_DIR}" \
  --sglang-src "${SGLANG_SRC}" \
  --megatron-src "${MEGATRON_LM_PATH}" \
  --teacher-model "${TEACHER_MODEL}" \
  --student-model "${STUDENT_MODEL}" \
  --raw-data "${RAW_DATA}"

# --- 数据准备 ---
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
        if not line: continue
        item = json.loads(line)
        messages = item.get("messages") or []
        user_message = next((m.get("content") for m in messages if m.get("role") == "user"), None)
        answer = item.get("answer")
        if not user_message or answer is None: continue
        format_instruction = (
            "\n\nThink through the problem carefully and write out your detailed reasoning. "
            "The very last non-empty line must be exactly in the format: Final Answer: \\boxed{A}\n"
            "Replace A with the single correct option letter only. Do not put any extra text inside \\boxed{}."
        )
        cleaned = {
            "messages": [{"role": "user", "content": user_message + format_instruction}],
            "answer": answer,
            "source": item.get("source"),
        }
        dst.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
        count += 1
if count == 0: raise SystemExit("Prepared dataset is empty.")
print(f"Prepared {count} prompt-only OPD samples at {prepared_path}")
PY

TEACHER_PID=""
PRUNER_PID=""
STARTED_RAY=0

cleanup() {
  local exit_code=$?
  [[ -n "${TEACHER_PID}" ]] && kill "${TEACHER_PID}" 2>/dev/null || true
  [[ -n "${PRUNER_PID}" ]] && kill "${PRUNER_PID}" 2>/dev/null || true
  [[ "${STARTED_RAY}" == "1" ]] && ray stop --force >/dev/null 2>&1 || true
  exit "${exit_code}"
}
trap cleanup EXIT INT TERM

# --- Checkpoint 管理 ---
checkpoint_pruner() {
  local ckpt_dir="$1"; local keep_count="$2"
  while true; do
    mapfile -t ckpts < <(find "${ckpt_dir}" -maxdepth 1 -mindepth 1 -type d -name 'iter_*' | sort)
    if (( ${#ckpts[@]} > keep_count )); then
      for ((i=0; i<${#ckpts[@]}-keep_count; i++)); do rm -rf "${ckpts[$i]}"; done
    fi
    sleep 30
  done
}
checkpoint_pruner "${RUN_ROOT}/checkpoints" "${MAX_CHECKPOINTS}" > /dev/null 2>&1 &
PRUNER_PID=$!

# --- 启动 Teacher Server (GPU 4) ---
CUDA_VISIBLE_DEVICES="${TEACHER_GPUS}" \
python -m sglang.launch_server \
  --model-path "${TEACHER_MODEL}" \
  --host 127.0.0.1 \
  --port "${TEACHER_PORT}" \
  --tp "${TEACHER_TP}" \
  --mem-fraction-static "${TEACHER_MEM_FRACTION}" \
  >"${TEACHER_LOG}" 2>&1 &
TEACHER_PID=$!

for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${TEACHER_PORT}/health_generate" >/dev/null; then
    echo "Teacher server is healthy on GPU ${TEACHER_GPUS}."
    break
  fi
  sleep 5
done

# --- 启动 Ray (GPU 5, 6, 7) ---
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

# --- 训练参数配置 ---
MODEL_ARGS=(
  --swiglu --num-layers 28 --hidden-size 2048 --ffn-hidden-size 6144
  --num-attention-heads 16 --group-query-attention --num-query-groups 8
  --use-rotary-position-embeddings --disable-bias-linear --normalization RMSNorm
  --norm-epsilon 1e-6 --rotary-base 1000000 --vocab-size 151936
  --kv-channels 128 --qk-layernorm --seq-length "${TRAIN_SEQ_LENGTH}"
)

printf -v TRAIN_ENV_VARS_JSON '{"PYTORCH_CUDA_ALLOC_CONF":"%s"}' "${TRAIN_PYTORCH_CUDA_ALLOC_CONF}"

TRAIN_ARGS=(
  --actor-num-nodes 1 --actor-num-gpus-per-node "${TRAIN_ACTOR_GPUS_PER_NODE}"
  --num-gpus-per-node "${RAY_NUM_GPUS}" --hf-checkpoint "${STUDENT_MODEL}"
  --ref-load "${STUDENT_MODEL}" --megatron-to-hf-mode bridge
  --save "${RUN_ROOT}/checkpoints" --save-interval "${SAVE_INTERVAL}"
  --save-rollout-trajectories-dir "${TRAJECTORY_DIR}"
  --prompt-data "${PREPARED_DATA}" --input-key messages --label-key answer
  --apply-chat-template --apply-chat-template-kwargs "${APPLY_CHAT_TEMPLATE_KWARGS}"
  --rollout-shuffle --rm-type math --reward-key accuracy --eval-reward-key accuracy
  --custom-rm-path slime.rollout.on_policy_distillation.reward_func
  --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
  --rm-url "http://127.0.0.1:${TEACHER_PORT}/generate"
  --eval-interval "${EVAL_INTERVAL}" --eval-config "${EVAL_CONFIG}"
  --n-samples-per-eval-prompt 1 --eval-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --num-epoch "${NUM_EPOCH}" --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}" --advantage-estimator grpo
  --train-env-vars "${TRAIN_ENV_VARS_JSON}" --train-memory-margin-bytes "${TRAIN_MEMORY_MARGIN_BYTES}"
  --use-opd --opd-type sglang --opd-kl-coef 1.0 --optimizer adam --lr 1e-6
  --tensor-model-parallel-size 2 --pipeline-model-parallel-size 1
  --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" --sglang-mem-fraction-static "${STUDENT_SGLANG_MEM_FRACTION}"
  --micro-batch-size "${MICRO_BATCH_SIZE}" --recompute-granularity "${RECOMPUTE_GRANULARITY}"
  --use-wandb --wandb-project "${WANDB_PROJECT}" --wandb-group "${RUN_ID}"
)

# ==========================================
# 关键修复：解决 Ray Worker 找不到 sglang 的环境配置
# ==========================================
RUNTIME_ENV_JSON="$(ROOT_DIR="${ROOT_DIR}" MEGATRON_LM_PATH="${MEGATRON_LM_PATH}" CONDA_ENV_PATH="${CONDA_ENV_PATH}" CUDA_HOME="${CUDA_HOME}" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" python - <<'PY'
import json
import os
print(json.dumps({
    "conda": os.environ['CONDA_ENV_PATH'], # 显式传递 Conda 环境路径
    "env_vars": {
        "PYTHONPATH": f"{os.environ['MEGATRON_LM_PATH']}:{os.environ['ROOT_DIR']}",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "CUDA_HOME": os.environ.get("CUDA_HOME", ""),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
        "no_proxy": "127.0.0.1,localhost",
        "NO_PROXY": "127.0.0.1,localhost",
    }
}))
PY
)"

# --- 提交 Job ---
echo "Submitting Ray Job for ID: ${RUN_ID} on GPUs ${STUDENT_GPUS}..."
CUDA_VISIBLE_DEVICES="${STUDENT_GPUS}" \
ray job submit \
  --address "http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
  --runtime-env-json "${RUNTIME_ENV_JSON}" \
  -- python "${ROOT_DIR}/train.py" "${MODEL_ARGS[@]}" "${TRAIN_ARGS[@]}" \
  2>&1 | tee "${MAIN_LOG}"