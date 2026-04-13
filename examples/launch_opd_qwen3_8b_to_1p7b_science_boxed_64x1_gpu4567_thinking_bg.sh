#!/usr/bin/env bash
# nohup bash /idfsdata/yexuyan/OPD/examples/launch_opd_qwen3_8b_to_1p7b_science_boxed_64x1_gpu4567_thinking_bg.sh > /idfsdata/yexuyan/OPD/runlogs/flat64x1_thinking_gpu4567_launcher_$(date -u +%Y%m%d_%H%M%S).log 2>&1 &

set -euo pipefail

export APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking": true}'
export RUN_ID_PREFIX="opd_qwen3_8b_to_1p7b_science_boxed_64x1_gpu4567_thinking"
export TEACHER_GPUS="4"
export STUDENT_GPUS="5,6,7"
export TEACHER_MEM_FRACTION="0.72"
export TEACHER_MAX_RUNNING_REQUESTS="1"
export TEACHER_CHUNKED_PREFILL_SIZE="1024"
export TEACHER_PORT="31086"
export RAY_PORT="6386"
export RAY_DASHBOARD_PORT="8271"
export RAY_DASHBOARD_AGENT_LISTEN_PORT="52980"
export RAY_DASHBOARD_AGENT_GRPC_PORT="52981"
export RAY_RUNTIME_ENV_AGENT_PORT="52982"
export RAY_METRICS_EXPORT_PORT="52983"
export RAY_TEMP_DIR="/idfsdata/yexuyan/ray4567t64x1"

exec bash /idfsdata/yexuyan/OPD/examples/launch_opd_qwen3_8b_to_1p7b_science_boxed_64x1_gpu0123_flat.sh
