#!/usr/bin/env bash
# nohup bash /idfsdata/yexuyan/OPD/examples/launch_opd_qwen3_8b_to_1p7b_science_boxed_64x1_gpu0123_thinking_bg.sh > /idfsdata/yexuyan/OPD/runlogs/flat64x1_thinking_launcher_$(date -u +%Y%m%d_%H%M%S).log 2>&1 &

set -euo pipefail

export APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking": true}'
export RUN_ID_PREFIX="opd_qwen3_8b_to_1p7b_science_boxed_64x1_gpu0123_thinking"
export TEACHER_PORT="31084"
export RAY_PORT="6385"
export RAY_DASHBOARD_PORT="8270"
export RAY_DASHBOARD_AGENT_LISTEN_PORT="52880"
export RAY_DASHBOARD_AGENT_GRPC_PORT="52881"
export RAY_RUNTIME_ENV_AGENT_PORT="52882"
export RAY_METRICS_EXPORT_PORT="52883"
export RAY_TEMP_DIR="/idfsdata/yexuyan/ray0123t64x1"

exec bash /idfsdata/yexuyan/OPD/examples/launch_opd_qwen3_8b_to_1p7b_science_boxed_64x1_gpu0123_flat.sh
