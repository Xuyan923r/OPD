#!/usr/bin/env bash
# Dedicated alias for 4,5,6,7 GRPO launch to avoid confusing with 0123-named file.
# nohup bash /idfsdata/yexuyan/OPD/examples/launch_grpo_qwen3_1p7b_rlvr_science_boxed_16x8_gpu4567_bg.sh > /idfsdata/yexuyan/OPD/runlogs/grpo_1p7b_rlvr_16x8_gpu4567_launcher_$(date -u +%Y%m%d_%H%M%S).log 2>&1 &

set -euo pipefail

exec bash /idfsdata/yexuyan/OPD/examples/launch_grpo_qwen3_1p7b_rlvr_science_boxed_16x8_gpu0123_bg.sh
