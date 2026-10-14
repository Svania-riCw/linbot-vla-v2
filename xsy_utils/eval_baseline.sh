#!/usr/bin/env bash
# Full RoboTwin eval of a baseline checkpoint: 7 tasks x {demo_clean, demo_randomized} x TEST_NUM
# trials on all GPUs. Thin wrapper around external/RoboTwin/policy/lingbot-vla-v2/eval_parallel.sh;
# every env var of that script can be overridden here as well.
#
#   bash xsy_utils/eval_baseline.sh [hf_ckpt]
#
# Output: external/RoboTwin/eval_result/<model>/run_<ts>/_result.txt (+ averages), videos, logs.
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${WS:-$(dirname "$REPO")}"

export MODEL_PATH="${1:-${MODEL_PATH:-$WS/outputs/baseline_gbs128_5k/checkpoints/global_step_5000/hf_ckpt}}"
export TEST_NUM="${TEST_NUM:-25}"
export TASK_CONFIGS="${TASK_CONFIGS:-demo_clean demo_randomized}"
export TASKS="${TASKS:-blocks_ranking_rgb hanging_mug open_laptop pick_diverse_bottles rotate_qrcode stack_blocks_three place_shoe}"
export GPUS="${GPUS:-0 1 2 3 4 5 6 7}"

start=$(date +%s)
echo "[eval_baseline] start $(date '+%F %T')  model=$MODEL_PATH  test_num=$TEST_NUM  gpus=$GPUS"
rc=0
bash "$REPO/external/RoboTwin/policy/lingbot-vla-v2/eval_parallel.sh" || rc=$?
el=$(( $(date +%s) - start ))
printf '[eval_baseline] end %s  rc=%d  elapsed %dh%02dm%02ds\n' "$(date '+%F %T')" "$rc" \
    $((el / 3600)) $((el % 3600 / 60)) $((el % 60))
exit "$rc"
