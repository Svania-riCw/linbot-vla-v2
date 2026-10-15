#!/bin/bash
# Single (task, task_config) eval on one GPU -- a thin front end over eval_parallel.sh.
#
# Usage:  bash eval.sh <hf_ckpt> [task_name] [task_config] [test_num] [gpu_id] [seed]
# Example (smoke test, 1 episode):
#   bash eval.sh /2025233210/xiesy/lingbot_ws/outputs/baseline_gbs128_5k/checkpoints/global_step_5000/hf_ckpt \
#       stack_bowls_three demo_clean 1 0
set -euo pipefail

export MODEL_PATH=${1:?usage: eval.sh <hf_ckpt> [task_name] [task_config] [test_num] [gpu_id] [seed]}
export TASKS=${2:-stack_bowls_three}
export TASK_CONFIGS=${3:-demo_clean}
export TEST_NUM=${4:-1}
export GPUS=${5:-0}
export SEED=${6:-0}

exec bash "$(dirname "${BASH_SOURCE[0]}")/eval_parallel.sh"
