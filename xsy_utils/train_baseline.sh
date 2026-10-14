#!/usr/bin/env bash
# Baseline (no RTC) LingBot-VLA v2 training on RoboTwin clean 50 tasks x 50 episodes.
# Code = this repo (linbot-vla-v2, branch xiesy). RTC stays off: training_time_rtc defaults
# to False. lmx's data / checkpoints are read-only.
#
# Usage: bash xsy_utils/train_baseline.sh [extra --model.* / --data.* / --train.* overrides]
# Env:   RUN_NAME (default baseline_<timestamp>), CUDA_VISIBLE_DEVICES, MICRO_BS, GLOBAL_BS, MASTER_PORT
set -euo pipefail

XSY_UTILS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WS="${WS:-$(cd -- "$XSY_UTILS/../.." && pwd)}"          # /2025233210/xiesy/lingbot_ws
CODE="$(cd -- "$XSY_UTILS/.." && pwd)"                 # this repo
CONFIG="${CONFIG:-$XSY_UTILS/configs/robotwin_clean_baseline.yaml}"
RUN_NAME="${RUN_NAME:-baseline_$(date +%Y%m%d_%H%M%S)}"
OUT="$WS/outputs/$RUN_NAME"
MICRO_BS="${MICRO_BS:-32}"

# Envs live on shared storage (container-local /root/miniconda3/envs does not survive restarts).
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-$WS/conda_envs}"
source /root/miniconda3/etc/profile.d/conda.sh
set +u; conda activate "${TRAIN_ENV:-lingbotvla}"; set -u

# PYTHONPATH takes precedence over any editable installs, so every module
# (lingbotvla, moge, mdm, utils3d) resolves to this checkout.
VM="$CODE/lingbotvla/models/vla/vision_models"
export PYTHONPATH="$CODE:$VM/MoGe:$VM/lingbot-depth:$CODE/third_party/utils3d"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DISABLE_TELEMETRY=1

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then NPROC=$(nvidia-smi -L | wc -l)
else NPROC=$(tr ',' '\n' <<<"$CUDA_VISIBLE_DEVICES" | wc -l); fi
GLOBAL_BS="${GLOBAL_BS:-$((MICRO_BS * NPROC))}"
GRAD_ACC=$((GLOBAL_BS / (MICRO_BS * NPROC)))
if (( GRAD_ACC * MICRO_BS * NPROC != GLOBAL_BS )); then echo "GLOBAL_BS must be a multiple of MICRO_BS*NPROC" >&2; exit 2; fi

mkdir -p "$OUT"
cp "$CONFIG" "$OUT/config.yaml"
{
    echo "date: $(date -Is)"; echo "host: $(hostname)"; echo "nproc: $NPROC"
    echo "micro_bs: $MICRO_BS grad_acc: $GRAD_ACC global_bs: $GLOBAL_BS"
    echo "code_rev: $(git -C "$CODE" rev-parse HEAD)"
    echo "extra_args: $*"
} > "$OUT/run_info.txt"
git -C "$CODE" diff > "$OUT/code.diff"

cd "$CODE"
torchrun --nnodes=1 --nproc-per-node "$NPROC" --master-port="${MASTER_PORT:-62511}" \
    tasks/vla/train_lingbotvla.py "$CONFIG" \
    --train.output_dir "$OUT" \
    --train.micro_batch_size "$MICRO_BS" \
    --train.global_batch_size "$GLOBAL_BS" \
    --train.gradient_accumulation_steps "$GRAD_ACC" \
    "$@" 2>&1 | tee -a "$OUT/train.log"
