#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/tools/rtc_paths.sh"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage: bash train_rtc.sh [--prepare-only] [--auto | CONFIG.yaml] [training arguments...]

The default recipe uses the local official LingBot 6B weights and clean
RoboTwin LeRobot datasets under $LMX_ROOT/training-data/lerobot. It also
enables the MoGe, MoRGBD, and DINO-video auxiliary teacher losses.
On 8 GPUs: micro batch 16 * GPU count 8 * accumulation 1 = global batch 128.
Checkpoints are saved every 1000 steps. The RTC model code is part of this
checkout; no patch is applied or reverted.

Options via environment: RTC_TRAIN_ENV, RTC_MAX_DELAY, RTC_MICRO_BATCH_SIZE,
RTC_GLOBAL_BATCH_SIZE, RTC_GRADIENT_CHECKPOINTING, RTC_MOGE_PATH, RTC_OUTPUT_DIR.
Set RTC_SAVE_STEPS to change the checkpoint interval. Set RTC_USE_TEACHERS=false
to disable the MoGe, MoRGBD, and DINO-video auxiliary losses.
EOF
    exit 0
fi

prepare_only=0
if [[ "${1:-}" == "--prepare-only" ]]; then
    prepare_only=1
    shift
fi
config_arg="--auto"
if [[ $# -gt 0 && "$1" != --* ]]; then
    config_arg="$1"
    shift
elif [[ "${1:-}" == "--auto" ]]; then
    shift
fi
rtc_activate_env

if [[ "$config_arg" == "--auto" ]]; then
    config_path="$RTC_REPO_ROOT/.runtime/rtc/robotwin_clean/robotwin_rtc.yaml"
    python "$RTC_REPO_ROOT/tools/prepare_rtc_training.py" --root "$LMX_ROOT" --output "$config_path"
else
    config_path="$(realpath "$config_arg")"
fi
if [[ ! -f "$config_path" ]]; then
    printf 'Training config not found: %s\n' "$config_path" >&2
    exit 2
fi
if [[ "$prepare_only" == 1 ]]; then
    printf 'Prepared training config: %s\n' "$config_path"
    exit 0
fi

export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 DISABLE_TELEMETRY=1
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        IFS=',' read -r -a rtc_devices <<< "$CUDA_VISIBLE_DEVICES"
        NPROC_PER_NODE="${#rtc_devices[@]}"
    else
        NPROC_PER_NODE="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
    fi
fi
cd "$RTC_REPO_ROOT"
rtc_log="${RTC_LOG_PATH:-$RTC_REPO_ROOT/.runtime/rtc/train.log}"
mkdir -p "$(dirname -- "$rtc_log")"
torchrun --nnodes="${NNODES:-1}" --nproc-per-node="$NPROC_PER_NODE" \
    --node-rank="${NODE_RANK:-0}" --master-addr="${MASTER_ADDR:-127.0.0.1}" \
    --master-port="${MASTER_PORT:-62500}" \
    tasks/vla/train_lingbotvla.py "$config_path" "$@" \
    --train.training_time_rtc true \
    --train.training_time_rtc_max_delay "${RTC_MAX_DELAY:-10}" 2>&1 | tee -a "$rtc_log"
