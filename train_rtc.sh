#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/tools/rtc_paths.sh"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage: bash train_rtc.sh [--name NAME] [--resume | --overwrite] [--prepare-only] [--auto | CONFIG.yaml] [training arguments...]

The default recipe uses the local official LingBot 6B weights and clean
RoboTwin LeRobot datasets under $LMX_ROOT/training-data/lerobot. It also
enables the MoGe, MoRGBD, and DINO-video auxiliary teacher losses.
On 8 GPUs: micro batch 16 * GPU count 8 * accumulation 1 = global batch 128.
Checkpoints are saved every 1000 steps. The RTC model code is part of this
checkout; no patch is applied or reverted.

--name NAME      Store this run under $LMX_ROOT/checkpoints/NAME.
--checkpoint-name NAME  Alias for --name.
--resume         Resume the latest global_step_N checkpoint in that run.
--overwrite      Delete that run directory and start from step 0.
Without either mode flag, an existing run directory is rejected.

Options via environment: RTC_TRAIN_ENV, RTC_MAX_DELAY, RTC_FIXED_DELAY, RTC_MICRO_BATCH_SIZE,
RTC_GLOBAL_BATCH_SIZE, RTC_GRADIENT_CHECKPOINTING, RTC_MOGE_PATH, RTC_OUTPUT_DIR.
Set RTC_SAVE_STEPS to change the checkpoint interval. Set RTC_USE_TEACHERS=false
to disable the MoGe, MoRGBD, and DINO-video auxiliary losses.
EOF
    exit 0
fi

prepare_only=0
config_arg="--auto"
config_selected=0
run_name="${RTC_RUN_NAME:-lingbot-vla-v2-6b-robotwin-rtc}"
explicit_name=0
run_mode="auto"
run_mode_selected=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prepare-only)
            prepare_only=1
            shift
            ;;
        --auto)
            config_arg="--auto"
            config_selected=1
            shift
            ;;
        --name|--run-name|--checkpoint-name|--ckpt-name)
            if [[ $# -lt 2 || -z "$2" ]]; then
                printf '%s requires a non-empty value.\n' "$1" >&2
                exit 2
            fi
            run_name="$2"
            explicit_name=1
            shift 2
            ;;
        --resume|--overwrite)
            if [[ "$run_mode_selected" == 1 ]]; then
                printf 'Choose only one of --resume or --overwrite.\n' >&2
                exit 2
            fi
            run_mode="${1#--}"
            run_mode_selected=1
            shift
            ;;
        --help|-h)
            cat <<'EOF'
Usage: bash train_rtc.sh [--name NAME] [--resume | --overwrite] [--prepare-only] [--auto | CONFIG.yaml] [training arguments...]

--name NAME      Store this run under $LMX_ROOT/checkpoints/NAME.
--resume         Resume the latest global_step_N checkpoint in that run.
--overwrite      Remove that run directory and start from step 0.
Without a mode flag, a non-empty existing run directory is rejected.
EOF
            exit 0
            ;;
        --*)
            break
            ;;
        *)
            if [[ "$config_selected" == 0 ]]; then
                config_arg="$1"
                config_selected=1
                shift
            else
                break
            fi
            ;;
    esac
done

checkpoint_root="$(realpath -m "$LMX_ROOT/checkpoints")"
if [[ "$explicit_name" == 0 && -n "${RTC_OUTPUT_DIR:-}" ]]; then
    if [[ -L "$RTC_OUTPUT_DIR" ]]; then
        printf 'Refusing to use a symlink as the run directory: %s\n' "$RTC_OUTPUT_DIR" >&2
        exit 2
    fi
    output_dir="$(realpath -m "$RTC_OUTPUT_DIR")"
    run_name="$(basename -- "$output_dir")"
    if [[ "$(dirname -- "$output_dir")" != "$checkpoint_root" ]]; then
        printf 'RTC_OUTPUT_DIR must be a direct child of %s.\n' "$checkpoint_root" >&2
        exit 2
    fi
else
    output_dir="$checkpoint_root/$run_name"
fi
if [[ ! "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ || "$run_name" == "." || "$run_name" == ".." ]]; then
    printf 'Invalid run name %q. Use only letters, numbers, dot, underscore, and hyphen.\n' "$run_name" >&2
    exit 2
fi
if [[ -L "$output_dir" ]]; then
    printf 'Refusing to use a symlink as the run directory: %s\n' "$output_dir" >&2
    exit 2
fi
export RTC_OUTPUT_DIR="$output_dir"

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
    printf 'Checkpoint run directory: %s\n' "$output_dir"
    exit 0
fi

mkdir -p "$checkpoint_root"
run_key="$(printf '%s' "$output_dir" | sha256sum | cut -c1-16)"
lock_file="${TMPDIR:-/tmp}/lingbot-vla-rtc-${run_key}.lock"
exec 9>"$lock_file"
if ! flock -n 9; then
    printf 'Another RTC launcher is using this run directory: %s\n' "$output_dir" >&2
    exit 2
fi

resume_enabled=false
case "$run_mode" in
    resume)
        latest_checkpoint="$(find "$output_dir/checkpoints" -mindepth 1 -maxdepth 1 -type d \
            -regextype posix-extended -regex '.*/global_step_[0-9]+' -print 2>/dev/null \
            | sort -V | tail -n 1 || true)"
        if [[ -z "$latest_checkpoint" ]]; then
            printf 'No global_step_N checkpoint found under %s.\n' "$output_dir/checkpoints" >&2
            exit 2
        fi
        resume_enabled=true
        printf 'Resuming from latest checkpoint in %s\n' "$output_dir/checkpoints"
        ;;
    overwrite)
        if [[ -e "$output_dir" ]]; then
            printf 'Overwriting RTC run directory: %s\n' "$output_dir"
            rm -rf -- "$output_dir"
        fi
        ;;
    auto)
        if [[ -e "$output_dir" ]] && find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit | rg -q .; then
            printf 'Run directory already exists: %s\nUse --resume or --overwrite explicitly.\n' "$output_dir" >&2
            exit 2
        fi
        ;;
    *)
        printf 'Unknown RTC run mode: %s\n' "$run_mode" >&2
        exit 2
        ;;
esac

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
    --train.training_time_rtc_max_delay "${RTC_MAX_DELAY:-10}" \
    --train.training_time_rtc_fixed_delay "${RTC_FIXED_DELAY:--1}" \
    --train.output_dir "$output_dir" \
    --train.enable_resume "$resume_enabled" 2>&1 | tee -a "$rtc_log"
