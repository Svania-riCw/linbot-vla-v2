#!/usr/bin/env bash
# RoboTwin evaluation wrapper for a LingBot-VLA RTC checkpoint.
# Defaults follow xsy_utils/eval_baseline.sh: seven representative tasks and
# 25 episodes per task. TASK_CONFIGS can include clean and randomized settings.
#
# Usage:
#   bash xsy_utils/eval_rtc_checkpoint.sh [--dry-run] [checkpoint_dir] [launcher options]
#
# Useful overrides: TASK_CONFIGS, TASKS, TEST_NUM, NUM_GPUS, NUM_PER_GPU,
# RTC_REPLAN_STEPS, RTC_DELAY, RUN_DIR, QWEN3VL_PATH, EVAL_WORKDIR, OUTPUT_BASE.
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
workspace=$(dirname -- "$repo")
checkpoint=${MODEL_PATH:-$workspace/checkpoints/lingbot-vla-v2-6b-robotwin-rtc/checkpoints/global_step_10000}
qwen=${QWEN3VL_PATH:-$workspace/checkpoints/Qwen3-VL-4B-Instruct}
eval_workdir=${EVAL_WORKDIR:-$workspace/RoboTwin}
output_base=${OUTPUT_BASE:-$eval_workdir/eval_results}
task_configs=${TASK_CONFIGS:-demo_clean demo_randomized}
selected_tasks=${TASKS:-blocks_ranking_rgb hanging_mug open_laptop pick_diverse_bottles rotate_qrcode stack_blocks_three place_shoe}
test_num=${TEST_NUM:-25}
num_gpus=${NUM_GPUS:-7}
num_per_gpu=${NUM_PER_GPU:-1}
rtc_replan_steps=${RTC_REPLAN_STEPS:-8}
rtc_delay=${RTC_DELAY:-6}
use_length=${USE_LENGTH:-50}
start_port=${START_PORT:-9330}
run_dir_override=${RUN_DIR:-}
video=${VIDEO:-true}
rtc_norm_stats=${LINGBOT_NORM_STATS_FILE:-$workspace/training-data/robotwin_clean_norm_stats.json}
dry_run=false

if [[ ${1:-} == --dry-run ]]; then
    dry_run=true
    shift
fi
if [[ ${1:-} == -h || ${1:-} == --help ]]; then
    sed -n '2,9p' "$0"
    cat <<'EOF'

Environment defaults:
  MODEL_PATH       checkpoint directory (defaults to this workspace's step 10000)
  TASK_CONFIGS     demo_clean demo_randomized
  TASKS            seven representative RoboTwin tasks
  TEST_NUM         25 episodes per task
  NUM_GPUS         7; uses logical GPUs 0..NUM_GPUS-1
  NUM_PER_GPU      1 inference server per GPU
  RTC_REPLAN_STEPS 8; RTC_DELAY 6
  LINGBOT_NORM_STATS_FILE defaults to this RTC checkpoint's clean training stats
  VIDEO            true; set false to disable video recording
  Pass --no_rtc after the checkpoint to use synchronous inference without RTC.
  Pass --rtc_mode inference_time --use_compile False for training-free RTC.
  --rtc_guidance_beta sets the inference-time guidance clip (default: 5.0).
  RUN_DIR          fixed directory to resume a prior run; with multiple settings,
                   each setting gets its own subdirectory

QWEN3VL_PATH, EVAL_WORKDIR, OUTPUT_BASE, CONDA_SH, INFERENCE_ENV, and SIM_ENV
are forwarded to eval_rtc.sh. Additional launcher options may follow the checkpoint.
EOF
    exit 0
fi
if [[ $# -gt 0 && ${1:-} != -* ]]; then
    checkpoint=$1
    shift
fi
launcher_args=("$@")
run_prefix=training_time_rtc
# eval_rtc.sh selects training-time RTC; explicit later options can override it.
for ((i=0; i<${#launcher_args[@]}; i++)); do
    case "${launcher_args[$i]}" in
        --no_rtc) run_prefix=sync ;;
        --rtc_async) run_prefix=training_time_rtc ;;
        --rtc_mode)
            case "${launcher_args[$((i+1))]:-}" in
                off) run_prefix=sync ;;
                training_time|fixed_prefix|async_fixed_prefix_v2) run_prefix=training_time_rtc ;;
                inference_time|paper) run_prefix=inference_time_rtc ;;
            esac
            i=$((i + 1))
            ;;
    esac
done

checkpoint=$(realpath -- "$checkpoint")
qwen=$(realpath -- "$qwen")
eval_workdir=$(realpath -- "$eval_workdir")
output_base=$(realpath -m -- "$output_base")
rtc_norm_stats=$(realpath -- "$rtc_norm_stats")
if [[ -d "$checkpoint/hf_ckpt" ]]; then
    checkpoint_label=$(basename -- "$checkpoint")
elif [[ $(basename -- "$checkpoint") == hf_ckpt ]]; then
    checkpoint_label=$(basename -- "$(dirname -- "$checkpoint")")
else
    checkpoint_label=$(basename -- "$checkpoint")
fi
[[ -d "$checkpoint" ]] || { echo "Checkpoint directory not found: $checkpoint" >&2; exit 2; }
[[ -f "$qwen/config.json" ]] || { echo "Qwen3-VL config not found under: $qwen" >&2; exit 2; }
[[ -f "$rtc_norm_stats" ]] || { echo "RTC normalization statistics not found: $rtc_norm_stats" >&2; exit 2; }
[[ -d "$eval_workdir/envs" && -d "$eval_workdir/task_config" ]] || {
    echo "RoboTwin repository not found at: $eval_workdir" >&2
    exit 2
}
export LINGBOT_NORM_STATS_FILE="$rtc_norm_stats"
echo "[eval_rtc_checkpoint] normalization statistics: $LINGBOT_NORM_STATS_FILE"
if [[ -n "${LINGBOT_INFERENCE_CONFIG:-}" ]]; then
    echo "[eval_rtc_checkpoint] using explicit inference config: $LINGBOT_INFERENCE_CONFIG (its data.norm_stats_file takes precedence)"
fi
[[ $test_num =~ ^[1-9][0-9]*$ && $num_gpus =~ ^[1-9][0-9]*$ && $num_per_gpu =~ ^[1-9][0-9]*$ ]] || {
    echo 'TEST_NUM, NUM_GPUS, and NUM_PER_GPU must be positive integers.' >&2
    exit 2
}
[[ $rtc_replan_steps =~ ^[1-9][0-9]*$ && $rtc_delay =~ ^[0-9]+$ && $use_length =~ ^[1-9][0-9]*$ ]] || {
    echo 'RTC_REPLAN_STEPS and USE_LENGTH must be positive; RTC_DELAY must be nonnegative.' >&2
    exit 2
}

read -r -a task_config_list <<< "${task_configs//,/ }"
read -r -a task_list <<< "${selected_tasks//,/ }"
(( ${#task_config_list[@]} > 0 && ${#task_list[@]} > 0 )) || {
    echo 'TASK_CONFIGS and TASKS must each contain at least one value.' >&2
    exit 2
}
for setting in "${task_config_list[@]}"; do
    case "$setting" in demo_clean|demo_randomized) ;; *) echo "Invalid task config: $setting" >&2; exit 2 ;; esac
done
tasks_csv=$(IFS=,; printf '%s' "${task_list[*]}")
video_args=()
case "$video" in
    true|True|TRUE|1|yes|YES) ;;
    false|False|FALSE|0|no|NO) video_args+=(--no_video) ;;
    *) echo 'VIDEO must be true or false.' >&2; exit 2 ;;
esac

timestamp=$(date +%Y%m%d_%H%M%S)
failed=0
for setting in "${task_config_list[@]}"; do
    if [[ -n "$run_dir_override" ]]; then
        if (( ${#task_config_list[@]} > 1 )); then
            run_dir="${run_dir_override%/}/$setting"
        else
            run_dir=$run_dir_override
        fi
    else
        run_dir="$output_base/${run_prefix}_${checkpoint_label}_${setting}_${timestamp}"
    fi

    cmd=(bash "$repo/eval_rtc.sh" "$checkpoint" "$qwen" "$setting"
        --output_base "$output_base" --run_dir "$run_dir"
        --start_port "$start_port" --num_gpus "$num_gpus" --num_per_gpu "$num_per_gpu"
        --num_episodes "$test_num" --tasks "$tasks_csv"
        --rtc_replan_steps "$rtc_replan_steps" --rtc_delay "$rtc_delay"
        --use_length "$use_length" "${video_args[@]}" "${launcher_args[@]}")

    echo "[eval_rtc_checkpoint] setting=$setting checkpoint=$checkpoint episodes/task=$test_num tasks=${task_list[*]} GPUs=$num_gpus run_dir=$run_dir"
    printf '[eval_rtc_checkpoint] command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    if $dry_run; then
        continue
    fi

    start=$(date +%s)
    rc=0
    "${cmd[@]}" || rc=$?
    elapsed=$(( $(date +%s) - start ))
    printf '[eval_rtc_checkpoint] finished setting=%s rc=%d elapsed=%dh%02dm%02ds\n' "$setting" "$rc" \
        $((elapsed / 3600)) $((elapsed % 3600 / 60)) $((elapsed % 60))
    if (( rc != 0 )); then
        failed=$((failed + 1))
    fi
done

if $dry_run; then
    echo '[eval_rtc_checkpoint] dry run complete; no evaluation was started.'
elif (( failed > 0 )); then
    echo "[eval_rtc_checkpoint] $failed setting(s) failed." >&2
else
    echo '[eval_rtc_checkpoint] all settings completed successfully.'
fi
exit "$failed"
