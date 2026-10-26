#!/usr/bin/env bash
# Evaluate the res1 checkpoint over all 50 RoboTwin tasks on three nodes.
# Each task runs in demo_clean and demo_randomized with the same fixed-delay RTC
# settings. Set EVAL_NODE_ID to 1, 2, or 3 before invoking this script.
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
workspace=$(dirname -- "$repo")

node_id=${EVAL_NODE_ID:-}
case "$node_id" in
    1|2|3) ;;
    *)
        echo 'Set EVAL_NODE_ID to 1, 2, or 3.' >&2
        exit 2
        ;;
esac

checkpoint=${MODEL_PATH:-/2025233210/xiesy/lingbot_ckpts/d4rt_q96_ctxvs_res1_augon/checkpoints/global_step_15000/hf_ckpt}
run_dir=${RUN_DIR:-/2025233210/lmx/eval_rtc_res1_all50_10_2_32_node${node_id}}
task_configs=${TASK_CONFIGS:-"demo_clean demo_randomized"}
test_num=${TEST_NUM:-10}
num_gpus=${NUM_GPUS:-8}
num_per_gpu=${NUM_PER_GPU:-1}
video=${VIDEO:-false}
start_port=${START_PORT:-9680}
image_size=${IMAGE_SIZE:-256}
rtc_mode=${RTC_MODE:-paper_fixed_delay}
rtc_replan_steps=${RTC_REPLAN_STEPS:-32}
rtc_delay=${RTC_DELAY:-2}
rtc_guidance_beta=${RTC_GUIDANCE_BETA:-5}
norm_stats=${LINGBOT_NORM_STATS_FILE:-$workspace/training-data/robotwin_clean_norm_stats.json}
dry_run=${DRY_RUN:-false}

case "$node_id" in
    1)
        tasks=(
            adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size
            click_alarmclock click_bell dump_bin_bigbin grab_roller handover_block
            handover_mic hanging_mug lift_pot move_can_pot move_pillbottle_pad
            move_playingcard_away move_stapler_pad open_laptop
        )
        ;;
    2)
        tasks=(
            open_microwave pick_diverse_bottles pick_dual_bottles place_a2b_left
            place_a2b_right place_bread_basket place_bread_skillet place_burger_fries
            place_can_basket place_cans_plasticbox place_container_plate place_dual_shoes
            place_empty_cup place_fan place_mouse_pad place_object_basket place_object_scale
        )
        ;;
    3)
        tasks=(
            place_object_stand place_phone_stand place_shoe press_stapler
            put_bottles_dustbin put_object_cabinet rotate_qrcode scan_object shake_bottle
            shake_bottle_horizontally stack_blocks_three stack_blocks_two stack_bowls_three
            stack_bowls_two stamp_seal turn_switch
        )
        ;;
esac

if (( ${#tasks[@]} != (node_id == 3 ? 16 : 17) )); then
    echo "Unexpected task count for node $node_id: ${#tasks[@]}" >&2
    exit 2
fi
for ((i = 0; i < ${#tasks[@]}; i++)); do
    for ((j = i + 1; j < ${#tasks[@]}; j++)); do
        if [[ "${tasks[$i]}" == "${tasks[$j]}" ]]; then
            echo "Duplicate task on node $node_id: ${tasks[$i]}" >&2
            exit 2
        fi
    done
done
if [[ ! "$image_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "IMAGE_SIZE must be a positive integer: $image_size" >&2
    exit 2
fi
if [[ ! "$rtc_guidance_beta" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "RTC_GUIDANCE_BETA must be a nonnegative number: $rtc_guidance_beta" >&2
    exit 2
fi
case "$video" in
    true|True|TRUE|1|yes|YES|false|False|FALSE|0|no|NO) ;;
    *) echo 'VIDEO must be true or false.' >&2; exit 2 ;;
esac
case "$dry_run" in
    true|True|TRUE|1|yes|YES) dry_run_args=(--dry-run) ;;
    false|False|FALSE|0|no|NO) dry_run_args=() ;;
    *) echo 'DRY_RUN must be true or false.' >&2; exit 2 ;;
esac

unset LINGBOT_INFERENCE_CONFIG
export MODEL_PATH="$checkpoint"
export QWEN3VL_PATH="${QWEN3VL_PATH:-$workspace/checkpoints/Qwen3-VL-4B-Instruct}"
export EVAL_WORKDIR="${EVAL_WORKDIR:-$workspace/RoboTwin}"

printf '[three-node eval] node=%s tasks=%d image_size=%s episodes/task=%s\n' \
    "$node_id" "${#tasks[@]}" "$image_size" "$test_num"
printf '[three-node eval] checkpoint=%s\n' "$checkpoint"
printf '[three-node eval] run_dir=%s\n' "$run_dir"
printf '[three-node eval] tasks=%s\n' "${tasks[*]}"

eval_args=(
    "${dry_run_args[@]}" "$checkpoint"
    --image_size "$image_size"
    --rtc_mode "$rtc_mode"
    --use_compile False
    --use_bf16 False
    --use_fp32 True
    --rtc_guidance_beta "$rtc_guidance_beta"
)

TASK_CONFIGS="$task_configs" \
TASKS="${tasks[*]}" \
TEST_NUM="$test_num" \
NUM_GPUS="$num_gpus" \
NUM_PER_GPU="$num_per_gpu" \
VIDEO="$video" \
START_PORT="$start_port" \
RUN_DIR="$run_dir" \
RTC_REPLAN_STEPS="$rtc_replan_steps" \
RTC_DELAY="$rtc_delay" \
LINGBOT_NORM_STATS_FILE="$norm_stats" \
    bash "$repo/xsy_utils/eval_rtc_checkpoint.sh" "${eval_args[@]}"
