#!/bin/bash
# ============================================================
# Inference + simulation launcher (queue-scheduled).
# num_per_gpu inference servers stay resident per GPU; sim tasks match
# inference slots from a queue. Finishing a task frees its slot and starts the next.
#
# Usage: bash start_robotwin_infer_and_eval.sh [options]
#   --model_path        inference model path (default: /path/to/your/checkpoint, or $MODEL_PATH)
#   --inference_script  inference-side module path (default: deploy/lingbot_vla_v2_policy.py)
#   --inference_workdir inference-side working dir (default: current working dir)
#   --eval_workdir      sim-side (RoboTwin repo) working dir (REQUIRED; or $EVAL_WORKDIR; default placeholder /path/to/RoboTwin)
#   --inference_env     inference-side conda env (default: lingbotvla, or $INFERENCE_ENV)
#   --sim_env           sim-side conda env (default: RoboTwin, or $SIM_ENV)
#   --conda_sh          conda.sh path to source (default: /path/to/miniconda3/etc/profile.d/conda.sh, or $CONDA_SH)
#   --output_base       result output path (default: /path/to/VLABenchmarkResult, or $OUTPUT_BASE)
#   --run_dir           fixed run directory; rerun with the same directory to resume
#   --start_port        starting port (default: 9330)
#   --pid_name          PID file prefix (default: test_pid)
#   --num_tasks         number of sim tasks, taken in order from the task list (default: 50, max: 50)
#   --task_name         evaluate one named task
#   --tasks             comma-separated or quoted space-separated named tasks (overrides --num_tasks)
#   --list_tasks        print available task names and exit
#   --num_gpus          total GPUs (default: 1)
#   --num_per_gpu       inference servers per GPU (default: 1)
#   --inference_gpu_ids comma-separated physical GPU IDs for inference servers
#   --sim_gpu_ids       comma-separated physical GPU IDs for RoboTwin clients
#   --use_length        chunk length (default: 50)
#   --robo_name         robot config name (default: robotwin)
#   --video_fps         video recording fps (default: 30)
#   --no_video          disable video recording to speed up simulation
#   --video             enable video recording (default: enabled)
#   --use_bf16          use bfloat16 inference (default: False)
#   --use_fp32          use float32 inference (default: True; release reproduction setting)
#   --use_compile       enable model compile in policy inference (default: True)
#   --task_config       RoboTwin task config (default: demo_clean, or $TASK_CONFIG)
#   --no_rtc            disable RTC (overrides an earlier RTC mode)
#   --keep_inference    keep inference servers resident after simulation
#
# Examples:
#   bash start_robotwin_infer_and_eval.sh --eval_workdir /path/to/RoboTwin --num_tasks 50 --num_gpus 4 --num_per_gpu 1
#   bash start_robotwin_infer_and_eval.sh --eval_workdir /path/to/RoboTwin --sim_env RoboTwin --num_tasks 20 --num_per_gpu 2
# ============================================================

# ===== Parse keyword arguments =====
export QWEN3VL_PATH="${QWEN3VL_PATH:-/path/to/your/checkpoints/Qwen3-VL-4B-Instruct}"

project_root="$(pwd)"
inference_workdir="${project_root}/"
inference_script="deploy/lingbot_vla_v2_policy.py"
model_path="${MODEL_PATH:-/path/to/your/checkpoint}"
eval_workdir="${EVAL_WORKDIR:-/path/to/RoboTwin}"
output_base="${OUTPUT_BASE:-/path/to/VLABenchmarkResult}"
run_dir=""
# Conda env names + conda.sh path for both sides (overridable via flags or env).
inference_env="${INFERENCE_ENV:-lingbotvla}"
sim_env="${SIM_ENV:-RoboTwin}"
conda_sh="${CONDA_SH:-/path/to/miniconda3/etc/profile.d/conda.sh}"
start_port=9330
pid_name="test_pid"
num_tasks=50
num_episodes=100
selected_tasks=""
list_tasks=false
num_gpus=1
num_per_gpu=1
inference_gpu_ids=""
sim_gpu_ids=""
use_length=50
use_bf16=False
use_fp32=True
use_compile=True
robo_name="robotwin"
video_fps=30
enable_video=True
rtc_async=False
rtc_mode=off
rtc_guidance_beta=5.0
rtc_replan_steps=8
rtc_delay=6
rtc_timeout=180
keep_inference=false
task_config="${TASK_CONFIG:-demo_clean}"
instruction_type="${INSTRUCTION_TYPE:-}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --inference_workdir) inference_workdir="$2"; shift 2 ;;
        --inference_script)  inference_script="$2";  shift 2 ;;
        --model_path)        model_path="$2";        shift 2 ;;
        --eval_workdir)      eval_workdir="$2";      shift 2 ;;
        --output_base)       output_base="$2";       shift 2 ;;
        --run_dir)           run_dir="$2";           shift 2 ;;
        --start_port)        start_port="$2";        shift 2 ;;
        --pid_name)          pid_name="$2";          shift 2 ;;
        --num_tasks)         num_tasks="$2";         shift 2 ;;
        --num_episodes)      num_episodes="$2";      shift 2 ;;
        --task_name|--tasks)
            if [[ $# -lt 2 || -z "${2//[[:space:],]/}" ]]; then
                echo "$1 requires at least one task name." >&2
                exit 2
            fi
            selected_tasks="${selected_tasks:+$selected_tasks,}$2"; shift 2 ;;
        --list_tasks)        list_tasks=true;         shift ;;
        --num_gpus)          num_gpus="$2";          shift 2 ;;
        --num_per_gpu)       num_per_gpu="$2";       shift 2 ;;
        --inference_gpu_ids) inference_gpu_ids="$2"; shift 2 ;;
        --sim_gpu_ids)       sim_gpu_ids="$2";       shift 2 ;;
        --rtc_async)         rtc_async=True; rtc_mode=training_time; shift ;;
        --no_rtc)            rtc_async=False; rtc_mode=off; shift ;;
        --rtc_mode)          rtc_mode="$2"; shift 2 ;;
        --rtc_guidance_beta) rtc_guidance_beta="$2"; shift 2 ;;
        --rtc_replan_steps)  rtc_replan_steps="$2"; shift 2 ;;
        --rtc_delay)         rtc_delay="$2"; shift 2 ;;
        --rtc_timeout)       rtc_timeout="$2"; shift 2 ;;
        --use_length)        use_length="$2";        shift 2 ;;
        --use_bf16)          use_bf16="$2";        shift 2 ;;
        --use_fp32)          use_fp32="$2";        shift 2 ;;
        --use_compile)       use_compile="$2";     shift 2 ;;
        --robo_name)         robo_name="$2";         shift 2 ;;
        --task_config)       task_config="$2";       shift 2 ;;
        --instruction_type)  instruction_type="$2";  shift 2 ;;
        --video_fps)         video_fps="$2";         shift 2 ;;
        --no_video)          enable_video=False;     shift ;;
        --video)             enable_video=True;      shift ;;
        --keep_inference)    keep_inference=true;    shift ;;
        --inference_env)     inference_env="$2";     shift 2 ;;
        --sim_env)           sim_env="$2";           shift 2 ;;
        --conda_sh)          conda_sh="$2";          shift 2 ;;
        -h|--help)
            echo "Usage: bash $0 [options]"
            echo "  --model_path        inference model path"
            echo "  --inference_script  inference-side module path"
            echo "  --inference_workdir inference-side working dir"
            echo "  --eval_workdir      sim-side (RoboTwin repo) working dir (REQUIRED; or \$EVAL_WORKDIR)"
            echo "  --inference_env     inference-side conda env (default: lingbotvla, or \$INFERENCE_ENV)"
            echo "  --sim_env           sim-side conda env (default: RoboTwin, or \$SIM_ENV)"
            echo "  --conda_sh          conda.sh path to source (default: /path/to/miniconda3/etc/profile.d/conda.sh, or \$CONDA_SH)"
            echo "  --output_base       result output path"
            echo "  --run_dir           fixed run directory; automatically resume saved episodes"
            echo "  --start_port        starting port (default: 9330)"
            echo "  --pid_name          PID file prefix (default: test_pid)"
            echo "  --num_tasks         number of sim tasks (default: 50, max: 50)"
            echo "  --num_episodes      evaluation episodes per task (default: 100)"
            echo "  --task_name         evaluate one named task"
            echo "  --tasks             comma-separated or quoted space-separated task names; overrides --num_tasks"
            echo "  --list_tasks        print available task names and exit"
            echo "  --num_gpus          total GPUs (default: 1)"
            echo "  --num_per_gpu       inference servers per GPU (default: 1)"
            echo "  --inference_gpu_ids physical GPU IDs for inference, e.g. 0,1,2,3"
            echo "  --sim_gpu_ids       physical GPU IDs for simulation, e.g. 4,5,6,7"
            echo "  --rtc_async         overlap inference with committed old-plan actions"
            echo "  --rtc_mode          off, training_time, inference_time, or paper_fixed_delay (fixed delay with paper guidance)"
            echo "  --rtc_guidance_beta maximum guidance weight for inference-time RTC (default: 5.0)"
            echo "  --no_rtc            disable RTC; overrides an earlier RTC mode"
            echo "  --rtc_replan_steps  target actions between inference requests, including delay (default: 8)"
            echo "  --rtc_delay         old-plan prefix actions executed during inference (default: 6; trained range: 0..9)"
            echo "  --rtc_timeout       request timeout seconds (default: 180)"
            echo "  --use_length        chunk length (default: 50)"
            echo "  --use_bf16          use bfloat16 inference (default: False)"
            echo "  --use_fp32          use float32 inference (default: True; release reproduction setting)"
            echo "  --use_compile       enable model compile (default: True)"
            echo "  --robo_name         robot config name (default: robotwin)"
            echo "  --task_config       RoboTwin task config (default: demo_clean)"
            echo "  --instruction_type  seen or unseen (defaults: clean=seen, randomized=unseen)"
            echo "  --keep_inference    keep inference servers resident after simulation"
            echo "  --video_fps         video recording fps (default: 30)"
            echo "  --no_video          disable video recording to speed up simulation"
            echo "  --video             enable video recording (default: enabled)"
            exit 0 ;;
        *)
            echo -e "\033[31mUnknown argument: $1\033[0m"; exit 1 ;;
    esac
done

case "$rtc_mode" in
    off) rtc_async=False ;;
    training_time|fixed_prefix|async_fixed_prefix_v2) rtc_mode=training_time; rtc_async=True ;;
    inference_time|paper) rtc_mode=inference_time; rtc_async=True ;;
    paper_fixed_delay) rtc_async=True ;;
    *) echo 'rtc_mode must be off, training_time, or inference_time' >&2; exit 2 ;;
esac
if [[ "$rtc_mode" == inference_time || "$rtc_mode" == paper_fixed_delay ]] && [[ "$use_compile" != False ]]; then
    echo 'Inference-time RTC requires --use_compile False for VJP autograd' >&2; exit 2
fi

if [ -z "$instruction_type" ]; then
    if [ "$task_config" = demo_clean ]; then
        instruction_type=seen
    else
        instruction_type=unseen
    fi
fi
case "$instruction_type" in
    seen|unseen) ;;
    *) echo "instruction_type must be seen or unseen" >&2; exit 2 ;;
esac
if ! [[ "$num_episodes" =~ ^[1-9][0-9]*$ ]]; then
    echo "num_episodes must be a positive integer." >&2
    exit 2
fi

if ! [[ "$rtc_replan_steps" =~ ^[1-9][0-9]*$ && "$rtc_delay" =~ ^[0-9]+$ ]]; then
    echo 'RTC replan steps must be positive and delay nonnegative integers' >&2; exit 2
fi
if [[ "$rtc_async" == True ]] && (( rtc_delay > rtc_replan_steps )); then
    echo 'RTC delay cannot exceed the interval between inference requests' >&2; exit 2
fi
# ===== Select and validate tasks before starting environments or services =====
task_list_all=("lift_pot" "hanging_mug" "stack_bowls_three" "scan_object" "handover_block" "click_bell" "put_object_cabinet" "open_microwave" "stack_blocks_three" "place_shoe" "adjust_bottle" "beat_block_hammer" "blocks_ranking_rgb" "blocks_ranking_size" "click_alarmclock" "dump_bin_bigbin" "grab_roller" "handover_mic" "move_can_pot" "move_pillbottle_pad" "move_playingcard_away" "place_cans_plasticbox" "place_container_plate" "place_dual_shoes" "place_empty_cup" "place_fan" "place_mouse_pad" "place_object_basket" "place_object_scale" "place_object_stand" "place_phone_stand" "move_stapler_pad" "open_laptop" "pick_diverse_bottles" "pick_dual_bottles" "place_a2b_left" "place_a2b_right" "place_bread_basket" "place_bread_skillet" "place_burger_fries" "place_can_basket" "press_stapler" "rotate_qrcode" "shake_bottle_horizontally" "shake_bottle" "stack_blocks_two" "stack_bowls_two" "stamp_seal" "turn_switch" "put_bottles_dustbin")
if $list_tasks; then
    printf '%s\n' "${task_list_all[@]}"
    exit 0
fi

task_queue=()
if [ -n "$selected_tasks" ]; then
    read -r -a requested_tasks <<< "${selected_tasks//,/ }"
    for task in "${requested_tasks[@]}"; do
        valid=false
        for known in "${task_list_all[@]}"; do
            if [ "$task" = "$known" ]; then valid=true; break; fi
        done
        if ! $valid; then
            echo "Unknown task: $task. Use --list_tasks to see available names." >&2
            exit 2
        fi
        for queued in "${task_queue[@]}"; do
            if [ "$task" = "$queued" ]; then
                echo "Duplicate task: $task" >&2
                exit 2
            fi
        done
        task_queue+=("$task")
    done
    num_tasks=${#task_queue[@]}
    if [ "$num_tasks" -eq 0 ]; then
        echo "Provide at least one task name." >&2
        exit 2
    fi
else
    if ! [[ "$num_tasks" =~ ^([1-9]|[1-4][0-9]|50)$ ]]; then
        echo "num_tasks must be an integer from 1 to 50." >&2
        exit 2
    fi
    task_queue=("${task_list_all[@]:0:num_tasks}")
fi
echo -e "\033[36mTasks this run (${num_tasks}): ${task_queue[*]}\033[0m"

# Optional dedicated GPU pools. Without these options the original behavior is
# preserved: each inference slot and its paired simulator share the same GPU.
declare -a inference_gpu_list=()
declare -a sim_gpu_list=()
if [[ -n "$inference_gpu_ids" || -n "$sim_gpu_ids" ]]; then
    if [[ -z "$inference_gpu_ids" || -z "$sim_gpu_ids" ]]; then
        echo "--inference_gpu_ids and --sim_gpu_ids must be provided together." >&2
        exit 2
    fi
    if ! [[ "$num_gpus" =~ ^[1-9][0-9]*$ && "$num_per_gpu" =~ ^[1-9][0-9]*$ ]]; then
        echo "num_gpus and num_per_gpu must be positive integers." >&2
        exit 2
    fi
    IFS=',' read -r -a inference_gpu_list <<< "$inference_gpu_ids"
    IFS=',' read -r -a sim_gpu_list <<< "$sim_gpu_ids"
    if (( ${#inference_gpu_list[@]} != num_gpus )); then
        echo "num_gpus (${num_gpus}) must match the number of --inference_gpu_ids (${#inference_gpu_list[@]})." >&2
        exit 2
    fi
    if (( ${#sim_gpu_list[@]} != ${#inference_gpu_list[@]} )); then
        echo "--sim_gpu_ids must contain the same number of IDs as --inference_gpu_ids." >&2
        exit 2
    fi
    # The inference and simulator pools may intentionally share a GPU when
    # running one slot on an isolated device. Reject duplicates within either
    # pool, but allow the same physical ID in both pools.
    declare -A seen_inference_gpu_ids=()
    declare -A seen_sim_gpu_ids=()
    for gpu_id in "${inference_gpu_list[@]}"; do
        if ! [[ "$gpu_id" =~ ^(0|[1-9][0-9]*)$ ]]; then
            echo "GPU IDs must be nonnegative integers; got '${gpu_id}'." >&2
            exit 2
        fi
        if [[ -n "${seen_inference_gpu_ids[$gpu_id]:-}" ]]; then
            echo "Inference GPU ID ${gpu_id} is repeated." >&2
            exit 2
        fi
        seen_inference_gpu_ids[$gpu_id]=1
    done
    for gpu_id in "${sim_gpu_list[@]}"; do
        if ! [[ "$gpu_id" =~ ^(0|[1-9][0-9]*)$ ]]; then
            echo "GPU IDs must be nonnegative integers; got '${gpu_id}'." >&2
            exit 2
        fi
        if [[ -n "${seen_sim_gpu_ids[$gpu_id]:-}" ]]; then
            echo "Simulation GPU ID ${gpu_id} is repeated." >&2
            exit 2
        fi
        seen_sim_gpu_ids[$gpu_id]=1
    done
fi

# Prepare resume state before loading the model or initializing the simulator.
requested_task_queue=("${task_queue[@]}")
batch_time=$(date +%Y%m%d_%H%M%S)
_exp_name=$(basename -- "$(dirname -- "$(dirname -- "$(dirname -- "$model_path")")")")
_step_num=$(echo "$model_path" | grep -oP 'global_step_\K\d+')
_step_k=$(( _step_num / 1000 ))k
run_dir="${run_dir:-${output_base}/${_exp_name}_${_step_k}_${task_config}_${batch_time}}"
mkdir -p "$run_dir"
run_dir=$(realpath "$run_dir")
# Lock the directory for this invocation and all child workers.
exec 9>"$run_dir/.run.lock"
flock -n 9 || { echo "Run directory already in use: $run_dir" >&2; exit 2; }
checkpoint_tool="${inference_workdir%/}/experiment/robotwin/eval_checkpoint.py"
inference_config="${LINGBOT_INFERENCE_CONFIG:-}"
if [ -z "$inference_config" ]; then
    inference_config=$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).parent.parent.parent / "lingbotvla_cli.yaml")' "$model_path")
fi
python3 "$checkpoint_tool" prepare --run-dir "$run_dir" --model "$model_path" \
    --config "$inference_config" --setting "$task_config" --instruction "$instruction_type" \
    --task-config-file "$eval_workdir/task_config/$task_config.yml" \
    --rtc-async "$rtc_async" --rtc-replan-steps "$rtc_replan_steps" --rtc-delay "$rtc_delay" \
    --rtc-mode "$rtc_mode" --rtc-guidance-beta "$rtc_guidance_beta" \
    --length "$use_length" --robo-name "$robo_name" \
    --precision "$use_bf16/$use_fp32/$use_compile" --episodes "$num_episodes" \
    --tasks "${requested_task_queue[@]}" || exit $?
run_id=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$run_dir/resume_plan.json")
mapfile -t task_queue < <(python3 -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["pending"]), end="")' "$run_dir/resume_plan.json")
if [ ${#task_queue[@]} -eq 0 ]; then
    echo "All selected tasks already complete; inference server is not started."
    python3 "$checkpoint_tool" stats --run-dir "$run_dir" --episodes "$num_episodes" --tasks "${requested_task_queue[@]}"
    exit $?
fi
num_tasks=${#task_queue[@]}

# ===== Common environment =====
# Cleanup: kill all child processes on exit / Ctrl-C / kill
cleanup() {
    echo ""
    echo -e "\033[33m=== Cleaning up all child processes ===\033[0m"
    # Kill inference servers
    for slot in $(seq 0 $((num_slots-1))); do
        local_pid=${inference_pids[$slot]:-0}
        if [ "$local_pid" != "0" ] && kill -0 "$local_pid" 2>/dev/null; then
            kill -TERM -- -"$(ps -o pgid= -p "$local_pid" 2>/dev/null | tr -d ' ')" 2>/dev/null \
                || kill -TERM "$local_pid" 2>/dev/null || true
        fi
    done
    # Kill eval workers
    for slot in $(seq 0 $((num_slots-1))); do
        local_pid=${slot_pid[$slot]:-0}
        if [ "$local_pid" != "0" ] && kill -0 "$local_pid" 2>/dev/null; then
            kill -TERM -- -"$(ps -o pgid= -p "$local_pid" 2>/dev/null | tr -d ' ')" 2>/dev/null \
                || kill -TERM "$local_pid" 2>/dev/null || true
        fi
    done
    sleep 1
    # Force kill survivors
    for slot in $(seq 0 $((num_slots-1))); do
        for local_pid in ${inference_pids[$slot]:-0} ${slot_pid[$slot]:-0}; do
            if [ "$local_pid" != "0" ] && kill -0 "$local_pid" 2>/dev/null; then
                kill -KILL -- -"$(ps -o pgid= -p "$local_pid" 2>/dev/null | tr -d ' ')" 2>/dev/null \
                    || kill -KILL "$local_pid" 2>/dev/null || true
            fi
        done
    done
    python3 "$checkpoint_tool" stats --run-dir "$run_dir" --episodes "$num_episodes" --tasks "${requested_task_queue[@]}" >/dev/null 2>&1 || true
    echo -e "\033[33m=== Cleanup done ===\033[0m"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM



# ===== Working dir (RoboTwin sim side) -- REQUIRED =====
# Must be supplied via --eval_workdir flag or $EVAL_WORKDIR env. No default.
if [ -z "$eval_workdir" ]; then
    eval_workdir="${EVAL_WORKDIR:-}"
fi
if [ -z "$eval_workdir" ]; then
    echo -e "\033[31mError: --eval_workdir (or \$EVAL_WORKDIR) is required (the RoboTwin repo root).\033[0m"
    exit 1
fi
if [ ! -d "$eval_workdir" ]; then
    echo -e "\033[31mError: sim workdir '${eval_workdir}' not found.\033[0m"
    echo "  Pass --eval_workdir <path> or set EVAL_WORKDIR=<path> (the RoboTwin repo root)."
    exit 1
fi
echo -e "\033[36mSim workdir: ${eval_workdir}\033[0m"

# ===== Sim-side python (conda env: ${sim_env}) =====
# conda_sh / sim_env / inference_env are resolved above (flag > env > builtin default).
if [ ! -f "$conda_sh" ]; then
    echo -e "\033[31mError: conda profile not found at ${conda_sh}\033[0m"
    exit 1
fi
# shellcheck disable=SC1090
source "$conda_sh"
if ! conda activate "$sim_env" 2>/dev/null; then
    echo -e "\033[31mError: conda env '${sim_env}' not found; create it first\033[0m"
    exit 1
fi
if ! command -v python >/dev/null 2>&1; then
    echo -e "\033[31mError: python command not found in conda env '${sim_env}'\033[0m"
    exit 1
fi
sim_python="$(command -v python)"
echo -e "\033[36mSim-side python (${sim_env}): ${sim_python} ($(${sim_python} --version 2>&1))\033[0m"
# Deactivate so later phases don't inherit sim env; the actual sim launch
# activates ${sim_env} itself via `setsid bash -c`.
conda deactivate 2>/dev/null || true

# ===== Sim-side args =====
policy_name=ACT
train_config_name=0
seed=0

# ===== Compute inference slot count =====
# actual slots = min(num_tasks, num_gpus * num_per_gpu)
# When tasks < max concurrency, spin up only as many servers as tasks to avoid
# the waste of starting then shutting down extras.
num_slots=$(( num_gpus * num_per_gpu ))
if [ "$num_tasks" -lt "$num_slots" ]; then
    echo -e "\033[36mnum_tasks ${num_tasks} < max slots ${num_slots}; starting servers by task count\033[0m"
    num_slots=$num_tasks
fi

if (( ${#inference_gpu_list[@]} > 0 )); then
    echo -e "\033[36mDedicated GPU pools: inference=[${inference_gpu_list[*]}], simulation=[${sim_gpu_list[*]}]\033[0m"
else
    echo -e "\033[36mInference/simulation share GPUs: 0..$((num_gpus - 1))\033[0m"
fi
echo -e "\033[36mInference config: ${num_gpus} GPU x ${num_per_gpu} servers/GPU = ${num_slots} slots\033[0m"
echo -e "\033[36mInference precision: use_bf16=${use_bf16}, use_fp32=${use_fp32}, use_compile=${use_compile}\033[0m"

# ===== Common variables =====
mkdir -p "${run_dir}/inference_logs" "${run_dir}/eval_logs"
log_dir="${run_dir}"
inference_pid_file="${run_dir}/inference_pids.txt"
eval_pid_file="${run_dir}/eval_pids.txt"
> "$inference_pid_file"
> "$eval_pid_file"
stats_file="${run_dir}/stats.txt"
echo -e "\033[36mRun directory: ${run_dir}\033[0m"

# Result collection arrays
declare -a result_task_names=()
declare -a result_durations=()
declare -a result_logs=()
declare -a result_status=()

# ============================================================
# Phase 1: start inference-side QwenPi servers (resident)
# ============================================================
echo -e "\033[32m========== Starting inference side: ${num_slots} QwenPi servers, ports ${start_port}~$((start_port + num_slots - 1)) ==========\033[0m"

cd "$inference_workdir" || { echo -e "\033[31mError: inference workdir ${inference_workdir} missing\033[0m"; exit 1; }

for slot in $(seq 0 $((num_slots-1))); do
    if (( ${#inference_gpu_list[@]} > 0 )); then
        gpu_id=${inference_gpu_list[$((slot % ${#inference_gpu_list[@]}))]}
    else
        gpu_id=$(( slot % num_gpus ))
    fi
    port=$(( start_port + slot ))

    export CUDA_VISIBLE_DEVICES=${gpu_id}

    log_file="${run_dir}/inference_logs/qwenpi_slot${slot}_gpu${gpu_id}_port${port}.log"

    echo -e "\033[33m[inf slot $slot] GPU: ${gpu_id}, PORT: ${port}, Log: ${log_file}\033[0m"

    # deploy/qwen3_5vl_dit_policy.py -> deploy.qwen3_5vl_dit_policy
    inference_script_for_module="${inference_script}"
    if [[ "${inference_script_for_module}" = /* ]]; then
        inference_script_for_module="${inference_script_for_module#${inference_workdir%/}/}"
    fi
    inference_module=$(echo "${inference_script_for_module}" | sed 's|/|.|g; s|\.py$||')
    setsid bash -c "source ${conda_sh} && conda activate ${inference_env} && SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 python -m ${inference_module} \
        --model_path '${model_path}' \
        --use_length '${use_length}' \
        --use_bf16 "${use_bf16}" \
        --use_fp32 "${use_fp32}" \
        --use_compile "${use_compile}" \
        --rtc_mode '${rtc_mode}' \
        --port '${port}'" > "$log_file" 2>&1 &

    pid=$!
    echo "${pid}" >> "$inference_pid_file"
done

echo -e "\033[32m${num_slots} inference servers started, PIDs saved to ${inference_pid_file}\033[0m"

# Record each slot's inference server PID for on-demand shutdown
declare -a inference_pids=()
for slot in $(seq 0 $((num_slots-1))); do
    inference_pids[$slot]=0
done
# Re-read PIDs (in launch order, mapping 1:1 to slots)
mapfile -t pid_list < "$inference_pid_file"
for slot in $(seq 0 $((num_slots-1))); do
    if [ $slot -lt ${#pid_list[@]} ]; then
        inference_pids[$slot]=${pid_list[$slot]}
    fi
done
active_inference=$num_slots

# ============================================================
# Phase 2: queue-scheduled sim tasks
# ============================================================
echo -e "\033[32m========== Starting sim side (queue-scheduled, ${num_slots} concurrent slots) ==========\033[0m"

cd "$eval_workdir" || { echo -e "\033[31mError: sim workdir ${eval_workdir} missing\033[0m"; exit 1; }

# ===== Ensure the lingbot eval client is present under RoboTwin/script =====
# The eval client resolves its ../task_config relative to its own location, so it
# must live at <RoboTwin>/script/ for _camera_config.yml to be found.
cp "$checkpoint_tool" "${eval_workdir}/script/eval_checkpoint.py"
eval_client_src="${inference_workdir%/}/experiment/robotwin/eval_policy_client_lingbotvla.py"
eval_client_dst="${eval_workdir}/script/eval_policy_client_lingbotvla.py"
if [ ! -f "$eval_client_dst" ] || ! cmp -s "$eval_client_src" "$eval_client_dst"; then
    if [ ! -f "$eval_client_src" ]; then
        echo -e "\033[31mError: eval client source not found: ${eval_client_src}\033[0m"
        exit 1
    fi
    if [ -f "$eval_client_dst" ]; then
        cp "$eval_client_dst" "${eval_client_dst}.bak.$(date +%Y%m%d_%H%M%S)"
    fi
    echo -e "\033[36mSyncing eval client -> ${eval_client_dst}\033[0m"
    cp "$eval_client_src" "$eval_client_dst"
fi

# ===== Ensure the deploy client helpers are present under RoboTwin/script/deploy =====
# The eval client does `from script.deploy.websocket_client_policy import WebsocketClientPolicy`,
# which pulls in a sibling msgpack_numpy. Copy any that are missing from the inference repo.
deploy_pkg_src="${inference_workdir}deploy"
deploy_pkg_dst="${eval_workdir}/script/deploy"
mkdir -p "$deploy_pkg_dst"
for f in __init__.py websocket_client_policy.py msgpack_numpy.py async_rtc.py rtc_protocol.py; do
    if [ ! -f "$deploy_pkg_dst/$f" ] || ! cmp -s "$deploy_pkg_src/$f" "$deploy_pkg_dst/$f"; then
        if [ ! -f "$deploy_pkg_src/$f" ]; then
            echo -e "\033[31mError: deploy helper source not found: ${deploy_pkg_src}/${f}\033[0m"
            exit 1
        fi
        echo -e "\033[36mCopying deploy helper -> ${deploy_pkg_dst}/${f}\033[0m"
        cp "$deploy_pkg_src/$f" "$deploy_pkg_dst/$f"
    fi
done

# ===== Ensure curobo absolute paths point at this RoboTwin checkout =====
# Two artifacts embed absolute paths and go stale when RoboTwin is moved to a
# new directory (both would break the curobo planner at sim start):
#   (1) assets/embodiments/*/curobo*.yml -- urdf_path / collision_spheres
#       (regenerated from *_tmp.yml templates via script/update_embodiment_config_path.py)
#   (2) the editable-install .pth in the RoboTwin conda env pointing at envs/curobo/src
# Both checks are idempotent: only act when the embedded path != current location.
_eval_resolved="$(pwd -P)"   # physical path of this RoboTwin checkout (after cd)
_rep_yml="assets/embodiments/aloha-agilex/curobo_left.yml"
if [ -f "$_rep_yml" ] && ! grep -qF "${_eval_resolved}/assets" "$_rep_yml" 2>/dev/null; then
    echo -e "\033[36mCurobo yml paths stale -> regenerating for ${_eval_resolved}\033[0m"
    ( source ${conda_sh} && conda activate ${sim_env} \
      && python script/update_embodiment_config_path.py ) \
      || { echo -e "\033[31mError: curobo path regeneration failed\033[0m"; exit 1; }
fi
_env_site="$( source ${conda_sh} && conda activate ${sim_env} \
    && python -c 'import site;print(site.getsitepackages()[0])' )"
_curobo_pth="$(echo "${_env_site}"/__editable__.nvidia_curobo-*.pth)"
_expected_curobo_src="${_eval_resolved}/envs/curobo/src"
if [ -f "$_curobo_pth" ]; then
    _cur_curobo_src="$(grep -v '^[[:space:]]*#' "$_curobo_pth" | head -1)"
    if [ "$_cur_curobo_src" != "$_expected_curobo_src" ]; then
        echo -e "\033[36mCurobo editable .pth stale (${_cur_curobo_src}) -> ${_expected_curobo_src}\033[0m"
        echo "$_expected_curobo_src" > "$_curobo_pth"
    fi
fi

total_start_time=$(date +%s)

# Slot state: free or busy
# slot_pid[slot]   = sim process PID (0 = free)
# slot_task[slot]  = current task name
# slot_start[slot] = task start time
# slot_log[slot]   = log file path
declare -a slot_pid=()
declare -a slot_task=()
declare -a slot_start=()
declare -a slot_log=()
for slot in $(seq 0 $((num_slots-1))); do
    slot_pid[$slot]=0
    slot_task[$slot]=""
    slot_start[$slot]=0
    slot_log[$slot]=""
done

# Queue pointer and retry counters
queue_idx=0
max_retries=3
declare -A task_retries=()
for t in "${task_queue[@]}"; do
    task_retries[$t]=0
done

# Launch a sim task onto a given slot
launch_task() {
    local slot=$1
    local task_name=$2
    local gpu_id
    if (( ${#sim_gpu_list[@]} > 0 )); then
        gpu_id=${sim_gpu_list[$((slot % ${#sim_gpu_list[@]}))]}
    else
        gpu_id=$(( slot % num_gpus ))
    fi
    local port=$(( start_port + slot ))

    export CUDA_VISIBLE_DEVICES=${gpu_id}

    local log_file="${run_dir}/eval_logs/${task_name}.log"

    local attempt=$((task_retries[$task_name]+1))
    echo -e "\033[33m  [launch] ${task_name} -> slot $slot (GPU: ${gpu_id}, PORT: ${port}, Log: ${log_file}) [attempt ${attempt}/${max_retries}]\033[0m"

    # On retry, append instead of overwriting the failed log; mark this attempt with a banner
    {
        echo ""
        echo "================================================================"
        echo "  [attempt ${attempt}/${max_retries}] $(date '+%Y-%m-%d %H:%M:%S')"
        echo "  task=${task_name}  slot=${slot}  gpu=${gpu_id}  port=${port}"
        echo "================================================================"
    } >> "$log_file"

    # Prepend the RoboTwin env's site-packages to PYTHONPATH (set inside the
    # bash -c after `conda activate`) so the env's numpy 1.26.4 -- which mplib /
    # sapien / open3d were compiled against -- is used instead of ~/.local's
    # numpy 2.x. The user-site numpy 2.x otherwise shadows it and segfaults
    # mplib's convert_physx_component during planner init.
    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore::UserWarning \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
    setsid bash -c "source ${conda_sh} && conda activate ${sim_env} && export PYTHONPATH=\"\$(python -c 'import site;print(site.getsitepackages()[0])')\${PYTHONPATH:+:\$PYTHONPATH}\" && PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore::UserWarning XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 python -u ${eval_client_dst} --config policy/${policy_name}/deploy_policy.yml \
        --overrides \
        --task_name ${task_name} \
        --num_episodes ${num_episodes} \
        --rtc_async ${rtc_async} \
        --rtc_mode ${rtc_mode} \
        --rtc_guidance_beta ${rtc_guidance_beta} \
        --rtc_replan_steps ${rtc_replan_steps} \
        --rtc_delay ${rtc_delay} \
        --rtc_timeout ${rtc_timeout} \
        --run_id ${run_id} \
        --task_config ${task_config} \
        --instruction_type ${instruction_type} \
        --train_config_name ${train_config_name} \
        --seed ${seed} \
        --policy_name ${policy_name} \
        --port ${port} \
        --robo_name ${robo_name} \
        --video_fps ${video_fps} \
        --eval_video_log ${enable_video} \
        --output_dir '${run_dir}/eval_results'" >> "$log_file" 2>&1 &

    local pid=$!
    echo "${pid}" >> "$eval_pid_file"
    slot_pid[$slot]=$pid
    slot_task[$slot]="$task_name"
    slot_start[$slot]=$(date +%s)
    slot_log[$slot]="$log_file"
}

# Read only a bounded log tail; camera rendering and model inference are untouched.
show_task_progress() {
    local slot=$1
    "$sim_python" "${inference_workdir}experiment/robotwin/eval_progress.py" \
        --log "${slot_log[$slot]}" --task "${slot_task[$slot]}" \
        --episodes "$num_episodes" --elapsed "$(( $(date +%s) - slot_start[$slot] ))"
}

# Initial fill: assign a task to each free slot
for slot in $(seq 0 $((num_slots-1))); do
    if [ $queue_idx -lt ${#task_queue[@]} ]; then
        launch_task $slot "${task_queue[$queue_idx]}"
        queue_idx=$((queue_idx + 1))
    fi
done

# Shut down the inference server on a given slot
shutdown_inference_slot() {
    local slot=$1
    local inf_pid=${inference_pids[$slot]}
    if [ "$inf_pid" != "0" ] && kill -0 "$inf_pid" 2>/dev/null; then
        kill "$inf_pid" 2>/dev/null
        echo -e "\033[36m  [release] slot $slot inference server (PID: ${inf_pid}) stopped, remaining: $((active_inference - 1))\033[0m"
    fi
    inference_pids[$slot]=0
    active_inference=$((active_inference - 1))
}

# ============================================================
# Phase 3: scheduling loop - detect finished slots, reclaim and dispatch
# ============================================================
completed=0
skipped=0
total_tasks=${#task_queue[@]}

# Exit when all tasks done/skipped and nothing running
has_running() {
    for slot in $(seq 0 $((num_slots-1))); do
        [ "${slot_pid[$slot]}" != "0" ] && return 0
    done
    return 1
}

# Count remaining tasks (running + queued)
remaining_tasks() {
    local running=0
    for slot in $(seq 0 $((num_slots-1))); do
        [ "${slot_pid[$slot]}" != "0" ] && running=$((running + 1))
    done
    local queued=$(( ${#task_queue[@]} - queue_idx ))
    echo $((running + queued))
}

while [ $((completed + skipped)) -lt $total_tasks ] && has_running; do
    for slot in $(seq 0 $((num_slots-1))); do
        pid=${slot_pid[$slot]}

        # Skip free slots
        [ "$pid" = "0" ] && continue

        # Check whether the process has exited
        if ! kill -0 "$pid" 2>/dev/null; then
            task_end_time=$(date +%s)
            task_duration=$(( task_end_time - slot_start[$slot] ))
            wait "$pid"
            exit_code=$?

            task_name="${slot_task[$slot]}"
            log_file="${slot_log[$slot]}"

            if [ $exit_code -eq 0 ]; then
                completed=$((completed + 1))
                echo -e "\033[32m  [done] ${task_name} slot $slot took ${task_duration}s (progress ${completed}/${total_tasks})\033[0m"
                result_status+=("done")
                result_task_names+=("$task_name")
                result_durations+=("$task_duration")
                result_logs+=("$log_file")

                # Free the slot
                slot_pid[$slot]=0
                slot_task[$slot]=""
                slot_log[$slot]=""

                # Dispatch next task from the queue
                if [ $queue_idx -lt ${#task_queue[@]} ]; then
                    launch_task $slot "${task_queue[$queue_idx]}"
                    queue_idx=$((queue_idx + 1))
                fi
            else
                task_retries[$task_name]=$((task_retries[$task_name] + 1))
                if [ ${task_retries[$task_name]} -lt $max_retries ]; then
                    echo -e "\033[31m  [fail] ${task_name} slot $slot took ${task_duration}s exit=${exit_code}, retrying on this slot (${task_retries[$task_name]}/${max_retries})\033[0m"
                    # Retry immediately on the same slot
                    launch_task $slot "$task_name"
                else
                    skipped=$((skipped + 1))
                    echo -e "\033[31m  [skip] ${task_name} failed ${max_retries} times, no more retries\033[0m"
                    result_status+=("skip:${exit_code}")
                    result_task_names+=("$task_name")
                    result_durations+=("$task_duration")
                    result_logs+=("$log_file")

                    # Free the slot
                    slot_pid[$slot]=0
                    slot_task[$slot]=""
                    slot_log[$slot]=""

                    # Dispatch next task from the queue
                    if [ $queue_idx -lt ${#task_queue[@]} ]; then
                        launch_task $slot "${task_queue[$queue_idx]}"
                        queue_idx=$((queue_idx + 1))
                    fi
                fi
            fi
        else
            show_task_progress "$slot"
        fi
    done

    # Adaptive shutdown: when remaining tasks < active inference servers, stop idle-slot servers
    remaining=$(remaining_tasks)
    for slot in $(seq 0 $((num_slots-1))); do
        [ $remaining -lt $active_inference ] || break
        if [ "${slot_pid[$slot]}" = "0" ] && [ "${inference_pids[$slot]}" != "0" ]; then
            shutdown_inference_slot $slot
        fi
    done

    sleep 5
done

total_end_time=$(date +%s)
total_duration=$(( total_end_time - total_start_time ))

if [ $skipped -gt 0 ]; then
    echo -e "\033[31m========== Sim finished: ${completed} done, ${skipped} skipped, total ${total_duration}s ==========\033[0m"
else
    echo -e "\033[32m========== All sim tasks done, total ${total_duration}s ==========\033[0m"
fi

# ============================================================
# Phase 4: generate the stats file
# ============================================================
echo -e "\033[36mGenerating stats file: ${stats_file}\033[0m"

python3 "$checkpoint_tool" stats --run-dir "$run_dir" --episodes "$num_episodes" \
    --tasks "${requested_task_queue[@]}"

# ============================================================
# Phase 5: handle inference servers (stop the remaining ones)
# ============================================================
if $keep_inference; then
    alive_count=0
    for slot in $(seq 0 $((num_slots-1))); do
        [ "${inference_pids[$slot]}" != "0" ] && alive_count=$((alive_count + 1))
    done
    echo -e "\033[36m========== Sim done, ${alive_count} inference servers kept resident ==========\033[0m"
    echo -e "\033[36mStop inference manually: kill \$(cat ${inference_pid_file})\033[0m"
else
    echo -e "\033[36m========== Stopping remaining inference servers ==========\033[0m"
    for slot in $(seq 0 $((num_slots-1))); do
        inf_pid=${inference_pids[$slot]}
        if [ "$inf_pid" != "0" ] && kill -0 "$inf_pid" 2>/dev/null; then
            kill "$inf_pid" 2>/dev/null
            echo -e "\033[36m  slot $slot inference server (PID: ${inf_pid}) stopped\033[0m"
        fi
    done
    echo -e "\033[32mAll inference servers stopped\033[0m"
fi

echo ""
echo -e "\033[32m========== All done ==========\033[0m"
echo -e "Run dir: ${run_dir}"
echo -e "Stats file: ${stats_file}"
echo -e "Inference logs: ${run_dir}/inference_logs/"
echo -e "Sim logs: ${run_dir}/eval_logs/"
echo -e "Eval results & videos: ${run_dir}/eval_results/"
if $keep_inference; then
    echo -e "Inference PIDs: ${inference_pid_file} (resident)"
else
    echo -e "Inference PIDs: ${inference_pid_file} (stopped)"
fi
echo -e "Sim PIDs: ${eval_pid_file}"

# Failed/incomplete tasks must propagate failure to suite scripts.
if [ "$skipped" -gt 0 ]; then exit 1; fi
