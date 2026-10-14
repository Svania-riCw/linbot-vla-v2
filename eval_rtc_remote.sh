#!/usr/bin/env bash
# Simulation only: connects to an already-running RTC server; never controls it.
set -euo pipefail
if [[ $# -lt 3 ]]; then
    echo 'Usage: bash eval_rtc_remote.sh HOST PORT demo_clean|demo_randomized --checkpoint_id NAME [--task_name TASK | --tasks CSV | --seven] [--num_episodes N] [--run_dir DIR]' >&2
    exit 2
fi
repo=$(cd -- "$(dirname -- "$0")" && pwd)
workspace=$(dirname -- "$repo")
host=$1 port=$2 setting=$3
shift 3
episodes=25 fps=30 video=True delay=6 replan=8 timeout=180
run_dir="" checkpoint_id="" selected=""
seven='stack_blocks_three,blocks_ranking_rgb,pick_diverse_bottles,place_shoe,rotate_qrcode,open_laptop,hanging_mug'
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint_id) checkpoint_id=${2:?Missing checkpoint identifier}; shift 2;;
        --task_name|--tasks) selected=${2:?Missing task selection}; shift 2;;
        --seven) selected=$seven; shift;;
        --num_episodes) episodes=${2:?Missing episode count}; shift 2;;
        --run_dir) run_dir=${2:?Missing run directory}; shift 2;;
        --rtc_delay) delay=${2:?Missing RTC delay}; shift 2;;
        --rtc_replan_steps) replan=${2:?Missing replan interval}; shift 2;;
        --rtc_timeout) timeout=${2:?Missing request timeout}; shift 2;;
        --video_fps) fps=${2:?Missing FPS}; shift 2;;
        --no_video) video=False; shift;;
        --video) video=True; shift;;
        *) echo "Unknown option: $1" >&2; exit 2;;
    esac
done
[[ -n "$checkpoint_id" ]] || { echo '--checkpoint_id is required for remote resume' >&2; exit 2; }
[[ "$episodes" =~ ^[1-9][0-9]*$ && "$replan" =~ ^[1-9][0-9]*$ && "$delay" =~ ^[0-9]+$ ]] || { echo 'Invalid episode count or RTC steps' >&2; exit 2; }
[[ "$port" =~ ^[0-9]+$ ]] && ((port > 0 && port < 65536)) || { echo 'Invalid port' >&2; exit 2; }
case "$setting" in demo_clean) instruction=seen;; demo_randomized) instruction=unseen;; *) echo 'Invalid task setting' >&2; exit 2;; esac
selected=${selected:-$seven}
read -r -a tasks <<< "${selected//,/ }"
eval_workdir=${EVAL_WORKDIR:-$workspace/RoboTwin}
for task in "${tasks[@]}"; do
    [[ "$task" =~ ^[a-z0-9_]+$ && -f "$eval_workdir/envs/$task.py" ]] || { echo "Invalid or missing task: $task" >&2; exit 2; }
done
conda_root=${CONDA_ROOT:-$(conda info --base 2>/dev/null || true)}
if [[ -z "$conda_root" && -d "$HOME/miniconda" ]]; then conda_root="$HOME/miniconda"; fi
conda_sh=${CONDA_SH:-$conda_root/etc/profile.d/conda.sh}
[[ -f "$conda_sh" ]] || { echo 'Set CONDA_SH to your conda.sh path' >&2; exit 2; }
source "$conda_sh"
conda activate "${SIM_ENV:-robotwin-eval}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
export PYTHONPATH="$(python -c 'import site;print(site.getsitepackages()[0])'):$repo${PYTHONPATH:+:$PYTHONPATH}"
identifier=$(printf '%s' "$checkpoint_id" | sha256sum | cut -c1-12)
run_dir=${run_dir:-$eval_workdir/eval_results/remote_rtc_${identifier}_${setting}_s${replan}_d${delay}}
mkdir -p "$run_dir/eval_logs"
run_dir=$(realpath "$run_dir")
exec 9>"$run_dir/.run.lock"
flock -n 9 || { echo "Run directory already in use: $run_dir" >&2; exit 2; }
# One evaluator per server endpoint on this machine.
mkdir -p "$repo/.runtime"
endpoint=$(printf '%s:%s' "$host" "$port" | sha256sum | cut -c1-16)
exec 8>"$repo/.runtime/remote-$endpoint.lock"
flock -n 8 || { echo "Remote server already in use locally: $host:$port" >&2; exit 2; }
helper="$repo/experiment/robotwin/eval_checkpoint.py"
pid=0
cleanup() {
    if [[ "$pid" != 0 ]] && kill -0 "$pid" 2>/dev/null; then
        kill -TERM -- "-$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
    python "$helper" stats --run-dir "$run_dir" --episodes "$episodes" --tasks "${tasks[@]}" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
python "$repo/experiment/robotwin/remote_eval.py" \
    --host "$host" --port "$port" --checkpoint-id "$checkpoint_id" \
    --run-dir "$run_dir" --setting "$setting" --instruction "$instruction" \
    --task-config-file "$eval_workdir/task_config/$setting.yml" \
    --episodes "$episodes" --tasks "${tasks[@]}" \
    --rtc-delay "$delay" --rtc-replan-steps "$replan" --timeout "$timeout"
run_id=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["run_id"])' "$run_dir/resume_plan.json")
mapfile -t pending < <(python -c 'import json,sys;print("\n".join(json.load(open(sys.argv[1]))["pending"]),end="")' "$run_dir/resume_plan.json")
if [[ ${#pending[@]} == 0 ]]; then
    echo 'All selected tasks already complete.'
    python "$helper" stats --run-dir "$run_dir" --episodes "$episodes" --tasks "${tasks[@]}"
    exit 0
fi
mkdir -p "$eval_workdir/script/deploy"
cp "$repo/experiment/robotwin/eval_policy_client_lingbotvla.py" "$eval_workdir/script/eval_policy_client_lingbotvla.py"
cp "$helper" "$eval_workdir/script/eval_checkpoint.py"
for file in __init__.py websocket_client_policy.py msgpack_numpy.py async_rtc.py; do
    cp "$repo/deploy/$file" "$eval_workdir/script/deploy/$file"
done
echo "RTC inference: $host:$port; run directory: $run_dir; pending tasks: ${#pending[@]}"
cd "$eval_workdir"
for task in "${pending[@]}"; do
    log="$run_dir/eval_logs/$task.log"
    start=$(date +%s)
    printf '\n[attempt 1/1] %s\n' "$(date -Iseconds)" >> "$log"
    setsid python -u script/eval_policy_client_lingbotvla.py --config policy/ACT/deploy_policy.yml --overrides \
        --host "$host" --port "$port" --task_name "$task" --run_id "$run_id" \
        --num_episodes "$episodes" --task_config "$setting" --instruction_type "$instruction" \
        --train_config_name 0 --seed 0 --policy_name ACT --robo_name robotwin \
        --rtc_async True --rtc_delay "$delay" --rtc_replan_steps "$replan" --rtc_timeout "$timeout" \
        --video_fps "$fps" --eval_video_log "$video" --output_dir "$run_dir/eval_results" \
        >> "$log" 2>&1 &
    pid=$!
    echo "$pid" > "$run_dir/eval_pids.txt"
    while kill -0 "$pid" 2>/dev/null; do
        python "$repo/experiment/robotwin/eval_progress.py" --log "$log" --task "$task" \
            --episodes "$episodes" --elapsed "$(( $(date +%s) - start ))"
        sleep 5
    done
    code=0
    wait "$pid" || code=$?
    pid=0
    if [[ $code != 0 ]]; then
        echo "Evaluation interrupted (exit=$code); rerun the same command to resume. Log: $log" >&2
        exit "$code"
    fi
    python "$helper" stats --run-dir "$run_dir" --episodes "$episodes" --tasks "${tasks[@]}"
done
