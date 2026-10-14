#!/bin/bash
# Parallel eval engine for LingBot-VLA-v2 on RoboTwin. Same work-unit scheme and output layout
# as policy/my-pi05/eval_parallel.sh (pi05-experiments), adapted to a client/server split:
#
#   * the model runs in the `lingbotvla` env (py3.12, flash-attn) as a websocket server
#     (serve.py), one per GPU, started once and reused for every unit that worker claims;
#   * the sim runs in the `robotwin-eval` env (py3.10, sapien/curobo) via script/eval_policy.py
#     with policy_name=lingbot-vla-v2, talking to that GPU's server on PORT_BASE + gpu_id.
#
# Work unit = one (task, task_config) pair running TEST_NUM trials. Workers claim the next
# unclaimed unit via an atomic `mkdir` lock, so fast workers pick up more units.
#
# All params are env vars. Only MODEL_PATH is required.
#
# Output (relative to external/RoboTwin):
#   eval_result/<MODEL_NAME>/<timestamp>/_result.txt        # one appended line per unit
#   eval_result/<MODEL_NAME>/<timestamp>/<task>_<case>/     # per-unit episode videos
#   eval_result/<MODEL_NAME>/<timestamp>/<task>_<case>.log  # per-unit sim stdout/err
#   eval_result/<MODEL_NAME>/<timestamp>/_server/gpu<id>.log
#   eval_result/<MODEL_NAME>/<timestamp>/_status/           # claim locks + done markers
set -uo pipefail

policy_name=lingbot-vla-v2
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
LINGBOT_ROOT="$(cd "${ROBOTWIN_ROOT}/../.." && pwd)"

# ---- Configurables (env vars) --------------------------------------------
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to .../checkpoints/global_step_N/hf_ckpt}
MODEL_NAME=${MODEL_NAME:-$(basename "$(dirname "$(dirname "$(dirname "${MODEL_PATH}")")")")_$(basename "$(dirname "${MODEL_PATH}")")}
QWEN3VL_PATH=${QWEN3VL_PATH:-/2025233210/lmx/checkpoints/Qwen3-VL-4B-Instruct}
NORM_STATS=${NORM_STATS:-}                 # empty => serve.py resolves from the run's lingbotvla_cli.yaml
ROBO_NAME=${ROBO_NAME:-robotwin}
USE_FP32=${USE_FP32:-True}                 # release numbers are fp32; False => bf16
USE_COMPILE=${USE_COMPILE:-False}
USE_LENGTH=${USE_LENGTH:-50}               # executed action-chunk length per inference
TEST_NUM=${TEST_NUM:-100}
SEED=${SEED:-0}
TASK_CONFIGS=${TASK_CONFIGS:-${TASK_CONFIG:-demo_clean demo_randomized}}
TASK_CONFIGS=${TASK_CONFIGS//,/ }
TASKS=${TASKS:-stack_bowls_three}
PORT_BASE=${PORT_BASE:-9330}
SERVER_TIMEOUT=${SERVER_TIMEOUT:-1800}     # seconds to wait for a server to load weights
UNIT_TIMEOUT=${UNIT_TIMEOUT:-21600}
WARP_CACHE_ROOT=${WARP_CACHE_ROOT:-/root/.cache/warp_par}
CONDA_SH=${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}
export CONDA_ENVS_PATH=${CONDA_ENVS_PATH:-/2025233210/xiesy/lingbot_ws/conda_envs}
INFERENCE_ENV=${INFERENCE_ENV:-lingbotvla}
SIM_ENV=${SIM_ENV:-robotwin-eval}
GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ')}
GPUS=${GPUS//,/ }
[ -z "${GPUS// /}" ] && { echo "ERROR: no GPU visible; set GPUS explicitly" >&2; exit 1; }
# --------------------------------------------------------------------------

if [ ! -f "${MODEL_PATH}/model.safetensors.index.json" ]; then
    echo "ERROR: no model.safetensors.index.json under ${MODEL_PATH}" >&2
    exit 1
fi

read -r -a GPU_ARR <<< "${GPUS}"
read -r -a TASK_ARR <<< "${TASKS}"
read -r -a CFG_ARR <<< "${TASK_CONFIGS}"
JOB_ARR=()
for t in "${TASK_ARR[@]}"; do
    for c in "${CFG_ARR[@]}"; do JOB_ARR+=("${t}|${c}"); done
done

cd "${ROBOTWIN_ROOT}" || exit 1
# Passed through --overrides, which eval()s every value: "2026-10-14_13-18-33" would be computed
# as arithmetic (14_13 is a valid int literal) and land in eval_result/<name>/552/. A leading
# identifier makes eval() raise NameError, so the string survives as-is.
run_timestamp=$(date +run_%Y%m%d_%H%M%S)
RUN_DIR="eval_result/${MODEL_NAME}/${run_timestamp}"
STATUS_DIR="${RUN_DIR}/_status"
SERVER_DIR="${RUN_DIR}/_server"
mkdir -p "${STATUS_DIR}" "${SERVER_DIR}"

# `set +u` around activation: conda's cuda-nvcc/gcc activate.d hooks read unbound vars.
set +u
source "${CONDA_SH}"
conda activate "${SIM_ENV}" || { echo "ERROR: cannot activate ${SIM_ENV}" >&2; exit 1; }
set -u
SIM_PY=$(command -v python)
SERVER_PY="${CONDA_ENVS_PATH}/${INFERENCE_ENV}/bin/python"
[ -x "${SERVER_PY}" ] || { echo "ERROR: ${SERVER_PY} not found" >&2; exit 1; }
export CUDA_HOME="${CONDA_PREFIX}"   # curobo JIT compiles against the env's nvcc

# Vulkan for sapien's camera rendering. Some containers mount only the NVIDIA *compute* libs
# (NVIDIA_DRIVER_CAPABILITIES without `graphics`): no libGLX_nvidia / libnvidia-glcore at all,
# and sapien fails with "failed to find a rendering device". NVIDIA_GL_DIR holds the userspace
# GL/Vulkan libs extracted (not installed) from the driver .run matching the kernel module
# version, plus ICD manifests with absolute library paths. Nothing outside it is touched.
DRIVER_VERSION=$(sed -n 's/.*Kernel Module *\([0-9.]*\).*/\1/p' /proc/driver/nvidia/version 2>/dev/null)
NVIDIA_GL_DIR=${NVIDIA_GL_DIR:-/2025233210/xiesy/lingbot_ws/nvidia-gl-${DRIVER_VERSION}}
if ! ldconfig -p | grep -q libGLX_nvidia.so.0 && [ -d "${NVIDIA_GL_DIR}/lib" ]; then
    export LD_LIBRARY_PATH="${NVIDIA_GL_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export VK_ICD_FILENAMES="${NVIDIA_GL_DIR}/icd/nvidia_icd.json"
    export __EGL_VENDOR_LIBRARY_FILENAMES="${NVIDIA_GL_DIR}/icd/10_nvidia.json"
elif ! ldconfig -p | grep -q libGLX_nvidia.so.0; then
    echo "WARNING: no NVIDIA GL/Vulkan libs and no ${NVIDIA_GL_DIR}; sapien rendering will fail" >&2
fi
# sapien's import-time probe also does a bare listdir() on /usr/share/glvnd/egl_vendor.d, which
# this image lacks. Point at the manifests sapien bundles.
if [ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ] && [ ! -d /usr/share/glvnd/egl_vendor.d ]; then
    SAPIEN_VK_DIR=$(python -c "import importlib.util, os; print(os.path.join(os.path.dirname(importlib.util.find_spec('sapien').origin), 'vulkan_library'))")
    export __EGL_VENDOR_LIBRARY_FILENAMES="${SAPIEN_VK_DIR}/10_nvidia.json"
    export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-${SAPIEN_VK_DIR}/nvidia_icd.json}"
fi

echo -e "\033[33mmodel_name   : ${MODEL_NAME}\033[0m"
echo -e "\033[33mcheckpoint   : ${MODEL_PATH}\033[0m"
echo -e "\033[33mprecision    : fp32=${USE_FP32} compile=${USE_COMPILE} use_length=${USE_LENGTH}\033[0m"
echo -e "\033[33mtasks        : ${TASK_ARR[*]}\033[0m"
echo -e "\033[33mcases        : ${CFG_ARR[*]}  | work units: ${#JOB_ARR[@]}\033[0m"
echo -e "\033[33mgpus         : ${GPU_ARR[*]}  (one server + one sim worker per GPU)\033[0m"
echo -e "\033[33mtest_num/unit: ${TEST_NUM} | seed: ${SEED} | total episodes: $(( ${#JOB_ARR[@]} * TEST_NUM ))\033[0m"
echo -e "\033[33mrun dir      : ${ROBOTWIN_ROOT}/${RUN_DIR}\033[0m"

start_server() {   # $1 = gpu_id ; prints the server pid
    local gpu_id="$1" port=$(( PORT_BASE + $1 ))
    local norm_args=()
    [ -n "${NORM_STATS}" ] && norm_args=(--norm_stats "${NORM_STATS}")
    CUDA_VISIBLE_DEVICES="${gpu_id}" QWEN3VL_PATH="${QWEN3VL_PATH}" PYTHONNOUSERSITE=1 \
    TOKENIZERS_PARALLELISM=false PYTHONPATH="${LINGBOT_ROOT}" \
    setsid "${SERVER_PY}" "${POLICY_DIR}/serve.py" \
        --model_path "${MODEL_PATH}" --port "${port}" --robo_name "${ROBO_NAME}" \
        --use_fp32 "${USE_FP32}" --use_compile "${USE_COMPILE}" --use_length "${USE_LENGTH}" \
        "${norm_args[@]}" > "${SERVER_DIR}/gpu${gpu_id}.log" 2>&1 &
    echo $!
}

wait_server() {   # $1 = gpu_id, $2 = server pid ; 0 once the port accepts connections
    local port=$(( PORT_BASE + $1 )) deadline=$(( $(date +%s) + SERVER_TIMEOUT ))
    while [ "$(date +%s)" -lt "${deadline}" ]; do
        kill -0 "$2" 2>/dev/null || return 1
        python -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(s.connect_ex(('127.0.0.1', ${port})))" && return 0
        sleep 5
    done
    return 1
}

run_one_task() {   # $1 = task_name, $2 = task_config, $3 = worker_tag, $4 = gpu_id
    local task_name="$1" task_config="$2" tag="$3" gpu_id="$4"
    local case_label="${task_config#demo_}"; [ "${case_label}" = "randomized" ] && case_label=random
    local unit="${task_name}_${case_label}"
    echo -e "\033[36m[${tag}] START ${unit}  ($(date +%H:%M:%S))\033[0m"
    CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONWARNINGS=ignore::UserWarning \
    WARP_CACHE_PATH="${WARP_CACHE_ROOT}/gpu${gpu_id}" \
    timeout --kill-after=60 "${UNIT_TIMEOUT}" \
    "${SIM_PY}" script/eval_policy.py --config "policy/${policy_name}/deploy_policy.yml" \
        --overrides \
        --task_name "${task_name}" \
        --task_config "${task_config}" \
        --ckpt_setting "${MODEL_NAME}" \
        --port "$(( PORT_BASE + gpu_id ))" \
        --robo_name "${ROBO_NAME}" \
        --test_num "${TEST_NUM}" \
        --seed "${SEED}" \
        --policy_name "${policy_name}" \
        --output_name "${MODEL_NAME}" \
        --run_timestamp "${run_timestamp}" \
        > "${RUN_DIR}/${unit}.log" 2>&1 &
    local job_pid=$!
    echo "${job_pid}" > "${STATUS_DIR}/${unit}.pid"
    wait "${job_pid}"
    local rc=$?
    rm -f "${STATUS_DIR}/${unit}.pid"
    # eval_policy.py can exit 0 without evaluating anything (e.g. script/test_render.py prints
    # "Render Error" and calls exit()), so success = the unit's line landed in _result.txt.
    if [ ${rc} -eq 0 ] && grep -qP "^${task_name}\t${case_label}\t" "${RUN_DIR}/_result.txt" 2>/dev/null; then
        echo -e "\033[32m[${tag}] DONE  ${unit}  ($(date +%H:%M:%S))\033[0m"
    elif [ ${rc} -eq 0 ]; then
        echo -e "\033[31m[${tag}] FAIL  ${unit} (exit 0 but no result line, see ${unit}.log)\033[0m"
    elif [ ${rc} -eq 124 ] || [ ${rc} -eq 137 ]; then
        echo -e "\033[31m[${tag}] TIMEOUT ${unit} after ${UNIT_TIMEOUT}s -- killed, see ${unit}.log\033[0m"
    else
        echo -e "\033[31m[${tag}] FAIL  ${unit} (exit ${rc}, see ${unit}.log)\033[0m"
    fi
}

# Tear down workers first (so none claims a new unit), then sims, then servers.
WORKER_PIDS=()
reap() {
    local f p
    for p in ${WORKER_PIDS[@]+"${WORKER_PIDS[@]}"}; do kill -TERM "${p}" 2>/dev/null; done
    for f in "${STATUS_DIR}"/*.pid "${SERVER_DIR}"/*.pid; do
        [ -f "${f}" ] || continue
        p=$(cat "${f}" 2>/dev/null) || continue
        [ -n "${p}" ] || continue
        { kill -TERM -- "-${p}" 2>/dev/null || kill -TERM "${p}" 2>/dev/null; }
        rm -f "${f}"
    done
}
trap 'reap; exit 130' INT TERM
trap reap EXIT

worker() {   # $1 = gpu_id
    local gpu_id="$1" tag="gpu${1}" server_pid
    server_pid=$(start_server "${gpu_id}")
    echo "${server_pid}" > "${SERVER_DIR}/gpu${gpu_id}.pid"
    echo -e "\033[36m[${tag}] server pid ${server_pid} on port $(( PORT_BASE + gpu_id )), loading...\033[0m"
    if ! wait_server "${gpu_id}" "${server_pid}"; then
        echo -e "\033[31m[${tag}] server failed to come up -- see ${SERVER_DIR}/gpu${gpu_id}.log\033[0m"
        return 1
    fi
    echo -e "\033[36m[${tag}] server ready\033[0m"
    while true; do
        local claimed=""
        for j in "${JOB_ARR[@]}"; do
            local key="${j/|/__}"
            [ -e "${STATUS_DIR}/${key}.done" ] && continue
            if mkdir "${STATUS_DIR}/${key}.lock" 2>/dev/null; then
                claimed="${j}"
                break
            fi
        done
        [ -z "${claimed}" ] && break
        run_one_task "${claimed%%|*}" "${claimed##*|}" "${tag}" "${gpu_id}"
        touch "${STATUS_DIR}/${claimed/|/__}.done"
    done
}

for gpu in "${GPU_ARR[@]}"; do
    mkdir -p "${WARP_CACHE_ROOT}/gpu${gpu}"
    worker "${gpu}" &
    WORKER_PIDS+=($!)
done
for p in "${WORKER_PIDS[@]}"; do wait "${p}"; done
WORKER_PIDS=()

# ---- Aggregate ----
RESULT="${RUN_DIR}/_result.txt"
echo -e "\n\033[32m======== EVAL SUMMARY (${MODEL_NAME}) ========\033[0m"
if [ -f "${RESULT}" ]; then
    # Rewrites _result.txt in place: unit lines sorted + all_clean/all_random/all averages.
    python script/summarize_result.py "${RESULT}"
    echo "units with a result: $(grep -vc '^#' "${RESULT}")/${#JOB_ARR[@]}"
else
    echo "no _result.txt produced (all units failed to run; check per-unit .log files)"
fi
echo -e "Results: ${ROBOTWIN_ROOT}/${RESULT}"
