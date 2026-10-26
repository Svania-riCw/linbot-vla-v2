#!/usr/bin/env bash
# Standalone local RTC evaluation; environments and paths can be overridden.
set -euo pipefail
if [[ $# -lt 3 ]]; then
    echo 'Usage: bash eval_rtc.sh HF_CKPT QWEN_DIR demo_clean|demo_randomized [launcher options]' >&2
    exit 2
fi
repo=$(cd -- "$(dirname -- "$0")" && pwd)
workspace=$(dirname -- "$repo")
checkpoint=$(realpath "$1")
qwen=$(realpath "$2")
setting=$3
shift 3
launcher_args=()
image_size_override=${LINGBOT_IMAGE_SIZE:-}
while (($#)); do
    case "$1" in
        --image_size|--image-size)
            [[ $# -ge 2 ]] || { echo "$1 requires a positive integer." >&2; exit 2; }
            image_size_override=$2
            shift 2
            ;;
        *)
            launcher_args+=("$1")
            shift
            ;;
    esac
done
if [[ -n "$image_size_override" && ! "$image_size_override" =~ ^[1-9][0-9]*$ ]]; then
    echo "image size must be a positive integer: $image_size_override" >&2
    exit 2
fi
[[ -d "$checkpoint/hf_ckpt" ]] && checkpoint="$checkpoint/hf_ckpt"
case "$setting" in demo_clean|demo_randomized) ;; *) echo 'Invalid task setting' >&2; exit 2;; esac
conda_root=${CONDA_ROOT:-$(conda info --base 2>/dev/null || true)}
if [[ -z "$conda_root" && -d "$HOME/miniconda" ]]; then conda_root="$HOME/miniconda"; fi
conda_sh=${CONDA_SH:-$conda_root/etc/profile.d/conda.sh}
[[ -f "$conda_sh" ]] || { echo 'Set CONDA_SH to your conda.sh path' >&2; exit 2; }
export INFERENCE_ENV=${INFERENCE_ENV:-lingbotvla}
export SIM_ENV=${SIM_ENV:-robotwin-eval}
nvidia_gl_runtime=${NVIDIA_GL_RUNTIME:-}
if [[ -z "$nvidia_gl_runtime" ]]; then
    for candidate in "$workspace"/.runtime/nvidia-gl-*; do
        if [[ -d "$candidate" && -f "$candidate/icd/10_nvidia.json" && -f "$candidate/icd/nvidia_icd.json" ]]; then
            nvidia_gl_runtime=$candidate
            break
        fi
    done
fi
if [[ -n "$nvidia_gl_runtime" && -d "$nvidia_gl_runtime/lib" ]]; then
    export LD_LIBRARY_PATH="$nvidia_gl_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export __EGL_VENDOR_LIBRARY_FILENAMES=${__EGL_VENDOR_LIBRARY_FILENAMES:-$nvidia_gl_runtime/icd/10_nvidia.json}
    export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-$nvidia_gl_runtime/icd/nvidia_icd.json}
    echo "NVIDIA EGL/Vulkan runtime: $nvidia_gl_runtime"
fi
source "$conda_sh"
conda activate "$INFERENCE_ENV"
export QWEN3VL_PATH="$qwen"
export LINGBOT_LOAD_DTYPE=${LINGBOT_LOAD_DTYPE:-bfloat16}
# Leave space for three cameras, inference activations and RoboTwin rendering.
if [[ -z "${LINGBOT_CPU_OFFLOAD_MLP_LAYERS:-}" ]]; then
    export LINGBOT_CPU_OFFLOAD_MLP_LAYERS=$(python -c 'import torch; gb=torch.cuda.get_device_properties(0).total_memory / 2**30; print(24 if gb <= 18 else 12 if gb <= 24 else 0)')
fi
echo "CPU offload MLP layers: $LINGBOT_CPU_OFFLOAD_MLP_LAYERS"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
if [[ -z "${LINGBOT_INFERENCE_CONFIG:-}" ]]; then
    config_id=$(printf '%s' "$checkpoint" | sha256sum | cut -c1-12)
    export LINGBOT_INFERENCE_CONFIG="$repo/.runtime/rtc-$config_id.yaml"
    render_args=(--checkpoint "$checkpoint" --qwen "$qwen" --output "$LINGBOT_INFERENCE_CONFIG")
    if [[ -n "$image_size_override" ]]; then
        render_args+=(--image_size "$image_size_override")
    fi
    python "$repo/tools/render_robotwin_inference_config.py" "${render_args[@]}"
elif [[ -n "$image_size_override" ]]; then
    echo '--image_size cannot override an existing LINGBOT_INFERENCE_CONFIG; unset it first.' >&2
    exit 2
fi
[[ -f "$LINGBOT_INFERENCE_CONFIG" ]] || { echo "Missing config: $LINGBOT_INFERENCE_CONFIG" >&2; exit 2; }
eval_workdir=${EVAL_WORKDIR:-$workspace/RoboTwin}
cd "$repo"
exec bash experiment/robotwin/start_robotwin_infer_and_eval.sh \
    --model_path "$checkpoint" --eval_workdir "$eval_workdir" \
    --output_base "$eval_workdir/eval_results" --conda_sh "$conda_sh" \
    --inference_env "$INFERENCE_ENV" --sim_env "$SIM_ENV" \
    --task_config "$setting" --num_gpus 1 --num_per_gpu 1 \
    --use_bf16 True --use_fp32 False --use_compile False --rtc_mode training_time \
    "${launcher_args[@]}"
