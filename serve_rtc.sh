#!/usr/bin/env bash
# Inference only. Run on the remote GPU server or locally.
set -euo pipefail
if [[ $# -lt 2 ]]; then
    echo 'Usage: bash serve_rtc.sh HF_CKPT QWEN_DIR [--host ADDRESS] [--port PORT]' >&2
    exit 2
fi
repo=$(cd -- "$(dirname -- "$0")" && pwd)
checkpoint=$(realpath "$1") qwen=$(realpath "$2")
shift 2
[[ -d "$checkpoint/hf_ckpt" ]] && checkpoint="$checkpoint/hf_ckpt"
conda_root=${CONDA_ROOT:-$(conda info --base 2>/dev/null || true)}
if [[ -z "$conda_root" && -d "$HOME/miniconda" ]]; then conda_root="$HOME/miniconda"; fi
conda_sh=${CONDA_SH:-$conda_root/etc/profile.d/conda.sh}
[[ -f "$conda_sh" ]] || { echo 'Set CONDA_SH to your conda.sh path' >&2; exit 2; }
source "$conda_sh"
conda activate "${INFERENCE_ENV:-lingbotvla}"
export QWEN3VL_PATH="$qwen" LINGBOT_LOAD_DTYPE=${LINGBOT_LOAD_DTYPE:-bfloat16}
# A 24GB+ remote GPU generally fits inference without offload.
if [[ -z "${LINGBOT_CPU_OFFLOAD_MLP_LAYERS:-}" ]]; then
    export LINGBOT_CPU_OFFLOAD_MLP_LAYERS=$(python -c 'import torch; gb=torch.cuda.get_device_properties(0).total_memory/2**30; print(12 if gb <= 18 else 0)')
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
if [[ -z "${LINGBOT_INFERENCE_CONFIG:-}" ]]; then
    config_id=$(printf '%s' "$checkpoint" | sha256sum | cut -c1-12)
    export LINGBOT_INFERENCE_CONFIG="$repo/.runtime/rtc-$config_id.yaml"
    python "$repo/tools/render_robotwin_inference_config.py" --checkpoint "$checkpoint" \
        --qwen "$qwen" --output "$LINGBOT_INFERENCE_CONFIG"
fi
cd "$repo"
exec python -u -m deploy.lingbot_vla_v2_policy --model_path "$checkpoint" \
    --host 127.0.0.1 --port 8000 --use_length 50 --chunk_ret True \
    --use_bf16 True --use_fp32 False --use_compile False "$@"
