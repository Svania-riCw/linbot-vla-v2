#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "$0")/rtc_paths.sh"
CONDA_SH="$(rtc_conda_sh)"
if [[ ! -f "$CONDA_SH" ]]; then
    printf 'Conda activation script not found: %s\n' "$CONDA_SH" >&2
    exit 2
fi
source "$CONDA_SH"

env_name="${RTC_TRAIN_ENV:-lingbotvla-train}"
source_env="${RTC_SOURCE_ENV:-lingbotvla}"
if conda env list | awk '{print $1}' | grep -Fxq "$env_name"; then
    if conda run -n "$env_name" python -c 'import torch, transformers, lerobot, flash_attn, moge, mdm, utils3d, torchcodec' >/dev/null 2>&1; then
        printf 'RTC training environment is ready: %s\n' "$env_name"
        conda run -n "$env_name" python -c 'import torch, transformers, lerobot; print("torch", torch.__version__, "transformers", transformers.__version__, "lerobot", lerobot.__version__)'
        exit 0
    fi
    printf 'Environment %s exists but is missing required packages. Remove it explicitly, then rerun this script.\n' "$env_name" >&2
    exit 2
fi
if ! conda env list | awk '{print $1}' | grep -Fxq "$source_env"; then
    printf 'Source environment not found: %s\n' "$source_env" >&2
    exit 2
fi
conda create -n "$env_name" --clone "$source_env" -y
conda run -n "$env_name" python -c 'import torch, transformers, lerobot, flash_attn, moge, mdm, utils3d, torchcodec; print("torch", torch.__version__, "transformers", transformers.__version__, "lerobot", lerobot.__version__)'
