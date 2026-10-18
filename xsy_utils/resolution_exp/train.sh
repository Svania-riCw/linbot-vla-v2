#!/usr/bin/env bash
# H20 resolution experiment launcher. --dry-run checks setup without starting CUDA.
# Usage: CUDA_VISIBLE_DEVICES=0,1,... bash xsy_utils/resolution_exp/train.sh 256 [--dry-run] [training overrides]
set -euo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SIZE="${1:-}"
case "$SIZE" in 256) LABEL=b0_256 ;; 384) LABEL=b1_384 ;; *) echo "Usage: $0 {256|384} [--dry-run] [training overrides]" >&2; exit 2 ;; esac
shift
DRY_RUN=false
if [[ "${1:-}" == --dry-run ]]; then DRY_RUN=true; shift; fi

export WS="${WS:-$(dirname "$CODE")}"
export TRAIN_ENV="${TRAIN_ENV:-/2025233210/xiesy/lingbot_ws/conda_envs/lingbotvla}"
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-$(dirname "$TRAIN_ENV")}"
export MICRO_BS="${MICRO_BS:-4}" GLOBAL_BS="${GLOBAL_BS:-128}"
NUM_WORKERS="${NUM_WORKERS:-1}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
export RUN_NAME="${RUN_NAME:-exp1_${LABEL}_lanczos_s42_$(date +%Y%m%d_%H%M%S)}"
export MASTER_PORT="${MASTER_PORT:-29581}"
export CONFIG="${CONFIG:-$CODE/.runtime/resolution_exp/$LABEL.yaml}"
[[ "$CONFIG" = /* ]] || CONFIG="$PWD/$CONFIG"
[[ -x "$TRAIN_ENV/bin/python" ]] || { echo "Missing training Python: $TRAIN_ENV/bin/python" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "Generate configs first with tools/prepare_resolution_baseline.py --run-root $WS/outputs" >&2; exit 2; }
[[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "Set CUDA_VISIBLE_DEVICES explicitly to the allocated GPU indices." >&2; exit 2; }
[[ "$RUN_NAME" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "RUN_NAME must be one directory name." >&2; exit 2; }
[[ "$MICRO_BS" =~ ^[1-9][0-9]*$ && "$GLOBAL_BS" =~ ^[1-9][0-9]*$ ]] || { echo "Batch sizes must be positive integers." >&2; exit 2; }
IFS=, read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
NPROC=${#GPUS[@]}
declare -A SEEN_GPUS=()
for GPU in "${GPUS[@]}"; do
    [[ -z "${SEEN_GPUS[$GPU]:-}" ]] || { echo "Duplicate GPU index: $GPU" >&2; exit 2; }
    SEEN_GPUS[$GPU]=1
done
[[ "$NUM_WORKERS" =~ ^[0-9]+$ && "$PREFETCH_FACTOR" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid loader worker/prefetch count." >&2; exit 2; }
(( GLOBAL_BS % (MICRO_BS * NPROC) == 0 )) || { echo "GLOBAL_BS must be divisible by MICRO_BS * GPU count." >&2; exit 2; }
[[ ! -e "$WS/outputs/$RUN_NAME" ]] || { echo "Run already exists; choose a new RUN_NAME: $WS/outputs/$RUN_NAME" >&2; exit 2; }

# Validate the chosen YAML before the generic launcher overrides its output/batch.
PYTHONNOUSERSITE=1 "$TRAIN_ENV/bin/python" - "$CONFIG" "$SIZE" <<'PY'
import sys, yaml
from pathlib import Path
config = yaml.safe_load(Path(sys.argv[1]).read_text())
data = config['data']
if (data.get('img_size') != int(sys.argv[2]) or data.get('image_resize_mode') != 'opencv_lanczos4'
        or data.get('preserve_image_resolution') is not True):
    raise SystemExit('Config must match the requested size and enable Lanczos + fixed resolution.')
PY
printf 'code=%s\nenvironment=%s\nconfig=%s\noutput=%s\ngpus=%s\nmicro_batch=%s global_batch=%s accumulation=%s\nworkers=%s prefetch=%s\n' \
    "$CODE" "$TRAIN_ENV" "$CONFIG" "$WS/outputs/$RUN_NAME" "$CUDA_VISIBLE_DEVICES" "$MICRO_BS" "$GLOBAL_BS" "$((GLOBAL_BS / (MICRO_BS * NPROC)))" "$NUM_WORKERS" "$PREFETCH_FACTOR"
if "$DRY_RUN"; then
    printf 'Validated launch settings. No training process started.\n'
    exit 0
fi
exec bash "$CODE/xsy_utils/train_baseline.sh" \
    --train.seed 42 --train.max_steps 10000 --train.save_steps 1000 \
    --data.num_workers "$NUM_WORKERS" --data.prefetch_factor "$PREFETCH_FACTOR" "$@"
