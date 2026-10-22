#!/usr/bin/env bash
# Resume only this experiment's DCP checkpoint, preserving its 10k LR horizon.
set -euo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CHECKPOINT="${1:?Pass the global_step_N DCP directory}"; shift
[[ "$CHECKPOINT" = /* && "$(basename "$CHECKPOINT")" =~ ^global_step_[1-9][0-9]*$ ]] || { echo 'Expected absolute global_step_N path' >&2; exit 2; }
[[ -f "$CHECKPOINT/model/.metadata" && -f "$CHECKPOINT/optimizer/.metadata" ]] || { echo 'Incomplete DCP checkpoint' >&2; exit 2; }
OUT="$(dirname -- "$(dirname -- "$CHECKPOINT")")"
export CONFIG="${CONFIG:?Set the pinned 384 training YAML}"
export WS="$(dirname -- "$(dirname -- "$OUT")")" RUN_NAME="$(basename -- "$OUT")"
# train_baseline.sh computes OUT=$WS/outputs/$RUN_NAME.
[[ "$(basename -- "$(dirname -- "$OUT")")" == outputs ]] || { echo 'Expected run under an outputs directory' >&2; exit 2; }
export MICRO_BS="${MICRO_BS:?Set the same micro batch as the first segment}" GLOBAL_BS=128
exec bash "$CODE/xsy_utils/train_baseline.sh" \
    --train.max_steps 10000 --train.save_steps 1000 --train.pause_after_steps 0 \
    --train.load_checkpoint_path "$CHECKPOINT" --train.enable_resume false \
    --data.num_workers 1 --data.prefetch_factor 1 "$@"
