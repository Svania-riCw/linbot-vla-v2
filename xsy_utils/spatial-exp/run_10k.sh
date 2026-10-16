#!/usr/bin/env bash
# Train one spatial-exp line for 10k steps, then run the standard 7-task RoboTwin eval on
# global_step_10000. Training arguments are exactly those of the baseline_gbs128_10k run
# (micro 16 x 8 GPUs = global 128, max_steps 10000, save every 2000, same cosine schedule);
# only the config file differs.
#
#   bash run_10k.sh d4rt      # configs/robotwin_clean_d4rt.yaml     -> outputs/spatial_d4rt_10k
#   bash run_10k.sh noalign   # configs/robotwin_clean_noalign.yaml  -> outputs/spatial_noalign_10k
#   SMOKE=1 bash run_10k.sh d4rt   # 6 steps, no checkpoint, no eval (outputs/smoke_spatial_d4rt)
set -uo pipefail

EXP=${1:?usage: run_10k.sh d4rt|noalign}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XSY="$(cd "$HERE/.." && pwd)"
WS=/2025233210/xiesy/lingbot_ws
CONFIG="$HERE/configs/robotwin_clean_${EXP}.yaml"
[ -f "$CONFIG" ] || { echo "no config $CONFIG" >&2; exit 2; }

if [ "${SMOKE:-0}" = 1 ]; then
    RUN=smoke_spatial_${EXP}
    rm -rf "$WS/outputs/$RUN"
    CONFIG="$CONFIG" RUN_NAME=$RUN MICRO_BS=16 bash "$XSY/train_baseline.sh" \
        --train.max_steps 6 --train.save_steps 100000
    exit $?
fi

RUN=spatial_${EXP}_10k
echo "[run_10k $EXP] train start $(date -Is)"
CONFIG="$CONFIG" RUN_NAME=$RUN MICRO_BS=16 bash "$XSY/train_baseline.sh" \
    --train.max_steps 10000 --train.save_steps 2000 > "$WS/logs/train_$RUN.log" 2>&1
echo "[run_10k $EXP] train end $(date -Is) rc=$?"
CKPT=$WS/outputs/$RUN/checkpoints/global_step_10000/hf_ckpt
if [ ! -f "$CKPT/model.safetensors.index.json" ]; then echo "[run_10k $EXP] no 10k hf_ckpt, stop"; exit 1; fi
echo "[run_10k $EXP] eval start $(date -Is)"
bash "$XSY/eval_baseline.sh" "$CKPT" > "$WS/logs/eval_$RUN.log" 2>&1
echo "[run_10k $EXP] eval end $(date -Is) rc=$?"
