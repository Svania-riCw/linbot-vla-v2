#!/usr/bin/env bash
# final_d4rt_q96_ctxvs_res2_augon_bs1024 -- the final run: q96, res2 (384 full432), full432 observation augmentation,
# global batch 1024 (official), 20k steps, compile off. DLC entry, 8 nodes x 8 PPU, the same command on every node:
#   bash /mnt/cpfs/linbot-vla-v2/xsy_utils/final_runs/train_final_d4rt_q96_ctxvs_res2_augon_bs1024.sh
# 64 GPUs x MBS 16 = 1024: no grad accumulation; ~7-7.5 s/step -> ~1.7 days (README section 3). Another node count
# stops here (it would accumulate gradients: same batch, slower); ALLOW_NNODES=1 to run it anyway.
# Saves every 1000 steps (~2 h; SAVE_STEPS overrides), hf_ckpt kept every 5k. The LR (cosine 1e-4 -> 5e-5) decays
# over MAX_STEPS: to go beyond 20k, set MAX_STEPS at the start.
# knobs and data: train.sh / README.md. RESUME=1 to continue after a restart (needs the same node count).
n=${NNODES:-${PET_NNODES:-${WORLD_SIZE:-1}}}  # as train.sh
if [ "$n" != 8 ] && [ "${ALLOW_NNODES:-0}" != 1 ]; then
    echo "[error] final_d4rt_q96_ctxvs_res2_augon_bs1024 is set up for 8 nodes, this job has $n (ALLOW_NNODES=1 to run anyway)" >&2
    exit 2
fi
export SAVE_STEPS=${SAVE_STEPS:-1000}
exec bash "$(dirname "${BASH_SOURCE[0]}")/train.sh" final_d4rt_q96_ctxvs_res2_augon_bs1024 "$@"
