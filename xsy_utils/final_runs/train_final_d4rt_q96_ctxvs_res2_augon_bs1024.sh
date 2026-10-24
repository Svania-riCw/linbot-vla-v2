#!/usr/bin/env bash
# final_d4rt_q96_ctxvs_res2_augon_bs1024 -- the final run: q96, res2 (384 full432), full432 observation augmentation,
# global batch 1024 (official), 20k steps, compile off. DLC entry (8 x PPU):
#   bash /mnt/cpfs/linbot-vla-v2/xsy_utils/final_runs/train_final_d4rt_q96_ctxvs_res2_augon_bs1024.sh
# 8 GPUs x MBS 16 x GA 8; ~53 s/step -> ~12.5 days (README section 3). The same command on 2 / 4 nodes trains the same
# batch with GA 4 / 2, ~2x / ~4x faster. Saves every 500 steps (~7.5 h; SAVE_STEPS overrides), hf_ckpt kept every 5k.
# The LR (cosine 1e-4 -> 5e-5) decays over MAX_STEPS: to go beyond 20k, set MAX_STEPS at the start.
# knobs and data: train.sh / README.md. RESUME=1 to continue after a restart (needs the same node count).
export SAVE_STEPS=${SAVE_STEPS:-500}
exec bash "$(dirname "${BASH_SOURCE[0]}")/train.sh" final_d4rt_q96_ctxvs_res2_augon_bs1024 "$@"
