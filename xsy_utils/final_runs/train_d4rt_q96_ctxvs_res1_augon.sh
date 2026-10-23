#!/usr/bin/env bash
# d4rt_q96_ctxvs_res1_augon -- q96, res1 (256 input), full432 observation augmentation; bs 1024 (official), 20k steps,
# compile off. DLC entry, 4 nodes x 8 PPU, the same command on every node:
#   bash /mnt/cpfs/linbot-vla-v2/xsy_utils/final_runs/train_d4rt_q96_ctxvs_res1_augon.sh
# 32 GPUs x MBS 32 = 1024: no grad accumulation. Another node count stops here (it would train the same batch with
# accumulation, 2-4x slower); ALLOW_NNODES=1 to run it anyway. knobs and data: train.sh / README.md. RESUME=1 to
# continue after a restart (needs the same node count).
n=${NNODES:-${PET_NNODES:-${WORLD_SIZE:-1}}}  # as train.sh
if [ "$n" != 4 ] && [ "${ALLOW_NNODES:-0}" != 1 ]; then
    echo "[error] d4rt_q96_ctxvs_res1_augon is set up for 4 nodes, this job has $n (ALLOW_NNODES=1 to run anyway)" >&2
    exit 2
fi
exec bash "$(dirname "${BASH_SOURCE[0]}")/train.sh" d4rt_q96_ctxvs_res1_augon "$@"
