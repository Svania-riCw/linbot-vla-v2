#!/usr/bin/env bash
# d4rt_q08_ctxvs_res1_augoff -- fallback hero. DLC entry (8 x PPU): bash /mnt/cpfs/linbot-vla-v2/xsy_utils/final_runs/train_d4rt_q08_ctxvs_res1_augoff.sh
# bs 256, 20k steps, compile off; knobs and data: train.sh / README.md. RESUME=1 to continue after a restart.
exec bash "$(dirname "${BASH_SOURCE[0]}")/train.sh" d4rt_q08_ctxvs_res1_augoff "$@"
