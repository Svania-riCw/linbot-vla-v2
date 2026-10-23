#!/usr/bin/env bash
# Multi-node test: run the SAME command on every node of a DLC job (e.g. 2 nodes x 8 PPU = 16 cards):
#
#   bash /mnt/cpfs/linbot-vla-v2/xsy_utils/final_runs/test_multinode.sh            # full shard over all ranks
#   DP_SHARD=8 bash .../test_multinode.sh                                           # HSDP (shard 8 x replicate nodes)
#
#   1. comm    comm_test.py: rank placement, all_reduce correctness, all_gather / reduce_scatter / all_reduce
#              bandwidth for all ranks, inside a node and between nodes (TCP fallback = a few GB/s)
#   2. train   train.sh on a real config (default the q96 hero) for TEST_STEPS steps at the real bs 1024,
#              save at the last step: multi-node data loading, FSDP/HSDP, DCP save from every node to OSS,
#              async HF export (~20 min on OSS: the stage waits for it at the end)
#   3. resume  RESUME=1 from that save, 2 more steps: the multi-node resume path
#   Then a summary (StepTime per stage, checkpoints) on node 0; the test checkpoints are deleted on success.
#
# Knobs: TEST_RUN=d4rt_q96_ctxvs_res2_augon  TEST_STEPS=4  TEST_DATASETS=4 (first N of the 50; 50 = all, needs
#   the full HF cache)  STAGES="comm train resume"  KEEP_TEST_CKPT=0  COMM_SIZES_MB=64,512  plus every train.sh knob
#   (DP_SHARD, MBS, GBS, ...). Nodes / master come from DLC (WORLD_SIZE = #nodes, RANK, MASTER_ADDR/PORT) as in train.sh.
# Uses ports MASTER_PORT (train), +1 (comm), +2 (resume).
# The body is one { } block: bash parses it fully before running, so editing this file while a job
# runs cannot break that job (bash otherwise reads a script incrementally).
{
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/../.." && pwd)"
source "$REPO/xsy_utils/env/ppu_env.sh"

TEST_RUN=${TEST_RUN:-d4rt_q96_ctxvs_res2_augon}
TEST_STEPS=${TEST_STEPS:-4}
TEST_DATASETS=${TEST_DATASETS:-4}
STAGES=${STAGES:-comm train resume}
KEEP_TEST_CKPT=${KEEP_TEST_CKPT:-0}
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
export NNODES=${NNODES:-${PET_NNODES:-${WORLD_SIZE:-1}}}
export NODE_RANK=${NODE_RANK:-${PET_NODE_RANK:-${RANK:-0}}}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
PORT0=${MASTER_PORT:-62500}
export NGPU
unset WORLD_SIZE RANK MASTER_PORT
CKPT_BASE=${CKPT_BASE:-/mnt/oss/xiesy/lingbot-ckpts}
OUT_BASE=${OUT_BASE:-$REPO/.runtime/final_runs}
export CKPT_BASE OUT_BASE
export RUN_SUFFIX="_mntest_${NNODES}x${NGPU}${DP_SHARD:+_hsdp$DP_SHARD}"
NAME="$TEST_RUN$RUN_SUFFIX"
OUT="$OUT_BASE/$NAME"
TLOG="$OUT/logs/test_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$OUT/logs"
exec > >(tee -a "$TLOG") 2>&1
echo "[test] $(date -Is) host $(hostname): node $NODE_RANK/$NNODES x $NGPU gpus, master $MASTER_ADDR:$PORT0, run $NAME, stages: $STAGES"

if [ "$TEST_DATASETS" -lt 50 ]; then
    export TRAIN_LIST="$OUT/train_list_first$TEST_DATASETS.txt"
    head -n "$TEST_DATASETS" "$REPO/xsy_utils/configs/ppu/robotwin_clean50.txt" > "$TRAIN_LIST.$NODE_RANK.tmp"
    mv -f "$TRAIN_LIST.$NODE_RANK.tmp" "$TRAIN_LIST"   # identical content from every node
fi

# node 0 removes a previous attempt's test checkpoints before anyone gets past the comm stage (whose rendezvous
# needs node 0), so train.sh's "already has checkpoints" guard passes on every node
if [ "$NODE_RANK" = 0 ] && [[ " $STAGES " == *" train "* ]]; then
    case "$CKPT_BASE/$NAME" in *_mntest_*) rm -rf "${CKPT_BASE:?}/$NAME" ;; esac
fi

run_torch() {  # port, script, args...
    local port=$1; shift
    python -m torch.distributed.run --nnodes="$NNODES" --nproc-per-node="$NGPU" --node-rank="$NODE_RANK" \
        --master-addr="$MASTER_ADDR" --master-port="$port" "$@"
}

for stage in $STAGES; do
    t0=$(date +%s)
    case "$stage" in
        comm)
            [ "$NNODES" -gt 1 ] && [ "${NCCL_IB_DISABLE:-0}" = 1 ] && \
                echo "[test][warn] NCCL_IB_DISABLE=1 in this environment: inter-node traffic over TCP sockets"
            echo "[test] comm env: $(env | grep -E '^(NCCL|PCCL|ACCL)_' | sort | tr '\n' ' ')"
            echo "[test] rdma devices: $(ls /sys/class/infiniband 2>/dev/null | tr '\n' ' ' || true)"
            run_torch $((PORT0 + 1)) "$HERE/comm_test.py"
            ;;
        train)
            MAX_STEPS=$TEST_STEPS SAVE_STEPS=$TEST_STEPS MASTER_PORT=$PORT0 KEEP_LAST=2 \
                bash "$HERE/train.sh" "$TEST_RUN"
            ;;
        resume)
            # the trainer saves at the last step anyway; skip its HF export (~20 min on OSS, tested in "train")
            RESUME=1 MAX_STEPS=$((TEST_STEPS + 2)) SAVE_STEPS=100000 MASTER_PORT=$((PORT0 + 2)) KEEP_LAST=2 \
                bash "$HERE/train.sh" "$TEST_RUN" --train.save_hf_weights false
            ;;
        *) echo "[test] unknown stage $stage" >&2; exit 2 ;;
    esac
    echo "[test] stage $stage ok on node $NODE_RANK ($(( $(date +%s) - t0 )) s)"
done

if [ "$NODE_RANK" = 0 ]; then
    echo "[test] ===== summary ($NAME) ====="
    for f in $(ls -tr "$OUT"/logs/train_node0_*.log 2>/dev/null | tail -n 2); do
        echo "[test] $(basename "$f"): StepTime $(grep -a -o 'StepTime [0-9.]*s' "$f" | awk '{printf "%s ", $2}')"
        grep -a -m1 "\[info\] nodes=" "$f" || true
        grep -a -m1 "\[info\] fsdp:" "$f" || true
        grep -a -o "max_memory_allocated[^,]*" "$f" | tail -n 1 || true
    done
    ls -d "$CKPT_BASE/$NAME/checkpoints"/global_step_* 2>/dev/null | sed 's/^/[test] saved: /' || true
    if [[ " $STAGES " == *" train "* ]] && [ "$KEEP_TEST_CKPT" != 1 ]; then
        sleep 60  # let the async HF export of the last save finish before deleting
        rm -rf "${CKPT_BASE:?}/$NAME" && echo "[test] removed test checkpoints $CKPT_BASE/$NAME"
    fi
fi
echo "[test] all stages passed on node $NODE_RANK; log $TLOG"
exit 0
}
