#!/usr/bin/env bash
# Final runs: LingBot-VLA v2 + d4rt probe (ctxvs) on RoboTwin clean-50, DLC 8 x PPU.
#
#   bash xsy_utils/final_runs/train.sh <run>      # run = a configs/<run>.yaml, or use the train_<run>.sh wrappers
#
# Common to every run (README.md): global batch 1024, 50k steps, official lr 1e-4 -> 5e-5 cosine, Muon,
# torch.compile OFF. Batch 1024 = MBS x world x grad-accum; memory depends on MBS only.
#
# Knobs (env):
#   MBS / GRAD_CKPT   per-GPU micro batch / gradient checkpointing (defaults per resolution, measured)
#   GBS=1024 MAX_STEPS=50000 SAVE_STEPS=2500   USE_COMPILE=false   NUM_WORKERS (default cpus/ranks - 2, max 12)
#   KEEP_LAST=2       full saves (DCP model + optimizer + hf_ckpt) kept for resume, the rest pruned ...
#   KEEP_HF_EVERY=10000  ... except the hf_ckpt of these steps (for eval)
#   CKPT_BASE=/mnt/oss/xiesy/lingbot-va-checkpoints/final_runs   checkpoints/ (OSS; ~75 GB per save)
#   OUT_BASE=$REPO/.runtime/final_runs   run dir on CPFS: logs, TensorBoard, cli yaml (ossfs2 cannot append)
#   RUN_SUFFIX        appended to the run name (e.g. _try2); RESUME=1 continues an existing run
#   DRYRUN=1          preflight + print, no training     HANG_DUMP_MIN=15  watchdog idle threshold
#   NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT   multi-node (DLC sets WORLD_SIZE / RANK / MASTER_*)
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/../.." && pwd)"
source "$REPO/xsy_utils/env/ppu_env.sh"

RUN=${1:?usage: train.sh <run>  (configs/<run>.yaml)}; shift
CONFIG=${CONFIG:-$HERE/configs/$RUN.yaml}  # override only for smoke tests
[ -f "$CONFIG" ] || { echo "no config $CONFIG" >&2; exit 2; }
NAME="$RUN${RUN_SUFFIX:-}"

die() { echo "[error] $*" >&2; exit 1; }

# ---- knobs ----
case "$RUN" in
    *_res1_*) MBS_DEF=32; CKPT_DEF=true ;;   # measured on 2 PPU, README.md
    *_res2_*) MBS_DEF=16; CKPT_DEF=true ;;
    *) die "run name must contain _res1_ or _res2_" ;;
esac
MBS=${MBS:-$MBS_DEF}
GRAD_CKPT=${GRAD_CKPT:-$CKPT_DEF}
GBS=${GBS:-1024}
MAX_STEPS=${MAX_STEPS:-50000}
SAVE_STEPS=${SAVE_STEPS:-2500}
USE_COMPILE=${USE_COMPILE:-false}
KEEP_LAST=${KEEP_LAST:-2}
KEEP_HF_EVERY=${KEEP_HF_EVERY:-10000}
RESUME=${RESUME:-0}
DRYRUN=${DRYRUN:-0}
HANG_DUMP_MIN=${HANG_DUMP_MIN:-15}
CKPT_BASE=${CKPT_BASE:-/mnt/oss/xiesy/lingbot-va-checkpoints/final_runs}
OUT_BASE=${OUT_BASE:-$REPO/.runtime/final_runs}
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
NNODES=${NNODES:-${WORLD_SIZE:-1}}
NODE_RANK=${NODE_RANK:-${RANK:-0}}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-62500}
unset WORLD_SIZE RANK  # torch.distributed.run sets the per-process values
WORLD=$((NNODES * NGPU))
[ $((GBS % (MBS * WORLD))) -eq 0 ] || die "GBS=$GBS not divisible by MBS x world = $MBS x $WORLD"
GA=$((GBS / (MBS * WORLD)))
read -r CPU_QUOTA CPU_PERIOD < /sys/fs/cgroup/cpu.max 2>/dev/null || CPU_QUOTA=max
NODE_CPUS=$(nproc)
[ "$CPU_QUOTA" != max ] && NODE_CPUS=$(( CPU_QUOTA / CPU_PERIOD < NODE_CPUS ? CPU_QUOTA / CPU_PERIOD : NODE_CPUS ))
NW=$(( NODE_CPUS / NGPU - 2 )); [ $NW -gt 12 ] && NW=12; [ $NW -lt 2 ] && NW=2
NUM_WORKERS=${NUM_WORKERS:-$NW}

OUT="$OUT_BASE/$NAME"
CKPT_DIR="$CKPT_BASE/$NAME/checkpoints"
LOG_DIR="$OUT/logs"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# ---- preflight ----
python - <<'PY' || die "venv broken on this node (wrong image?): need PPU torch + lingbotvla + lerobot 0.4.2"
import torch, lingbotvla, lerobot
assert torch.cuda.is_available(), "no GPU visible"
print(f"[info] python ok: torch {torch.__version__}, lerobot {lerobot.__version__}, gpus {torch.cuda.device_count()}")
PY
[ "$(nvidia-smi -L | wc -l)" -ge "$NGPU" ] || die "NGPU=$NGPU but fewer GPUs visible"
python "$HERE/check_data.py" "$CONFIG" || die "data check failed (README.md: what to copy where)"
mkdir -p "$CKPT_DIR" && touch "$CKPT_DIR/.w" && rm -f "$CKPT_DIR/.w" || die "cannot write $CKPT_DIR (is /mnt/oss mounted?)"
if ls -d "$CKPT_DIR"/global_step_* >/dev/null 2>&1 && [ "$RESUME" != 1 ]; then
    die "$CKPT_DIR already has checkpoints; RESUME=1 to continue, or set RUN_SUFFIX"
fi
RESUME_ARGS=()
if [ "$RESUME" = 1 ]; then
    # Newest COMPLETE save (DCP writes .metadata last; extra_state has one file per rank, written after
    # the DCP parts). Passed explicitly, so a failed load stops the job instead of the trainer's silent
    # "start from scratch" fallback.
    for d in $(ls -d "$CKPT_DIR"/global_step_* 2>/dev/null | sort -rV); do
        if [ -f "$d/model/.metadata" ] && [ -f "$d/optimizer/.metadata" ] && \
           [ "$(ls "$d"/extra_state/extra_state_rank_*.pt 2>/dev/null | wc -l)" -eq "$WORLD" ]; then
            RESUME_ARGS=(--train.load_checkpoint_path "$d"); break
        fi
    done
    [ ${#RESUME_ARGS[@]} -gt 0 ] || die "RESUME=1 but no complete checkpoint for world size $WORLD in $CKPT_DIR"
fi
mkdir -p "$OUT" "$LOG_DIR"
if [ -e "$OUT/checkpoints" ] && [ "$(readlink -f "$OUT/checkpoints")" != "$(readlink -f "$CKPT_DIR")" ]; then
    die "$OUT/checkpoints exists and is not a link to $CKPT_DIR"
fi
ln -sfn "$CKPT_DIR" "$OUT/checkpoints"   # trainer writes <output_dir>/checkpoints -> OSS

echo "[info] run=$NAME config=$CONFIG"
echo "[info] out=$OUT (CPFS)  checkpoints -> $CKPT_DIR"
echo "[info] nodes=$NNODES (rank $NODE_RANK) x gpus=$NGPU = world $WORLD; MBS=$MBS GA=$GA GBS=$GBS grad_ckpt=$GRAD_CKPT compile=$USE_COMPILE"
[ "$RESUME" = 1 ] && echo "[info] resume from ${RESUME_ARGS[1]}"
echo "[info] steps=$MAX_STEPS (~$(python -c "print(round($MAX_STEPS*$GBS/549787, 1))") epochs) save=$SAVE_STEPS keep_last=$KEEP_LAST keep_hf_every=$KEEP_HF_EVERY workers/rank=$NUM_WORKERS cpus=$NODE_CPUS"
if [ "$DRYRUN" = 1 ]; then echo "[info] DRYRUN=1, stop."; exit 0; fi

# ---- warm the PPU math libraries (legacy DLC fix, 2026-10-13: all ranks stuck in D state for 30+ min
# inside the first GPU torch.linalg call, which maps ~800 MB of PPU SDK libs from the image) ----
TORCH_LIB="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
t0=$(date +%s)
for f in /usr/local/PPU_SDK/targets/x86_64-linux/lib/*.so* /usr/local/PPU_SDK/CUDA_SDK/targets/x86_64-linux/lib/*.so* "$TORCH_LIB"/*.so*; do
    [ -f "$f" ] && [ ! -L "$f" ] || continue
    timeout 300 cat "$f" > /dev/null || die "reading $f stalled/failed: container image I/O on this node is broken, resubmit"
done
timeout 600 python - <<'PY' || die "GPU linalg warmup failed or hung > 600 s"
import torch
x = torch.randn(4, 64, 4, device="cuda") + 0.1
torch.linalg.inv(torch.eye(4, device="cuda") * 2 + 0.01 * x[:, :4, :4]); torch.linalg.svd(x, full_matrices=False)
torch.cuda.synchronize()
PY
echo "[info] PPU lib pre-read + linalg warmup $(( $(date +%s) - t0 )) s"

BG=""
# ---- checkpoint pruner (node 0): newest KEEP_LAST saves stay complete; older ones keep only hf_ckpt
# when step % KEEP_HF_EVERY == 0, else are removed (the trainer never deletes saves) ----
[ "$KEEP_LAST" -ge 2 ] || die "KEEP_LAST must be >= 2 (the newest save may still be being written)"
if [ "$NODE_RANK" = 0 ]; then
    (
        while sleep 300; do
            ls -d "$CKPT_DIR"/global_step_* 2>/dev/null | sort -V | head -n -"$KEEP_LAST" | while read -r d; do
                step=${d##*_}
                if [ $((step % KEEP_HF_EVERY)) -eq 0 ]; then
                    keep="hf_ckpt"  # without a finished HF export keep the DCP model to convert later
                    [ -f "$d/hf_ckpt/model.safetensors.index.json" ] || keep="hf_ckpt model extra_state"
                    for sub in "$d"/*; do
                        case " $keep " in *" $(basename "$sub") "*) ;; *) echo "[pruner] rm $sub"; rm -rf "$sub" ;; esac
                    done
                else
                    echo "[pruner] rm $d"; rm -rf "$d"
                fi
            done
        done
    ) &
    BG="$!"
fi
# ---- hang watchdog: train log idle for HANG_DUMP_MIN min -> node snapshot (py-spy, /proc) into LOG_DIR ----
(
    last=0
    while sleep 60; do
        f=$(ls -t "$LOG_DIR"/train_node"$NODE_RANK"_*.log 2>/dev/null | head -1) || true
        [ -n "$f" ] || continue
        now=$(date +%s); idle=$(( now - $(stat -c %Y "$f") ))
        if [ $idle -ge $((HANG_DUMP_MIN * 60)) ] && [ $((now - last)) -ge $((HANG_DUMP_MIN * 60)) ]; then
            last=$now
            echo "[watchdog] log idle ${idle}s -> $(bash "$HERE/diag_node.sh" "$LOG_DIR")"
        fi
    done
) &
BG="$BG $!"
trap 'kill $BG 2>/dev/null || true' EXIT

LOG="$LOG_DIR/train_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"
{
    echo "date: $(date -Is)  host: $(hostname)  code_rev: $(git -C "$REPO" rev-parse HEAD)"
    echo "MBS=$MBS GA=$GA GBS=$GBS world=$WORLD grad_ckpt=$GRAD_CKPT compile=$USE_COMPILE steps=$MAX_STEPS save=$SAVE_STEPS"
    echo "extra args: $*"
} >> "$OUT/run_info.txt"
git -C "$REPO" diff > "$OUT/code_$(date +%Y%m%d_%H%M%S).diff"
cd "$REPO"
python -m torch.distributed.run --nnodes="$NNODES" --nproc-per-node="$NGPU" --node-rank="$NODE_RANK" \
    --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
    tasks/vla/train_lingbotvla.py "$CONFIG" \
    --train.output_dir "$OUT" \
    --train.micro_batch_size "$MBS" \
    --train.global_batch_size "$GBS" \
    --train.gradient_accumulation_steps "$GA" \
    --train.max_steps "$MAX_STEPS" \
    --train.save_steps "$SAVE_STEPS" \
    --train.use_compile "$USE_COMPILE" \
    --train.enable_gradient_checkpointing "$GRAD_CKPT" \
    --data.num_workers "$NUM_WORKERS" \
    "${RESUME_ARGS[@]}" "$@" 2>&1 | tee --output-error=warn "$LOG"
exit "${PIPESTATUS[0]}"
