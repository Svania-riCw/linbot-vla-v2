#!/usr/bin/env bash
# Run one GPU stage of the d4rt data prep on all GPUs: one rank per GPU, round-robin shards.
# Resumable (finished files are skipped). Logs: $LOG_DIR/<stage>_r<rank>.log
#
#   bash run_all.sh depth  [extra prep_depth.py args]
#   bash run_all.sh tracks [extra prep_tracks.py args]
#   GPUS="0 1" TASKS=place_shoe bash run_all.sh depth
#
# prep_ee.py is CPU-only and takes seconds; run it directly.
set -euo pipefail

STAGE=${1:?usage: run_all.sh depth|tracks [args]}; shift
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')}
TASKS=${TASKS:-all}
LOG_DIR=${LOG_DIR:-/2025233210/xiesy/lingbot_ws/logs/d4rt_data}
export CONDA_ENVS_PATH=${CONDA_ENVS_PATH:-/2025233210/xiesy/lingbot_ws/conda_envs}
set +u; source /root/miniconda3/etc/profile.d/conda.sh; conda activate d4rt-prep; set -u
mkdir -p "$LOG_DIR"

read -r -a G <<< "$GPUS"
pids=()
for r in "${!G[@]}"; do
    CUDA_VISIBLE_DEVICES=${G[$r]} python "$HERE/prep_$STAGE.py" --tasks "$TASKS" \
        --rank "$r" --world "${#G[@]}" --device cuda:0 "$@" > "$LOG_DIR/${STAGE}_r$r.log" 2>&1 &
    pids+=($!)
done
trap 'kill "${pids[@]}" 2>/dev/null' INT TERM
start=$(date +%s); rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
el=$(( $(date +%s) - start ))
printf '[run_all %s] rc=%d elapsed %dh%02dm\n' "$STAGE" "$rc" $((el / 3600)) $((el % 3600 / 60))
exit $rc
