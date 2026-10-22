#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export RTC_MICRO_BATCH_SIZE="${RTC_MICRO_BATCH_SIZE:-64}"
export RTC_GLOBAL_BATCH_SIZE="${RTC_GLOBAL_BATCH_SIZE:-512}"
export RTC_MAX_DELAY="${RTC_MAX_DELAY:-8}"
export RTC_MAX_STEPS="${RTC_MAX_STEPS:-10000}"

# train_rtc.sh activates lingbotvla-train and prepares the RTC config.
# Forward --resume, --overwrite, or other trainer arguments unchanged.
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    exec bash "$repo_root/train_rtc.sh" "$@"
fi
if [[ "$#" -eq 0 ]]; then
    echo "Launching RTC training: run=rtc_moge_run01, GPUs=$CUDA_VISIBLE_DEVICES, micro_batch=$RTC_MICRO_BATCH_SIZE, global_batch=$RTC_GLOBAL_BATCH_SIZE, max_delay=$RTC_MAX_DELAY, max_steps=$RTC_MAX_STEPS" >&2
fi
exec bash "$repo_root/train_rtc.sh" --name "${RTC_RUN_NAME:-rtc_moge_run01}" "$@" --train.max_steps "$RTC_MAX_STEPS"
