#!/usr/bin/env bash
# Shared paths for the source repository's RTC scripts.
export RTC_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export LMX_ROOT="${LMX_ROOT:-$(dirname -- "$RTC_REPO_ROOT")}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$RTC_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

rtc_conda_sh() {
    if [[ -n "${CONDA_SH:-}" ]]; then
        printf '%s\n' "$CONDA_SH"
    elif command -v conda >/dev/null 2>&1; then
        printf '%s/etc/profile.d/conda.sh\n' "$(conda info --base)"
    else
        local candidate
        for candidate in /root/miniconda3 "$HOME/miniconda3" "$HOME/miniconda"; do
            if [[ -f "$candidate/etc/profile.d/conda.sh" ]]; then
                printf '%s/etc/profile.d/conda.sh\n' "$candidate"
                return
            fi
        done
        printf 'Set CONDA_SH to your Conda activation script.\n' >&2
        return 2
    fi
}

rtc_activate_env() {
    local activation restore_nounset=false status=0
    activation="$(rtc_conda_sh)" || return
    if [[ ! -f "$activation" ]]; then
        printf 'Conda activation script not found: %s\n' "$activation" >&2
        return 2
    fi
    [[ $- == *u* ]] && restore_nounset=true
    set +u
    source "$activation" && conda activate "${RTC_TRAIN_ENV:-lingbotvla-train}" || status=$?
    if $restore_nounset; then set -u; fi
    return "$status"
}
