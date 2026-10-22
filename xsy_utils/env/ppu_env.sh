# Source me. Runtime env for this repo on the PPU box / DLC nodes (venv built by setup_venv_ppu.sh).
# Plays the role of `conda activate lingbotvla` + the exports in train_baseline.sh.
_XSY_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export LINGBOT_VENV="${LINGBOT_VENV:-/mnt/cpfs/venvs/lingbotvla}"
export PATH="$LINGBOT_VENV/bin:$PATH"
export VIRTUAL_ENV="$LINGBOT_VENV"
# PYTHONPATH takes precedence over the editable installs, so lingbotvla / moge / mdm / utils3d always
# resolve to the checkout this file lives in (same as train_baseline.sh).
_VM="$_XSY_REPO/lingbotvla/models/vla/vision_models"
export PYTHONPATH="$_XSY_REPO:$_VM/MoGe:$_VM/lingbot-depth:$_XSY_REPO/third_party/utils3d"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DISABLE_TELEMETRY=1
# HF datasets turns every LeRobot parquet into an Arrow cache on first load (~141 GB for clean-50);
# keep it on CPFS (git-ignored .runtime/), never in ~ (lost on DLC nodes).
export HF_HOME="${HF_HOME:-$_XSY_REPO/.runtime/huggingface}"
# torch.compile cache shared across runs: a cold compile is minutes of silent startup.
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$_XSY_REPO/.runtime/torch_compile}"
unset _XSY_REPO _VM
