#!/usr/bin/env bash
# Build the `lingbotvla` training venv on the PPU (PPU-ZW810E) box -- the counterpart of setup.sh's
# deps + model_env steps, which assume conda and torch 2.8/cu128 wheels that do not run on PPU.
#
#   bash xsy_utils/env/setup_venv_ppu.sh            # VENV=/mnt/cpfs/venvs/lingbotvla by default
#
# Same approach as the pi05 / rt50 envs on this box: a venv on the system python 3.12 with
# --system-site-packages, so the vendor PPU stack (torch 2.9.0, torchvision 0.24, triton 3.5,
# flash-attn 2.7.4) is inherited, never reinstalled. Everything else is pinned by lingbotvla.lock.txt
# (the H20 env snapshot), installed --no-deps. torchcodec is skipped: the 0.6.0 wheel is built for
# torch 2.8 and no PPU build loads (the pi05 env's copy fails too), so train on image-mode LeRobot data
# (video_utils falls back to pyav otherwise).
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${VENV:-/mnt/cpfs/venvs/lingbotvla}"
SYS_PY="${SYS_PY:-/usr/local/bin/python}"
LOCK="$REPO/xsy_utils/env/lingbotvla.lock.txt"
VM="$REPO/lingbotvla/models/vla/vision_models"
export PYTHONNOUSERSITE=1

[ -d "$REPO/third_party/utils3d/.git" ] || git clone -q https://github.com/EasternJournalist/utils3d.git "$REPO/third_party/utils3d"
git -C "$REPO/third_party/utils3d" checkout -q 3fab839f0be9931dac7c8488eb0e1600c236e183

[ -x "$VENV/bin/python" ] || "$SYS_PY" -m venv --system-site-packages "$VENV"
P="$VENV/bin/python"
torch_before=$("$P" -c 'import torch; print(torch.__version__, torch.__file__)')

"$P" -m pip install --no-deps -r <(grep -vE '^(#|torchcodec==)' "$LOCK")
"$P" -m pip install --no-deps 'lerobot @ https://github.com/huggingface/lerobot/archive/refs/tags/v0.4.2.tar.gz'
"$P" -m pip install --no-deps py-spy==0.4.2  # hang watchdog of xsy_utils/final_runs/train.sh
for p in "$REPO" "$REPO/third_party/utils3d" "$VM/lingbot-depth" "$VM/MoGe"; do
    "$P" -m pip install --no-deps --no-build-isolation -e "$p"
done

torch_after=$("$P" -c 'import torch; print(torch.__version__, torch.__file__)')
[ "$torch_before" = "$torch_after" ] || { echo "PPU torch was replaced: $torch_before -> $torch_after" >&2; exit 1; }
"$P" - <<'PY'
import torch, flash_attn, transformers, lerobot, lingbotvla, moge, mdm, utils3d, cv2, mlflow, trimesh
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), torch.cuda.device_count(),
      "| flash_attn", flash_attn.__version__, "| transformers", transformers.__version__,
      "| lerobot", lerobot.__version__)
PY
echo "venv ready: $VENV"
