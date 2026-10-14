#!/usr/bin/env bash
# One-shot setup of everything the RoboTwin eval / training need that is NOT tracked in git.
# Idempotent per step; pass step names to run a subset:
#   bash xsy_utils/env/setup.sh [deps] [model_env] [sim_env] [assets] [gl]
#
#   deps       third_party/utils3d @3fab839 and external/RoboTwin/envs/curobo @d64c4b0
#   model_env  conda env `lingbotvla`    (py3.12, torch 2.8 cu128, flash-attn 2.8.3, this repo -e)
#   sim_env    conda env `robotwin-eval` (py3.10, sapien/mplib/curobo, cuda-nvcc 12.8)
#   assets     RoboTwin assets: objects + background_texture symlinked from ASSETS_SRC (read-only),
#              embodiments copied (its curobo yml gets absolute paths rewritten to this checkout)
#   gl         userspace NVIDIA GL/Vulkan libs for containers mounted without the `graphics`
#              driver capability (sapien: "failed to find a rendering device"). Extracted, not
#              installed, from the driver .run matching the kernel module into $WS/nvidia-gl-<ver>.
#
# Envs go to CONDA_ENVS_PATH on shared storage; /root/miniconda3/envs does not survive restarts.
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
WS="${WS:-$(dirname "$REPO")}"
ROBOTWIN="$REPO/external/RoboTwin"
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-$WS/conda_envs}"
CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-/2025233210/lmx/wheels/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl}"
ASSETS_SRC="${ASSETS_SRC:-/2025233210/lmx/RoboTwin/assets}"
LOCKS="$REPO/xsy_utils/env"
steps=("$@"); [ ${#steps[@]} -eq 0 ] && steps=(deps model_env sim_env assets gl)

clone_at() {   # $1 = url, $2 = dir, $3 = rev
    [ -d "$2/.git" ] || git clone -q "$1" "$2"
    git -C "$2" checkout -q "$3"
}

activate() {   # conda's cuda-nvcc / gcc activate.d hooks read unset variables
    set +u; source "$CONDA_SH"; conda activate "$1"; set -u
}

ensure_env() {   # $1 = name, $2 = python version
    set +u; source "$CONDA_SH"; set -u
    [ -x "$CONDA_ENVS_PATH/$1/bin/python" ] || conda create -y -n "$1" "python=$2" pip
    activate "$1"
}

for step in "${steps[@]}"; do
    echo "==== $step"
    case "$step" in
    deps)
        clone_at https://github.com/EasternJournalist/utils3d.git "$REPO/third_party/utils3d" 3fab839f0be9931dac7c8488eb0e1600c236e183
        clone_at https://github.com/NVlabs/curobo.git "$ROBOTWIN/envs/curobo" d64c4b005459db10c5dd867d8b30a87d5bda9bdb
        ;;
    model_env)
        ensure_env lingbotvla 3.12
        pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
        pip install --no-deps -r "$LOCKS/lingbotvla.lock.txt"
        pip install --no-deps "$FLASH_ATTN_WHEEL"
        pip install --no-deps 'lerobot @ https://github.com/huggingface/lerobot/archive/refs/tags/v0.4.2.tar.gz'
        VM="$REPO/lingbotvla/models/vla/vision_models"
        for p in "$REPO" "$REPO/third_party/utils3d" "$VM/lingbot-depth" "$VM/MoGe"; do
            pip install --no-deps --no-build-isolation -e "$p"
        done
        ;;
    sim_env)
        ensure_env robotwin-eval 3.10
        set +u
        conda install -y --override-channels -c nvidia/label/cuda-12.8.1 -c conda-forge \
            cuda-nvcc=12.8.93 cuda-cudart-dev=12.8.90 cuda-cccl=12.8.90 cuda-driver-dev=12.8.90
        conda install -y --override-channels -c conda-forge gcc_linux-64=11 gxx_linux-64=11 ffmpeg
        set -u
        activate robotwin-eval
        pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
        pip install --no-deps -r "$LOCKS/robotwin-eval.lock.txt"
        CUDA_HOME="$CONDA_PREFIX" MAX_JOBS="${MAX_JOBS:-16}" \
            pip install --no-deps --no-build-isolation -e "$ROBOTWIN/envs/curobo"
        # Pinned-RoboTwin package fixes (utf-8 URDF/SRDF reads; mplib screw-motion collision check).
        python - <<'PY'
import importlib.util
from pathlib import Path
sapien_dir = Path(importlib.util.find_spec("sapien").origin).parent   # find_spec: no sapien import
mplib_dir = Path(importlib.util.find_spec("mplib").origin).parent
loader = sapien_dir / "wrapper/urdf_loader.py"
loader.write_text(loader.read_text()
    .replace('open(urdf_file, "r")', 'open(urdf_file, "r", encoding="utf-8")')
    .replace('open(srdf_file, "r")', 'open(srdf_file, "r", encoding="utf-8")'))
planner = mplib_dir / "planner.py"
planner.write_text(planner.read_text().replace(
    "if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:",
    "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:"))
print("applied sapien/mplib fixes")
PY
        ;;
    assets)
        for d in objects background_texture; do
            [ -e "$ROBOTWIN/assets/$d" ] || ln -s "$ASSETS_SRC/$d" "$ROBOTWIN/assets/$d"
        done
        [ -d "$ROBOTWIN/assets/embodiments" ] || cp -r "$ASSETS_SRC/embodiments" "$ROBOTWIN/assets/embodiments"
        (cd "$ROBOTWIN" && python3 -I script/update_embodiment_config_path.py < /dev/null)
        ;;
    gl)
        ver=$(sed -n 's/.*Kernel Module *\([0-9.]*\).*/\1/p' /proc/driver/nvidia/version)
        dst="$WS/nvidia-gl-$ver"
        if [ -e "$dst/lib/libGLX_nvidia.so.0" ]; then echo "already staged: $dst"; continue; fi
        mkdir -p "$dst/_dl" "$dst/lib" "$dst/icd"
        run="$dst/_dl/NVIDIA-Linux-x86_64-$ver.run"
        [ -f "$run" ] || curl -sSfL -o "$run" "https://us.download.nvidia.com/tesla/$ver/NVIDIA-Linux-x86_64-$ver.run"
        sh "$run" --extract-only --target "$dst/_dl/x" > /dev/null
        for l in libEGL_nvidia libGLX_nvidia libnvidia-eglcore libnvidia-glcore libnvidia-glsi \
                 libnvidia-glvkspirv libnvidia-gpucomp libnvidia-rtcore libnvidia-tls libnvidia-allocator; do
            cp "$dst/_dl/x/$l.so.$ver" "$dst/lib/"
        done
        ln -sf "libEGL_nvidia.so.$ver" "$dst/lib/libEGL_nvidia.so.0"
        ln -sf "libGLX_nvidia.so.$ver" "$dst/lib/libGLX_nvidia.so.0"
        ln -sf "libnvidia-allocator.so.$ver" "$dst/lib/libnvidia-allocator.so.1"
        api=$(sed -n 's/.*"api_version" *: *"\([0-9.]*\)".*/\1/p' "$dst/_dl/x/nvidia_icd.json")
        printf '{\n    "file_format_version" : "1.0.0",\n    "ICD": {\n        "library_path": "%s",\n        "api_version" : "%s"\n    }\n}\n' \
            "$dst/lib/libGLX_nvidia.so.0" "$api" > "$dst/icd/nvidia_icd.json"
        printf '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "%s"\n    }\n}\n' \
            "$dst/lib/libEGL_nvidia.so.0" > "$dst/icd/10_nvidia.json"
        rm -rf "$dst/_dl/x"
        echo "staged $dst"
        ;;
    *) echo "unknown step: $step" >&2; exit 2 ;;
    esac
done
