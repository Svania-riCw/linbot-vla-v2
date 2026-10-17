#!/usr/bin/env bash
# Run inside tmux; repeat the same command after interruption to resume.
set -euo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${MODEL_PYTHON:-/2025233210/xiesy/lingbot_ws/conda_envs/lingbotvla/bin/python}"
cd "$CODE"
exec "$PYTHON" "$CODE/ysj_utils/resume_joint_delta.py" "$@"
