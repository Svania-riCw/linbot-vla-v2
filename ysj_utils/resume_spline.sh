#!/usr/bin/env bash
# Run in tmux. Repeating this command selects the latest complete continuation DCP.
set -euo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${MODEL_PYTHON:-/2025233210/xiesy/lingbot_ws/conda_envs/lingbotvla/bin/python}"
cd "$CODE"
exec "$PYTHON" "$CODE/ysj_utils/resume_spline.py" "$@"
