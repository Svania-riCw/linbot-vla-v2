#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${LINGBOT_PYTHON:-/2025233210/xiesy/lingbot_ws/conda_envs/lingbotvla/bin/python}"
exec "$PYTHON" "$REPO/ysj_utils/train_baseline_bs256.py" \
  --run-dir /2025233210/ysj/lingbot_ws/outputs/baseline_bs256_20k_20261020 "$@"
