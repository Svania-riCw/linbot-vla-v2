#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/rtc_paths.sh"
rtc_activate_env
cd "$RTC_REPO_ROOT"
python -m unittest discover -s tests -p 'test_rtc*.py' -v
