#!/usr/bin/env bash
set -euo pipefail

root="${LMX_ROOT:-/2025233210/lmx}"
repo="Ruicheng/moge-2-vitb-normal"
revision="ca5f0e07ff01d3e5a364c1d954ed12ee1814b368"
endpoint="${RTC_MOGE_ENDPOINT:-https://hf-mirror.com}"
target="${RTC_MOGE_PATH:-$root/checkpoints/moge2-vitb-normal.pt}"
expected_bytes=419110160
expected_sha256=16b8110e86d5dc5a849db120ca96ef3a223fd30b0c9146d1d81db504073da5f6
mkdir -p "$(dirname -- "$target")"
if [[ -s "$target" ]]; then
    actual_bytes=$(stat -c%s "$target")
    if [[ "$actual_bytes" -eq "$expected_bytes" ]]; then
        printf '%s  %s\n' "$expected_sha256" "$target" | sha256sum -c -
        printf 'MoGe checkpoint already present: %s (%s bytes)\n' "$target" "$actual_bytes"
        sha256sum "$target" | tee "${target}.sha256"
        exit 0
    fi
    printf 'Existing MoGe checkpoint has unexpected size: %s (%s bytes)\n' "$target" "$actual_bytes" >&2
    exit 2
fi
partial="${target}.part"
url="${endpoint%/}/${repo}/resolve/${revision}/model.pt"
curl -L --fail --retry 4 --retry-delay 3 --connect-timeout 30 -C - "$url" -o "$partial"
actual_bytes=$(stat -c%s "$partial")
if [[ "$actual_bytes" -ne "$expected_bytes" ]]; then
    printf 'Downloaded MoGe checkpoint has unexpected size: %s bytes (expected %s)\n' "$actual_bytes" "$expected_bytes" >&2
    exit 2
fi
printf '%s  %s\n' "$expected_sha256" "$partial" | sha256sum -c -
mv "$partial" "$target"
sha256sum "$target" | tee "${target}.sha256"
printf 'Installed MoGe checkpoint: %s (%s bytes)\n' "$target" "$actual_bytes"
