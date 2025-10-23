#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --upgrade pip
pip install -r requirements.txt

if ! command -v ffmpeg >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
  else
    echo "ffmpeg not found and apt-get unavailable; please install ffmpeg manually."
  fi
fi

echo "v2_core setup completed."
