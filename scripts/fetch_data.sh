#!/usr/bin/env bash
# Download the original CoLight/CityFlow benchmark files (Jinan 3x4, Hangzhou 4x4, New York 28x7).
set -euo pipefail
DEST="${1:-data/raw/colight}"
if [ ! -f "$DEST/data/Jinan/3_4/roadnet_3_4.json" ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/wingsweihua/colight.git "$DEST"
  git -C "$DEST" sparse-checkout set data/Jinan data/Hangzhou data/NewYork/28_7
fi
git -C "$DEST" rev-parse HEAD
