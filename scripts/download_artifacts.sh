#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_URL="${MODEL_URL:-https://huggingface.co/zwh20081/yolo26-onnx/resolve/main/yolo26n.onnx}"
MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/experiments/detection/artifacts/yolo26n.onnx}"

mkdir -p "$(dirname "$MODEL_PATH")"

if command -v curl >/dev/null 2>&1; then
  curl -L "$MODEL_URL" -o "$MODEL_PATH"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$MODEL_PATH" "$MODEL_URL"
else
  python - "$MODEL_URL" "$MODEL_PATH" <<'PY'
import sys
from pathlib import Path
from urllib.request import urlopen

url = sys.argv[1]
target = Path(sys.argv[2])
target.parent.mkdir(parents=True, exist_ok=True)
with urlopen(url, timeout=240) as response:
    target.write_bytes(response.read())
PY
fi

echo "Saved model artifact to $MODEL_PATH"
echo "Dataset download is optional for the main measurements."
echo "For detection validation data, run:"
echo "python experiments/detection/download_hf_yolo_dataset.py --repo-id LibreYOLO/road-traffic --output-dir experiments/detection/datasets/road-traffic"
