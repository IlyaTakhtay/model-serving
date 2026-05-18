#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
YOLO_URL="${YOLO_URL:-${MODEL_URL:-https://huggingface.co/zwh20081/yolo26-onnx/resolve/main/yolo26n.onnx}}"
YOLO_PATH="${YOLO_PATH:-${MODEL_PATH:-$ROOT_DIR/experiments/detection/artifacts/yolo26n.onnx}}"
EFFICIENTNET_URL="${EFFICIENTNET_URL:-https://huggingface.co/onnx/EfficientNet-Lite4/resolve/main/efficientnet-lite4-11.onnx}"
EFFICIENTNET_PATH="${EFFICIENTNET_PATH:-$ROOT_DIR/experiments/classification/artifacts/efficientnet-lite4-11.onnx}"

download() {
  local url="$1"
  local target="$2"
  mkdir -p "$(dirname "$target")"

  if command -v curl >/dev/null 2>&1; then
    curl -L "$url" -o "$target"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$target" "$url"
  else
    python - "$url" "$target" <<'PY'
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
  echo "Saved model artifact to $target"
}

download "$YOLO_URL" "$YOLO_PATH"
download "$EFFICIENTNET_URL" "$EFFICIENTNET_PATH"

echo "Dataset download is optional for the main measurements."
echo "For detection validation data, run:"
echo "python experiments/detection/download_hf_yolo_dataset.py --repo-id LibreYOLO/road-traffic --output-dir experiments/detection/datasets/road-traffic"
