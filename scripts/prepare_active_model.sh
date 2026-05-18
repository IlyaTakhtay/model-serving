#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/env.sh"

ARTIFACT="${1:-$ROOT_DIR/experiments/detection/artifacts/yolo26n.onnx}"
METADATA="$ROOT_DIR/experiments/detection/yolo26n.execution.json"
TARGET_DIR="$ROOT_DIR/models/yolo26n/v1"

if [[ ! -f "$ARTIFACT" ]]; then
  echo "Model artifact is missing: $ARTIFACT" >&2
  echo "Run: bash scripts/download_artifacts.sh" >&2
  exit 1
fi

mkdir -p "$TARGET_DIR" "$ROOT_DIR/config"
cp "$ARTIFACT" "$TARGET_DIR/model.onnx"

python - "$ARTIFACT" "$METADATA" "$TARGET_DIR/model.json" <<'PY'
import json
import sys
from pathlib import Path

import onnxruntime as ort

artifact = Path(sys.argv[1])
metadata_override = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
output = Path(sys.argv[3])

datatype = {
    "tensor(bool)": "BOOL",
    "tensor(uint8)": "UINT8",
    "tensor(uint16)": "UINT16",
    "tensor(uint32)": "UINT32",
    "tensor(uint64)": "UINT64",
    "tensor(int8)": "INT8",
    "tensor(int16)": "INT16",
    "tensor(int32)": "INT32",
    "tensor(int64)": "INT64",
    "tensor(float16)": "FP16",
    "tensor(float)": "FP32",
    "tensor(double)": "FP64",
}

options = ort.SessionOptions()
execution = metadata_override.get("execution", {})
options.intra_op_num_threads = int(execution.get("intra_op_num_threads", 4))
options.inter_op_num_threads = int(execution.get("inter_op_num_threads", 1))
session = ort.InferenceSession(
    str(artifact),
    sess_options=options,
    providers=execution.get("providers", ["CPUExecutionProvider"]),
)

def node_spec(node):
    return {
        "name": node.name,
        "datatype": datatype.get(node.type, node.type),
        "shape": [dim if isinstance(dim, int) else -1 for dim in node.shape],
    }

metadata = {
    "name": "yolo26n",
    "version": "v1",
    "runtime": "onnxruntime",
    "artifact": "model.onnx",
    "inputs": [node_spec(node) for node in session.get_inputs()],
    "outputs": [node_spec(node) for node in session.get_outputs()],
    "execution": execution,
}
output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
PY

cat > "$ROOT_DIR/config/active_models.json" <<'JSON'
{
  "yolo26n": {
    "active": "v1",
    "previous": null
  }
}
JSON

echo "Prepared active model at $TARGET_DIR"
echo "Active version config: $ROOT_DIR/config/active_models.json"
