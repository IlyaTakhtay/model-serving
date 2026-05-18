#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/env.sh"

EXPERIMENT="${1:?Usage: bash scripts/run_single_experiment.sh <latency|resources|update|rollback|recovery|diagnostics> [extra args...]}"
shift || true

cd "$ROOT_DIR"
python -m experiments.stage6.run_experiment \
  --experiment "$EXPERIMENT" \
  --artifact experiments/detection/artifacts/yolo26n.onnx \
  --metadata experiments/detection/yolo26n.execution.json \
  --output-root experiments/results \
  "$@"
