#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/env.sh"

cd "$ROOT_DIR"
python -m experiments.measurements.two_model_concurrent \
  --output-root experiments/results \
  "$@"
