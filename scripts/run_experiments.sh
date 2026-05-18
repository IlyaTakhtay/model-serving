#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/env.sh"

cd "$ROOT_DIR"

for experiment in latency resources update rollback recovery diagnostics; do
  echo "== $experiment =="
  bash scripts/run_single_experiment.sh "$experiment" "$@"
done

echo "Reports are stored under experiments/results/"
