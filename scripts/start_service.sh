#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/env.sh"

PID_FILE="$ROOT_DIR/logs/service.pid"
STDOUT_LOG="$ROOT_DIR/logs/service.stdout.log"
STDERR_LOG="$ROOT_DIR/logs/service.stderr.log"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Service is already running with PID $(cat "$PID_FILE")"
  exit 0
fi

cd "$ROOT_DIR"
nohup python -m app.main >"$STDOUT_LOG" 2>"$STDERR_LOG" &
echo "$!" > "$PID_FILE"
echo "Started service on http://${SERVING_HOST}:${SERVING_PORT} with PID $(cat "$PID_FILE")"
