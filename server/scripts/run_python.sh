#!/usr/bin/env bash
set -euo pipefail

if [ -x "server/.venv/bin/python" ]; then
  PYTHON_BIN="server/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON:-python3}"
fi

exec "$PYTHON_BIN" "$@"
