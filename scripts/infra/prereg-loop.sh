#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$ROOT/.venv-wsl"
PYTHON="${CANOMPX3_PYTHON:-}"

if [[ -z "$PYTHON" ]]; then
  if [[ -x "$VENV/bin/python" ]]; then
    PYTHON="$VENV/bin/python"
  else
    PYTHON="python3"
  fi
fi

cd "$ROOT"
exec "$PYTHON" "$ROOT/scripts/tools/prereg_front_door.py" "$@"
