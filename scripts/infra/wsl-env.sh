#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$ROOT/.venv-wsl"

if [[ ! -f "$VENV/bin/python" ]]; then
  echo "ERROR: .venv-wsl/bin/python not found." >&2
  echo "Run 'uv sync --frozen' inside WSL to create the venv." >&2
  echo "Windows Claude Code uses .venv/ — this script is WSL-only." >&2
  exit 1
fi

cd "$ROOT"
export JOBLIB_MULTIPROCESSING=0
source "$VENV/bin/activate"

if [[ "$#" -eq 0 ]]; then
  exec "${SHELL:-/bin/bash}" -i
fi

exec "$@"
