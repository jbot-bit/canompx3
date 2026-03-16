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
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"

exec codex \
  -C "$ROOT" \
  -p canompx3 \
  --sandbox workspace-write \
  --ask-for-approval on-request \
  "$@"
