#!/usr/bin/env bash
set -euo pipefail

DEFAULT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${CANOMPX3_ROOT:-$DEFAULT_ROOT}"
VENV="$ROOT/.venv-wsl"
PREFLIGHT="$ROOT/scripts/tools/session_preflight.py"

if [[ ! -f "$VENV/bin/python" ]]; then
  echo "ERROR: .venv-wsl/bin/python not found." >&2
  echo "Run 'UV_PROJECT_ENVIRONMENT=.venv-wsl uv sync --frozen --python 3.13 --group dev' inside WSL to create the venv." >&2
  echo "Windows Claude Code uses .venv/ - this script is WSL-only." >&2
  exit 1
fi

cd "$ROOT"
export JOBLIB_MULTIPROCESSING=0
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"

if [[ -d "$HOME/.nvm/versions/node" ]]; then
  NVM_CODEX="$(find "$HOME/.nvm/versions/node" -path '*/bin/codex' -print 2>/dev/null | sort | tail -n1 || true)"
  if [[ -n "$NVM_CODEX" ]]; then
    export PATH="$(dirname "$NVM_CODEX"):$PATH"
  fi
fi

if [[ "${CANOMPX3_SKIP_PREFLIGHT:-0}" != "1" && -f "$PREFLIGHT" ]]; then
  "$VENV/bin/python" "$PREFLIGHT" --quiet --context codex-wsl --claim codex || true
fi

exec codex \
  -C "$ROOT" \
  -p canompx3 \
  --sandbox workspace-write \
  --ask-for-approval on-request \
  "$@"
