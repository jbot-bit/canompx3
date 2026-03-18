#!/usr/bin/env bash
set -euo pipefail

DEFAULT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${CANOMPX3_ROOT:-$DEFAULT_ROOT}"
VENV="$ROOT/.venv-wsl"
PREFLIGHT="$ROOT/scripts/tools/session_preflight.py"

if [[ "${CANOMPX3_SKIP_PREFLIGHT:-0}" != "1" && -f "$PREFLIGHT" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    python3 "$PREFLIGHT" --quiet --context codex-wsl --claim codex-search || true
  elif command -v python >/dev/null 2>&1; then
    python "$PREFLIGHT" --quiet --context codex-wsl --claim codex-search || true
  fi
fi

if [[ ! -f "$VENV/bin/python" ]]; then
  echo "ERROR: .venv-wsl/bin/python not found." >&2
  echo "Run 'uv sync --frozen' inside WSL to create the venv." >&2
  echo "Windows Claude Code uses .venv/ - this script is WSL-only." >&2
  exit 1
fi

cd "$ROOT"
export JOBLIB_MULTIPROCESSING=0
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"

exec codex \
  -C "$ROOT" \
  -p canompx3_search \
  --sandbox workspace-write \
  --ask-for-approval on-request \
  --search \
  "$@"
