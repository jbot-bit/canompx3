#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PY="$ROOT/.venv-wsl/bin/python"

if [[ -x "$VENV_PY" ]]; then
  PYTHON_BIN="$VENV_PY"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "ERROR: no usable Python found for repo-state MCP." >&2
  echo "Expected .venv-wsl/bin/python or python3 in PATH." >&2
  exit 1
fi

cd "$ROOT"
exec "$PYTHON_BIN" scripts/tools/repo_state_mcp_server.py
