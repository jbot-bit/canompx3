#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PY="$ROOT/.venv-wsl/bin/python"
TMP_SITE="/tmp/canompx3-fastmcp"

if [[ -x "$VENV_PY" ]]; then
  PYTHON_BIN="$VENV_PY"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "ERROR: no usable Python found for gold-db MCP." >&2
  echo "Expected .venv-wsl/bin/python or python3 in PATH." >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c "import fastmcp" >/dev/null 2>&1; then
  if [[ -d "$TMP_SITE" ]]; then
    # Dependency cache only: this is not a second gold.db or market-data store.
    export PYTHONPATH="$TMP_SITE${PYTHONPATH:+:$PYTHONPATH}"
  fi
fi

if ! "$PYTHON_BIN" -c "import fastmcp" >/dev/null 2>&1; then
  echo "Setting up gold-db MCP dependencies..." >&2
  "$PYTHON_BIN" -m pip install --upgrade --target "$TMP_SITE" -c "$ROOT/constraints.txt" fastmcp >&2
  export PYTHONPATH="$TMP_SITE${PYTHONPATH:+:$PYTHONPATH}"
fi

if ! "$PYTHON_BIN" -c "import fastmcp" >/dev/null 2>&1; then
  echo "ERROR: gold-db MCP requires the 'fastmcp' package, but dependency setup failed." >&2
  echo "Interpreter: $PYTHON_BIN" >&2
  echo "Dependency cache checked: $TMP_SITE (not a gold.db copy)" >&2
  exit 1
fi

cd "$ROOT"
exec "$PYTHON_BIN" trading_app/mcp_server.py
