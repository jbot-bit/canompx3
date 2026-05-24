#!/usr/bin/env bash
set -euo pipefail

DEFAULT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${CANOMPX3_ROOT:-$DEFAULT_ROOT}"
VENV="$ROOT/.venv-wsl"
PREFLIGHT="$ROOT/scripts/tools/session_preflight.py"
SHARED_CODEX_HOME_HELPER="$ROOT/scripts/infra/codex_shared_home.sh"
PROFILE="${CANOMPX3_CODEX_PROFILE:-canompx3_power}"

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
if [[ -f "$SHARED_CODEX_HOME_HELPER" ]]; then
  # Keep app/CLI auth, config, and session state aligned across WSL terminals.
  source "$SHARED_CODEX_HOME_HELPER"
  setup_shared_codex_home
fi

"$VENV/bin/python" "$ROOT/scripts/tools/wsl_mount_guard.py" --root "$ROOT"

if [[ -d "$HOME/.nvm/versions/node" ]]; then
  NVM_CODEX="$(find "$HOME/.nvm/versions/node" -path '*/bin/codex' -print 2>/dev/null | sort | tail -n1 || true)"
  if [[ -n "$NVM_CODEX" ]]; then
    export PATH="$(dirname "$NVM_CODEX"):$PATH"
  fi
fi

if [[ "${CANOMPX3_SKIP_PREFLIGHT:-0}" != "1" && -f "$PREFLIGHT" ]]; then
  CANOMPX3_SESSION_OWNER="pid:$$" \
    "$VENV/bin/python" "$PREFLIGHT" --context codex-wsl --claim codex-review || true
fi

CODEX_ARGS=(
  -c 'mcp_servers.repo-state.command="bash"' \
  -c 'mcp_servers.repo-state.args=["scripts/infra/run-repo-state-mcp.sh"]' \
  -c 'mcp_servers.research-catalog.command="bash"' \
  -c 'mcp_servers.research-catalog.args=["scripts/infra/run-research-catalog-mcp.sh"]' \
  -c 'mcp_servers.strategy-lab.command="bash"' \
  -c 'mcp_servers.strategy-lab.args=["scripts/infra/run-strategy-lab-mcp.sh"]' \
)

if declare -F append_codex_profile_arg >/dev/null 2>&1; then
  append_codex_profile_arg "$PROFILE" CODEX_ARGS
fi

exec codex \
  "${CODEX_ARGS[@]}" \
  review --uncommitted "$@"
