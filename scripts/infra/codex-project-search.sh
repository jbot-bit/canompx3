#!/usr/bin/env bash
set -euo pipefail

DEFAULT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${CANOMPX3_ROOT:-$DEFAULT_ROOT}"
VENV="$ROOT/.venv-wsl"
PREFLIGHT="$ROOT/scripts/tools/session_preflight.py"
TASK_ROUTE_PACKET="$ROOT/scripts/tools/task_route_packet.py"
PROFILE="${CANOMPX3_CODEX_PROFILE:-canompx3_search}"
SHARED_CODEX_HOME_HELPER="$ROOT/scripts/infra/codex_shared_home.sh"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Setting up .venv-wsl for Codex..."
  if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is not installed in WSL PATH." >&2
    echo "Install uv once, then run 'codex' again from this repo." >&2
    exit 1
  fi

  export UV_PROJECT_ENVIRONMENT=.venv-wsl
  export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
  export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/tmp/uv-python}"
  export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
  mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"
  cd "$ROOT"
  uv sync --frozen --python 3.13 --group dev
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
    "$VENV/bin/python" "$PREFLIGHT" --quiet --context codex-wsl --claim codex-search --mode read-only || true
fi

if [[ -f "$TASK_ROUTE_PACKET" ]]; then
  if [[ -n "${CANOMPX3_STARTUP_TASK:-}" ]]; then
    "$VENV/bin/python" "$TASK_ROUTE_PACKET" \
      --root "$ROOT" \
      --tool codex \
      --task "$CANOMPX3_STARTUP_TASK" \
      --briefing-level read_only >/dev/null || true
  else
    "$VENV/bin/python" "$TASK_ROUTE_PACKET" --root "$ROOT" --clear >/dev/null || true
  fi
fi

CODEX_ARGS=(
  -C "$ROOT"
  --sandbox workspace-write
  --ask-for-approval on-request
  --search
  -c 'mcp_servers.repo-state.command="bash"'
  -c 'mcp_servers.repo-state.args=["scripts/infra/run-repo-state-mcp.sh"]'
  -c 'mcp_servers.research-catalog.command="bash"'
  -c 'mcp_servers.research-catalog.args=["scripts/infra/run-research-catalog-mcp.sh"]'
  -c 'mcp_servers.strategy-lab.command="bash"'
  -c 'mcp_servers.strategy-lab.args=["scripts/infra/run-strategy-lab-mcp.sh"]'
  -c 'mcp_servers.gold-db.command="bash"'
  -c 'mcp_servers.gold-db.args=["scripts/infra/run-gold-db-mcp.sh"]'
)

if declare -F append_codex_profile_arg >/dev/null 2>&1; then
  append_codex_profile_arg "$PROFILE" CODEX_ARGS
fi

exec codex \
  "${CODEX_ARGS[@]}" \
  "$@"
