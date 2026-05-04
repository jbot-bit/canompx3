#!/usr/bin/env bash
set -euo pipefail

DEFAULT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${CANOMPX3_ROOT:-$DEFAULT_ROOT}"
PROFILE="${CANOMPX3_CODEX_PROFILE:-canompx3}"
TASK_TEXT="${CANOMPX3_STARTUP_TASK:-}"
ROUTER="$ROOT/scripts/tools/session_router.py"

if [[ "${CANOMPX3_SESSION_AUTO_ROUTE:-1}" != "0" && -f "$ROUTER" ]]; then
  ROUTE_ARGS=(--root "$ROOT" --tool codex --mode mutating)
  if [[ -n "$TASK_TEXT" ]]; then
    ROUTE_ARGS+=(--task "$TASK_TEXT")
  fi
  ROUTED_ROOT="$(python3 "$ROUTER" "${ROUTE_ARGS[@]}" 2> >(cat >&2))"
  if [[ -n "$ROUTED_ROOT" ]]; then
    ROOT="$ROUTED_ROOT"
  fi
fi

VENV="$ROOT/.venv-wsl"
PREFLIGHT="$ROOT/scripts/tools/session_preflight.py"
TASK_ROUTE_PACKET="$ROOT/scripts/tools/task_route_packet.py"

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

"$VENV/bin/python" "$ROOT/scripts/tools/wsl_mount_guard.py" --root "$ROOT"

if [[ -d "$HOME/.nvm/versions/node" ]]; then
  NVM_CODEX="$(find "$HOME/.nvm/versions/node" -path '*/bin/codex' -print 2>/dev/null | sort | tail -n1 || true)"
  if [[ -n "$NVM_CODEX" ]]; then
    export PATH="$(dirname "$NVM_CODEX"):$PATH"
  fi
fi

if [[ "${CANOMPX3_SKIP_PREFLIGHT:-0}" != "1" && -f "$PREFLIGHT" ]]; then
  "$VENV/bin/python" "$PREFLIGHT" --quiet --context codex-wsl --claim codex --mode mutating
fi

if [[ -f "$TASK_ROUTE_PACKET" ]]; then
  if [[ -n "$TASK_TEXT" ]]; then
    "$VENV/bin/python" "$TASK_ROUTE_PACKET" \
      --root "$ROOT" \
      --tool codex \
      --task "$TASK_TEXT" \
      --briefing-level mutating >/dev/null || true
  else
    "$VENV/bin/python" "$TASK_ROUTE_PACKET" --root "$ROOT" --clear >/dev/null || true
  fi
fi

CODEX_ARGS=(
  -C "$ROOT"
  -p "$PROFILE"
  --sandbox workspace-write
  --ask-for-approval on-request
  -c 'mcp_servers.repo-state.command="bash"'
  -c 'mcp_servers.repo-state.args=["scripts/infra/run-repo-state-mcp.sh"]'
  -c 'mcp_servers.research-catalog.command="bash"'
  -c 'mcp_servers.research-catalog.args=["scripts/infra/run-research-catalog-mcp.sh"]'
)

if [[ "${CANOMPX3_CODEX_ENABLE_GOLD_DB:-0}" == "1" ]]; then
  CODEX_ARGS+=(
    -c 'mcp_servers.gold-db.command="bash"'
    -c 'mcp_servers.gold-db.args=["scripts/infra/run-gold-db-mcp.sh"]'
  )
fi

exec codex \
  "${CODEX_ARGS[@]}" \
  "$@"
