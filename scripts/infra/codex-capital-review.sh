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

"$VENV/bin/python" "$ROOT/scripts/tools/wsl_mount_guard.py" --root "$ROOT"

if [[ -d "$HOME/.nvm/versions/node" ]]; then
  NVM_CODEX="$(find "$HOME/.nvm/versions/node" -path '*/bin/codex' -print 2>/dev/null | sort | tail -n1 || true)"
  if [[ -n "$NVM_CODEX" ]]; then
    export PATH="$(dirname "$NVM_CODEX"):$PATH"
  fi
fi

if [[ "${CANOMPX3_SKIP_PREFLIGHT:-0}" != "1" && -f "$PREFLIGHT" ]]; then
  "$VENV/bin/python" "$PREFLIGHT" --context codex-wsl --claim codex-capital-review || true
fi

PROMPT=$(cat <<'EOF'
Use the canompx3-capital-review skill. Review the current uncommitted changes
as capital-at-risk software, not as a style diff. Classify the route stack
first, then report findings with MEASURED / INFERRED / UNSUPPORTED labels.
Apply live-audit, deploy-readiness, research/evidence, security, threat-model,
or supply-chain scrutiny whenever the diff touches those surfaces. High and
critical findings require PREMISE -> TRACE -> EVIDENCE -> VERDICT. End with
Decision: BLOCK, FIX_REQUIRED, VERIFY_MORE, ACCEPT_WITH_RISK, or CLEAR.
EOF
)

exec codex -p canompx3_max review --uncommitted "$PROMPT" "$@"
