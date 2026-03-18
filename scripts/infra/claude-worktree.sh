#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANAGER="$ROOT/scripts/tools/worktree_manager.py"

usage() {
  cat <<'EOF'
Usage:
  scripts/infra/claude-worktree.sh open <workstream-name> [-- claude args...]
  scripts/infra/claude-worktree.sh close <workstream-name> [--force] [--drop-branch]
  scripts/infra/claude-worktree.sh list
  scripts/infra/claude-worktree.sh prune
EOF
}

cmd="${1:-}"
if [[ -z "$cmd" ]]; then
  usage
  exit 2
fi
shift || true

case "$cmd" in
  open)
    workstream="${1:-}"
    if [[ -z "$workstream" ]]; then
      echo "Workstream name required." >&2
      exit 2
    fi
    shift || true
    if [[ "${1:-}" == "--" ]]; then
      shift
    fi
    PURPOSE="${CANOMPX3_WORKSTREAM_PURPOSE:-}"
    CREATE_ARGS=(create --tool claude --name "$workstream")
    if [[ -n "$PURPOSE" ]]; then
      CREATE_ARGS+=(--purpose "$PURPOSE")
    fi
    WT="$(python3 "$MANAGER" "${CREATE_ARGS[@]}")"
    cd "$WT"
    PREFLIGHT="$WT/scripts/tools/session_preflight.py"
    if [[ "${CANOMPX3_SKIP_PREFLIGHT:-0}" != "1" && -f "$PREFLIGHT" ]]; then
      if command -v python3 >/dev/null 2>&1; then
        python3 "$PREFLIGHT" --quiet --context generic --claim claude || true
      elif command -v python >/dev/null 2>&1; then
        python "$PREFLIGHT" --quiet --context generic --claim claude || true
      fi
    fi
    exec env CANOMPX3_ROOT="$WT" claude "$@"
    ;;
  close)
    workstream="${1:-}"
    if [[ -z "$workstream" ]]; then
      echo "Workstream name required." >&2
      exit 2
    fi
    shift || true
    exec python3 "$MANAGER" close --tool claude --name "$workstream" "$@"
    ;;
  list)
    exec python3 "$MANAGER" list --managed-only
    ;;
  prune)
    exec python3 "$MANAGER" prune
    ;;
  *)
    usage
    exit 2
    ;;
esac
