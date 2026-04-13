#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANAGER="$ROOT/scripts/tools/worktree_manager.py"

usage() {
  cat <<'EOF'
Usage:
  scripts/infra/codex-worktree.sh open <workstream-name> [-- codex args...]
  scripts/infra/codex-worktree.sh search <workstream-name> [-- codex args...]
  scripts/infra/codex-worktree.sh close <workstream-name> [--force] [--drop-branch]
  scripts/infra/codex-worktree.sh list
  scripts/infra/codex-worktree.sh prune
EOF
}

cmd="${1:-}"
if [[ -z "$cmd" ]]; then
  usage
  exit 2
fi
shift || true

case "$cmd" in
  open|search)
    workstream="${1:-}"
    if [[ -z "$workstream" ]]; then
      echo "Workstream name required." >&2
      exit 2
    fi
    python3 "$ROOT/scripts/tools/wsl_mount_guard.py" --root "$ROOT"
    shift || true
    if [[ "${1:-}" == "--" ]]; then
      shift
    fi
    PURPOSE="${CANOMPX3_WORKSTREAM_PURPOSE:-}"
    CREATE_ARGS=(create --tool codex --name "$workstream")
    if [[ -n "$PURPOSE" ]]; then
      CREATE_ARGS+=(--purpose "$PURPOSE")
    fi
    WT="$(python3 "$MANAGER" "${CREATE_ARGS[@]}")"
    if [[ "$cmd" == "search" ]]; then
      exec env CANOMPX3_ROOT="$WT" "$ROOT/scripts/infra/codex-project-search.sh" "$@"
    fi
    exec env CANOMPX3_ROOT="$WT" "$ROOT/scripts/infra/codex-project.sh" "$@"
    ;;
  close)
    workstream="${1:-}"
    if [[ -z "$workstream" ]]; then
      echo "Workstream name required." >&2
      exit 2
    fi
    shift || true
    exec python3 "$MANAGER" close --tool codex --name "$workstream" "$@"
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
