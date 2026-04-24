#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANAGER="$ROOT/scripts/tools/worktree_manager.py"
TASK_ROUTE_PACKET="$ROOT/scripts/tools/task_route_packet.py"

usage() {
  cat <<'EOF'
Usage:
  scripts/infra/codex-worktree.sh open <workstream-name> [--queue-item <id>] [--override-note <note>] [--startup-task <text>] [-- codex args...]
  scripts/infra/codex-worktree.sh search <workstream-name> [--startup-task <text>] [-- codex args...]
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
    PURPOSE="${CANOMPX3_WORKSTREAM_PURPOSE:-}"
    TASK_TEXT="${CANOMPX3_STARTUP_TASK:-}"
    QUEUE_ITEM="${CANOMPX3_QUEUE_ITEM:-}"
    QUEUE_OVERRIDE_NOTE="${CANOMPX3_QUEUE_OVERRIDE_NOTE:-}"
    PASSTHROUGH_ARGS=()
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --startup-task)
          TASK_TEXT="${2:-}"
          shift 2
          ;;
        --queue-item)
          QUEUE_ITEM="${2:-}"
          shift 2
          ;;
        --override-note)
          QUEUE_OVERRIDE_NOTE="${2:-}"
          shift 2
          ;;
        --)
          shift
          PASSTHROUGH_ARGS+=("$@")
          break
          ;;
        *)
          PASSTHROUGH_ARGS+=("$@")
          break
          ;;
      esac
    done
    CREATE_ARGS=(create --tool codex --name "$workstream")
    if [[ -n "$PURPOSE" ]]; then
      CREATE_ARGS+=(--purpose "$PURPOSE")
    fi
    WT="$(python3 "$MANAGER" "${CREATE_ARGS[@]}")"
    if [[ -z "$TASK_TEXT" ]]; then
      TASK_TEXT="$workstream"
      if [[ -n "$PURPOSE" ]]; then
        TASK_TEXT="$PURPOSE: $workstream"
      fi
    fi
    if [[ -f "$TASK_ROUTE_PACKET" ]]; then
      BRIEFING_LEVEL="mutating"
      if [[ "$cmd" == "search" ]]; then
        BRIEFING_LEVEL="read_only"
      fi
      PACKET_ARGS=(
        --root "$WT"
        --tool codex
        --task "$TASK_TEXT"
        --briefing-level "$BRIEFING_LEVEL"
      )
      if [[ -n "$QUEUE_ITEM" ]]; then
        PACKET_ARGS+=(--queue-item "$QUEUE_ITEM")
      fi
      if [[ -n "$QUEUE_OVERRIDE_NOTE" ]]; then
        PACKET_ARGS+=(--override-note "$QUEUE_OVERRIDE_NOTE")
      fi
      python3 "$TASK_ROUTE_PACKET" "${PACKET_ARGS[@]}" >/dev/null || true
    fi
    if [[ "$cmd" == "search" ]]; then
      exec env CANOMPX3_ROOT="$WT" "$ROOT/scripts/infra/codex-project-search.sh" \
        --startup-task "$TASK_TEXT" \
        -- "${PASSTHROUGH_ARGS[@]}"
    fi
    PROJECT_ARGS=(--startup-task "$TASK_TEXT")
    if [[ -n "$QUEUE_ITEM" ]]; then
      PROJECT_ARGS+=(--queue-item "$QUEUE_ITEM")
    fi
    if [[ -n "$QUEUE_OVERRIDE_NOTE" ]]; then
      PROJECT_ARGS+=(--override-note "$QUEUE_OVERRIDE_NOTE")
    fi
    exec env CANOMPX3_ROOT="$WT" "$ROOT/scripts/infra/codex-project.sh" "${PROJECT_ARGS[@]}" -- "${PASSTHROUGH_ARGS[@]}"
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
