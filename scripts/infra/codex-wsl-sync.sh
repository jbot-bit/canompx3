#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/infra/codex-wsl-sync.sh --source <source-repo> --target <target-repo>
EOF
}

SOURCE_ROOT=""
TARGET_ROOT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE_ROOT="${2:-}"
      shift 2
      ;;
    --target)
      TARGET_ROOT="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$SOURCE_ROOT" || -z "$TARGET_ROOT" ]]; then
  usage >&2
  exit 2
fi

if [[ ! -d "$SOURCE_ROOT/.git" ]]; then
  echo "ERROR: source repo not found: $SOURCE_ROOT" >&2
  exit 1
fi

if [[ ! -d "$TARGET_ROOT/.git" ]]; then
  echo "ERROR: WSL Codex repo not found: $TARGET_ROOT" >&2
  exit 1
fi

source_branch="$(git -C "$SOURCE_ROOT" rev-parse --abbrev-ref HEAD)"
target_branch="$(git -C "$TARGET_ROOT" rev-parse --abbrev-ref HEAD)"

if [[ "$source_branch" == "HEAD" ]]; then
  echo "ERROR: source repo is in detached HEAD state: $SOURCE_ROOT" >&2
  exit 1
fi

if [[ "$target_branch" == "HEAD" ]]; then
  echo "ERROR: WSL Codex repo is in detached HEAD state: $TARGET_ROOT" >&2
  exit 1
fi

source_dirty="$(git -C "$SOURCE_ROOT" status --short)"
if [[ -n "$source_dirty" ]]; then
  echo "ERROR: source repo has uncommitted changes, so the WSL Codex clone would be stale." >&2
  echo "Commit or stash the current checkout before launching Codex on the WSL clone." >&2
  git -C "$SOURCE_ROOT" status --short >&2
  exit 1
fi

target_dirty="$(git -C "$TARGET_ROOT" status --short)"
if [[ -n "$target_dirty" ]]; then
  echo "ERROR: WSL Codex repo has uncommitted changes and cannot be auto-synced safely." >&2
  echo "Clean or archive the WSL Codex repo before launching from it again." >&2
  git -C "$TARGET_ROOT" status --short >&2
  exit 1
fi

if [[ "$source_branch" != "$target_branch" ]]; then
  echo "ERROR: branch mismatch between source and WSL Codex repo." >&2
  echo "Source: $SOURCE_ROOT [$source_branch]" >&2
  echo "Target: $TARGET_ROOT [$target_branch]" >&2
  exit 1
fi

source_head="$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
target_head="$(git -C "$TARGET_ROOT" rev-parse HEAD)"
sync_message=""

if [[ "$source_head" == "$target_head" ]]; then
  sync_message="WSL Codex repo already current at ${source_branch} @ ${source_head:0:12}"
else
  git -C "$TARGET_ROOT" fetch --quiet "$SOURCE_ROOT" "$source_branch"
  fetch_head="$(git -C "$TARGET_ROOT" rev-parse FETCH_HEAD)"

  if [[ "$fetch_head" != "$source_head" ]]; then
    echo "ERROR: fetched WSL state does not match source HEAD." >&2
    echo "Source: $source_head" >&2
    echo "Fetched: $fetch_head" >&2
    exit 1
  fi

  if ! git -C "$TARGET_ROOT" merge --ff-only --quiet FETCH_HEAD; then
    echo "ERROR: WSL Codex repo cannot fast-forward to the current checkout." >&2
    echo "Source: $SOURCE_ROOT [$source_branch @ ${source_head:0:12}]" >&2
    echo "Target: $TARGET_ROOT [$target_branch @ ${target_head:0:12}]" >&2
    exit 1
  fi

  sync_message="Synced WSL Codex repo to ${source_branch} @ ${source_head:0:12}"
fi

preflight_py="$TARGET_ROOT/.venv-wsl/bin/python"
if [[ ! -x "$preflight_py" ]]; then
  preflight_py="python3"
fi

"$preflight_py" "$TARGET_ROOT/scripts/tools/session_preflight.py" \
  --quiet \
  --root "$TARGET_ROOT" \
  --related-root "$SOURCE_ROOT" \
  --context codex-wsl \
  --claim codex \
  --mode mutating

echo "$sync_message"
