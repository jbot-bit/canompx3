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
  echo "WARNING: source repo has uncommitted changes; WSL Codex will use committed HEAD only." >&2
  echo "Uncommitted Windows-checkout changes are not mirrored into the WSL-home clone:" >&2
  git -C "$SOURCE_ROOT" status --short >&2
fi

target_dirty="$(git -C "$TARGET_ROOT" status --short)"
if [[ -n "$target_dirty" ]]; then
  echo "ERROR: WSL Codex repo has uncommitted changes and cannot be auto-synced safely." >&2
  echo "MEASURED: dirty WSL Codex clone at $TARGET_ROOT" >&2
  echo "This is a fail-closed guard, not a Codex install failure." >&2
  echo "" >&2
  echo "Why this guard exists:" >&2
  echo "  Microsoft WSL and OpenAI Codex both recommend keeping Linux-tool repos under /home." >&2
  echo "  The Windows desktop launcher is only the front door; Codex work runs in the WSL-home clone." >&2
  echo "" >&2
  echo "Manual remedy:" >&2
  echo "  1. In WSL: cd ~/canompx3" >&2
  echo "  2. Inspect: git status --short --branch" >&2
  echo "  3. Commit, stash, move to a named worktree, or otherwise preserve the changes." >&2
  echo "  4. Retry: codex.bat" >&2
  echo "" >&2
  echo "For parallel mutable work, prefer: codex.bat task <name>" >&2
  echo "" >&2
  echo "Dirty files:" >&2
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

repo_knows_commit() {
  local repo_root="$1"
  local commit="$2"
  git -C "$repo_root" cat-file -e "${commit}^{commit}" >/dev/null 2>&1
}

if [[ "$source_head" == "$target_head" ]]; then
  sync_message="WSL Codex repo already current at ${source_branch} @ ${source_head:0:12}"
else
  comparison_repo=""
  if repo_knows_commit "$TARGET_ROOT" "$source_head" && repo_knows_commit "$TARGET_ROOT" "$target_head"; then
    comparison_repo="$TARGET_ROOT"
  elif repo_knows_commit "$SOURCE_ROOT" "$source_head" && repo_knows_commit "$SOURCE_ROOT" "$target_head"; then
    comparison_repo="$SOURCE_ROOT"
  else
    echo "ERROR: source and target HEADs differ, but local history is insufficient to compare them safely." >&2
    echo "Source: $SOURCE_ROOT [$source_branch @ ${source_head:0:12}]" >&2
    echo "Target: $TARGET_ROOT [$target_branch @ ${target_head:0:12}]" >&2
    echo "Sync the repos manually, then retry `codex.bat`." >&2
    exit 1
  fi

  if git -C "$comparison_repo" merge-base --is-ancestor "$target_head" "$source_head"; then
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
  elif git -C "$comparison_repo" merge-base --is-ancestor "$source_head" "$target_head"; then
    echo "ERROR: source repo is behind the WSL Codex repo on the same branch." >&2
    echo "Source: $SOURCE_ROOT [$source_branch @ ${source_head:0:12}]" >&2
    echo "Target: $TARGET_ROOT [$target_branch @ ${target_head:0:12}]" >&2
    echo "Update the source checkout before launching Codex so the smart path does not reopen stale code." >&2
    exit 1
  else
    echo "ERROR: source repo and WSL Codex repo diverged on the same branch." >&2
    echo "Source: $SOURCE_ROOT [$source_branch @ ${source_head:0:12}]" >&2
    echo "Target: $TARGET_ROOT [$target_branch @ ${target_head:0:12}]" >&2
    echo "Reconcile the two repos manually, then retry `codex.bat`." >&2
    exit 1
  fi
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
