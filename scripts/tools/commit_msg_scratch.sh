#!/usr/bin/env bash
# Write a commit message (from stdin) into the *worktree's own git dir* and echo
# the path, so a long-running pre-commit hook (~6 min: drift + tests) cannot lose
# the file to Windows /tmp eviction mid-commit.
#
# Why not /tmp: on Windows /tmp is a volatile temp dir the OS may clean during the
# hook window; `git commit -F /tmp/msg` then aborts with exit 128 AFTER all checks
# pass (nothing committed). The git dir is on git's own NTFS volume and is durable
# for the hook's lifetime. See:
#   memory/feedback_commit_msgfile_in_tmp_evicted_during_long_hook_use_gitdir_scratch_2026_06_05.md
#
# Why --git-path: in a worktree, `.git` is a *file* (gitdir pointer), so
# `mkdir .git/...` fails ("Not a directory"). `git rev-parse --git-path` resolves
# the real git dir in both the main checkout and any worktree.
#
# Usage:
#   MSG=$(printf '%s\n' "subject line" "" "body...") ; echo "$MSG" | scripts/tools/commit_msg_scratch.sh
#   git commit -F "$(scripts/tools/commit_msg_scratch.sh < /path/to/draft)"   # --shared-state-ack
#
# Echoes the absolute message-file path on stdout (and nothing else) on success.
set -euo pipefail

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "commit_msg_scratch: not inside a git work tree" >&2
  exit 1
fi

scratch_dir="$(git rev-parse --git-path cc-scratch)"
mkdir -p "$scratch_dir"
msg_path="$scratch_dir/commit_msg.txt"

# Read the full message from stdin (heredoc, pipe, or redirect).
cat > "$msg_path"

if [ ! -s "$msg_path" ]; then
  echo "commit_msg_scratch: refusing to write an empty commit message" >&2
  rm -f "$msg_path"
  exit 1
fi

echo "$msg_path"
