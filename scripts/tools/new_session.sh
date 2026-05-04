#!/usr/bin/env bash
# Auto-isolate a new working session with a git worktree off origin/main.
# Solves the recurring "open 2 terminals, start working, get CRLF noise + lost
# stashes + merge conflicts" problem documented in feedback_parallel_session_awareness.md.
#
# Usage: scripts/tools/new_session.sh [<descriptor>]
#   descriptor: optional short name for branch/dir (e.g. "hwm-fix")
#               default: timestamp.
#
# Each Claude Code session should work in its OWN worktree. This is the
# zero-memory escape — no need to remember `git worktree add` invocations.

set -e

DESCRIPTOR="${1:-$(date +%Y%m%d-%H%M%S)}"
USER_PART="${USER:-$(whoami | tr -d '\r')}"
BRANCH="session/${USER_PART}-${DESCRIPTOR}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
REPO_PARENT="$(dirname "$REPO_ROOT")"
REPO_NAME="$(basename "$REPO_ROOT")"
WT_PATH="${REPO_PARENT}/${REPO_NAME}-${DESCRIPTOR}"

git fetch origin --quiet
git worktree add -b "$BRANCH" "$WT_PATH" origin/main

cat <<EOM

=========================================
Worktree spawned: $WT_PATH
Branch:           $BRANCH (from origin/main)
=========================================

Open a NEW terminal there:

    cd "$WT_PATH"

This session stays untouched — no edit collisions, no CRLF noise, no lost stashes.
EOM
