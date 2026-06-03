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

# Per-worktree venv isolation (Stage 1, 2026-06-03).
# Each worktree gets its OWN .venv so a peer's `uv sync` can never strip this
# tree's dev group or contend its DLLs. uv resolves a RELATIVE .venv from the
# workspace root (the worktree), so running `uv sync` from inside $WT_PATH with
# UV_PROJECT_ENVIRONMENT UNSET (or =.venv) creates an isolated environment.
# Never point UV_PROJECT_ENVIRONMENT at an absolute path — the uv docs warn it
# "will be overwritten by each project", which is exactly the shared-venv bug.
# Doctrine: .claude/rules/worktree-venv-isolation.md
VENV_OK=0
if command -v uv >/dev/null 2>&1; then
    echo "Bootstrapping isolated venv in $WT_PATH/.venv ..."
    # Subshell: do not leak any inherited UV_PROJECT_ENVIRONMENT into this sync.
    if ( cd "$WT_PATH" && unset UV_PROJECT_ENVIRONMENT && \
         uv sync --locked --group dev ); then
        VENV_OK=1
    else
        echo "WARNING: isolated venv bootstrap failed — worktree will fall back" >&2
        echo "         to the shared canonical venv (pre-commit probe handles this)." >&2
    fi
else
    echo "WARNING: 'uv' not on PATH — skipping isolated venv bootstrap." >&2
fi

cat <<EOM

=========================================
Worktree spawned: $WT_PATH
Branch:           $BRANCH (from origin/main)
Isolated venv:    $([ "$VENV_OK" -eq 1 ] && echo "$WT_PATH/.venv (created)" || echo "FAILED — shares canonical venv (see warning above)")
=========================================

Open a NEW terminal there:

    cd "$WT_PATH"

This session stays untouched — no edit collisions, no CRLF noise, no lost stashes.
Its venv is isolated — a peer's \`uv sync\` cannot strip this tree's dev deps.
EOM
