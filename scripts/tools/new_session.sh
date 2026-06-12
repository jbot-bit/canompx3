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

# Strip trailing CR: on Windows `git rev-parse` stdout carries \r, which leaks
# into WT_PATH as a literal `?` and makes `git worktree add` fail with
# "could not create leading directories … Invalid argument" (n=1 2026-06-12).
# whoami above is already stripped; this is the matching fix for git output.
REPO_ROOT="$(git rev-parse --show-toplevel | tr -d '\r')"
REPO_PARENT="$(dirname "$REPO_ROOT")"
REPO_NAME="$(basename "$REPO_ROOT")"
WT_PATH="${REPO_PARENT}/${REPO_NAME}-${DESCRIPTOR}"

# Auto-heal a phantom-cwd husk at this path: a force-removed worktree leaves the
# dir behind (no .git, not in `git worktree list`). reconcile-launch-path rmtrees
# a provably scratch-only husk (CLEANED) or re-paths one that may hold uncommitted
# work (REPATHED) — so we never launch into a dead folder. set -e is active, so
# capture without aborting on a benign non-zero, then re-read FINALPATH/ACTION.
RECONCILE_OUT="$(python "$(dirname "$0")/worktree_manager.py" reconcile-launch-path --path "$WT_PATH" 2>/dev/null || true)"
while IFS='=' read -r _key _val; do
    case "$_key" in
        FINALPATH) [ -n "$_val" ] && WT_PATH="$_val" ;;
        ACTION) echo "Reconcile: $_val" ;;
    esac
done <<< "$RECONCILE_OUT"

# Auto-sweep OTHER stale husks while we're here. reconcile-launch-path above only
# heals THIS launch path; husks left at other paths by past `worktree remove
# --force` accumulate as parent-dir noise. reap-graveyards rmtrees only provably
# scratch-only husks (registered worktrees + any husk holding real source are
# skipped, so uncommitted work is never at risk). Cheap (one git call). Worktree
# create is the right seam to clean — no per-session-start cost. Best-effort.
python "$(dirname "$0")/worktree_manager.py" reap-graveyards --execute 2>/dev/null || true

git fetch origin --quiet
# Branch may already exist (e.g. the dead worktree's branch). Retry attaching to
# it WITHOUT -b, mirroring create_worktree:381-385. Guard with || so the first
# attempt's non-zero does not abort under `set -e`.
if ! git worktree add -b "$BRANCH" "$WT_PATH" origin/main 2>/dev/null; then
    echo "WARNING: branch '$BRANCH' already exists — reattaching worktree to it (not a fresh origin/main checkout)" >&2
    git worktree add "$WT_PATH" "$BRANCH"
fi

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
