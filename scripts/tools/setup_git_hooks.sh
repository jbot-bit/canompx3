#!/usr/bin/env bash
#
# One-shot: configure this clone to use the repo's .githooks/ directory.
#
# Must be run once per clone/worktree. Without it, the pre-commit / commit-msg
# / pre-push hooks are silently inactive, which is exactly how the 2026-04-20
# codex/live-book-reaudit CRLF + empty-body + stale-base incident happened:
# the hooks existed but were never wired up on that machine.
#
# Idempotent. Safe to re-run.
#

set -e

cd "$(dirname "$0")/../.."

CURRENT=$(git config --get core.hooksPath 2>/dev/null || echo "<unset>")

if [ "$CURRENT" = ".githooks" ]; then
    echo "OK: core.hooksPath is already .githooks"
else
    echo "Setting core.hooksPath = .githooks (was: $CURRENT)"
    git config core.hooksPath .githooks
fi

# Make all hook files executable (needed on fresh clones from some platforms)
for hook in .githooks/pre-commit .githooks/commit-msg .githooks/pre-push .githooks/post-commit; do
    if [ -f "$hook" ] && [ ! -x "$hook" ]; then
        chmod +x "$hook"
        echo "chmod +x $hook"
    fi
done

# Verify python is resolvable for hooks that depend on it
PYTHON="python"
if [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON=".venv/Scripts/python.exe"
elif [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
fi

if ! "$PYTHON" --version >/dev/null 2>&1; then
    echo "WARNING: python not resolvable — hooks that run python will fail"
    exit 2
fi

echo "Hooks are active. Verify with:"
echo "  git config --get core.hooksPath    # should print: .githooks"
echo ""
echo "See docs/governance/agent_handoff_protocol.md for what each hook enforces."
