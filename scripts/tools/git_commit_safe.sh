#!/usr/bin/env bash
#
# git_commit_safe.sh — commit with one automatic retry on the concurrent-peer
# ref-lock race.
#
# WHY
# ---
# In a multi-terminal repo, a peer can advance HEAD during this session's
# ~2-minute pre-commit gate. git's final ref write then dies with:
#     fatal: cannot lock ref 'HEAD': is at <peer> but expected <old>
# ...discarding all the pre-commit work. The pre-commit serializer
# (scripts/tools/commit_serialize.py, step 0 of .githooks/pre-commit) prevents
# the common case by stopping two gates running at once. This wrapper is the
# SAFETY NET for the residual race (a commit that bypasses the hook, or a peer
# landing in the narrow ref-write window): on a ref-lock failure it rebases onto
# the peer's new HEAD and retries the commit exactly ONCE.
#
# It retries ONLY on the ref-lock signature — never on a real pre-commit BLOCK
# (lint/drift/test failure), so genuine gate failures still surface immediately.
#
# USAGE
#   scripts/tools/git_commit_safe.sh -m "message"        # any `git commit` args
#   scripts/tools/git_commit_safe.sh -m "msg" --no-verify # args passed through
#
# EXIT CODES
#   0  commit succeeded (first try or after one rebase+retry)
#   N  the underlying `git commit` exit code on a non-ref-lock failure, or on a
#      second consecutive ref-lock failure (structural, not a transient race).
#
set -u

# Signatures that identify the concurrent-peer ref-lock race (git wording across
# versions). Anchored on the specific failure so a normal pre-commit BLOCK
# (which prints "BLOCKED:" from the hook, not these) never triggers a retry.
_REF_LOCK_RE='cannot lock ref|failed to lock|ref updates forbidden|unable to update|cannot update the ref|reference already exists'

_attempt_commit() {
    # Capture combined output so we can scan it, but still show it to the user.
    local out
    out="$(git commit "$@" 2>&1)"
    local rc=$?
    printf '%s\n' "$out"
    # Stash the output for the caller's race-detection via a global.
    LAST_COMMIT_OUTPUT="$out"
    return $rc
}

LAST_COMMIT_OUTPUT=""

# First attempt.
_attempt_commit "$@"
rc=$?
if [ $rc -eq 0 ]; then
    exit 0
fi

# Did it fail on the ref-lock race specifically?
if printf '%s\n' "$LAST_COMMIT_OUTPUT" | grep -qiE "$_REF_LOCK_RE"; then
    echo ""
    echo "[git_commit_safe] ref-lock race detected (a peer advanced HEAD)."
    echo "[git_commit_safe] rebasing onto the new HEAD and retrying once..."
    # Pull the peer's commits and replay our index on top. --autostash protects
    # any unstaged changes; the staged commit content is preserved by git.
    if git pull --rebase --autostash; then
        _attempt_commit "$@"
        rc=$?
        if [ $rc -eq 0 ]; then
            echo "[git_commit_safe] retry succeeded."
            exit 0
        fi
        echo "[git_commit_safe] retry still failed (rc=$rc) — not a transient race; stopping." >&2
        exit $rc
    else
        echo "[git_commit_safe] rebase failed — resolve manually, then re-commit." >&2
        exit $rc
    fi
fi

# Non-ref-lock failure (real pre-commit BLOCK, bad args, etc.) — surface as-is.
exit $rc
