#!/usr/bin/env python3
"""Head-flip guard: PostToolUse(Bash) — surface silent HEAD SHA rewrites.

Companion to `branch-flip-guard.py`. That hook catches branch *name* changes
(`git checkout` / `switch` / `worktree`). This hook catches the harder class:
the branch name stays the same but HEAD SHA changes — `git pull --rebase`,
`git reset --hard`, `git commit --amend`, or a session hook silently
amending. Any commit SHA Claude quoted earlier in the session may now be
unreachable from HEAD (reachable via reflog only, until GC ~90 days).

Triggered by n=1 incident 2026-05-28: a background `git pull --rebase` mid-
session rewrote 4 Lane B commit SHAs after Claude had already captured the
pre-rebase SHAs from session-start `git log`. Claude wrote the dead SHAs
into durable memory before the divergence was caught in code-review.

Shape: ADVISORY, not blocking. Legitimate operator rebase/amend operations
are common; hard-blocking them would self-DOS. Instead this hook emits
`additionalContext` via stdout JSON so Claude sees the rewrite on the next
turn and re-resolves any SHA before writing it durably.

Fail-safe: any read error, missing lock, corrupted JSON, or non-git context
exits 0 silently. The guard must never inject false warnings on a session
it can't read accurately.

See: `.claude/rules/branch-flip-protection.md` (companion rule) +
`memory/feedback_silent_mid_session_pull_rebase_invalidates_sha_quotes_2026_05_28.md`.
"""

from __future__ import annotations

import importlib.util as _importlib_util
import json
import sys
from pathlib import Path

_HOOKS_DIR = Path(__file__).resolve().parent
_SPEC = _importlib_util.spec_from_file_location(
    "_branch_state", _HOOKS_DIR / "_branch_state.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_branch_state = _importlib_util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_branch_state)


_git_dir = _branch_state.git_dir
_current_branch = _branch_state.current_branch
_current_head_sha = _branch_state.current_head_sha
_branch_at_start = _branch_state.branch_at_start
_head_at_start = _branch_state.head_at_start


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    if event.get("tool_name", "") != "Bash":
        sys.exit(0)

    # Scope to the worktree the Bash command ran in, not the hook process's
    # cwd (always the main checkout). Advisory-only, but a cross-worktree
    # mismatch would inject a false "SHA rewritten" warning every turn.
    cwd = _branch_state.invoking_cwd(event)

    git_dir = _git_dir(cwd)
    if git_dir is None:
        sys.exit(0)

    lock_path = git_dir / ".claude.pid"
    if not lock_path.exists():
        sys.exit(0)

    branch_at_start = _branch_at_start(git_dir)
    head_at_start = _head_at_start(git_dir)
    if not branch_at_start or not head_at_start:
        sys.exit(0)

    current_branch = _current_branch(cwd)
    current_head = _current_head_sha(cwd)
    if current_branch is None or current_head is None:
        sys.exit(0)

    # If the branch flipped, `branch-flip-guard.py` already handles it.
    # Don't double-warn — only fire on the branch-stable-but-HEAD-moved case.
    if current_branch != branch_at_start:
        sys.exit(0)

    if current_head == head_at_start:
        sys.exit(0)

    # HEAD moved while branch name held. Could be:
    #   - benign: a normal `git commit` adding work this session.
    #   - dangerous: pull --rebase, reset --hard, commit --amend, hook amend.
    # We can't reliably distinguish without parsing reflog (and the user may
    # have legitimately rebased). So always advise — re-resolve SHAs before
    # quoting them in durable memory/HANDOFF/commit messages.
    message = (
        f"HEAD SHA changed since session start while branch unchanged. "
        f"branch={current_branch} head_at_start={head_at_start[:8]} "
        f"head_now={current_head[:8]}. "
        "If you are about to write a commit SHA into MEMORY.md, HANDOFF.md, "
        "a memory/*.md file, or a commit message, run `git rev-parse <sha>` "
        "AND `git log -1 --oneline <sha>` first to confirm the SHA is still "
        "reachable from HEAD. Pre-rebase SHAs survive in reflog ~90d but are "
        "unreachable from `git log`. "
        "See .claude/rules/branch-flip-protection.md and "
        "feedback_silent_mid_session_pull_rebase_invalidates_sha_quotes_2026_05_28.md."
    )

    response = {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": message,
        }
    }
    print(json.dumps(response))
    sys.exit(0)


if __name__ == "__main__":
    main()
