#!/usr/bin/env python3
"""Shared branch-state helpers for branch-flip protection hooks.

Canonical source for the three primitives both `branch-flip-guard.py`
(PostToolUse/Bash) and `mcp-git-guard.py` (PostToolUse/mcp__git__.*) need:

  - `git_dir()`           : resolve the per-worktree `.git` directory.
  - `current_branch()`    : `git branch --show-current`.
  - `branch_at_start(d)`  : read the `branch_at_start` field written by
                            `session-start.py` into `<git_dir>/.claude.pid`.

Per `.claude/rules/institutional-rigor.md` rule 4 ("Delegate to canonical
sources — never re-encode"), both guards MUST import these helpers rather
than maintain parallel copies. The guards previously diverged-by-design on
this point (only one existed); adding `mcp-git-guard.py` as a sibling
without a shared module would create exactly the silent-divergence class
that rule 4 forbids.

Each helper is fail-safe: any error path returns `None`. The branch-flip
protection layer must never block a session it can't read (see
`.claude/rules/branch-flip-protection.md` § "Fail-safe guarantee").
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def git_dir() -> Path | None:
    """Return the per-worktree `.git` directory, or None outside a repo.

    Worktrees resolve to `<repo>/.git/worktrees/<name>/`, NOT the shared
    common dir. This is intentional: `.claude.pid` is per-worktree state
    written by `session-start.py` into exactly this directory.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        d = Path(result.stdout.strip())
        return d if d.is_absolute() else Path.cwd() / d
    except Exception:
        # fail-safe: any subprocess / environment failure -> None;
        # caller must treat as "not in a git repo" and exit 0.
        return None


def current_branch() -> str | None:
    """Return the current branch name, or None on detached HEAD / failure."""
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None
    except Exception:
        # fail-safe: subprocess failure -> None; caller exits 0.
        return None


def branch_at_start(git_dir_path: Path) -> str | None:
    """Read `branch_at_start` from `<git_dir>/.claude.pid`.

    Returns None if the lock file is missing, unreadable, not JSON, or
    has no `branch_at_start` field. The file is written by
    `session-start.py` `_session_lock_lines()` on every fresh session.
    """
    lock = git_dir_path / ".claude.pid"
    if not lock.exists():
        return None
    try:
        data = json.loads(lock.read_text(encoding="utf-8"))
    except Exception:
        # fail-safe: corrupted JSON -> None; caller exits 0.
        return None
    value = data.get("branch_at_start", "")
    return value or None
