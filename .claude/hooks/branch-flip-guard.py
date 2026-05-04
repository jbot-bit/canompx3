#!/usr/bin/env python3
"""Branch-flip guard: PostToolUse(Bash) — warn if current branch drifted from session start.

Reads the session lock file (.git/.claude.pid) written by session-start.py to
retrieve branch_at_start, then compares it to the current git branch. If they
differ, prints a hard BLOCK message via stderr and exits 2.

Fail-safe design: any read error, missing lock, corrupted JSON, or non-git
context exits 0 (pass). The guard must never block a session it can't read.

Why this exists: a mid-session `git checkout <other-branch>` followed by file
edits deposits commits on the wrong branch. The pre-commit hook has the same
check, but catching it early (after every Bash call) gives the user a chance
to course-correct before work accumulates.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _git_dir() -> Path | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        git_dir = Path(result.stdout.strip())
        return git_dir if git_dir.is_absolute() else Path.cwd() / git_dir
    except Exception:
        return None


def _current_branch() -> str | None:
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
        return None


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    tool_name = event.get("tool_name", "")
    if tool_name != "Bash":
        sys.exit(0)

    # Only inspect commands that touch git branch state
    command = event.get("tool_input", {}).get("command", "")
    branch_ops = ("checkout", "switch", "worktree")
    if not any(op in command for op in branch_ops):
        sys.exit(0)

    git_dir = _git_dir()
    if git_dir is None:
        sys.exit(0)

    lock_path = git_dir / ".claude.pid"
    if not lock_path.exists():
        sys.exit(0)

    try:
        lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
        branch_at_start = lock_data.get("branch_at_start", "")
    except Exception:
        sys.exit(0)

    if not branch_at_start:
        sys.exit(0)

    current = _current_branch()
    if current is None:
        sys.exit(0)

    if current == branch_at_start:
        sys.exit(0)

    # Branch has flipped mid-session — BLOCK
    print(
        "",
        file=sys.stderr,
    )
    print(
        "  ====================================================================",
        file=sys.stderr,
    )
    print(
        "  BLOCKED: Branch changed mid-session.",
        file=sys.stderr,
    )
    print(
        "  --------------------------------------------------------------------",
        file=sys.stderr,
    )
    print(
        f"  Session started on: {branch_at_start}",
        file=sys.stderr,
    )
    print(
        f"  Current branch:     {current}",
        file=sys.stderr,
    )
    print(
        "  --------------------------------------------------------------------",
        file=sys.stderr,
    )
    print(
        "  Edits on the wrong branch corrupt history. Options:",
        file=sys.stderr,
    )
    print(
        f"    1. Switch back:   git checkout {branch_at_start}",
        file=sys.stderr,
    )
    print(
        f"    2. New worktree:  scripts/tools/new_session.sh",
        file=sys.stderr,
    )
    print(
        f"    3. Accept flip:   rm '{lock_path}'  then restart session",
        file=sys.stderr,
    )
    print(
        "  See: .claude/rules/branch-flip-protection.md",
        file=sys.stderr,
    )
    print(
        "  ====================================================================",
        file=sys.stderr,
    )
    print("", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
