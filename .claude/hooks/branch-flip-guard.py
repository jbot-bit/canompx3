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

Helpers (`_git_dir`, `_current_branch`, `_branch_at_start`) live in
`_branch_state.py` so this hook and `mcp-git-guard.py` share one canonical
implementation per `.claude/rules/institutional-rigor.md` rule 4.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Import shared helpers via spec_from_file_location because the dotted hooks
# directory is not a package (`.claude/hooks/` is a flat script tree, not
# importable as `claude.hooks`). spec_from_file_location is the
# stdlib-sanctioned way to import a sibling .py without packaging it.
import importlib.util as _importlib_util

_HOOKS_DIR = Path(__file__).resolve().parent
_SPEC = _importlib_util.spec_from_file_location(
    "_branch_state", _HOOKS_DIR / "_branch_state.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_branch_state = _importlib_util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_branch_state)


# Test-friendly re-exports: tests/test_hooks/test_branch_flip_guard.py
# monkeypatches `hook._git_dir` and `hook._current_branch`. Keeping these
# module-level names preserves the test contract while delegating the
# implementation to the shared canonical source.
_git_dir = _branch_state.git_dir
_current_branch = _branch_state.current_branch


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # fail-safe: malformed event from harness -> pass

    tool_name = event.get("tool_name", "")
    if tool_name != "Bash":
        sys.exit(0)  # fail-safe: matcher misfire -> pass

    # Only inspect commands that touch git branch state
    command = event.get("tool_input", {}).get("command", "")
    branch_ops = ("checkout", "switch", "worktree")
    if not any(op in command for op in branch_ops):
        sys.exit(0)

    git_dir = _git_dir()
    if git_dir is None:
        sys.exit(0)  # fail-safe: not in a git repo -> pass

    lock_path = git_dir / ".claude.pid"
    if not lock_path.exists():
        sys.exit(0)  # fail-safe: no session lock -> pass

    branch_at_start = _branch_state.branch_at_start(git_dir)
    if not branch_at_start:
        sys.exit(0)  # fail-safe: corrupted/empty lock -> pass

    current = _current_branch()
    if current is None:
        sys.exit(0)  # fail-safe: detached HEAD / git failure -> pass

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
