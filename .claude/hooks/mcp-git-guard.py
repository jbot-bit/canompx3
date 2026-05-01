#!/usr/bin/env python3
"""MCP git guard: PostToolUse(mcp__git__.*) — branch-flip protection for MCP git calls.

Mirrors `branch-flip-guard.py` for the MCP-server-git path. Two layers of the
existing protection stack are bypassed by `mcp-server-git`:

  1. `branch-flip-guard.py` PostToolUse(Bash) — bypassed because the matcher
     is "Bash" and the script early-exits on `tool_name != "Bash"`.
  2. `.githooks/pre-commit` step 0c — bypassed because upstream
     `mcp-server-git/server.py` `git_commit()` uses `repo.index.commit()`
     (GitPython direct index API) instead of shelling out to `git commit`,
     so the pre-commit hook never fires.

This guard closes both gaps for MCP git tool calls. See the eval verdict at
`docs/external/git-mcp/eval-2026-05-01/safety-precheck.md` for the full
analysis.

Behavior:

  - Read-only MCP git tools on a flipped branch -> exit 0 (no harm; matches
    the existing guard's intent of only firing on branch-mutating ops).
  - Write MCP git tools on a flipped branch -> exit 2 with BLOCK message.
    The write list is the explicit state-mutating subset enumerated in Q3
    of the safety pre-check.

Fail-safe: any read error / missing lock / non-git context / unexpected
tool name -> exit 0. The guard must never block a session it can't read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Shared canonical helpers — see _branch_state.py for the rationale
# (institutional-rigor.md rule 4: delegate, never re-encode).
import importlib.util as _importlib_util

_HOOKS_DIR = Path(__file__).resolve().parent
_SPEC = _importlib_util.spec_from_file_location(
    "_branch_state", _HOOKS_DIR / "_branch_state.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_branch_state = _importlib_util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_branch_state)


# Tool-name classification. Sourced from upstream
# `mcp-server-git/src/mcp_server_git/server.py` (verified 2026-05-01 in
# safety-precheck.md Q2). If upstream adds new tools, the safe default is
# "unknown -> read-only treatment" via the matcher prefix check below; the
# explicit WRITE_TOOLS set is conservative and defends the documented
# state-mutating subset.

WRITE_TOOLS = frozenset(
    {
        "mcp__git__git_commit",
        "mcp__git__git_checkout",
        "mcp__git__git_create_branch",
        "mcp__git__git_reset",
        "mcp__git__git_add",
        "mcp__git__git_branch",
    }
)

# Read-only tools listed for documentation only — they take no action when
# branches drift, so they don't need to be enumerated in code. The default
# branch (any non-WRITE mcp__git__ tool) exits 0.


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # fail-safe: malformed event from harness -> pass

    tool_name = event.get("tool_name", "")

    # Self-check: settings.json matcher already filters to mcp__git__.* but
    # defend against misconfiguration and against direct CLI invocation.
    if not tool_name.startswith("mcp__git__"):
        sys.exit(0)  # fail-safe: matcher misfire / wrong tool family -> pass

    # Read-only tools: no harm done by reads on the wrong branch. This
    # matches the existing Bash guard's intent (it only fires on commands
    # containing checkout/switch/worktree).
    if tool_name not in WRITE_TOOLS:
        sys.exit(0)

    git_dir = _branch_state.git_dir()
    if git_dir is None:
        sys.exit(0)  # fail-safe: not in a git repo -> pass

    lock_path = git_dir / ".claude.pid"
    if not lock_path.exists():
        sys.exit(0)  # fail-safe: no session lock -> pass

    branch_at_start = _branch_state.branch_at_start(git_dir)
    if not branch_at_start:
        sys.exit(0)  # fail-safe: corrupted/empty lock -> pass

    current = _branch_state.current_branch()
    if current is None:
        sys.exit(0)  # fail-safe: detached HEAD / git failure -> pass

    if current == branch_at_start:
        sys.exit(0)

    # Branch has flipped mid-session AND tool is a state-mutating MCP git
    # call — BLOCK. For mcp__git__git_commit specifically, this hook is the
    # ONLY layer of protection: the pre-commit backstop is bypassed because
    # `repo.index.commit()` does not invoke client-side hooks.
    print("", file=sys.stderr)
    print(
        "  ====================================================================",
        file=sys.stderr,
    )
    print(
        f"  BLOCKED: MCP git tool `{tool_name}` on a flipped branch.",
        file=sys.stderr,
    )
    print(
        "  --------------------------------------------------------------------",
        file=sys.stderr,
    )
    print(f"  Session started on: {branch_at_start}", file=sys.stderr)
    print(f"  Current branch:     {current}", file=sys.stderr)
    print(
        "  --------------------------------------------------------------------",
        file=sys.stderr,
    )
    print(
        "  MCP git commits/branch-mutations corrupt history when the branch",
        file=sys.stderr,
    )
    print(
        "  has changed since session start. Note: pre-commit hook does NOT",
        file=sys.stderr,
    )
    print(
        "  fire for `mcp__git__git_commit` (GitPython index API), so this",
        file=sys.stderr,
    )
    print("  hook is the only safety net. Options:", file=sys.stderr)
    print(
        f"    1. Switch back:   mcp__git__git_checkout {branch_at_start}",
        file=sys.stderr,
    )
    print(
        f"                  or: git checkout {branch_at_start}",
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
        "       docs/external/git-mcp/eval-2026-05-01/safety-precheck.md",
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
