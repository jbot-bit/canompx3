#!/usr/bin/env python3
"""Worktree-guard PreToolUse hook: BLOCK Edit/Write/MultiEdit/Bash when another
live Claude session holds the worktree lease.

## Hook contract (Anthropic Claude Code official)

  - PreToolUse event payload arrives on stdin as JSON.
  - exit 0 -> allow the tool call
  - exit 2 -> BLOCK; stderr is shown to the user verbatim
  - exit anything else -> error, treated as allow

Source: Anthropic Claude Code documentation, "Hooks" section, PreToolUse +
exit-code semantics. Mirrors the structural template of every sibling hook
(`stage-gate-guard.py`, `pre-edit-guard.py`, `judgment-review-soft-block.py`).

## Decision logic

  1. If `acquire_or_refresh()` returns "acquired" / "refreshed" / "reclaimed"
     -> ALLOW. This process holds the lease and the sidecar is fresh.
  2. If it returns "blocked" -> a live peer holds the OS-level FileLock for
     this exact worktree. Print structured BLOCK message and exit 2.
  3. If "skipped" (not in a git repo) or "error" (FS transient) -> ALLOW.
     Fail-open per `.claude/rules/branch-flip-protection.md` § "Fail-safe
     guarantee" — a guard that can't reason about state must not block.

## Auto-clear of stale leases

Handled inside `scripts/tools/worktree_guard.py:acquire()` — when the OS lock
is free (i.e. the prior holder process exited), this hook simply acquires it
fresh. When the OS lock is HELD but the sidecar heartbeat is older than
STALE_HEARTBEAT_SECONDS (30 min), the OS lock is the source of truth — we
BLOCK regardless, because some process IS still holding the file. The
operator's recovery path in that case is `--force-release`.

## Canonical-source delegation (institutional-rigor §4)

ALL state I/O delegates to `scripts/tools/worktree_guard.py`. This hook only
shapes the stderr message and selects the exit code. Drift parity check
`check_worktree_guard_lease_path_parity` guards against future inlining.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_MODULE = PROJECT_ROOT / "scripts" / "tools" / "worktree_guard.py"


def _load_canonical():
    """Import `scripts/tools/worktree_guard.py` by absolute path.

    Direct `from scripts.tools.worktree_guard import ...` would work but the
    sibling hooks (`judgment-review-soft-block.py`, `pre-edit-guard.py`) use
    the same importlib pattern so we stay consistent. Fail-open on any error.
    """
    spec = importlib.util.spec_from_file_location("worktree_guard", CANONICAL_MODULE)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except BaseException:
        return None
    return module


def _matched_tool(event: dict) -> bool:
    """The settings.json matcher already filters by tool name; this is belt-
    and-braces against future matcher misconfig."""
    tool = event.get("tool_name") or event.get("tool", "")
    return tool in {"Edit", "Write", "MultiEdit", "Bash"}


def _emit_block(peer_lease: dict | None, lease_module) -> None:
    """Structured stderr BLOCK message (mirrors the template used by
    `shared-state-commit-guard.py`)."""
    pid = (peer_lease or {}).get("pid", "?")
    worktree = (peer_lease or {}).get("worktree", "?")
    branch = (peer_lease or {}).get("branch", "?")
    started = (peer_lease or {}).get("iso_started", "?")
    heartbeat = (peer_lease or {}).get("iso_heartbeat", "?")
    lp = lease_module.lease_path(Path.cwd())
    lk = lease_module.lock_path(Path.cwd())

    lines = [
        "",
        "  ====================================================================",
        "  BLOCKED: Another Claude session holds the worktree concurrency lease.",
        "  --------------------------------------------------------------------",
        f"  Peer PID:     {pid}",
        f"  Worktree:     {worktree}",
        f"  Branch:       {branch}",
        f"  Started:      {started}",
        f"  Heartbeat:    {heartbeat}",
        f"  Lock file:    {lk}",
        f"  Lease sidecar:{lp}",
        "  --------------------------------------------------------------------",
        "  Two Claudes in one worktree corrupt .git/index and lose work. See:",
        "  .claude/rules/parallel-session-isolation.md",
        "",
        "  Resolutions (pick one):",
        "    1. Switch to the other Claude session and continue there.",
        "    2. Spawn a fresh worktree for parallel work:",
        "         scripts/tools/new_session.sh",
        "    3. If the peer session is confirmed gone (process exited, not just",
        "       quiet), force-release the lease and retry:",
        "         python scripts/tools/worktree_guard.py --force-release",
        "  ====================================================================",
        "",
    ]
    print("\n".join(lines), file=sys.stderr)


def main() -> int:
    # Read the hook event (fail-open on malformed JSON — never block what we
    # can't parse).
    try:
        event = json.load(sys.stdin) if not sys.stdin.isatty() else {}
    except (json.JSONDecodeError, ValueError):
        return 0

    if event and not _matched_tool(event):
        return 0  # matcher misconfig safety

    lease_module = _load_canonical()
    if lease_module is None:
        return 0  # canonical module unavailable -> fail-open

    # Bypass entirely when the env-var test seam is engaged (so the hook's
    # own pytest suite can run without acquiring a real OS lock on the dev
    # machine).
    if os.environ.get("WORKTREE_GUARD_BYPASS") == "1":
        return 0

    try:
        status_str, payload, _ = lease_module.acquire(Path.cwd())
    except BaseException:
        # Any unexpected failure in the canonical module -> fail-open. This
        # mirrors the fail-open contract in branch-flip-protection.md.
        return 0

    if status_str == "blocked":
        _emit_block(payload, lease_module)
        return 2

    # acquired / refreshed / skipped / error -> allow
    return 0


if __name__ == "__main__":
    sys.exit(main())
