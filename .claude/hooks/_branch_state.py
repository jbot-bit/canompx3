#!/usr/bin/env python3
"""Shared branch-state helpers for branch-flip protection hooks.

Canonical source for the primitives the three branch-protection guards
(`branch-flip-guard.py` PostToolUse/Bash, `mcp-git-guard.py`
PostToolUse/mcp__git__.*, `head-flip-guard.py` PostToolUse/Bash) need:

  - `invoking_cwd(event)` : resolve the worktree a tool call ran in from the
                            PostToolUse payload's `cwd` field.
  - `git_dir(cwd)`        : resolve the per-worktree `.git` directory.
  - `current_branch(cwd)` : `git branch --show-current`.
  - `current_head_sha(cwd)`: `git rev-parse HEAD`.
  - `branch_at_start(d)`  : read the `branch_at_start` field written by
                            `session-start.py` into `<git_dir>/.claude.pid`.
  - `head_at_start(d)`    : read the `head_at_start` field from the same lock.

Per `.claude/rules/institutional-rigor.md` rule 4 ("Delegate to canonical
sources — never re-encode"), all three guards MUST import these helpers
rather than maintain parallel copies. Adding a guard as a sibling without a
shared module would create exactly the silent-divergence class rule 4
forbids — and the F4-A cwd-scoping fix lands once here for all three.

Each helper is fail-safe: any error path returns `None`. The branch-flip
protection layer must never block a session it can't read (see
`.claude/rules/branch-flip-protection.md` § "Fail-safe guarantee").
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def invoking_cwd(event: dict) -> str | None:
    """Resolve the directory a PostToolUse tool call actually ran in.

    The branch-flip guards are registered in `settings.json` with a command
    hardcoded to the main checkout, so the hook *process* cwd is always the
    main checkout regardless of which worktree the tool ran in. The harness
    payload carries the real invoking directory in `event["cwd"]` (the same
    field `session-heartbeat.py` reads); use it to scope the guard to the
    correct worktree. Canonicalised via `Path.resolve()` so an 8.3 short name
    or junction compares equal to the worktree's own path.

    Returns None when no usable cwd is present, so callers fall back to the
    helpers' `cwd=None` (hook-process cwd) default — the historical behavior.
    """
    raw = (event.get("cwd") or "").strip()
    if not raw:
        return None
    try:
        return str(Path(raw).resolve())
    except Exception:
        # fail-safe: an unresolvable path -> None -> caller uses default cwd.
        return None


def git_dir(cwd: str | Path | None = None) -> Path | None:
    """Return the per-worktree `.git` directory, or None outside a repo.

    Worktrees resolve to `<repo>/.git/worktrees/<name>/`, NOT the shared
    common dir. This is intentional: `.claude.pid` is per-worktree state
    written by `session-start.py` into exactly this directory.

    `cwd` selects WHICH worktree to resolve. The branch-flip guards run from
    a `settings.json` command hardcoded to the main checkout, so the hook
    process's own cwd is always the main checkout — blind to the worktree the
    tool actually ran in. Passing the PostToolUse payload's `cwd` (the real
    invoking directory, the same signal `session-heartbeat.py` reads) makes
    the guard inspect the correct worktree. Default `None` = hook-process cwd
    = the historical behavior, preserved for callers that don't pass it.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(cwd) if cwd else None,
        )
        if result.returncode != 0:
            return None
        d = Path(result.stdout.strip())
        if d.is_absolute():
            return d
        # A relative result (".git" / "worktrees/<name>") is relative to the
        # directory `git` ran in — anchor it to that same `cwd`, NOT the hook
        # process's cwd, else a relative dir from another worktree resolves
        # against the wrong base.
        base = Path(cwd) if cwd else Path.cwd()
        return base / d
    except Exception:
        # fail-safe: any subprocess / environment failure -> None;
        # caller must treat as "not in a git repo" and exit 0.
        return None


def current_branch(cwd: str | Path | None = None) -> str | None:
    """Return the current branch name, or None on detached HEAD / failure.

    `cwd` selects the worktree to inspect — see `git_dir` for why the guards
    must pass the invoking worktree's path. Default `None` = hook-process cwd.
    """
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(cwd) if cwd else None,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None
    except Exception:
        # fail-safe: subprocess failure -> None; caller exits 0.
        return None


def current_head_sha(cwd: str | Path | None = None) -> str | None:
    """Return the current HEAD SHA, or None on failure.

    Used by `head-flip-guard.py` to detect silent ref rewrites (pull --rebase,
    reset --hard, commit --amend by a session hook) that preserve the branch
    name but invalidate any commit SHA Claude quoted earlier in the session.

    `cwd` selects the worktree to inspect — see `git_dir`. Default `None` =
    hook-process cwd.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(cwd) if cwd else None,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None
    except Exception:
        return None


def head_at_start(git_dir_path: Path) -> str | None:
    """Read `head_at_start` from `<git_dir>/.claude.pid`.

    Companion to `branch_at_start`: the same lock file already carries the
    head SHA at session start (written by `session-start.py:485`). Returns
    None on any read/parse failure so callers can fail-open.
    """
    lock = git_dir_path / ".claude.pid"
    if not lock.exists():
        return None
    try:
        data = json.loads(lock.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = data.get("head_at_start", "")
    return value or None


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
