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

import importlib.util as _ilu
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

# A lock whose holder PID is PROVEN dead AND whose `iso_started` is at least
# this many hours old is a corpse: the session that wrote it crashed/force-
# exited and never cleaned up. Reading `head_at_start`/`branch_at_start` from a
# corpse made the flip-guards advise on a DEAD session's stale SHA every turn
# (n=1 2026-06-07: a 124.8h corpse held by dead PID 13716 nagged for ~6 days).
#
# The 12h floor MIRRORS `session-start.py:_STALE_LOCK_RECLAIM_HOURS` — the same
# threshold the session-lock reclaim path already trusts. It is doubly load-
# bearing: (1) a crashed-and-restarted shell within 12h KEEPS its flip-guard;
# (2) existing tests write locks with no `iso_started` (age unknown), so the
# floor's "can't prove stale -> treat as live -> return the value" default
# preserves their historical behavior. Age-unknown NEVER silences a guard.
_CORPSE_LOCK_MIN_AGE_HOURS = 12

_GUARD_PATH = Path(__file__).resolve().parents[2] / "scripts" / "tools" / "worktree_guard.py"


def _canonical_pid_is_alive(pid: int) -> bool | None:
    """Delegate liveness to the canonical `worktree_guard._pid_is_alive`.

    institutional-rigor §4 (never re-encode canonical logic): liveness is
    Windows-load-bearing and the project ALREADY solved it. On Windows,
    `os.kill(pid, 0)` does NOT reliably raise for a dead PID (it returns no
    error for a gone-but-recycled PID — the exact false-alive that let the
    n=1 124.8h corpse survive). `worktree_guard` probes via OpenProcess +
    GetExitCodeProcess, the only Windows-correct oracle in this repo (origin:
    reaper corpse-lock fix 3e9aec96). Returns the bool verdict, or None when
    the canonical module can't be loaded so the caller can fall back.
    """
    try:
        spec = _ilu.spec_from_file_location("worktree_guard", str(_GUARD_PATH))
        if spec is None or spec.loader is None:
            return None
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return bool(mod._pid_is_alive(pid))
    except BaseException:
        return None


def _pid_is_alive(pid: int) -> bool:
    """Cross-platform PID liveness — canonical-delegated, conservative fallback.

    Primary path delegates to `worktree_guard._pid_is_alive` (the repo's only
    Windows-correct probe). When that import fails, fall back to a conservative
    local check that returns True unless the PID is PROVEN dead. The fallback's
    asymmetry (false-alive is safe, false-dead is catastrophic) means a fallback
    can only ever KEEP a guard, never silence one on an unproven-dead lock — so
    a degraded environment loses the new corpse-detection but never the old
    flip-protection.
    """
    if not isinstance(pid, int) or pid <= 0:
        return False
    verdict = _canonical_pid_is_alive(pid)
    if verdict is not None:
        return verdict
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by another user — treat as alive
    except (OSError, AttributeError):
        return True  # cannot prove dead -> do not silence the guard


def _lock_is_corpse(data: dict) -> bool:
    """True iff the lock's holder is PROVEN dead AND it is >= 12h old.

    Both conditions are required. A live PID at any age is a real session. A
    dead PID within the freshness window may be a /clear-and-restart of the
    same shell whose guard should still hold. Only a stale corpse — dead AND
    old — is ignored. Returns False (not a corpse) whenever either condition
    cannot be established, so the reader fails toward KEEPING the guard.
    """
    pid = data.get("pid")
    if not isinstance(pid, int) or _pid_is_alive(pid):
        return False  # no PID, or alive -> not a corpse
    iso = data.get("iso_started", "")
    try:
        ts = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except (ValueError, AttributeError, TypeError):
        return False  # age unknown -> can't prove stale -> not a corpse
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    age_hours = (datetime.now(UTC) - ts).total_seconds() / 3600.0
    return age_hours >= _CORPSE_LOCK_MIN_AGE_HOURS


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
    if not isinstance(data, dict) or _lock_is_corpse(data):
        return None  # corpse lock -> treat as absent -> guard fails open
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
    if not isinstance(data, dict) or _lock_is_corpse(data):
        return None  # corpse lock -> treat as absent -> guard fails open
    value = data.get("branch_at_start", "")
    return value or None
