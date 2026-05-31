#!/usr/bin/env python3
"""PostToolUse heartbeat writer — proves THIS session is live, right now.

Why this exists
---------------
The same-worktree concurrency guards in `session-start.py`
(`_session_lock_lines`, `_active_sibling_lines`) only fire at session START and
infer liveness from either (a) a PID — unreliable on Windows, documented in that
module — or (b) a tracked file edited in the last 15 min. Both miss the case the
operator actually hits: two terminals already open in the SAME git tree, where
the second started during a quiet moment and neither was mid-edit. After
startup, nothing re-checks, so the two sessions coexist silently and the
operator redoes work the other terminal already did.

A heartbeat closes that gap with a FACT instead of a guess: every live session
stamps `<git-common-dir>/.claude-heartbeats/<session_id>.beat` on each tool call.
The next session's startup (`_live_heartbeat_lines` in session-start.py) reads
those beats and shouts if a *different* session touched THIS tree seconds ago —
no PID liveness guess required.

Design constraints (grounded, not assumed)
-------------------------------------------
- Fires on PostToolUse, matcher "*". Per the official hooks reference, PostToolUse
  fires after every successful tool call and carries `session_id` + `cwd` on
  stdin. There is NO timer/cron hook event, so PostToolUse is the most frequent
  liveness signal available.
- Beats live in the git COMMON dir (shared across all worktrees of the repo) so
  worktree A and worktree B can see each other. Per-worktree `.git` dirs would
  make them blind to one another.
- One `git rev-parse --git-common-dir` per write. This is a local (no-network)
  call; the repo's existing `branch-flip-guard.py` already runs git on every
  PostToolUse:Bash with timeout=5, so the pattern is proven acceptable here.
- A 20s self-throttle skips the write when our own beat is fresh, so a burst of
  tool calls does not hammer the disk.
- Fail-open on EVERY path: malformed stdin, missing git, IO error -> exit 0,
  no output. A heartbeat writer must never disturb a tool call.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# How recently THIS session must have beaten before we skip a redundant write.
# Bursty tool calls (several per second during active work) would otherwise
# rewrite the file constantly; 20s is well inside the reader's liveness window.
_WRITE_THROTTLE_SECS = 20

# Subdir under the git common dir holding one .beat file per live session.
_BEAT_DIRNAME = ".claude-heartbeats"


def _git_common_dir(cwd: str) -> Path | None:
    """Resolve the repo's shared git dir from ``cwd``. Returns None on any error.

    Uses ``--git-common-dir`` (not ``--git-dir``) so every worktree of the repo
    resolves to the SAME absolute path — the precondition for cross-worktree
    heartbeat visibility.
    """
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=cwd or None,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError, ValueError):
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    p = Path(r.stdout.strip())
    if not p.is_absolute():
        # Relative (".git") means cwd IS the main checkout — anchor to cwd.
        base = Path(cwd) if cwd else Path.cwd()
        p = (base / p).resolve()
    return p


def _current_branch(cwd: str) -> str:
    try:
        r = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=cwd or None,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError, ValueError):
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError, OSError):
        sys.exit(0)

    if not isinstance(event, dict):
        sys.exit(0)

    # session_id: prefer stdin (authoritative per hooks reference), fall back to
    # the env var the harness sets. Without an id we cannot key the beat file
    # uniquely (would collide with peers) — fail-open silent rather than write a
    # mis-keyed beat that produces false alarms.
    session_id = (
        event.get("session_id")
        or os.environ.get("CLAUDE_CODE_SESSION_ID")
        or os.environ.get("CLAUDE_SESSION_ID")
        or ""
    ).strip()
    if not session_id:
        sys.exit(0)

    # Sanitise: a session_id is a UUID, but never trust input into a filename.
    safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")
    if not safe_id:
        sys.exit(0)

    cwd = (event.get("cwd") or os.getcwd() or "").strip()
    # Canonicalise so the stored cwd matches the reader's
    # `git rev-parse --show-toplevel` (long-path, resolved) form. Without this an
    # 8.3 short name or junction `cwd` would compare unequal to the reader's
    # canonical path, making a SAME-tree peer look like a sibling and dropping
    # the loud warning. Best-effort: keep the raw value if resolve fails.
    if cwd:
        try:
            cwd = str(Path(cwd).resolve())
        except (OSError, ValueError, RuntimeError):
            pass

    common_dir = _git_common_dir(cwd)
    if common_dir is None:
        sys.exit(0)  # not a git repo / git unavailable — nothing to coordinate

    beat_dir = common_dir / _BEAT_DIRNAME
    beat_path = beat_dir / f"{safe_id}.beat"

    now = time.time()

    # Self-throttle: skip redundant writes when our own beat is younger than the
    # throttle window. os.stat is far cheaper than git + a file write.
    try:
        if beat_path.exists():
            age = now - beat_path.stat().st_mtime
            if 0 <= age < _WRITE_THROTTLE_SECS:
                sys.exit(0)
    except OSError:
        pass  # stat failure -> fall through and attempt the write

    payload = json.dumps(
        {
            "session_id": session_id,
            "cwd": cwd,
            "branch": _current_branch(cwd),
            "pid": os.getpid(),
            "ts": now,
        }
    ).encode("utf-8")

    try:
        beat_dir.mkdir(parents=True, exist_ok=True)
        # Atomic write: temp + os.replace so a reader never sees a half-written
        # beat (which would parse-fail and be ignored — not fatal, but avoidable).
        tmp = beat_dir / f"{safe_id}.beat.{os.getpid()}.tmp"
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
        os.replace(str(tmp), str(beat_path))
    except OSError:
        sys.exit(0)  # disk full, perms, race — fail-open, never disturb the tool

    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        sys.exit(0)
