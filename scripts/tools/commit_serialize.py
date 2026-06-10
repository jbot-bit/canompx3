#!/usr/bin/env python3
"""Serialize concurrent pre-commit runs in one repo (cross-platform, no `flock`).

Why this exists
---------------
The pre-commit hook (`.githooks/pre-commit`) takes ~2 minutes (CRG update +
~172 drift checks + behavioral audit + staged tests). In a multi-terminal repo,
two terminals can run that 2-minute gate at the same time. Whichever finishes
second loses git's final ref write:

    fatal: cannot lock ref 'HEAD': is at <peer> but expected <old>

...and all ~2 minutes of pre-commit work is thrown away. This script serializes
the *expensive pre-commit window* so two gates never run concurrently in the
same repo — the collision is prevented at its source instead of being retried
after the waste.

`flock` is ABSENT in this repo's Git Bash (verified 2026-05-31), so bash-level
locking is not an option. We use the same atomic `O_EXCL` create-or-fail pattern
as `session-start.py:_acquire_lock_fd` — portable to Windows and POSIX.

Contract (called from `.githooks/pre-commit` step 0 and `.githooks/pre-push`)
----------------------------------------------------------------------------
  acquire [lock_name]  -> exit 0 if we took the lock (lock file written with
              our pid+ts), exit 1 if a LIVE peer holds it (caller aborts the
              commit/push cleanly).
  release [lock_name]  -> exit 0 always (idempotent; only removes a lock owned
              by this hook shell).

  `lock_name` is optional and defaults to ``.commit-in-progress.lock`` so the
  pre-commit caller (which passes no name) is unchanged. pre-push passes
  ``.push-in-progress.lock``.

The lock file lives at ``<git-common-dir>/.commit-in-progress.lock`` so all
worktrees of the repo serialize against ONE lock (a peer worktree committing is
exactly the HEAD-moving collision we guard against).

The lock filename is parameterizable so the SAME serializer guards the pre-PUSH
seam too. ``.githooks/pre-push`` runs a ~3-min full-drift gate then pushes
``HEAD:main``; two terminals pushing at once each burn that drift and the loser
hits ``cannot lock ref 'refs/heads/main'``. Passing ``.push-in-progress.lock``
as the lock name reuses every line of liveness/staleness logic below for that
seam instead of re-encoding it (institutional-rigor §4 — delegate, never fork).
The default stays ``.commit-in-progress.lock`` so pre-commit is unchanged.

Staleness / crash safety
------------------------
A crashed pre-commit must not wedge every future commit. On contention we read
the holder's recorded pid and timestamp:
  - holder pid is provably dead         -> steal the lock (acquire).
  - lock older than ``_STALE_LOCK_SECS`` -> steal it (a 2-min gate that ran 5+
    min ago is gone; the floor sits well above the real pre-commit runtime).
  - holder alive AND lock fresh          -> a real concurrent gate: abort.

Fail-open: any unexpected error in ACQUIRE returns exit 0 (allow the commit).
A serializer that can't reason about state must never block a legitimate commit
— mirrors the fail-open contract in `.claude/rules/branch-flip-protection.md`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_LOCK_FILENAME = ".commit-in-progress.lock"
OWNER_ENV = "CANOMPX3_PRECOMMIT_LOCK_OWNER"

# The hook shell exports its OWN native pid here. This is the pid whose liveness
# actually gates staleness — it lives for the WHOLE gate (the ~2-min commit /
# ~3-min push window), unlike this helper subprocess whose pid dies the instant
# `acquire()` returns. Recording the helper's own pid (the original design) meant
# a concurrent gate, running after the first helper exited, always saw a DEAD
# holder pid and stole the lock — so the "live peer blocks" guard never fired in
# practice (proven by live-repro 2026-06-09). On Git Bash the shell's MSYS `$$`
# is NOT resolvable by `_pid_alive` (OpenProcess needs a native Windows pid), so
# the hook resolves its native pid via `/proc/$$/winpid` before exporting it;
# POSIX shells fall back to `$$` (already native). When the env var is absent
# (old lock files, direct in-process unit tests) we fall back to the recorded
# `pid` — there the calling Python process IS the live owner, so that is correct.
LIVE_PID_ENV = "CANOMPX3_HOOK_NATIVE_PID"

# A held lock older than this is treated as abandoned (crashed pre-commit). The
# real gate runs ~2 min; 5 min is a comfortable floor above it that still frees
# a wedged repo quickly.
_STALE_LOCK_SECS = 5 * 60


def _git_common_dir() -> Path | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    p = Path(r.stdout.strip())
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return p


def _pid_alive(pid: int) -> bool:
    """Best-effort liveness. Conservative: ambiguity → alive (never steal a lock
    we can't prove is dead). Windows uses OpenProcess exit-code; POSIX os.kill."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            ERROR_INVALID_PARAMETER = 87
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return kernel32.GetLastError() != ERROR_INVALID_PARAMETER
            try:
                code = ctypes.c_ulong()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return True
                return code.value == STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return True  # cannot probe → assume alive (no false steal)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return True


def _read_holder(lock_path: Path) -> dict | None:
    try:
        return json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _holder_is_dead_or_stale(lock_path: Path) -> bool:
    """True iff the current lock holder is provably gone or the lock is stale."""
    holder = _read_holder(lock_path)
    if not isinstance(holder, dict):
        # Unreadable/garbage lock — treat as stale so a corrupt file can't wedge.
        return True
    ts = holder.get("ts")
    if isinstance(ts, (int, float)) and (time.time() - ts) >= _STALE_LOCK_SECS:
        return True
    # Liveness keys on the hook shell's native pid when recorded (it lives for the
    # whole gate); else the helper's own pid (old locks / in-process tests, where
    # the calling process is the live owner). Checking the ephemeral helper pid for
    # a lock written by a now-exited helper was the original false-steal bug.
    live_pid = holder.get("live_pid")
    gate_pid = live_pid if isinstance(live_pid, int) else holder.get("pid")
    if isinstance(gate_pid, int) and not _pid_alive(gate_pid):
        return True
    return False


def _try_create(lock_path: Path) -> bool:
    """Atomic O_EXCL create. Returns True if we created (own) the lock."""
    owner = os.environ.get(OWNER_ENV) or f"ppid:{os.getppid()}"
    record = {"pid": os.getpid(), "ppid": os.getppid(), "owner": owner, "ts": time.time()}
    # The hook shell's native pid (lives for the whole gate) is the real liveness
    # key; record it when present. Absent (in-process tests / old locks) → liveness
    # falls back to `pid` below, where this Python process is itself the owner.
    live_pid_env = os.environ.get(LIVE_PID_ENV)
    if live_pid_env and live_pid_env.strip().isdigit():
        record["live_pid"] = int(live_pid_env.strip())
    payload = json.dumps(record).encode("utf-8")
    try:
        fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        return False
    except OSError:
        return False
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)
    return True


def acquire(lock_name: str = DEFAULT_LOCK_FILENAME) -> int:
    common = _git_common_dir()
    if common is None:
        return 0  # not a git repo / git unavailable — fail-open, allow commit
    lock_path = common / lock_name

    if _try_create(lock_path):
        return 0  # clean acquire

    # Contention. Steal iff the holder is provably gone or the lock is stale.
    if _holder_is_dead_or_stale(lock_path):
        try:
            lock_path.unlink()
        except OSError:
            pass
        if _try_create(lock_path):
            return 0
        # Lost the race to another stealer that just acquired — treat as live peer.

    # A live peer holds a fresh lock — a real concurrent gate. Abort cleanly so
    # the caller does NOT burn the expensive gate just to lose the ref write.
    # The seam word ("commit"/"push") is derived from the lock name so the one
    # helper gives an accurate message for either caller.
    seam = "push" if "push" in lock_name else "commit"
    holder = _read_holder(lock_path) or {}
    sys.stderr.write(
        f"BLOCKED: another pre-{seam} is already running in this repo "
        f"(holder pid {holder.get('pid', '?')}, started "
        f"{int(time.time() - holder.get('ts', time.time()))}s ago).\n"
        f"  Two concurrent pre-{seam} gates race git's final ref write and waste the\n"
        f"  full gate. Wait for the other {seam} to finish, then retry.\n"
        f"  If you are certain no other {seam} is running: rm '{lock_path}'\n"
    )
    return 1


def _holder_matches_this_hook(holder: object) -> bool:
    """Return True iff this helper belongs to the hook shell that owns the lock."""
    if not isinstance(holder, dict):
        return False
    owner = os.environ.get(OWNER_ENV) or f"ppid:{os.getppid()}"
    if holder.get("owner") == owner:
        return True
    # Backward compatibility for lock files written before the owner token
    # existed, plus direct in-process unit tests.
    return holder.get("pid") == os.getpid() or (holder.get("owner") is None and holder.get("ppid") == os.getppid())


def release(lock_name: str = DEFAULT_LOCK_FILENAME) -> int:
    common = _git_common_dir()
    if common is None:
        return 0
    lock_path = common / lock_name
    holder = _read_holder(lock_path)
    # acquire/release are separate helper processes. The helper pid matches only
    # for direct in-process tests; the parent shell pid is the real hook owner.
    if _holder_matches_this_hook(holder):
        try:
            lock_path.unlink()
        except OSError:
            pass
    return 0


def main(argv: list[str]) -> int:
    if not (2 <= len(argv) <= 3) or argv[1] not in {"acquire", "release"}:
        sys.stderr.write("usage: commit_serialize.py {acquire|release} [lock_name]\n")
        return 2
    lock_name = argv[2] if len(argv) == 3 else DEFAULT_LOCK_FILENAME
    try:
        return acquire(lock_name) if argv[1] == "acquire" else release(lock_name)
    except BaseException:
        # Fail-open on ACQUIRE (allow commit/push); release is already best-effort.
        return 0 if argv[1] == "acquire" else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
