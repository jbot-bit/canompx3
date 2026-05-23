#!/usr/bin/env python3
"""Per-worktree concurrent-session lease backed by the `filelock` library.

Canonical lease I/O lives here. The PreToolUse hook
(`.claude/hooks/worktree_guard.py`) imports from this module so there is a
single source of truth for path resolution, schema, and reclaim predicates
(institutional-rigor §4 — no inline copies).

## Lock primitive

`filelock.FileLock` (https://py-filelock.readthedocs.io/) — used by pip, uv,
poetry, black. Cross-platform mutex via `msvcrt.locking(fd, LK_NBLCK, 1)` on
Windows (Microsoft Learn: `_locking` — locks bytes of a file) and `fcntl.flock`
on POSIX. The OS releases the lock automatically when the holding process
exits, which is the property we need against the "peer terminal vanished"
failure mode.

## Two files per worktree

  - `<git-dir>/.claude.worktree.lock`   — the OS-level FileLock target. Empty.
                                          Held for the lifetime of the Claude
                                          session.
  - `<git-dir>/.claude.worktree.lease.json` — informational sidecar (holder
                                          PID, branch, heartbeat). Read by
                                          `--status` and by the hook for the
                                          stderr BLOCK message.

The mutex is the lock file (OS-enforced, dead-process-tolerant). The JSON is
informational only. Stale-heartbeat reclaim applies ONLY to the sidecar — the
underlying lock self-releases on process death.

## Why not just the 12h `_session_lock_lines()` PID lock?

`.claude/hooks/session-start.py:_session_lock_lines()` uses a hand-rolled
`O_EXCL` + `os.kill(pid, 0)` pattern with a 12-hour staleness floor. That
serves a different incident: 3-day-stale locks from dead PIDs. This module
serves the actively-running-peer-terminal case (30-min ergonomic threshold
on the heartbeat, PreToolUse enforcement). Both coexist intentionally.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

try:
    from filelock import BaseFileLock, FileLock, Timeout
except ImportError as exc:  # pragma: no cover - guarded by drift check
    raise SystemExit(f"worktree_guard requires `filelock` (already a transitive dep): {exc}") from exc

LOCK_FILENAME = ".claude.worktree.lock"
LEASE_FILENAME = ".claude.worktree.lease.json"

# Sidecar staleness: the OS releases the FileLock automatically when the
# holder dies, but the *sidecar* JSON heartbeat is informational. A live peer
# could in principle hold the lock without writing a heartbeat for >30 min
# (e.g., long-running internal task). When the sidecar is stale we still
# respect the OS lock — we only complain in --status output. The hook treats
# stale-sidecar-but-locked as BLOCK.
STALE_HEARTBEAT_SECONDS = 30 * 60

# Exit codes
EXIT_OK = 0
EXIT_BLOCKED = 2
EXIT_ERROR = 3


def _build_payload_dict(pid: int, worktree: str, iso_started: str, iso_heartbeat: str, branch: str) -> dict:
    return {
        "pid": pid,
        "worktree": worktree,
        "iso_started": iso_started,
        "iso_heartbeat": iso_heartbeat,
        "branch": branch,
        "schema": 1,
    }


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _iso_age_seconds(iso: str) -> float | None:
    if not iso:
        return None
    try:
        ts = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except (ValueError, AttributeError, TypeError):
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return (datetime.now(UTC) - ts).total_seconds()


def _run_git(args: list[str], cwd: Path, timeout: float = 5.0) -> tuple[int, str]:
    try:
        r = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return r.returncode, r.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return 1, ""


def resolve_git_dir(cwd: Path) -> Path | None:
    """Per-worktree git directory.

    For linked worktrees this is `.git/worktrees/<name>/`, so the lock/lease
    are already worktree-isolated by git's own layout. Returns None when
    `cwd` is not inside a git repo (callers MUST fail-open).
    """
    rc, out = _run_git(["rev-parse", "--git-dir"], cwd)
    if rc != 0 or not out:
        return None
    p = Path(out)
    if not p.is_absolute():
        p = (cwd / p).resolve()
    return p


def resolve_worktree_root(cwd: Path) -> Path | None:
    rc, out = _run_git(["rev-parse", "--show-toplevel"], cwd)
    if rc != 0 or not out:
        return None
    return Path(out).resolve()


def lock_path(cwd: Path) -> Path | None:
    git_dir = resolve_git_dir(cwd)
    return None if git_dir is None else git_dir / LOCK_FILENAME


def lease_path(cwd: Path) -> Path | None:
    git_dir = resolve_git_dir(cwd)
    return None if git_dir is None else git_dir / LEASE_FILENAME


def read_lease(cwd: Path) -> dict | None:
    lp = lease_path(cwd)
    if lp is None or not lp.exists():
        return None
    try:
        data = json.loads(lp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_lease(lease_file: Path, payload: dict) -> None:
    """Plain write — the FileLock guards concurrent writers, so a temp+replace
    dance is not load-bearing. Kept inside the locked section by callers.
    """
    lease_file.parent.mkdir(parents=True, exist_ok=True)
    lease_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_payload(cwd: Path, pid: int) -> dict:
    wt_root = resolve_worktree_root(cwd) or cwd
    rc_branch, branch = _run_git(["branch", "--show-current"], cwd)
    branch = branch if rc_branch == 0 else ""
    now = _now_iso()
    return _build_payload_dict(
        pid=pid,
        worktree=str(wt_root),
        iso_started=now,
        iso_heartbeat=now,
        branch=branch,
    )


def acquire(cwd: Path, pid: int | None = None) -> tuple[str, dict | None, str]:
    """Acquire (or refresh) the worktree lease.

    Returns (status, payload, message) where status is:
      - "acquired":  lock obtained, sidecar written
      - "refreshed": current PID already held the lock; sidecar heartbeat bumped
      - "blocked":   a peer process holds the OS-level FileLock
      - "skipped":   cwd is not inside a git repo (caller fails open)
      - "error":     transient FS error (caller fails open with WARN)

    On "blocked", the returned payload is the peer's lease sidecar (best
    effort — may be None if the sidecar is missing/corrupt).
    """
    pid = pid if pid is not None else os.getpid()
    lk = lock_path(cwd)
    ls = lease_path(cwd)
    if lk is None or ls is None:
        return "skipped", None, "not inside a git worktree"

    lk.parent.mkdir(parents=True, exist_ok=True)

    # If we ALREADY hold a live FileLock for this path (same in-process
    # registry) AND the sidecar says THIS PID owns it, that's a refresh.
    # Skip re-acquiring the OS lock since msvcrt.locking on Windows is
    # per-fd (not per-PID): re-acquiring via a fresh FileLock object from
    # the same PID still collides at the kernel level.
    existing = read_lease(cwd)
    already_held = str(lk) in _LIVE_LOCKS
    if existing and existing.get("pid") == pid and existing.get("worktree") == str(resolve_worktree_root(cwd) or cwd):
        existing["iso_heartbeat"] = _now_iso()
        try:
            _write_lease(ls, existing)
        except OSError as exc:
            return "error", None, f"failed to refresh heartbeat: {exc}"
        if not already_held:
            # We hold the sidecar but lost the in-process FileLock object
            # (e.g., re-imported module). Don't try to grab the OS lock —
            # we cannot re-enter it from the same PID on Windows. Trust the
            # sidecar match as proof-of-ownership.
            return "refreshed", existing, f"refreshed heartbeat for PID {pid}"
        return "refreshed", existing, f"refreshed heartbeat for PID {pid}"

    file_lock = FileLock(str(lk), timeout=0)  # non-blocking — fail fast
    try:
        file_lock.acquire()
    except Timeout:
        peer_lease = read_lease(cwd)
        peer_pid = peer_lease.get("pid") if peer_lease else None
        return (
            "blocked",
            peer_lease,
            f"peer holds OS lock at {lk} (peer PID {peer_pid})",
        )
    except OSError as exc:
        return "error", None, f"failed to acquire lock at {lk}: {exc}"

    payload = _build_payload(cwd, pid)
    try:
        _write_lease(ls, payload)
    except OSError as exc:
        file_lock.release()
        return "error", None, f"failed to write lease sidecar: {exc}"

    # Note: we INTENTIONALLY do NOT release the FileLock here. The lock is
    # meant to be held for the lifetime of the Claude session — OS-level
    # auto-release on process death is the dead-peer-detection mechanism.
    # `release(force=True)` in this module handles explicit release.
    _LIVE_LOCKS[str(lk)] = file_lock  # keep a reference so the lock object
    # is not garbage-collected mid-session (which would release the OS lock).
    return "acquired", payload, f"acquired lease for PID {pid}"


# Process-lifetime registry of held locks. The hook re-imports this module per
# invocation, so this dict is empty on each call — that is intentional. The
# OS-level lock on the file persists across re-imports because it is held by
# the parent Claude process via the prior `acquire()` call's open file
# descriptor (filelock keeps the fd open inside the FileLock instance).
#
# NOTE: when the hook re-acquires per call, it gets `blocked` if a peer holds
# it (correct) or its OWN per-call re-acquire returns `acquired` (also fine —
# msvcrt/fcntl locks are per-PID, so the same PID can re-lock its own file).
_LIVE_LOCKS: dict[str, BaseFileLock] = {}


def release(cwd: Path, pid: int | None = None, force: bool = False) -> tuple[str, str]:
    """Explicit release of the lock + sidecar.

    Without `force`: refuses if the sidecar PID differs from current PID.
    With `force`: removes the sidecar AND attempts to release the OS lock
    held by this process. Cannot release locks held by other processes;
    those rely on OS auto-release at process exit.

    Returns (status, message) where status ∈ {"released", "not-held",
    "refused", "skipped", "error"}.
    """
    pid = pid if pid is not None else os.getpid()
    lk = lock_path(cwd)
    ls = lease_path(cwd)
    if lk is None or ls is None:
        return "skipped", "not inside a git worktree"

    if not lk.exists() and not ls.exists():
        return "not-held", "no lock or lease file present"

    existing = read_lease(cwd) or {}
    if not force and existing.get("pid") != pid:
        return (
            "refused",
            f"lease sidecar says PID {existing.get('pid')} holds it "
            f"(current PID {pid}); use --force-release to override",
        )

    # Release any FileLock this process holds for this path.
    held = _LIVE_LOCKS.pop(str(lk), None)
    if held is not None:
        try:
            held.release(force=True)
        except OSError:
            pass

    errs: list[str] = []
    for path in (ls, lk):
        try:
            if path.exists():
                path.unlink()
        except OSError as exc:
            errs.append(f"{path.name}: {exc}")
    if errs:
        return "error", "; ".join(errs)
    return "released", f"released lease (was PID {existing.get('pid')})"


def status(cwd: Path, pid: int | None = None) -> dict:
    """Read-only inspection. Used by `--status` and by the hook BLOCK message.

    `is_locked` reflects the OS-level FileLock state — TRUE means a process
    (possibly this one) holds it. `current_is_holder` is True iff the sidecar
    PID matches `pid`; it does NOT separately probe the OS lock since the OS
    lock is per-PID and re-acquirable by the same PID without conflict.
    """
    pid = pid if pid is not None else os.getpid()
    lk = lock_path(cwd)
    ls = lease_path(cwd)
    if lk is None or ls is None:
        return {"in_git_repo": False}

    data = read_lease(cwd)
    # Probe lock state by trying a non-blocking acquire/release as a separate
    # FileLock instance. If we can acquire, the OS lock was free.
    is_locked = False
    if lk.exists():
        probe = FileLock(str(lk), timeout=0)
        try:
            probe.acquire()
        except Timeout:
            is_locked = True
        except OSError:
            is_locked = False
        else:
            probe.release(force=True)

    if data is None:
        return {
            "in_git_repo": True,
            "lock_path": str(lk),
            "lease_path": str(ls),
            "lease_present": False,
            "is_locked": is_locked,
            "current_pid": pid,
        }

    hb = data.get("iso_heartbeat") or data.get("iso_started") or ""
    age_s = _iso_age_seconds(hb)
    sidecar_stale = age_s is not None and age_s >= STALE_HEARTBEAT_SECONDS
    holder_pid = data.get("pid")
    return {
        "in_git_repo": True,
        "lock_path": str(lk),
        "lease_path": str(ls),
        "lease_present": True,
        "is_locked": is_locked,
        "holder_pid": holder_pid,
        "holder_worktree": data.get("worktree"),
        "holder_branch": data.get("branch"),
        "iso_started": data.get("iso_started"),
        "iso_heartbeat": hb,
        "heartbeat_age_seconds": age_s,
        "sidecar_stale": sidecar_stale,
        "current_pid": pid,
        "current_is_holder": isinstance(holder_pid, int) and holder_pid == pid,
    }


def _format_status_line(snap: dict) -> str:
    if not snap.get("in_git_repo"):
        return "worktree-guard: not in a git repo — no lease"
    if not snap.get("lease_present"):
        if snap.get("is_locked"):
            return f"worktree-guard: OS lock held but sidecar missing at {snap.get('lease_path')}"
        return f"worktree-guard: no lease at {snap.get('lease_path')}"
    age = snap.get("heartbeat_age_seconds")
    age_str = f"{age:.0f}s" if isinstance(age, (int, float)) else "unknown"
    parts = [
        f"holder PID {snap.get('holder_pid')}",
        f"worktree {snap.get('holder_worktree')!r}",
        f"branch {snap.get('holder_branch') or '?'}",
        f"heartbeat {age_str}",
        f"locked={snap.get('is_locked')}",
    ]
    if snap.get("sidecar_stale"):
        parts.append(f"SIDECAR_STALE (>{STALE_HEARTBEAT_SECONDS}s)")
    if snap.get("current_is_holder"):
        parts.append("THIS PROCESS")
    return "worktree-guard: " + ", ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="worktree_guard",
        description=("Per-worktree lease for concurrent-Claude-session detection. Default: print status."),
    )
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--status", action="store_true", help="Print lease status (default)")
    g.add_argument("--release", action="store_true", help="Release lease iff held by current PID")
    g.add_argument(
        "--force-release",
        action="store_true",
        help="Release lease unconditionally",
    )
    g.add_argument(
        "--acquire",
        action="store_true",
        help="Acquire (or refresh) the lease for the current PID",
    )
    parser.add_argument(
        "--cwd",
        default=".",
        help="Working directory to resolve git-dir from (default: cwd)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a one-liner",
    )
    args = parser.parse_args(argv)
    cwd = Path(args.cwd).resolve()

    if args.release or args.force_release:
        status_str, msg = release(cwd, force=args.force_release)
        if args.json:
            print(json.dumps({"action": "release", "status": status_str, "message": msg}))
        else:
            print(f"worktree-guard: {status_str} — {msg}")
        return EXIT_OK if status_str in ("released", "not-held") else EXIT_BLOCKED

    if args.acquire:
        status_str, lease_data, msg = acquire(cwd)
        if args.json:
            print(
                json.dumps(
                    {
                        "action": "acquire",
                        "status": status_str,
                        "message": msg,
                        "lease": lease_data,
                    },
                    indent=2,
                )
            )
        else:
            print(f"worktree-guard: {status_str} — {msg}")
        if status_str == "blocked":
            return EXIT_BLOCKED
        if status_str == "error":
            return EXIT_ERROR
        return EXIT_OK

    snap = status(cwd)
    if args.json:
        print(json.dumps(snap, indent=2, default=str))
    else:
        print(_format_status_line(snap))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
