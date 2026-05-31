#!/usr/bin/env python3
"""Per-worktree concurrent-session mutex backed by (session_id, ppid) + heartbeat.

Canonical lease I/O lives here. The PreToolUse hook
(`.claude/hooks/worktree_guard.py`) imports from this module so there is a
single source of truth for path resolution, schema, and ownership predicates
(institutional-rigor §4 — no inline copies).

## Why this is NOT an OS file-lock anymore (n=2 incident 2026-05-29/30)

The previous design held a `filelock.FileLock` for the worktree. That provided
ZERO mutual exclusion in practice: the PreToolUse hook runs as a NEW `python`
subprocess per tool call, so the lock holder was an ephemeral subprocess that
exited microseconds later — and `filelock` auto-releases the OS lock on holder
process exit. Two concurrent Claude/Codex sessions therefore never contended;
the sidecar merely recorded "whoever ran a tool last" (the rotating-PID storm
seen during the incident). The mutex was theatre.

The real holder we need to track is the long-lived **Claude session process**,
which is the PARENT of every hook subprocess. So ownership is keyed on:

  - `session_id` — the Claude session id from the hook event payload (stable
    within one session; the transcript filename IS this id). A NEW id is minted
    on every start including /clear & /compact, so it alone cannot be the key
    (a /clear-restart would false-block itself against its own fresh sidecar).
  - `ppid` — `os.getppid()` from the hook subprocess == the Claude session
    process. Stable for one running session; reliable to liveness-probe because
    it is the immediate parent (same user, same machine) — unlike an arbitrary
    stored PID, which the old 12h mutex's own comments admit is unreliable on
    Windows.

A peer holds the worktree IFF the sidecar records a DIFFERENT session_id whose
heartbeat is fresh (< STALE_HEARTBEAT_SECONDS) AND whose recorded ppid is still
alive. Stale heartbeat OR dead ppid → reclaimable immediately (no 12h wait);
this is what makes a /clear-restart safe: the prior session's Claude process is
gone, so its ppid reads dead and we reclaim.

## Two files per worktree

  - `<git-dir>/.claude.worktree.lock`   — best-effort OS FileLock target. Empty.
                                          Belt-and-braces only; the heartbeat +
                                          ppid-liveness is the actual mutex.
  - `<git-dir>/.claude.worktree.lease.json` — the authoritative ownership
                                          record: holder session_id, ppid, pid,
                                          branch, heartbeat. Read by `--status`
                                          and by the hook for the BLOCK message.

For linked worktrees `<git-dir>` is `.git/worktrees/<name>/`, so the lease is
already worktree-isolated by git's own layout.

## Why not just the 12h `_session_lock_lines()` PID lock?

`.claude/hooks/session-start.py:_session_lock_lines()` uses a hand-rolled
`O_EXCL` + `os.kill(pid, 0)` pattern with a 12-hour staleness floor. It serves
a different, longer-lived incident class (3-day-stale dead-PID locks). This
module serves the actively-running-peer case with a 90s ergonomic heartbeat and
PreToolUse enforcement. Both coexist intentionally.
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

# Heartbeat freshness window. A peer's lease is "live" only if its heartbeat is
# newer than this AND its ppid is alive. 90s comfortably exceeds the gap between
# a session's tool calls (each PreToolUse refreshes the heartbeat) while keeping
# a /clear-restart's worst-case wait short if ppid-liveness somehow can't be
# determined.
STALE_HEARTBEAT_SECONDS = 90

# Cross-session heartbeat directory — written by `.claude/hooks/session-heartbeat.py`
# (one `<session_id>.beat` file per live session, stamped on every tool call) and
# read at startup by `_live_heartbeat_lines` in `.claude/hooks/session-start.py`.
# We consult it here too: a fresh beat from a DIFFERENT session is a positive
# liveness FACT that does not depend on the single lease sidecar's ppid. This is
# the load-bearing Windows fix — `os.kill(ppid, 0)` / OpenProcess can momentarily
# read a live peer's ppid as dead, which would let `acquire()` RECLAIM the single
# lease (handing it session-to-session) instead of BLOCKing. A fresh peer beat
# overrides that false-dead verdict so the block stands. Source of the writer's
# constants: the two hook files above (this is the canonical READER for the gate).
#
# The directory lives under the git COMMON dir so every worktree of the repo
# shares it; the live window matches the hooks' `_HEARTBEAT_LIVE_WINDOW_SECS`.
HEARTBEAT_DIRNAME = ".claude-heartbeats"
HEARTBEAT_LIVE_WINDOW_SECONDS = 10 * 60

# Exit codes
EXIT_OK = 0
EXIT_BLOCKED = 2
EXIT_ERROR = 3


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _time_now() -> float:
    """Wall-clock epoch seconds — wrapper so tests can monkeypatch liveness time
    without touching ``datetime`` (heartbeat mtimes are epoch-based)."""
    import time

    return time.time()


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


def _pid_is_alive(pid: int | None, expected_create_time: int | None = None) -> bool:
    """Reliable cross-platform liveness for a recorded ppid.

    Windows is the load-bearing case: ``os.kill(pid, 0)`` does NOT reliably
    raise ``ProcessLookupError`` for a dead pid there — it commonly raises a
    generic ``OSError`` (ERROR_INVALID_PARAMETER / ERROR_ACCESS_DENIED), which a
    naive ``except OSError: return True`` would misread as "alive" and so NEVER
    reclaim a dead session's lease — re-creating the exact wedge this fix
    exists to remove (a /clear-restart would block on its own dead predecessor
    forever). So on Windows we probe via ``OpenProcess`` and inspect the exit
    code: a live process has exit code STILL_ACTIVE (259); a dead one has any
    other value (or fails to open with a not-found error).

    `expected_create_time` is the low-DWORD of the process's Windows FILETIME at
    lease-write time. When provided, a mismatch proves PID reuse — the original
    session is gone even if exit code says STILL_ACTIVE.

    POSIX: ``os.kill(pid, 0)`` — ProcessLookupError = dead, PermissionError =
    alive-but-other-user.

    The ppid we probe is the hook subprocess's parent — the Claude session
    process — so it is the SAME user on the SAME machine, which is why this
    probe is trustworthy here (unlike the old code's probe of an arbitrary
    stored PID across process trees).
    """
    if not isinstance(pid, int) or pid <= 0:
        return False

    if os.name == "nt":
        return _pid_is_alive_windows(pid, expected_create_time=expected_create_time)

    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, AttributeError):
        return True  # POSIX: cannot prove dead → assume alive (no false reclaim)


def _get_process_create_time_windows(pid: int) -> int | None:
    """Return the Windows process creation time (FILETIME low DWORD) or None on failure.

    Used to detect PID reuse: if the recorded creation time doesn't match the
    live process's creation time, the PID was recycled and the old session is gone.
    Returns None on any failure (caller falls back to exit-code check alone).
    """
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return None
        try:
            creation = wintypes.FILETIME()
            exit_ft = wintypes.FILETIME()
            kernel_ft = wintypes.FILETIME()
            user_ft = wintypes.FILETIME()
            ok = kernel32.GetProcessTimes(
                handle,
                ctypes.byref(creation),
                ctypes.byref(exit_ft),
                ctypes.byref(kernel_ft),
                ctypes.byref(user_ft),
            )
            if not ok:
                return None
            return creation.dwLowDateTime  # low 32 bits — sufficient for reuse detection
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        return None


def _pid_is_alive_windows(pid: int, expected_create_time: int | None = None) -> bool:
    """Windows liveness via OpenProcess + GetExitCodeProcess.

    Returns False ONLY when we can positively prove the process is gone
    (OpenProcess fails with ERROR_INVALID_PARAMETER == the pid doesn't exist,
    or the process reports a non-STILL_ACTIVE exit code). Any ambiguity →
    True (conservative: never false-reclaim a possibly-live session).

    When `expected_create_time` is provided (from the lease sidecar), it is
    cross-checked against the live process's creation time. A mismatch means
    the PID was reused by a different process — the original session is gone.
    This closes the Windows PID-reuse false-block (adversarial audit HIGH, 2026-05-30).
    """
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        ERROR_INVALID_PARAMETER = 87
        ERROR_ACCESS_DENIED = 5
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259

        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            err = kernel32.GetLastError()
            if err == ERROR_INVALID_PARAMETER:
                return False  # pid does not exist → dead
            if err == ERROR_ACCESS_DENIED:
                return True  # exists but not queryable → alive
            return True  # any other open failure → assume alive
        try:
            exit_code = wintypes.DWORD()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            if not ok:
                return True  # could not read exit code → assume alive
            if exit_code.value != STILL_ACTIVE:
                return False  # process has exited

            # Process is running. Cross-check creation time to detect PID reuse.
            if expected_create_time is not None:
                live_create = _get_process_create_time_windows(pid)
                if live_create is not None and live_create != expected_create_time:
                    return False  # PID reused by a different process → original is gone

            return True
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        # ctypes unavailable / unexpected — fall back to os.kill, then to alive.
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except OSError:
            return True


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


def _build_payload(cwd: Path, pid: int, ppid: int, session_id: str, iso_started: str) -> dict:
    wt_root = resolve_worktree_root(cwd) or cwd
    rc_branch, branch = _run_git(["branch", "--show-current"], cwd)
    branch = branch if rc_branch == 0 else ""
    payload: dict = {
        "pid": pid,
        "ppid": ppid,
        "session_id": session_id,
        "worktree": str(wt_root),
        "iso_started": iso_started,
        "iso_heartbeat": _now_iso(),
        "branch": branch,
        "schema": 3,
    }
    # Store ppid creation time for PID-reuse detection on Windows (schema 3+).
    if os.name == "nt":
        ct = _get_process_create_time_windows(ppid)
        if ct is not None:
            payload["ppid_create_time"] = ct
    return payload


def _write_lease(lease_file: Path, payload: dict) -> None:
    lease_file.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: write to a temp file then os.replace() so read_lease()
    # never sees a partial JSON during a concurrent heartbeat refresh. This
    # is the torn-read fix (adversarial audit CRITICAL, 2026-05-30).
    tmp = lease_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, lease_file)


def _same_session(existing: dict, session_id: str, pid: int) -> bool:
    """True iff the existing lease belongs to the caller's session.

    Primary key is `session_id` (stable within a running session). When the
    caller has no session_id (CLI invocation, or a hook event without the
    field), fall back to the recorded pid — preserving the legacy CLI/test
    contract where the same `pid` re-acquires its own lease as a refresh.
    """
    existing_sid = existing.get("session_id")
    if session_id and existing_sid:
        return existing_sid == session_id
    # No session identity on one side → fall back to pid identity.
    return existing.get("pid") == pid


def _git_common_dir(cwd: Path) -> Path | None:
    """Resolve the repo's SHARED git dir from ``cwd`` (``--git-common-dir``).

    Unlike ``resolve_git_dir`` (per-worktree), this resolves to the SAME path
    for every worktree of the repo — the precondition for reading the shared
    `.claude-heartbeats/` directory. Returns None on any error (callers treat
    a missing common dir as "no heartbeat evidence", never as a hard failure).
    """
    rc, out = _run_git(["rev-parse", "--git-common-dir"], cwd)
    if rc != 0 or not out:
        return None
    p = Path(out)
    if not p.is_absolute():
        p = (cwd / p).resolve()
    return p


def _fresh_peer_heartbeat(cwd: Path, exclude_session_id: str = "") -> bool:
    """True iff a DIFFERENT live session has a fresh beat in this repo's tree.

    Reads `<git-common-dir>/.claude-heartbeats/*.beat`. A beat counts as a live
    peer when ALL hold:
      - its `session_id` differs from ``exclude_session_id`` (never self-block),
      - its `cwd` canonicalises to THIS worktree root (a beat from a sibling
        worktree is sanctioned parallel work, not a same-tree collision),
      - its file mtime is within ``HEARTBEAT_LIVE_WINDOW_SECONDS`` (older beats
        are crashed/abandoned sessions).

    Fail-CLOSED-to-safe means: any error → False (no heartbeat evidence). That is
    the SAFE direction here because this function only ever STRENGTHENS the block
    decision in ``_peer_is_live`` — a False return just falls back to the existing
    ppid logic, it never forces a reclaim. Excluding our own session is essential:
    a single live session beats constantly, and self-counting would false-block
    every `/clear`-restart.
    """
    common = _git_common_dir(cwd)
    if common is None:
        return False
    beat_dir = common / HEARTBEAT_DIRNAME
    try:
        if not beat_dir.is_dir():
            return False
        beat_files = list(beat_dir.glob("*.beat"))
    except OSError:
        return False

    wt_root = resolve_worktree_root(cwd)
    current_norm = str(wt_root).lower() if wt_root is not None else None

    now = _time_now()
    me = (exclude_session_id or "").strip()
    for bf in beat_files:
        try:
            age = now - bf.stat().st_mtime
        except OSError:
            continue
        # Reject clock-skew (future-dated beyond 60s) and beats outside the
        # live window. Mirrors `_live_heartbeat_lines` in session-start.py.
        if age < -60 or age > HEARTBEAT_LIVE_WINDOW_SECONDS:
            continue
        try:
            beat = json.loads(bf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, ValueError):
            continue
        if not isinstance(beat, dict):
            continue
        beat_sid = str(beat.get("session_id", "")).strip()
        if beat_sid and me and beat_sid == me:
            continue  # our own beat — not a peer
        # Same-tree gate: only a beat in THIS worktree is a collision. A beat
        # whose cwd canonicalises elsewhere is a sanctioned parallel worktree.
        if current_norm is not None:
            beat_cwd = str(beat.get("cwd", "")).strip()
            try:
                beat_norm = str(Path(beat_cwd).resolve()).lower() if beat_cwd else None
            except (OSError, ValueError, RuntimeError):
                beat_norm = beat_cwd.lower() if beat_cwd else None
            if beat_norm != current_norm:
                continue
        return True  # a different session is beating in this tree, right now
    return False


def _peer_is_live(existing: dict, cwd: Path | None = None, exclude_session_id: str = "") -> bool:
    """True iff the existing lease is held by a still-live session.

    Live = (recorded ppid alive) OR (a different session has a fresh beat in
    this tree's `.claude-heartbeats/`). A stale heartbeat AND a dead ppid AND
    no fresh peer beat means the holder is gone → reclaimable. When the lease
    predates schema 2 (no ppid), fall back to heartbeat freshness alone
    (conservative: a fresh legacy heartbeat still blocks).

    The heartbeat-directory cross-check (``cwd`` provided) is the Windows fix:
    ``os.kill``/OpenProcess can momentarily read a live peer's ppid as dead,
    which without this would RECLAIM the single lease instead of blocking. A
    fresh peer beat is an independent liveness FACT that keeps the block. When
    ``cwd`` is None (legacy callers / tests), the heartbeat check is skipped and
    behaviour is exactly the prior ppid-only logic — a pure superset.
    """
    ppid = existing.get("ppid")
    ppid_create_time = existing.get("ppid_create_time")  # schema 3+; None on older leases

    # ppid-liveness is AUTHORITATIVE. A provably-alive Claude session process
    # holds the worktree no matter how stale its heartbeat — because the
    # heartbeat only refreshes on PreToolUse, and a single long-running tool
    # call (e.g. `python pipeline/check_drift.py` at ~180s, well past the 90s
    # window) legitimately stops refreshing it. Reclaiming on stale-heartbeat
    # ALONE would steal a live session's lease mid-call and let two sessions
    # write — the exact corruption this guard exists to prevent (adversarial
    # audit MED finding, 2026-05-30). So: a live ppid always blocks.
    # ppid_create_time cross-check prevents Windows PID-reuse false-blocks
    # (adversarial audit HIGH, 2026-05-30).
    if ppid is not None and _pid_is_alive(ppid, expected_create_time=ppid_create_time):
        return True

    # ppid reads dead (or absent). Before falling back to the lease's own
    # heartbeat timestamp, consult the shared `.claude-heartbeats/` directory:
    # a fresh beat from a DIFFERENT session in this tree is an independent
    # liveness FACT that the (unreliable on Windows) ppid probe just missed.
    # This is what stops the single lease being reclaimed session-to-session
    # while three terminals are genuinely live. Skipped when cwd is None
    # (legacy callers / unit tests) — a pure superset of the prior logic.
    if cwd is not None and _fresh_peer_heartbeat(cwd, exclude_session_id=exclude_session_id):
        return True

    # No fresh peer beat either. Now the lease's OWN heartbeat is the tiebreaker:
    #   - fresh heartbeat + no ppid  → legacy schema-1 lease; conservatively
    #     treat as live and block (we cannot prove the holder gone).
    #   - stale/unparseable heartbeat → holder is gone → reclaimable.
    hb = existing.get("iso_heartbeat") or existing.get("iso_started") or ""
    age = _iso_age_seconds(hb)
    if age is None or age >= STALE_HEARTBEAT_SECONDS:
        return False  # ppid dead AND heartbeat stale AND no peer beat → reclaim
    if ppid is None:
        return True  # legacy lease, fresh heartbeat, no ppid → block (cautious)
    return False  # ppid proven dead, even if heartbeat still fresh → reclaim


def acquire(
    cwd: Path,
    pid: int | None = None,
    *,
    session_id: str = "",
    ppid: int | None = None,
) -> tuple[str, dict | None, str]:
    """Acquire (or refresh) the worktree lease using (session_id, ppid)+heartbeat.

    Returns (status, payload, message) where status is:
      - "acquired":  no existing lease; we wrote the sidecar as owner
      - "refreshed": the lease already belongs to THIS session; heartbeat bumped
      - "reclaimed": prior holder was stale/dead; we took ownership
      - "blocked":   a DIFFERENT session holds a live lease (fresh hb + live ppid)
      - "skipped":   cwd is not inside a git repo (caller fails open)
      - "error":     transient FS error (caller fails open with WARN)

    On "blocked", the returned payload is the peer's lease sidecar.

    `session_id` is the stable session identity; when empty (CLI/legacy) the
    function falls back to `pid` identity so existing callers keep working.
    """
    pid = pid if pid is not None else os.getpid()
    ppid = ppid if ppid is not None else os.getppid()
    lk = lock_path(cwd)
    ls = lease_path(cwd)
    if lk is None or ls is None:
        return "skipped", None, "not inside a git worktree"

    lk.parent.mkdir(parents=True, exist_ok=True)

    existing = read_lease(cwd)
    if existing is not None:
        # (a) Our own lease → refresh heartbeat, keep started timestamp.
        if _same_session(existing, session_id, pid):
            payload = _build_payload(
                cwd,
                pid,
                ppid,
                session_id or str(existing.get("session_id") or ""),
                existing.get("iso_started") or _now_iso(),
            )
            try:
                _write_lease(ls, payload)
            except OSError as exc:
                return "error", None, f"failed to refresh heartbeat: {exc}"
            return "refreshed", payload, f"refreshed lease for session {session_id or pid}"

        # (b) Someone else's lease and it is LIVE → block. Pass cwd + our own
        # session_id so the heartbeat cross-check can see a fresh peer beat
        # (Windows ppid-false-dead fix) while never counting our own beat.
        if _peer_is_live(existing, cwd=cwd, exclude_session_id=session_id):
            peer_pid = existing.get("pid")
            peer_sid = existing.get("session_id")
            return (
                "blocked",
                existing,
                f"peer session {peer_sid or peer_pid} holds a live lease (ppid "
                f"{existing.get('ppid')} alive, heartbeat fresh)",
            )

        # (c) Someone else's lease but it is STALE/DEAD → reclaim.
        payload = _build_payload(cwd, pid, ppid, session_id, _now_iso())
        try:
            _write_lease(ls, payload)
        except OSError as exc:
            return "error", None, f"failed to reclaim stale lease: {exc}"
        prior = existing.get("session_id") or existing.get("pid")
        return "reclaimed", payload, f"reclaimed stale lease (was {prior})"

    # No existing lease → fresh acquire.
    payload = _build_payload(cwd, pid, ppid, session_id, _now_iso())
    try:
        _write_lease(ls, payload)
    except OSError as exc:
        return "error", None, f"failed to write lease sidecar: {exc}"

    # Best-effort belt: take the OS FileLock too. Not load-bearing (it releases
    # when this process exits, which for a hook subprocess is immediate) — but
    # harmless. Never fail on it.
    try:
        file_lock = FileLock(str(lk), timeout=0)
        file_lock.acquire()
        _LIVE_LOCKS[str(lk)] = file_lock
    except (Timeout, OSError):
        pass

    return "acquired", payload, f"acquired lease for session {session_id or pid}"


# Process-lifetime registry of best-effort OS locks. Empty per hook-subprocess
# invocation by design — the heartbeat sidecar, not this dict, is the mutex.
_LIVE_LOCKS: dict[str, BaseFileLock] = {}


def release(cwd: Path, pid: int | None = None, force: bool = False) -> tuple[str, str]:
    """Explicit release of the lease (+ best-effort OS lock).

    Without `force`: refuses if the sidecar belongs to a different PID.
    With `force`: removes the sidecar unconditionally.

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
            f"lease sidecar says PID {existing.get('pid')} / session "
            f"{existing.get('session_id')} holds it (current PID {pid}); "
            f"use --force-release to override",
        )

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
    """Read-only inspection. Used by `--status` and the hook BLOCK message.

    `peer_live` reflects the real mutex semantics: a holder with a fresh
    heartbeat and a live ppid. `current_is_holder` is True iff the sidecar
    pid matches `pid` (CLI/legacy identity).
    """
    pid = pid if pid is not None else os.getpid()
    lk = lock_path(cwd)
    ls = lease_path(cwd)
    if lk is None or ls is None:
        return {"in_git_repo": False}

    data = read_lease(cwd)
    if data is None:
        return {
            "in_git_repo": True,
            "lock_path": str(lk),
            "lease_path": str(ls),
            "lease_present": False,
            "current_pid": pid,
        }

    hb = data.get("iso_heartbeat") or data.get("iso_started") or ""
    age_s = _iso_age_seconds(hb)
    sidecar_stale = age_s is not None and age_s >= STALE_HEARTBEAT_SECONDS
    holder_pid = data.get("pid")
    holder_ppid = data.get("ppid")
    return {
        "in_git_repo": True,
        "lock_path": str(lk),
        "lease_path": str(ls),
        "lease_present": True,
        "holder_pid": holder_pid,
        "holder_ppid": holder_ppid,
        "holder_session_id": data.get("session_id"),
        "holder_worktree": data.get("worktree"),
        "holder_branch": data.get("branch"),
        "iso_started": data.get("iso_started"),
        "iso_heartbeat": hb,
        "heartbeat_age_seconds": age_s,
        "sidecar_stale": sidecar_stale,
        "holder_ppid_alive": _pid_is_alive(holder_ppid) if holder_ppid is not None else None,
        # peer_live consults the heartbeat dir too (cwd passed) so --status
        # reflects the real mutex truth even when the holder's ppid reads
        # false-dead on Windows. No session excluded — status is an external
        # observer, so any fresh same-tree beat counts as a live peer.
        "peer_live": _peer_is_live(data, cwd=cwd),
        "fresh_peer_heartbeat": _fresh_peer_heartbeat(cwd),
        "current_pid": pid,
        "current_is_holder": isinstance(holder_pid, int) and holder_pid == pid,
    }


def _format_status_line(snap: dict) -> str:
    if not snap.get("in_git_repo"):
        return "worktree-guard: not in a git repo — no lease"
    if not snap.get("lease_present"):
        return f"worktree-guard: no lease at {snap.get('lease_path')}"
    age = snap.get("heartbeat_age_seconds")
    age_str = f"{age:.0f}s" if isinstance(age, (int, float)) else "unknown"
    parts = [
        f"holder PID {snap.get('holder_pid')}",
        f"session {snap.get('holder_session_id') or '?'}",
        f"ppid {snap.get('holder_ppid')} alive={snap.get('holder_ppid_alive')}",
        f"worktree {snap.get('holder_worktree')!r}",
        f"branch {snap.get('holder_branch') or '?'}",
        f"heartbeat {age_str}",
        f"peer_live={snap.get('peer_live')}",
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
    g.add_argument("--force-release", action="store_true", help="Release lease unconditionally")
    g.add_argument("--acquire", action="store_true", help="Acquire (or refresh) the lease for the current PID")
    parser.add_argument("--cwd", default=".", help="Working directory to resolve git-dir from (default: cwd)")
    parser.add_argument("--session-id", default="", help="Claude session id (identity key)")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of a one-liner")
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
        status_str, lease_data, msg = acquire(cwd, session_id=args.session_id)
        if args.json:
            print(
                json.dumps({"action": "acquire", "status": status_str, "message": msg, "lease": lease_data}, indent=2)
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
