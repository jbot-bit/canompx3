"""
Single-instance lock for live trading bot.

Prevents two bot instances from trading the same account simultaneously.
Uses an exclusive file lock (msvcrt on Windows, fcntl on Unix).
Supports multiple instruments (multi-instrument mode acquires one lock per instrument).
"""

import atexit
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

log = logging.getLogger(__name__)

_LOCK_DIR = Path(tempfile.gettempdir()) / "canompx3"
# Track all acquired locks: {instrument: (fd, path)}
_locks: dict[str, tuple[int, Path]] = {}

# When the orchestrator console is closed with the window [X] instead of
# Ctrl+C, the process is terminated without running atexit cleanup. Per the
# Windows file-locking docs, the OS releases the process's byte-range lock and
# closes its handle on termination, but NOT instantaneously — there is a brief
# window where a stale handle to the orphan lock file lingers, so an immediate
# unlink/open raises PermissionError [WinError 32]. We retry with a short
# backoff to ride out that window instead of crashing the next launch.
_ORPHAN_RETRY_ATTEMPTS = 10
_ORPHAN_RETRY_DELAY_S = 0.3


def _lock_file_for(instrument: str) -> Path:
    _LOCK_DIR.mkdir(parents=True, exist_ok=True)
    return _LOCK_DIR / f"bot_{instrument}.lock"


def _unlink_with_retry(lock_path: Path) -> None:
    """Remove an orphaned lock file, tolerating a briefly-leaked OS handle.

    On a graceful exit the file is already gone; on an X-closed prior session
    the OS may not have released the dead process's handle yet (WinError 32).
    Retry with backoff; if it still cannot be removed after the window, fall
    through and let the subsequent os.open() acquire attempt surface the real
    contention (which itself retries).
    """
    last_exc: OSError | None = None
    for _attempt in range(_ORPHAN_RETRY_ATTEMPTS):
        try:
            lock_path.unlink(missing_ok=True)
            return
        except OSError as exc:  # PermissionError is a subclass
            last_exc = exc
            time.sleep(_ORPHAN_RETRY_DELAY_S)
    log.warning(
        "Orphan lock %s still not removable after %d attempts (%s); "
        "proceeding to acquire — the OS handle should clear shortly.",
        lock_path,
        _ORPHAN_RETRY_ATTEMPTS,
        last_exc,
    )


def acquire_instance_lock(instrument: str) -> None:
    """Acquire exclusive lock for one instrument. Raises SystemExit if held by another process."""
    if instrument in _locks:
        log.info("Lock already held for %s — skipping re-acquire", instrument)
        return

    lock_path = _lock_file_for(instrument)

    # Check for stale PID in existing lock file.
    #
    # X-close handling: closing the orchestrator's console with the window [X]
    # (instead of Ctrl+C) kills the process WITHOUT running atexit cleanup, so
    # the lock file is left orphaned — and on Windows it is frequently left
    # EMPTY (killed after create/truncate but before the PID is written) with
    # the OS-level msvcrt handle briefly leaked. The old code only cleared the
    # lock when it contained a LIVE-vs-dead PID; an empty orphan fell straight
    # through to os.open() and crashed with PermissionError [WinError 32].
    if lock_path.exists():
        try:
            content = lock_path.read_text().strip()
        except OSError:
            content = ""
        if content:
            try:
                old_pid = int(content)
            except ValueError:
                old_pid = None
            if old_pid is not None and is_pid_alive(old_pid):
                log.critical(
                    "Another bot instance is running (PID %d). If stale, delete %s and retry.",
                    old_pid,
                    lock_path,
                )
                print(
                    f"\n!!! CRITICAL: Another bot instance for {instrument} is running (PID {old_pid}).\n"
                    f"    If stale, delete {lock_path} and retry.\n"
                )
                sys.exit(1)
            log.info("Stale lock file from dead/invalid PID %r — removing", content)
        else:
            # Empty orphan from an X-closed (non-graceful) prior session.
            log.info("Empty orphan lock file (likely X-closed prior session) — removing")
        # Remove the orphan, retrying briefly in case Windows has not yet
        # released a leaked handle from the hard-killed process.
        _unlink_with_retry(lock_path)

    # Acquire exclusive lock, then write PID.
    #
    # We already established above that no LIVE holder owns this lock (a live
    # PID exits at sys.exit(1)). So any OSError here is the X-close orphan-handle
    # race, not real contention — retry with backoff to ride out the OS's
    # not-yet-released handle before giving up.
    lock_fd = None
    last_exc: OSError | None = None
    for attempt in range(_ORPHAN_RETRY_ATTEMPTS):
        try:
            lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(lock_fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

            # Truncate and write PID after lock is held (avoids race on read)
            os.ftruncate(lock_fd, 0)
            os.lseek(lock_fd, 0, os.SEEK_SET)
            os.write(lock_fd, str(os.getpid()).encode())
            os.fsync(lock_fd)
            _locks[instrument] = (lock_fd, lock_path)
            log.info("Instance lock acquired: %s (PID %d)", lock_path, os.getpid())
            break

        except OSError as e:
            last_exc = e
            if lock_fd is not None:
                try:
                    os.close(lock_fd)
                except OSError:
                    pass
                lock_fd = None
            if attempt < _ORPHAN_RETRY_ATTEMPTS - 1:
                log.info(
                    "Lock acquire for %s blocked by a stale handle (attempt %d/%d: %s) — retrying after %.1fs",
                    instrument,
                    attempt + 1,
                    _ORPHAN_RETRY_ATTEMPTS,
                    e,
                    _ORPHAN_RETRY_DELAY_S,
                )
                time.sleep(_ORPHAN_RETRY_DELAY_S)
    else:
        log.critical("Failed to acquire instance lock for %s: %s", instrument, last_exc)
        print(
            f"\n!!! CRITICAL: Cannot acquire lock for {instrument} after "
            f"{_ORPHAN_RETRY_ATTEMPTS} attempts.\n    Another instance may be running. "
            f"Error: {last_exc}\n"
        )
        sys.exit(1)

    # Register cleanup only once
    if len(_locks) == 1:
        atexit.register(release_instance_lock)


def release_instance_lock() -> None:
    """Release ALL held locks and clean up files."""
    for instrument in list(_locks.keys()):
        _release_one(instrument)


def _release_one(instrument: str) -> None:
    """Release lock for a single instrument."""
    if instrument not in _locks:
        return
    lock_fd, lock_path = _locks.pop(instrument)
    try:
        if sys.platform == "win32":
            import msvcrt

            try:
                os.lseek(lock_fd, 0, os.SEEK_SET)
                msvcrt.locking(lock_fd, msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        else:
            import fcntl

            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    except OSError:
        pass
    lock_path.unlink(missing_ok=True)


def is_pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes

        # Rationale: Windows quirk — OpenProcess can return a valid handle for
        # an already-exited ("zombie") process whose PID slot has not yet been
        # recycled. We must additionally call GetExitCodeProcess and check for
        # STILL_ACTIVE (259). Without this check, `acquire_instance_lock`
        # refuses to start the bot after a crash until the operator manually
        # deletes the lock file. Both numeric values (259 and 0x1000) are
        # Win32 API constants dictated by the OS — not arbitrary tunables.
        STILL_ACTIVE = 259
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            if not ok:
                return False  # can't query exit code — treat as dead (fail-open for restart)
            return exit_code.value == STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
