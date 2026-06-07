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
# closes its handle on termination, but not instantaneously. Retry briefly so
# stale handles from hard-killed processes do not crash the next launch.
_ORPHAN_RETRY_ATTEMPTS = 10
_ORPHAN_RETRY_DELAY_S = 0.3


class _LockBusy(OSError):
    """Raised when the existing lock file is currently OS-locked."""


class _AmbiguousLock(RuntimeError):
    """Raised when a lock file exists but does not prove a dead owner."""


class _LiveLock(RuntimeError):
    """Raised when a lock file names a live owner PID."""

    def __init__(self, pid: int) -> None:
        super().__init__(f"live PID {pid}")
        self.pid = pid


def _lock_file_for(instrument: str) -> Path:
    _LOCK_DIR.mkdir(parents=True, exist_ok=True)
    return _LOCK_DIR / f"bot_{instrument}.lock"


def _try_lock_fd(lock_fd: int) -> None:
    """Try to acquire the platform lock on an already-open lock file."""
    if sys.platform == "win32":
        import msvcrt

        os.lseek(lock_fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(lock_fd, msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise _LockBusy(*exc.args) from exc
    else:
        import fcntl

        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise _LockBusy(*exc.args) from exc


def _unlock_fd(lock_fd: int) -> None:
    """Release the platform lock for a held lock fd."""
    if sys.platform == "win32":
        import msvcrt

        os.lseek(lock_fd, 0, os.SEEK_SET)
        msvcrt.locking(lock_fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(lock_fd, fcntl.LOCK_UN)


def _parse_lock_pid(content: str) -> int | None:
    """Parse PID text from a lock file, returning None for absent/invalid text."""
    content = content.strip()
    if not content:
        return None
    try:
        return int(content)
    except ValueError:
        return None


def _read_lock_text_from_fd(lock_fd: int) -> str:
    """Read lock metadata from the exact fd whose OS lock was tested/held."""
    os.lseek(lock_fd, 0, os.SEEK_SET)
    return os.read(lock_fd, 128).decode("utf-8", errors="replace").strip()


def _read_lock_pid_from_fd(lock_fd: int) -> int | None:
    """Return the PID encoded on an already-open lock fd, or None."""
    return _parse_lock_pid(_read_lock_text_from_fd(lock_fd))


def _read_lock_pid(lock_path: Path) -> int | None:
    """Return the PID encoded in a lock file path, or None when absent/invalid."""
    try:
        return _parse_lock_pid(lock_path.read_text(encoding="utf-8"))
    except OSError:
        return None


def _classify_lock_content(lock_path: Path, content: str) -> None:
    """Fail closed on live/ambiguous lock metadata content."""
    if not content:
        raise _AmbiguousLock(f"empty lock file: {lock_path}")
    pid = _parse_lock_pid(content)
    if pid is None:
        raise _AmbiguousLock(f"invalid lock PID in {lock_path}: {content!r}")
    if is_pid_alive(pid):
        raise _LiveLock(pid)


def is_lock_file_active_or_ambiguous(lock_path: Path) -> tuple[bool, int | None, str]:
    """Read-only status for lock observers.

    Returns (active_or_ambiguous, pid, reason). Dead-PID locks return False.
    Live PID, OS-locked, empty, invalid, and unreadable locks return True to
    preserve the fail-closed single-writer invariant during startup races. The
    observer briefly tries the platform lock and immediately releases it when
    successful; a busy OS lock is active even if the PID text is stale because
    a live process may be between lock acquisition and PID rewrite.
    """
    lock_fd: int | None = None
    try:
        lock_fd = os.open(str(lock_path), os.O_RDWR)
        try:
            _try_lock_fd(lock_fd)
        except _LockBusy:
            pid = _read_lock_pid(lock_path)
            return True, pid, "os-locked"

        try:
            content = _read_lock_text_from_fd(lock_fd)
            _classify_lock_content(lock_path, content)
        except _LiveLock as exc:
            return True, exc.pid, "live"
        except _AmbiguousLock as exc:
            return True, None, str(exc)
        finally:
            try:
                _unlock_fd(lock_fd)
            except OSError:
                pass
        return False, _parse_lock_pid(content), "dead"
    except FileNotFoundError:
        return False, None, "absent"
    except OSError as exc:
        return True, None, f"unreadable lock file: {lock_path} ({exc})"
    finally:
        _close_fd_quietly(lock_fd)


def acquire_instance_lock(instrument: str) -> None:
    """Acquire exclusive lock for one instrument. Raises SystemExit if held by another process."""
    if instrument in _locks:
        log.info("Lock already held for %s - skipping re-acquire", instrument)
        return

    lock_path = _lock_file_for(instrument)

    # Correctness rule: ownership is proved by the OS lock, not by lock-file
    # contents. Empty files are possible both for true X-close orphans and for a
    # live process in the tiny window after create/lock but before PID write. So
    # never unlink/replace an existing path before first acquiring that same
    # file's OS lock. On Unix, unlinking a locked empty file creates a new inode
    # and can let a second bot acquire a separate lock for the same instrument.
    lock_fd: int | None = None
    last_exc: BaseException | None = None
    for attempt in range(_ORPHAN_RETRY_ATTEMPTS):
        try:
            lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
            _try_lock_fd(lock_fd)

            # From this point onward we own the OS lock on this exact file, so a
            # stale/empty/invalid PID is safe to replace. A live PID here means
            # stale metadata from a different live process that no longer holds
            # the OS lock; fail closed anyway rather than risk double trading.
            old_pid = _read_lock_pid_from_fd(lock_fd)
            if old_pid is not None and is_pid_alive(old_pid):
                try:
                    _unlock_fd(lock_fd)
                except OSError:
                    pass
                os.close(lock_fd)
                lock_fd = None
                _abort_live_holder(instrument, lock_path, old_pid)

            os.ftruncate(lock_fd, 0)
            os.lseek(lock_fd, 0, os.SEEK_SET)
            os.write(lock_fd, str(os.getpid()).encode())
            os.fsync(lock_fd)
            _locks[instrument] = (lock_fd, lock_path)
            log.info("Instance lock acquired: %s (PID %d)", lock_path, os.getpid())
            break

        except _LockBusy as exc:
            last_exc = exc
            _close_fd_quietly(lock_fd)
            lock_fd = None
            pid = _read_lock_pid(lock_path)
            if pid is not None and is_pid_alive(pid):
                _abort_live_holder(instrument, lock_path, pid)
            if attempt < _ORPHAN_RETRY_ATTEMPTS - 1:
                log.info(
                    "Lock acquire for %s found an active/ambiguous lock (attempt %d/%d: %s) - retrying after %.1fs",
                    instrument,
                    attempt + 1,
                    _ORPHAN_RETRY_ATTEMPTS,
                    exc,
                    _ORPHAN_RETRY_DELAY_S,
                )
                time.sleep(_ORPHAN_RETRY_DELAY_S)
        except OSError as exc:
            last_exc = exc
            _close_fd_quietly(lock_fd)
            lock_fd = None
            pid = _read_lock_pid(lock_path)
            if pid is not None and is_pid_alive(pid):
                _abort_live_holder(instrument, lock_path, pid)
            if attempt < _ORPHAN_RETRY_ATTEMPTS - 1:
                log.info(
                    "Lock acquire for %s blocked by a stale handle (attempt %d/%d: %s) - retrying after %.1fs",
                    instrument,
                    attempt + 1,
                    _ORPHAN_RETRY_ATTEMPTS,
                    exc,
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


def _close_fd_quietly(lock_fd: int | None) -> None:
    if lock_fd is None:
        return
    try:
        os.close(lock_fd)
    except OSError:
        pass


def _abort_live_holder(instrument: str, lock_path: Path, pid: int) -> None:
    log.critical(
        "Another bot instance is running (PID %d). If stale, delete %s and retry.",
        pid,
        lock_path,
    )
    print(
        f"\n!!! CRITICAL: Another bot instance for {instrument} is running (PID {pid}).\n"
        f"    If stale, delete {lock_path} and retry.\n"
    )
    sys.exit(1)


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
        try:
            _unlock_fd(lock_fd)
        except OSError:
            pass
        os.close(lock_fd)
    except OSError:
        pass
    lock_path.unlink(missing_ok=True)


def is_pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes

        # Rationale: Windows quirk - OpenProcess can return a valid handle for
        # an already-exited ("zombie") process whose PID slot has not yet been
        # recycled. We must additionally call GetExitCodeProcess and check for
        # STILL_ACTIVE (259). Without this check, `acquire_instance_lock`
        # refuses to start the bot after a crash until the operator manually
        # deletes the lock file. Both numeric values (259 and 0x1000) are
        # Win32 API constants dictated by the OS, not arbitrary tunables.
        STILL_ACTIVE = 259
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            if not ok:
                return False
            return exit_code.value == STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
