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
from pathlib import Path

log = logging.getLogger(__name__)

_LOCK_DIR = Path(tempfile.gettempdir()) / "canompx3"
# Track all acquired locks: {instrument: (fd, path)}
_locks: dict[str, tuple[int, Path]] = {}


def _lock_file_for(instrument: str) -> Path:
    _LOCK_DIR.mkdir(parents=True, exist_ok=True)
    return _LOCK_DIR / f"bot_{instrument}.lock"


def acquire_instance_lock(instrument: str) -> None:
    """Acquire exclusive lock for one instrument. Raises SystemExit if held by another process."""
    if instrument in _locks:
        log.info("Lock already held for %s — skipping re-acquire", instrument)
        return

    lock_path = _lock_file_for(instrument)

    # Check for stale PID in existing lock file
    if lock_path.exists():
        try:
            content = lock_path.read_text().strip()
            if content:
                old_pid = int(content)
                if _is_pid_alive(old_pid):
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
                else:
                    log.info("Stale lock file from dead PID %d — removing", old_pid)
                    lock_path.unlink(missing_ok=True)
        except (ValueError, OSError):
            lock_path.unlink(missing_ok=True)

    # Acquire exclusive lock, then write PID
    lock_fd = None
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

    except OSError as e:
        if lock_fd is not None:
            os.close(lock_fd)
        log.critical("Failed to acquire instance lock for %s: %s", instrument, e)
        print(
            f"\n!!! CRITICAL: Cannot acquire lock for {instrument}.\n    Another instance may be running. Error: {e}\n"
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


def _is_pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
