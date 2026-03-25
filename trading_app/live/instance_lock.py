"""
Single-instance lock for live trading bot.

Prevents two bot instances from trading the same account simultaneously.
Uses an exclusive file lock (msvcrt on Windows, fcntl on Unix).
"""

import atexit
import logging
import os
import sys
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

_LOCK_DIR = Path(tempfile.gettempdir()) / "canompx3"
_lock_fd: int | None = None
_lock_path: Path | None = None


def _lock_file_for(instrument: str) -> Path:
    _LOCK_DIR.mkdir(parents=True, exist_ok=True)
    return _LOCK_DIR / f"bot_{instrument}.lock"


def acquire_instance_lock(instrument: str) -> None:
    """Acquire exclusive lock. Raises SystemExit if another instance is running."""
    global _lock_fd, _lock_path

    _lock_path = _lock_file_for(instrument)

    # Check for stale PID in existing lock file
    if _lock_path.exists():
        try:
            content = _lock_path.read_text().strip()
            if content:
                old_pid = int(content)
                if _is_pid_alive(old_pid):
                    log.critical(
                        "Another bot instance is running (PID %d). "
                        "If stale, delete %s and retry.",
                        old_pid,
                        _lock_path,
                    )
                    print(
                        f"\n!!! CRITICAL: Another bot instance for {instrument} is running (PID {old_pid}).\n"
                        f"    If stale, delete {_lock_path} and retry.\n"
                    )
                    sys.exit(1)
                else:
                    log.info("Stale lock file from dead PID %d — removing", old_pid)
                    _lock_path.unlink(missing_ok=True)
        except (ValueError, OSError):
            # Corrupt lock file — remove and proceed
            _lock_path.unlink(missing_ok=True)

    # Write our PID and acquire exclusive lock
    try:
        _lock_fd = os.open(str(_lock_path), os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(_lock_fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

        os.write(_lock_fd, str(os.getpid()).encode())
        os.fsync(_lock_fd)
        log.info("Instance lock acquired: %s (PID %d)", _lock_path, os.getpid())

    except (OSError, IOError) as e:
        if _lock_fd is not None:
            os.close(_lock_fd)
            _lock_fd = None
        log.critical("Failed to acquire instance lock: %s", e)
        print(
            f"\n!!! CRITICAL: Cannot acquire lock for {instrument}.\n"
            f"    Another instance may be running. Error: {e}\n"
        )
        sys.exit(1)

    # Register cleanup
    atexit.register(release_instance_lock)


def release_instance_lock() -> None:
    """Release lock and clean up file."""
    global _lock_fd, _lock_path

    if _lock_fd is not None:
        try:
            if sys.platform == "win32":
                import msvcrt

                try:
                    msvcrt.locking(_lock_fd, msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                import fcntl

                fcntl.flock(_lock_fd, fcntl.LOCK_UN)
            os.close(_lock_fd)
        except OSError:
            pass
        _lock_fd = None

    if _lock_path is not None:
        _lock_path.unlink(missing_ok=True)
        _lock_path = None


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
