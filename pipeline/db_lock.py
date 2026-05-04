"""
Advisory DB write lock for gold.db.

PID-based file lock that prevents concurrent pipeline write operations.
Uses a JSON lock file alongside gold.db — no external dependencies.

Usage:
    from pipeline.db_lock import PipelineLock

    with PipelineLock("outcome_builder"):
        # exclusive write access to gold.db
        ...

    # Or check lock status:
    if PipelineLock.is_locked():
        print("DB is locked by another process")
"""

import ctypes
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from pipeline.paths import GOLD_DB_PATH


class PipelineLockError(RuntimeError):
    """Raised when the pipeline lock cannot be acquired."""


def is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running (Windows-compatible)."""
    if sys.platform == "win32":
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


class PipelineLock:
    """Advisory file lock for pipeline write operations.

    Creates a .lock file alongside gold.db containing:
        {"pid": <int>, "script": <str>, "started": <ISO timestamp>}

    Stale locks (PID no longer alive) are automatically reclaimed.
    If the lock is held by a live process, waits up to `timeout` seconds
    before raising PipelineLockError.
    """

    def __init__(
        self,
        script_name: str,
        db_path: Path | None = None,
        timeout: int = 30,
    ):
        self.script_name = script_name
        self.db_path = db_path or GOLD_DB_PATH
        self.lock_path = self.db_path.with_suffix(".db.lock")
        self.timeout = timeout
        self._held = False

    def acquire(self) -> None:
        """Acquire the lock. Raises PipelineLockError if unable."""
        deadline = time.monotonic() + self.timeout

        while True:
            # Try to acquire
            if self._try_acquire():
                self._held = True
                return

            # Check timeout
            if time.monotonic() >= deadline:
                holder = self._read_lock_info()
                holder_desc = (
                    f"pid={holder.get('pid')}, script={holder.get('script')}, started={holder.get('started')}"
                    if holder
                    else "unknown"
                )
                raise PipelineLockError(
                    f"Cannot acquire pipeline lock after {self.timeout}s. "
                    f"Held by: {holder_desc}. "
                    f"Lock file: {self.lock_path}"
                )

            time.sleep(1)

    def release(self) -> None:
        """Release the lock by deleting the lock file."""
        if self._held and self.lock_path.exists():
            try:
                # Verify we own it before deleting
                info = self._read_lock_info()
                if info and info.get("pid") == os.getpid():
                    self.lock_path.unlink()
            except OSError:
                pass  # Lock file already gone — fine
        self._held = False

    def _try_acquire(self) -> bool:
        """Attempt to create the lock file. Returns True if successful."""
        if self.lock_path.exists():
            info = self._read_lock_info()
            if info is None:
                # Corrupt lock file — reclaim
                self._remove_stale_lock()
            elif not is_pid_alive(info.get("pid", -1)):
                # Stale lock — process is dead
                print(
                    f"[LOCK] Reclaiming stale lock from dead process "
                    f"(pid={info.get('pid')}, script={info.get('script')})",
                    file=sys.stderr,
                )
                self._remove_stale_lock()
            else:
                # Lock is held by a live process
                return False

        # Write our lock
        try:
            lock_data = {
                "pid": os.getpid(),
                "script": self.script_name,
                "started": datetime.now(UTC).isoformat(),
            }
            # Use exclusive create mode to avoid race conditions
            fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, json.dumps(lock_data, indent=2).encode())
            finally:
                os.close(fd)
            return True
        except FileExistsError:
            # Another process beat us to it
            return False
        except OSError as e:
            raise PipelineLockError(f"Cannot create lock file: {e}") from e

    def _read_lock_info(self) -> dict | None:
        """Read and parse the lock file. Returns None if corrupt or missing."""
        try:
            return json.loads(self.lock_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def _remove_stale_lock(self) -> None:
        """Remove a stale lock file."""
        try:
            self.lock_path.unlink()
        except OSError:
            pass

    @classmethod
    def is_locked(cls, db_path: Path | None = None) -> bool:
        """Check if the DB is currently locked by a live process."""
        lock_path = (db_path or GOLD_DB_PATH).with_suffix(".db.lock")
        if not lock_path.exists():
            return False
        try:
            info = json.loads(lock_path.read_text())
            return is_pid_alive(info.get("pid", -1))
        except (json.JSONDecodeError, OSError):
            return False

    @classmethod
    def lock_info(cls, db_path: Path | None = None) -> dict | None:
        """Return lock info dict if locked, None otherwise."""
        lock_path = (db_path or GOLD_DB_PATH).with_suffix(".db.lock")
        if not lock_path.exists():
            return None
        try:
            info = json.loads(lock_path.read_text())
            if is_pid_alive(info.get("pid", -1)):
                return info
        except (json.JSONDecodeError, OSError):
            pass
        return None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False
