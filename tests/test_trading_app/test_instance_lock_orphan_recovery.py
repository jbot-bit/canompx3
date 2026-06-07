"""Regression tests: instance-lock recovery from X-close orphan lock files.

Closing the orchestrator console with the window [X] (instead of Ctrl+C)
terminates the process without atexit cleanup, leaving an orphaned lock file —
frequently EMPTY (killed after create/truncate, before the PID write). The
previous logic only cleared a lock that held a dead-vs-live PID, so an empty
orphan fell through to os.open() and crashed the next launch with
PermissionError [WinError 32]. These tests lock in the self-healing behaviour
while proving a genuinely-live holder is still refused.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

from trading_app.live import instance_lock
from trading_app.live.instance_lock import (
    acquire_instance_lock,
    is_lock_file_active_or_ambiguous,
    release_instance_lock,
)

_LOCK_DIR = Path(tempfile.gettempdir()) / "canompx3"


def _lock_path(instrument: str) -> Path:
    return _LOCK_DIR / f"bot_{instrument}.lock"


@pytest.fixture(autouse=True)
def _cleanup():
    """Ensure no held locks / stray files leak between tests."""
    yield
    release_instance_lock()
    for name in ("ORPHANEMPTY", "ORPHANDEAD", "ORPHANLIVE", "ORPHANLOCKED", "ORPHANLOCKEDDEAD"):
        _lock_path(name).unlink(missing_ok=True)


def test_acquires_over_empty_orphan_from_x_close():
    """Empty orphan (the X-close signature) is cleared; acquire succeeds."""
    _LOCK_DIR.mkdir(parents=True, exist_ok=True)
    orphan = _lock_path("ORPHANEMPTY")
    orphan.write_text("")  # X-close: file exists, no PID written

    acquire_instance_lock("ORPHANEMPTY")

    assert "ORPHANEMPTY" in instance_lock._locks
    release_instance_lock()
    assert not orphan.exists()


def test_acquires_over_dead_pid_orphan():
    """A non-empty orphan whose PID is dead is cleared; acquire succeeds."""
    _LOCK_DIR.mkdir(parents=True, exist_ok=True)
    orphan = _lock_path("ORPHANDEAD")
    orphan.write_text("999999")  # implausible, dead PID

    acquire_instance_lock("ORPHANDEAD")

    assert "ORPHANDEAD" in instance_lock._locks


def test_live_holder_is_still_refused():
    """A lock owned by a LIVE pid must still abort (double-trade guard intact)."""
    _LOCK_DIR.mkdir(parents=True, exist_ok=True)
    orphan = _lock_path("ORPHANLIVE")
    orphan.write_text(str(os.getpid()))  # our own pid = provably alive

    with pytest.raises(SystemExit) as exc:
        acquire_instance_lock("ORPHANLIVE")

    assert exc.value.code == 1
    assert "ORPHANLIVE" not in instance_lock._locks


def test_empty_file_with_live_os_lock_is_refused(monkeypatch):
    """An empty-but-locked file may be a live process before PID write; fail closed."""
    if sys.platform == "win32":
        pytest.skip("fcntl race regression is Unix-specific")
    import fcntl

    monkeypatch.setattr(instance_lock, "_ORPHAN_RETRY_ATTEMPTS", 2)
    monkeypatch.setattr(instance_lock, "_ORPHAN_RETRY_DELAY_S", 0)
    _LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = _lock_path("ORPHANLOCKED")
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before_inode = lock_path.stat().st_ino

        with pytest.raises(SystemExit) as exc:
            acquire_instance_lock("ORPHANLOCKED")

        assert exc.value.code == 1
        assert "ORPHANLOCKED" not in instance_lock._locks
        assert lock_path.stat().st_ino == before_inode
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        lock_path.unlink(missing_ok=True)


def test_observer_treats_os_locked_dead_pid_as_active(monkeypatch):
    """Observer must respect OS lock even while PID text is stale/dead."""
    if sys.platform == "win32":
        pytest.skip("fcntl race regression is Unix-specific")
    import fcntl

    monkeypatch.setattr(instance_lock, "is_pid_alive", lambda _pid: False)
    _LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = _lock_path("ORPHANLOCKEDDEAD")
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    try:
        os.write(fd, b"999999")
        os.fsync(fd)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

        locked, pid, reason = is_lock_file_active_or_ambiguous(lock_path)

        assert locked is True
        assert pid == 999999
        assert reason == "os-locked"
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        lock_path.unlink(missing_ok=True)
