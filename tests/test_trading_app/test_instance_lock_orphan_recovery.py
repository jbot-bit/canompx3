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
import tempfile
from pathlib import Path

import pytest

from trading_app.live import instance_lock
from trading_app.live.instance_lock import (
    acquire_instance_lock,
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
    for name in ("ORPHANEMPTY", "ORPHANDEAD", "ORPHANLIVE"):
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
