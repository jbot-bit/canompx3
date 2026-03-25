"""Tests for single-instance lock."""

import os
from unittest.mock import patch

import pytest

from trading_app.live.instance_lock import (
    _is_pid_alive,
    _lock_file_for,
    acquire_instance_lock,
    release_instance_lock,
)


@pytest.fixture(autouse=True)
def _clean_lock_state():
    """Reset module-level lock state between tests."""
    import trading_app.live.instance_lock as mod

    mod._locks.clear()
    yield
    release_instance_lock()


class TestInstanceLock:
    def test_acquire_and_release(self):
        import trading_app.live.instance_lock as mod

        acquire_instance_lock("TEST_INST")
        lock_path = _lock_file_for("TEST_INST")
        assert lock_path.exists()
        assert "TEST_INST" in mod._locks

        release_instance_lock()
        assert not lock_path.exists()
        assert len(mod._locks) == 0

    def test_stale_pid_cleaned_up(self):
        """Lock file from dead process should be cleaned up."""
        import trading_app.live.instance_lock as mod

        lock_path = _lock_file_for("TEST_STALE")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("99999999")

        with patch(
            "trading_app.live.instance_lock._is_pid_alive", return_value=False
        ):
            acquire_instance_lock("TEST_STALE")

        assert "TEST_STALE" in mod._locks
        release_instance_lock()

    def test_live_pid_blocks(self):
        """Lock file from live process should block acquisition."""
        lock_path = _lock_file_for("TEST_BLOCK")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(str(os.getpid()))

        with pytest.raises(SystemExit) as exc_info:
            with patch(
                "trading_app.live.instance_lock._is_pid_alive", return_value=True
            ):
                acquire_instance_lock("TEST_BLOCK")

        assert exc_info.value.code == 1
        lock_path.unlink(missing_ok=True)

    def test_multi_instrument_locks(self):
        """Multi-instrument: each instrument gets its own lock, all released together."""
        import trading_app.live.instance_lock as mod

        acquire_instance_lock("MGC")
        acquire_instance_lock("MNQ")
        acquire_instance_lock("MES")

        assert len(mod._locks) == 3
        assert _lock_file_for("MGC").exists()
        assert _lock_file_for("MNQ").exists()
        assert _lock_file_for("MES").exists()

        release_instance_lock()

        assert len(mod._locks) == 0
        assert not _lock_file_for("MGC").exists()
        assert not _lock_file_for("MNQ").exists()
        assert not _lock_file_for("MES").exists()

    def test_reacquire_same_instrument_is_noop(self):
        """Acquiring the same instrument twice should not fail."""
        import trading_app.live.instance_lock as mod

        acquire_instance_lock("TEST_DUP")
        acquire_instance_lock("TEST_DUP")  # should not raise
        assert len(mod._locks) == 1

    def test_pid_alive_current_process(self):
        assert _is_pid_alive(os.getpid()) is True

    def test_pid_alive_dead_process(self):
        assert _is_pid_alive(99999999) is False
