"""Tests for single-instance lock."""

import os
import sys
from unittest.mock import patch

import pytest

from trading_app.live.instance_lock import (
    _lock_file_for,
    acquire_instance_lock,
    is_pid_alive,
    release_instance_lock,
)


def _open_and_lock(path) -> int:
    """Acquire the SAME byte-range lock a live holder would hold; leave it held."""
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR)
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


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

        with patch("trading_app.live.instance_lock.is_pid_alive", return_value=False):
            acquire_instance_lock("TEST_STALE")

        assert "TEST_STALE" in mod._locks
        release_instance_lock()

    def test_live_pid_blocks(self):
        """Lock file from live process should block acquisition."""
        lock_path = _lock_file_for("TEST_BLOCK")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(str(os.getpid()))

        with pytest.raises(SystemExit) as exc_info:
            with patch("trading_app.live.instance_lock.is_pid_alive", return_value=True):
                acquire_instance_lock("TEST_BLOCK")

        assert exc_info.value.code == 1
        lock_path.unlink(missing_ok=True)

    def test_multi_instrument_locks(self):
        """Multi-instrument: each instrument gets its own lock, all released together."""
        import trading_app.live.instance_lock as mod

        acquire_instance_lock("TEST_MGC")
        acquire_instance_lock("TEST_MNQ")
        acquire_instance_lock("TEST_MES")

        assert len(mod._locks) == 3
        assert _lock_file_for("TEST_MGC").exists()
        assert _lock_file_for("TEST_MNQ").exists()
        assert _lock_file_for("TEST_MES").exists()

        release_instance_lock()

        assert len(mod._locks) == 0
        assert not _lock_file_for("TEST_MGC").exists()
        assert not _lock_file_for("TEST_MNQ").exists()
        assert not _lock_file_for("TEST_MES").exists()

    def test_reacquire_same_instrument_is_noop(self):
        """Acquiring the same instrument twice should not fail."""
        import trading_app.live.instance_lock as mod

        acquire_instance_lock("TEST_DUP")
        acquire_instance_lock("TEST_DUP")  # should not raise
        assert len(mod._locks) == 1

    def test_pid_alive_current_process(self):
        assert is_pid_alive(os.getpid()) is True

    def test_pid_alive_dead_process(self):
        assert is_pid_alive(99999999) is False

    def test_pid_alive_zombie_process_windows(self):
        """Regression test: Windows zombie PIDs (OpenProcess returns handle but
        GetExitCodeProcess != STILL_ACTIVE) must be treated as dead.

        Before the GetExitCodeProcess check, a bot crash followed by a restart
        within seconds-to-minutes could fail because Windows had not yet recycled
        the PID slot of the crashed process. The stale lock would report "alive"
        and block acquisition."""
        import sys

        if sys.platform != "win32":
            pytest.skip("Windows-specific zombie PID handling")

        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        # Scan a range of PIDs to find one where OpenProcess returns a handle
        # but the exit code is not STILL_ACTIVE — a zombie we can observe.
        # If none found in the scan window, the test skips (can't fabricate the
        # OS state reliably). On a busy dev box this reliably finds several.
        zombie_pid = None
        for pid in range(1000, 20000, 4):
            h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if h:
                exit_code = ctypes.c_ulong()
                ok = ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(exit_code))
                ctypes.windll.kernel32.CloseHandle(h)
                if ok and exit_code.value != 259:  # STILL_ACTIVE
                    zombie_pid = pid
                    break
        if zombie_pid is None:
            pytest.skip("No zombie PID found in scan window (OS state)")
        assert is_pid_alive(zombie_pid) is False, (
            f"PID {zombie_pid} is a zombie (exit_code != STILL_ACTIVE) "
            "but is_pid_alive returned True. Would block bot restart after crash."
        )

    def test_empty_dead_orphan_acquired_in_place(self):
        """Capital review 2026-06-07: an EMPTY lock file with no live holder is a
        dead X-closed orphan → acquired in place (no unlink), acquire succeeds."""
        import trading_app.live.instance_lock as mod

        lock_path = _lock_file_for("TEST_EMPTY_DEAD")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("")  # empty, no holder

        acquire_instance_lock("TEST_EMPTY_DEAD")  # must not raise
        assert "TEST_EMPTY_DEAD" in mod._locks
        # Lock is held in place (not unlinked): the file still exists and is the
        # one we acquired. We can't read its bytes here — the held byte-range
        # lock makes a cross-process read raise PermissionError on Windows, which
        # itself confirms the lock is held in place.
        assert lock_path.exists()

    def test_empty_lock_with_live_holder_refuses_start(self):
        """Capital review 2026-06-07: an EMPTY lock file held by a LIVE process
        must REFUSE acquisition (anti double-instance) — the old code removed it
        unconditionally, which would let a second bot trade the same account.

        The fix does NOT probe-and-release-then-reacquire (that left a TOCTOU
        window caught in adversarial audit). The empty branch falls straight
        through to the acquire loop, whose msvcrt/flock acquire IS the mutex.

        Exercises the REAL OS lock (msvcrt/fcntl), not a mock — and the holder is
        held for the WHOLE acquire attempt, so a probe-release-reacquire design
        would wrongly succeed here. This proves there is no release window."""
        import trading_app.live.instance_lock as mod

        lock_path = _lock_file_for("TEST_EMPTY_LIVE")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        holder_fd = _open_and_lock(lock_path)  # live holder, held throughout
        # Shrink retry budget so the test doesn't wait 10×0.3s on the live block.
        with patch.object(mod, "_ORPHAN_RETRY_ATTEMPTS", 2), patch.object(mod, "_ORPHAN_RETRY_DELAY_S", 0.01):
            try:
                with pytest.raises(SystemExit) as exc:
                    acquire_instance_lock("TEST_EMPTY_LIVE")
                assert exc.value.code == 1
                # The guard must NOT have stolen the lock from the live holder.
                assert "TEST_EMPTY_LIVE" not in mod._locks
            finally:
                os.close(holder_fd)
                lock_path.unlink(missing_ok=True)
