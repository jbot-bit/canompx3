"""Tests for pipeline.db_lock — advisory PID-based write lock."""

import json
import os

import pytest

from pipeline.db_lock import PipelineLock, PipelineLockError


@pytest.fixture
def lock_dir(tmp_path):
    """Create a fake DB file + lock path for testing."""
    db_path = tmp_path / "test.db"
    db_path.write_text("fake db")
    return db_path


def test_acquire_and_release(lock_dir):
    """Lock can be acquired and released cleanly."""
    lock = PipelineLock("test_script", db_path=lock_dir)
    lock.acquire()
    assert lock._held
    assert lock.lock_path.exists()

    # Lock file contains our PID
    info = json.loads(lock.lock_path.read_text())
    assert info["pid"] == os.getpid()
    assert info["script"] == "test_script"

    lock.release()
    assert not lock._held
    assert not lock.lock_path.exists()


def test_context_manager(lock_dir):
    """Context manager acquires on enter, releases on exit."""
    with PipelineLock("ctx_test", db_path=lock_dir) as lock:
        assert lock._held
        assert lock.lock_path.exists()
    # After exit
    assert not lock._held
    assert not lock.lock_path.exists()


def test_context_manager_releases_on_exception(lock_dir):
    """Lock is released even if the body raises an exception."""
    with pytest.raises(ValueError):
        with PipelineLock("error_test", db_path=lock_dir) as lock:
            assert lock._held
            raise ValueError("boom")
    assert not lock._held
    assert not lock.lock_path.exists()


def test_stale_lock_reclaimed(lock_dir):
    """A lock held by a dead PID is automatically reclaimed."""
    lock_path = lock_dir.with_suffix(".db.lock")
    # Write a lock file with a definitely-dead PID
    lock_path.write_text(
        json.dumps(
            {
                "pid": 999999999,
                "script": "dead_process",
                "started": "2026-01-01T00:00:00+00:00",
            }
        )
    )

    # Should be able to acquire despite existing lock file
    lock = PipelineLock("reclaim_test", db_path=lock_dir)
    lock.acquire()
    assert lock._held

    info = json.loads(lock.lock_path.read_text())
    assert info["pid"] == os.getpid()
    assert info["script"] == "reclaim_test"
    lock.release()


def test_corrupt_lock_reclaimed(lock_dir):
    """A corrupt lock file is treated as stale and reclaimed."""
    lock_path = lock_dir.with_suffix(".db.lock")
    lock_path.write_text("not valid json {{{")

    lock = PipelineLock("corrupt_test", db_path=lock_dir)
    lock.acquire()
    assert lock._held
    lock.release()


def test_live_lock_blocks(lock_dir):
    """A lock held by our own PID (alive) blocks a second acquire with short timeout."""
    lock_path = lock_dir.with_suffix(".db.lock")
    # Write a lock file with OUR PID (definitely alive)
    lock_path.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "script": "other_script",
                "started": "2026-01-01T00:00:00+00:00",
            }
        )
    )

    lock = PipelineLock("blocked_test", db_path=lock_dir, timeout=2)
    with pytest.raises(PipelineLockError, match="Cannot acquire pipeline lock"):
        lock.acquire()


def test_is_locked_class_method(lock_dir):
    """is_locked returns correct status."""
    assert not PipelineLock.is_locked(db_path=lock_dir)

    with PipelineLock("status_test", db_path=lock_dir):
        assert PipelineLock.is_locked(db_path=lock_dir)

    assert not PipelineLock.is_locked(db_path=lock_dir)


def test_lock_info_class_method(lock_dir):
    """lock_info returns dict when locked, None when not."""
    assert PipelineLock.lock_info(db_path=lock_dir) is None

    with PipelineLock("info_test", db_path=lock_dir):
        info = PipelineLock.lock_info(db_path=lock_dir)
        assert info is not None
        assert info["script"] == "info_test"
        assert info["pid"] == os.getpid()

    assert PipelineLock.lock_info(db_path=lock_dir) is None
