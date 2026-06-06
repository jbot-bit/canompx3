"""Tests for scripts/tools/commit_serialize.py — the pre-commit serializer.

Strategy: each test runs against a synthetic git repo under tmp_path so the
real repo's commit lock is never touched. We invoke the helper as a subprocess
(its real entry contract) and also import it for the unit-level liveness logic.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "tools"))
import commit_serialize as cs  # noqa: E402  # type: ignore[import-not-found]

_HELPER = Path(__file__).resolve().parents[2] / "scripts" / "tools" / "commit_serialize.py"


def _init_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    return tmp_path


def _run(arg: str, cwd: Path, *, env: dict[str, str] | None = None) -> tuple[int, str]:
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    r = subprocess.run(
        [sys.executable, str(_HELPER), arg],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=run_env,
    )
    return r.returncode, r.stderr.strip()


def _lock(repo: Path) -> Path:
    return repo / ".git" / cs.LOCK_FILENAME


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return _init_repo(tmp_path)


def test_clean_acquire_writes_lock(repo: Path):
    token = "test-precommit-owner"
    rc, _ = _run("acquire", repo, env={cs.OWNER_ENV: token})
    assert rc == 0
    assert _lock(repo).exists()
    holder = json.loads(_lock(repo).read_text(encoding="utf-8"))
    assert isinstance(holder["pid"], int) and holder["pid"] > 0
    assert isinstance(holder["ts"], (int, float))
    assert holder["owner"] == token


def test_live_peer_blocks(repo: Path):
    """A fresh lock held by a LIVE pid (this test process) blocks acquire."""
    _lock(repo).write_text(
        json.dumps({"pid": os.getpid(), "ppid": os.getppid(), "ts": time.time()}),
        encoding="utf-8",
    )
    rc, err = _run("acquire", repo)
    assert rc == 1
    assert "BLOCKED" in err
    assert "another pre-commit is already running" in err


def test_stale_lock_is_stolen(repo: Path):
    """A lock older than the stale floor is abandoned → stolen (acquire ok)."""
    _lock(repo).write_text(
        json.dumps({"pid": os.getpid(), "ppid": 1, "ts": time.time() - (cs._STALE_LOCK_SECS + 60)}),
        encoding="utf-8",
    )
    rc, _ = _run("acquire", repo)
    assert rc == 0


def test_dead_pid_lock_is_stolen(repo: Path):
    """A fresh lock whose holder pid is provably dead is stolen."""
    _lock(repo).write_text(
        json.dumps({"pid": 2147480000, "ppid": 1, "ts": time.time()}),
        encoding="utf-8",
    )
    rc, _ = _run("acquire", repo)
    assert rc == 0


def test_garbage_lock_is_stolen(repo: Path):
    """An unreadable/garbage lock must not wedge commits — treated as stale."""
    _lock(repo).write_text("}{ not json", encoding="utf-8")
    rc, _ = _run("acquire", repo)
    assert rc == 0


def test_release_only_removes_own_lock(repo: Path):
    """release() must never clobber a peer's lock — only one matching our pid."""
    _lock(repo).write_text(json.dumps({"pid": 12345, "ppid": 1, "ts": time.time()}), encoding="utf-8")
    rc, _ = _run("release", repo)
    assert rc == 0
    assert _lock(repo).exists()  # not ours → left intact


def test_release_only_removes_matching_owner_token(repo: Path):
    _lock(repo).write_text(
        json.dumps({"pid": 12345, "ppid": os.getpid(), "owner": "peer-token", "ts": time.time()}),
        encoding="utf-8",
    )
    rc, _ = _run("release", repo, env={cs.OWNER_ENV: "our-token"})
    assert rc == 0
    assert _lock(repo).exists()


def test_acquire_then_release_roundtrip_same_process(repo: Path):
    """In-process acquire then release removes the lock (pid matches)."""
    os.chdir(repo)
    try:
        assert cs.acquire() == 0
        assert _lock(repo).exists()
        assert cs.release() == 0
        assert not _lock(repo).exists()
    finally:
        os.chdir(Path(__file__).resolve().parents[2])


def test_acquire_then_release_roundtrip_subprocess_same_parent(repo: Path):
    """Hook contract: acquire/release are separate helpers under one pre-commit owner."""
    token = "same-precommit-shell"
    rc, _ = _run("acquire", repo, env={cs.OWNER_ENV: token})
    assert rc == 0
    assert _lock(repo).exists()

    rc, _ = _run("release", repo, env={cs.OWNER_ENV: token})
    assert rc == 0
    assert not _lock(repo).exists()


def test_outside_git_repo_fails_open(tmp_path: Path):
    """Not a git repo → acquire fails open (exit 0, allow commit)."""
    plain = tmp_path / "plain"
    plain.mkdir()
    rc, _ = _run("acquire", plain)
    assert rc == 0


def test_bad_usage_returns_2(repo: Path):
    r = subprocess.run([sys.executable, str(_HELPER), "bogus"], cwd=repo, capture_output=True, text=True)
    assert r.returncode == 2


def test_pid_alive_false_for_dead_pid():
    assert cs._pid_alive(2147480000) is False


def test_pid_alive_true_for_self():
    assert cs._pid_alive(os.getpid()) is True
