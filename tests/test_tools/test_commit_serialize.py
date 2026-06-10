"""Tests for scripts/tools/commit_serialize.py — the pre-commit serializer.

Strategy: each test runs against a synthetic git repo under tmp_path so the
real repo's commit lock is never touched. We invoke the helper as a subprocess
(its real entry contract) and also import it for the unit-level liveness logic.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "tools"))
import commit_serialize as cs  # noqa: E402  # type: ignore[import-not-found]

_HELPER = Path(__file__).resolve().parents[2] / "scripts" / "tools" / "commit_serialize.py"


def _init_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    return tmp_path


# Defense-in-depth: git exports these location vars into a hook's environment. If
# they leak into the helper subprocess, `git rev-parse --git-common-dir` resolves
# to the REAL repo regardless of `cwd`, defeating the per-test tmp_path isolation
# this file promises (module docstring). The pre-commit hook already scrubs
# GIT_DIR/GIT_WORK_TREE/GIT_INDEX_FILE before pytest, so this is NOT the root-cause
# fix for test_outside_git_repo_fails_open (that is the OS-temp dir below — a
# path-location issue env-stripping cannot solve). We strip here anyway so a manual
# `GIT_DIR=… pytest` invocation honors `cwd` exactly like a clean shell would.
_GIT_LOCATION_ENV = (
    "GIT_DIR",
    "GIT_COMMON_DIR",
    "GIT_INDEX_FILE",
    "GIT_WORK_TREE",
    "GIT_PREFIX",
    "GIT_INDEX_VERSION",
)


def _run(arg: str, cwd: Path, *, lock_name: str | None = None, env: dict[str, str] | None = None) -> tuple[int, str]:
    run_env = os.environ.copy()
    for _git_var in _GIT_LOCATION_ENV:
        run_env.pop(_git_var, None)
    if env:
        run_env.update(env)
    cmd = [sys.executable, str(_HELPER), arg]
    if lock_name is not None:
        cmd.append(lock_name)
    r = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        env=run_env,
    )
    return r.returncode, r.stderr.strip()


def _lock(repo: Path, name: str = cs.DEFAULT_LOCK_FILENAME) -> Path:
    return repo / ".git" / name


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


def test_outside_git_repo_fails_open():
    """Not a git repo → acquire fails open (exit 0, allow commit).

    Deliberately NOT parameterized on the `tmp_path` fixture: the pre-commit hook
    runs pytest with --basetemp INSIDE <repo>/.git/pytest-tmp/, so a fixture-based
    tmp_path is literally under the repo's .git dir. Anything under .git/ IS part of
    the repo by git's own rules (no env var changes this — it is path-location, not
    env-inheritance), so `git rev-parse --git-common-dir` succeeds there and the
    "outside any repo" premise silently breaks. We instead mkdtemp under the OS temp
    root (verified outside any repo), so the helper's _git_common_dir() returns None
    and we exercise the real fail-open branch regardless of basetemp location.
    """
    plain = Path(tempfile.mkdtemp(prefix="commit_serialize_outside_repo_"))
    try:
        rc, _ = _run("acquire", plain)
        assert rc == 0
    finally:
        shutil.rmtree(plain, ignore_errors=True)


def test_bad_usage_returns_2(repo: Path):
    r = subprocess.run([sys.executable, str(_HELPER), "bogus"], cwd=repo, capture_output=True, text=True)
    assert r.returncode == 2


def test_pid_alive_false_for_dead_pid():
    assert cs._pid_alive(2147480000) is False


def test_pid_alive_true_for_self():
    assert cs._pid_alive(os.getpid()) is True


# --- push-lock seam (parameterized lock name, reused by .githooks/pre-push) ---

_PUSH_LOCK = ".push-in-progress.lock"


def test_no_arg_uses_default_commit_lock(repo: Path):
    """Regression guard: omitting the lock name keeps the pre-commit default,
    so the existing pre-commit caller is byte-identical."""
    rc, _ = _run("acquire", repo, env={cs.OWNER_ENV: "default-tok"})
    assert rc == 0
    assert _lock(repo).exists()  # .commit-in-progress.lock
    assert not _lock(repo, _PUSH_LOCK).exists()


def test_push_lock_acquires_its_own_file(repo: Path):
    rc, _ = _run("acquire", repo, lock_name=_PUSH_LOCK, env={cs.OWNER_ENV: "push:tok"})
    assert rc == 0
    assert _lock(repo, _PUSH_LOCK).exists()
    assert not _lock(repo).exists()  # commit lock untouched


def test_push_lock_live_peer_blocks_with_push_wording(repo: Path):
    """A live peer holding the push lock blocks, and the message says 'pre-push'."""
    _lock(repo, _PUSH_LOCK).write_text(
        json.dumps({"pid": os.getpid(), "ppid": os.getppid(), "ts": time.time()}),
        encoding="utf-8",
    )
    rc, err = _run("acquire", repo, lock_name=_PUSH_LOCK)
    assert rc == 1
    assert "BLOCKED" in err
    assert "another pre-push is already running" in err


def test_commit_and_push_locks_are_independent(repo: Path):
    """Holding the commit lock must NOT block a push (different ref-write seam),
    and a push release must not free the commit lock."""
    rc, _ = _run("acquire", repo, env={cs.OWNER_ENV: "commit:tok"})
    assert rc == 0
    # Push can still acquire even while a commit lock is held.
    rc, _ = _run("acquire", repo, lock_name=_PUSH_LOCK, env={cs.OWNER_ENV: "push:tok"})
    assert rc == 0
    assert _lock(repo).exists() and _lock(repo, _PUSH_LOCK).exists()
    # Releasing the push lock leaves the commit lock intact.
    rc, _ = _run("release", repo, lock_name=_PUSH_LOCK, env={cs.OWNER_ENV: "push:tok"})
    assert rc == 0
    assert _lock(repo).exists()
    assert not _lock(repo, _PUSH_LOCK).exists()


def test_push_lock_acquire_release_roundtrip_subprocess(repo: Path):
    token = "same-push-shell"
    rc, _ = _run("acquire", repo, lock_name=_PUSH_LOCK, env={cs.OWNER_ENV: token})
    assert rc == 0
    assert _lock(repo, _PUSH_LOCK).exists()
    rc, _ = _run("release", repo, lock_name=_PUSH_LOCK, env={cs.OWNER_ENV: token})
    assert rc == 0
    assert not _lock(repo, _PUSH_LOCK).exists()


# --- native-pid liveness fix (the live-peer block that the helper-pid model missed) ---


def test_live_pid_recorded_when_env_set(repo: Path):
    """When the hook exports its native pid, the lock records it as `live_pid`."""
    rc, _ = _run("acquire", repo, env={cs.LIVE_PID_ENV: str(os.getpid()), cs.OWNER_ENV: "tok"})
    assert rc == 0
    holder = json.loads(_lock(repo).read_text(encoding="utf-8"))
    assert holder["live_pid"] == os.getpid()


def test_live_pid_alive_blocks_even_when_helper_pid_dead(repo: Path):
    """The core fix: a lock whose `live_pid` is ALIVE blocks, even though the
    recorded helper `pid` is dead. Under the old model (liveness on `pid`) this
    would have falsely stolen the lock during the gate window."""
    _lock(repo).write_text(
        json.dumps({"pid": 2147480000, "ppid": 1, "live_pid": os.getpid(), "ts": time.time()}),
        encoding="utf-8",
    )
    rc, err = _run("acquire", repo)
    assert rc == 1
    assert "BLOCKED" in err


def test_live_pid_dead_is_stolen(repo: Path):
    """Once the gate shell exits, its `live_pid` is dead → lock is stealable."""
    _lock(repo).write_text(
        json.dumps({"pid": os.getpid(), "ppid": 1, "live_pid": 2147480000, "ts": time.time()}),
        encoding="utf-8",
    )
    rc, _ = _run("acquire", repo)
    assert rc == 0


def test_legacy_lock_without_live_pid_falls_back_to_pid(repo: Path):
    """Old lock files (no `live_pid`) still gate on `pid` — a live pid blocks."""
    _lock(repo).write_text(
        json.dumps({"pid": os.getpid(), "ppid": os.getppid(), "ts": time.time()}),
        encoding="utf-8",
    )
    rc, err = _run("acquire", repo)
    assert rc == 1
    assert "BLOCKED" in err
