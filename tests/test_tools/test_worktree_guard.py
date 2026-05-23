"""Tests for `scripts/tools/worktree_guard.py` (the canonical lease module).

Strategy: every test runs against a synthetic git worktree under `tmp_path`
so the developer's real `<git-dir>/.claude.worktree.lock` is never touched.
We `subprocess.run(["git", "init", "-q"])` once per test (fast on Windows
because tmp_path is per-test).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "tools"))

import worktree_guard as wg  # noqa: E402  # type: ignore[import-not-found]


def _init_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    return tmp_path


def _release_all_for_cwd(cwd: Path) -> None:
    """Belt-and-braces cleanup of any FileLock we still hold for this cwd."""
    lk = wg.lock_path(cwd)
    if lk is None:
        return
    held = wg._LIVE_LOCKS.pop(str(lk), None)
    if held is not None:
        try:
            held.release(force=True)
        except OSError:
            pass


@pytest.fixture
def repo(tmp_path: Path):
    repo_path = _init_repo(tmp_path)
    yield repo_path
    _release_all_for_cwd(repo_path)


def test_skipped_outside_git_repo(tmp_path: Path):
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    status_str, _, _ = wg.acquire(not_a_repo)
    assert status_str == "skipped"


def test_acquire_fresh_writes_sidecar(repo: Path):
    status_str, payload, _ = wg.acquire(repo)
    assert status_str == "acquired"
    assert payload is not None and payload["pid"] == os.getpid()
    ls = wg.lease_path(repo)
    assert ls is not None and ls.exists()
    data = json.loads(ls.read_text(encoding="utf-8"))
    assert data["pid"] == os.getpid()
    assert data["worktree"] == str(repo.resolve())


def test_acquire_then_refresh_same_pid(repo: Path):
    s1, _, _ = wg.acquire(repo)
    s2, payload, _ = wg.acquire(repo)
    assert s1 == "acquired"
    assert s2 == "refreshed"
    assert payload is not None and payload["pid"] == os.getpid()


def test_blocked_by_live_peer(repo: Path):
    # Acquire as "this" PID, then try to acquire as a different PID -> blocked.
    s1, _, _ = wg.acquire(repo, pid=99991)
    # Force the live registry to retain the lock object so the subsequent
    # acquire from a different PID hits the held OS lock.
    assert s1 == "acquired"

    s2, payload, msg = wg.acquire(repo, pid=99992)
    assert s2 == "blocked", msg
    assert payload is not None
    assert payload["pid"] == 99991


def test_status_shows_lease_state(repo: Path):
    wg.acquire(repo)
    snap = wg.status(repo)
    assert snap["in_git_repo"] is True
    assert snap["lease_present"] is True
    assert snap["holder_pid"] == os.getpid()
    assert snap["current_is_holder"] is True


def test_status_no_lease(repo: Path):
    snap = wg.status(repo)
    assert snap["in_git_repo"] is True
    assert snap["lease_present"] is False
    assert snap["is_locked"] is False


def test_release_by_current_pid(repo: Path):
    wg.acquire(repo)
    status_str, msg = wg.release(repo)
    assert status_str == "released", msg
    assert wg.read_lease(repo) is None


def test_release_refuses_other_pid(repo: Path):
    wg.acquire(repo, pid=42424)
    status_str, msg = wg.release(repo, pid=99999, force=False)
    assert status_str == "refused", msg


def test_force_release_always_clears(repo: Path):
    wg.acquire(repo, pid=42424)
    status_str, msg = wg.release(repo, pid=99999, force=True)
    assert status_str == "released", msg
    assert wg.read_lease(repo) is None


def test_release_when_not_held(repo: Path):
    status_str, _ = wg.release(repo)
    assert status_str == "not-held"


def test_sidecar_stale_heartbeat_reported(repo: Path):
    wg.acquire(repo)
    ls = wg.lease_path(repo)
    assert ls is not None
    data = json.loads(ls.read_text(encoding="utf-8"))
    # Backdate the heartbeat past the staleness threshold.
    backdated = datetime.now(UTC) - timedelta(seconds=wg.STALE_HEARTBEAT_SECONDS + 60)
    data["iso_heartbeat"] = backdated.isoformat()
    ls.write_text(json.dumps(data), encoding="utf-8")

    snap = wg.status(repo)
    assert snap["sidecar_stale"] is True
    assert snap["heartbeat_age_seconds"] >= wg.STALE_HEARTBEAT_SECONDS


def test_cli_status_default(repo: Path, capsys: pytest.CaptureFixture):
    rc = wg.main(["--cwd", str(repo), "--status"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "worktree-guard:" in out


def test_cli_acquire_then_status_then_release(repo: Path, capsys: pytest.CaptureFixture):
    rc1 = wg.main(["--cwd", str(repo), "--acquire"])
    rc2 = wg.main(["--cwd", str(repo), "--status"])
    rc3 = wg.main(["--cwd", str(repo), "--force-release"])
    assert rc1 == 0
    assert rc2 == 0
    assert rc3 == 0
    out = capsys.readouterr().out
    assert "acquired" in out or "refreshed" in out
    assert "released" in out


def test_cli_json_output(repo: Path, capsys: pytest.CaptureFixture):
    wg.main(["--cwd", str(repo), "--acquire", "--json"])
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["action"] == "acquire"
    assert data["status"] in ("acquired", "refreshed")


def test_iso_age_seconds_parses_z_suffix():
    # Both `+00:00` and `Z` should parse to the same age.
    base = datetime.now(UTC) - timedelta(seconds=120)
    iso_offset = base.isoformat()
    iso_z = base.isoformat().replace("+00:00", "Z")
    a = wg._iso_age_seconds(iso_offset)
    b = wg._iso_age_seconds(iso_z)
    assert a is not None and b is not None
    assert abs(a - b) < 1


def test_iso_age_seconds_handles_garbage():
    assert wg._iso_age_seconds("") is None
    assert wg._iso_age_seconds("not-a-date") is None


def test_lease_path_isolation_across_worktrees(tmp_path: Path):
    """Two separate git repos must have separate lease paths."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=a, check=True)
    subprocess.run(["git", "init", "-q"], cwd=b, check=True)
    pa = wg.lease_path(a)
    pb = wg.lease_path(b)
    assert pa is not None and pb is not None
    assert pa != pb
