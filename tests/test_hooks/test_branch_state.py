"""Tests for .claude/hooks/_branch_state.py — shared branch-flip helpers.

Covers the three primitives:

  - git_dir()           : Path | None
  - current_branch()    : str | None
  - branch_at_start(d)  : str | None

Each must be fail-safe (return None on any error) per
`.claude/rules/branch-flip-protection.md` § "Fail-safe guarantee".
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / ".claude" / "hooks" / "_branch_state.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_branch_state", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _init_git(repo: Path, branch: str = "main") -> None:
    subprocess.run(
        ["git", "init", "-q", f"--initial-branch={branch}"],
        cwd=repo,
        check=True,
        capture_output=True,
    )


class TestGitDir:
    def test_inside_repo_returns_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = _load_module()
        _init_git(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = mod.git_dir()
        assert result is not None
        assert result.name == ".git"
        assert result.exists()

    def test_outside_repo_returns_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = _load_module()
        # Walk up to the drive root via a non-git tmp dir. `git rev-parse`
        # walks parents; on the worktree's drive there is always a parent
        # `.git`, so simulate the no-repo case via env override.
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
        # tmp_path itself has no .git; with the ceiling set, rev-parse fails.
        result = mod.git_dir()
        assert result is None


class TestGitDirCwd:
    """`cwd` selects the worktree to inspect — the F4-A scoping fix."""

    def test_cwd_selects_repo_regardless_of_process_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = _load_module()
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git(repo)
        outside = tmp_path / "outside"
        outside.mkdir()
        # Process cwd is OUTSIDE the repo; cwd= points INTO it.
        monkeypatch.chdir(outside)
        monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
        result = mod.git_dir(str(repo))
        assert result is not None
        assert result.name == ".git"
        assert result.exists()

    def test_cwd_none_falls_back_to_process_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = _load_module()
        _init_git(tmp_path)
        monkeypatch.chdir(tmp_path)
        # Default (no cwd) == historical behavior == process cwd.
        assert mod.git_dir() == mod.git_dir(str(tmp_path))


class TestCurrentBranch:
    def test_inside_repo_returns_branch(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = _load_module()
        _init_git(tmp_path, branch="main")
        monkeypatch.chdir(tmp_path)
        result = mod.current_branch()
        assert result == "main"

    def test_cwd_selects_branch_of_target_repo(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = _load_module()
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git(repo, branch="feature/x")
        outside = tmp_path / "outside"
        outside.mkdir()
        monkeypatch.chdir(outside)
        monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
        assert mod.current_branch(str(repo)) == "feature/x"

    def test_outside_repo_returns_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = _load_module()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
        result = mod.current_branch()
        assert result is None


class TestInvokingCwd:
    def test_reads_event_cwd(self, tmp_path: Path) -> None:
        mod = _load_module()
        result = mod.invoking_cwd({"cwd": str(tmp_path)})
        assert result == str(tmp_path.resolve())

    def test_missing_cwd_returns_none(self) -> None:
        mod = _load_module()
        assert mod.invoking_cwd({}) is None

    def test_empty_cwd_returns_none(self) -> None:
        mod = _load_module()
        assert mod.invoking_cwd({"cwd": "   "}) is None


class TestBranchAtStart:
    def test_reads_field_from_lock(self, tmp_path: Path) -> None:
        mod = _load_module()
        git_dir = tmp_path
        (git_dir / ".claude.pid").write_text(
            json.dumps({"pid": 999, "branch_at_start": "feature/foo"}),
            encoding="utf-8",
        )
        assert mod.branch_at_start(git_dir) == "feature/foo"

    def test_missing_lock_returns_none(self, tmp_path: Path) -> None:
        mod = _load_module()
        assert mod.branch_at_start(tmp_path) is None

    def test_corrupt_json_returns_none(self, tmp_path: Path) -> None:
        mod = _load_module()
        (tmp_path / ".claude.pid").write_text("not json {{{", encoding="utf-8")
        assert mod.branch_at_start(tmp_path) is None

    def test_missing_field_returns_none(self, tmp_path: Path) -> None:
        mod = _load_module()
        (tmp_path / ".claude.pid").write_text(
            json.dumps({"pid": 999}),
            encoding="utf-8",
        )
        assert mod.branch_at_start(tmp_path) is None

    def test_empty_field_returns_none(self, tmp_path: Path) -> None:
        mod = _load_module()
        (tmp_path / ".claude.pid").write_text(
            json.dumps({"pid": 999, "branch_at_start": ""}),
            encoding="utf-8",
        )
        assert mod.branch_at_start(tmp_path) is None


def _write_lock(git_dir: Path, **fields: object) -> None:
    """Write a `.claude.pid` lock with the given fields."""
    (git_dir / ".claude.pid").write_text(json.dumps(fields), encoding="utf-8")


def _iso_hours_ago(hours: float) -> str:
    return (datetime.now(UTC) - timedelta(hours=hours)).isoformat()


class TestCorpseLockIgnored:
    """A lock held by a PROVEN-dead PID >=12h old is a corpse — both readers
    must return None so the flip-guards fall silent instead of advising on a
    dead session's stale SHA. Origin: n=1 2026-06-07, a 124.8h corpse held by
    dead PID 13716 nagged for ~6 days.
    """

    def test_dead_and_old_silences_head_reader(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = _load_module()
        monkeypatch.setattr(mod, "_pid_is_alive", lambda pid: False, raising=True)
        _write_lock(
            tmp_path,
            pid=13716,
            iso_started=_iso_hours_ago(125),
            head_at_start="deadbeef" * 5,
            branch_at_start="main",
        )
        assert mod.head_at_start(tmp_path) is None
        assert mod.branch_at_start(tmp_path) is None

    def test_live_pid_keeps_guard(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = _load_module()
        monkeypatch.setattr(mod, "_pid_is_alive", lambda pid: True, raising=True)
        _write_lock(
            tmp_path,
            pid=4242,
            iso_started=_iso_hours_ago(125),  # old, but the holder is ALIVE
            head_at_start="abc123" + "0" * 34,
            branch_at_start="feature/x",
        )
        assert mod.head_at_start(tmp_path) == "abc123" + "0" * 34
        assert mod.branch_at_start(tmp_path) == "feature/x"

    def test_dead_but_fresh_keeps_guard(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dead PID but within the 12h freshness window (e.g. /clear-and-
        restart of the same shell) must KEEP its guard."""
        mod = _load_module()
        monkeypatch.setattr(mod, "_pid_is_alive", lambda pid: False, raising=True)
        _write_lock(
            tmp_path,
            pid=13716,
            iso_started=_iso_hours_ago(2),  # only 2h old
            head_at_start="cafe" * 10,
            branch_at_start="main",
        )
        assert mod.head_at_start(tmp_path) == "cafe" * 10
        assert mod.branch_at_start(tmp_path) == "main"

    def test_dead_but_no_timestamp_keeps_guard(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Age unknown (no iso_started) must NEVER silence a guard — this is
        the backward-compat property that keeps the legacy `pid:999` tests
        (which have no timestamp) returning their values."""
        mod = _load_module()
        monkeypatch.setattr(mod, "_pid_is_alive", lambda pid: False, raising=True)
        _write_lock(
            tmp_path,
            pid=999,
            head_at_start="f00d" * 10,
            branch_at_start="main",
        )
        assert mod.head_at_start(tmp_path) == "f00d" * 10
        assert mod.branch_at_start(tmp_path) == "main"

    def test_dead_and_unparseable_timestamp_keeps_guard(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A garbage timestamp can't prove staleness -> not a corpse."""
        mod = _load_module()
        monkeypatch.setattr(mod, "_pid_is_alive", lambda pid: False, raising=True)
        _write_lock(
            tmp_path,
            pid=13716,
            iso_started="not-a-date",
            head_at_start="dead" * 10,
            branch_at_start="main",
        )
        assert mod.head_at_start(tmp_path) == "dead" * 10
        assert mod.branch_at_start(tmp_path) == "main"


class TestPidIsAliveSelf:
    def test_self_pid_is_alive(self) -> None:
        mod = _load_module()
        assert mod._pid_is_alive(os.getpid()) is True

    def test_invalid_pid_is_dead(self) -> None:
        mod = _load_module()
        assert mod._pid_is_alive(0) is False
        assert mod._pid_is_alive(-1) is False


class TestCanonicalLivenessDelegation:
    """`_pid_is_alive` MUST delegate to the canonical worktree_guard probe.

    The original draft re-encoded `os.kill(pid, 0)`, which on Windows returns
    no error for a dead-but-recycled PID — so the live 124.8h corpse read as
    ALIVE and the fix did nothing on the operator's actual platform. Delegating
    to worktree_guard's OpenProcess probe is the only Windows-correct oracle
    (institutional-rigor §4). These tests pin the delegation contract.
    """

    def test_delegates_to_canonical_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = _load_module()
        # Canonical says ALIVE -> _pid_is_alive must return True regardless of
        # what the os.kill fallback would say for this (real, dead) PID.
        monkeypatch.setattr(mod, "_canonical_pid_is_alive", lambda pid: True)
        assert mod._pid_is_alive(999_999_321) is True
        # Canonical says DEAD -> _pid_is_alive must return False.
        monkeypatch.setattr(mod, "_canonical_pid_is_alive", lambda pid: False)
        assert mod._pid_is_alive(os.getpid()) is False

    def test_falls_back_when_canonical_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Canonical import failure -> conservative os.kill fallback. The
        fallback must still report THIS live process as alive (never silence a
        guard on an unproven-dead lock in a degraded environment)."""
        mod = _load_module()
        monkeypatch.setattr(mod, "_canonical_pid_is_alive", lambda pid: None)
        assert mod._pid_is_alive(os.getpid()) is True

    def test_canonical_helper_returns_none_on_missing_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If the canonical guard path doesn't exist, the delegate helper
        returns None (signalling the caller to fall back), never raises."""
        mod = _load_module()
        monkeypatch.setattr(mod, "_GUARD_PATH", Path("/nonexistent/worktree_guard.py"))
        assert mod._canonical_pid_is_alive(os.getpid()) is None
