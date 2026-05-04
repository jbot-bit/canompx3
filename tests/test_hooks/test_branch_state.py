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
import subprocess
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


class TestCurrentBranch:
    def test_inside_repo_returns_branch(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = _load_module()
        _init_git(tmp_path, branch="main")
        monkeypatch.chdir(tmp_path)
        result = mod.current_branch()
        assert result == "main"

    def test_outside_repo_returns_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = _load_module()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
        result = mod.current_branch()
        assert result is None


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
