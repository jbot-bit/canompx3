"""Tests for .claude/hooks/branch-flip-guard.py.

Covers:
- No-op when tool_name is not Bash
- No-op when command has no branch-related ops
- No-op when no lock file present (fail-safe)
- No-op when lock file is corrupted (fail-safe)
- No-op when lock branch == current branch
- BLOCK when branch has changed mid-session
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from io import StringIO
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOOK_PATH = PROJECT_ROOT / ".claude" / "hooks" / "branch-flip-guard.py"


def _load_hook() -> ModuleType:
    spec = importlib.util.spec_from_file_location("branch_flip_guard", HOOK_PATH)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_event(tool_name: str = "Bash", command: str = "git checkout main") -> dict:
    return {"tool_name": tool_name, "tool_input": {"command": command}}


def _init_git(repo: Path, branch: str = "main") -> None:
    subprocess.run(["git", "init", "-q", f"--initial-branch={branch}"], cwd=repo, check=True, capture_output=True)


def _write_lock(git_dir: Path, branch_at_start: str) -> None:
    lock = git_dir / ".claude.pid"
    lock.write_text(
        json.dumps({"pid": 999, "branch_at_start": branch_at_start}),
        encoding="utf-8",
    )


class TestBranchFlipGuard:
    def test_non_bash_tool_exits_0(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hook = _load_hook()
        event = _make_event(tool_name="Edit")
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(event)))
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 0

    def test_bash_without_branch_op_exits_0(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hook = _load_hook()
        event = _make_event(command="python pipeline/check_drift.py")
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(event)))
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 0

    def test_no_lock_file_exits_0(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        hook = _load_hook()
        _init_git(tmp_path)
        event = _make_event(command="git checkout feature")

        def fake_git_dir():
            return tmp_path / ".git"

        monkeypatch.setattr(hook, "_git_dir", fake_git_dir)
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(event)))
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 0

    def test_corrupted_lock_exits_0(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        hook = _load_hook()
        _init_git(tmp_path)
        git_dir = tmp_path / ".git"
        (git_dir / ".claude.pid").write_text("not json", encoding="utf-8")
        event = _make_event(command="git checkout feature")

        def fake_git_dir():
            return git_dir

        monkeypatch.setattr(hook, "_git_dir", fake_git_dir)
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(event)))
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 0

    def test_same_branch_exits_0(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        hook = _load_hook()
        _init_git(tmp_path)
        git_dir = tmp_path / ".git"
        _write_lock(git_dir, "main")
        event = _make_event(command="git checkout main")

        def fake_git_dir():
            return git_dir

        def fake_current_branch():
            return "main"

        monkeypatch.setattr(hook, "_git_dir", fake_git_dir)
        monkeypatch.setattr(hook, "_current_branch", fake_current_branch)
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(event)))
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 0

    def test_branch_flip_blocks_with_exit_2(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
        hook = _load_hook()
        _init_git(tmp_path)
        git_dir = tmp_path / ".git"
        _write_lock(git_dir, "main")
        event = _make_event(command="git checkout feature/new-thing")

        def fake_git_dir():
            return git_dir

        def fake_current_branch():
            return "feature/new-thing"

        monkeypatch.setattr(hook, "_git_dir", fake_git_dir)
        monkeypatch.setattr(hook, "_current_branch", fake_current_branch)
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(event)))
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 2
        captured = capsys.readouterr()
        assert "BLOCKED" in captured.err
        assert "main" in captured.err
        assert "feature/new-thing" in captured.err
