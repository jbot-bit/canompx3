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

        def fake_git_dir(cwd=None):
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

        def fake_git_dir(cwd=None):
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

        def fake_git_dir(cwd=None):
            return git_dir

        def fake_current_branch(cwd=None):
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

        def fake_git_dir(cwd=None):
            return git_dir

        def fake_current_branch(cwd=None):
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


class TestCwdScoping:
    """F4-A regression: the guard must inspect the worktree the Bash command
    actually ran in (payload `cwd`), NOT the hook process's cwd (always the
    main checkout per the hardcoded settings.json command). Before the fix, a
    peer flipping the main checkout's branch false-fired ~20x/session against
    every isolated worktree.
    """

    def _make_worktree(self, tmp_path: Path) -> tuple[Path, Path, ModuleType]:
        """Build a real repo + a linked worktree on a different branch."""
        hook = _load_hook()
        main = tmp_path / "main"
        main.mkdir()
        _init_git(main, branch="main")
        (main / "f.txt").write_text("x", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=main, check=True, capture_output=True)
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
            cwd=main,
            check=True,
            capture_output=True,
        )
        wt = tmp_path / "wt"
        subprocess.run(
            ["git", "worktree", "add", "-q", "-b", "session/work", str(wt)],
            cwd=main,
            check=True,
            capture_output=True,
        )
        # Simulate a peer flipping the MAIN checkout to another branch.
        subprocess.run(
            ["git", "checkout", "-q", "-b", "codex/peer-flip"],
            cwd=main,
            check=True,
            capture_output=True,
        )
        return main, wt, hook

    def test_no_false_fire_in_isolated_worktree(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        main, wt, hook = self._make_worktree(tmp_path)
        wt_gitdir = hook._branch_state.git_dir(str(wt))
        assert wt_gitdir is not None
        # Lock in the WORKTREE records its OWN branch — no real flip here.
        (wt_gitdir / ".claude.pid").write_text(
            json.dumps({"pid": 1, "branch_at_start": "session/work"}),
            encoding="utf-8",
        )
        # Hook process cwd is the MAIN checkout (the false-fire trigger).
        monkeypatch.chdir(main)
        event = {
            "tool_name": "Bash",
            "tool_input": {"command": "git worktree list"},
            "cwd": str(wt),
        }
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(event)))
        with pytest.raises(SystemExit) as exc:
            hook.main()
        # cwd-scoped: worktree branch == lock branch -> exit 0 (no false fire).
        assert exc.value.code == 0

    def test_real_flip_in_worktree_still_blocks(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
        main, wt, hook = self._make_worktree(tmp_path)
        wt_gitdir = hook._branch_state.git_dir(str(wt))
        assert wt_gitdir is not None
        # Lock says the worktree STARTED on a different branch -> a REAL flip.
        (wt_gitdir / ".claude.pid").write_text(
            json.dumps({"pid": 1, "branch_at_start": "session/original"}),
            encoding="utf-8",
        )
        monkeypatch.chdir(main)
        event = {
            "tool_name": "Bash",
            "tool_input": {"command": "git checkout session/work"},
            "cwd": str(wt),
        }
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(event)))
        with pytest.raises(SystemExit) as exc:
            hook.main()
        # Real same-worktree flip is still caught.
        assert exc.value.code == 2
        assert "session/original" in capsys.readouterr().err
