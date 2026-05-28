"""Tests for .claude/hooks/head-flip-guard.py.

Covers the silent-rebase class: branch name unchanged but HEAD SHA moved
(pull --rebase, reset --hard, commit --amend, silent hook amend). Verifies:

- Non-Bash tool: no-op.
- No lock file / corrupted lock / missing fields: fail-safe pass.
- Branch flipped (delegated to branch-flip-guard.py): no double-warn.
- Branch stable AND HEAD stable: silent pass.
- Branch stable AND HEAD moved: emits `additionalContext` JSON to stdout
  carrying both SHAs and the re-resolve instruction.
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
HOOK_PATH = PROJECT_ROOT / ".claude" / "hooks" / "head-flip-guard.py"


def _load_hook() -> ModuleType:
    spec = importlib.util.spec_from_file_location("head_flip_guard", HOOK_PATH)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_event(tool_name: str = "Bash", command: str = "git pull --rebase") -> dict:
    return {"tool_name": tool_name, "tool_input": {"command": command}}


def _init_git(repo: Path, branch: str = "main") -> None:
    subprocess.run(
        ["git", "init", "-q", f"--initial-branch={branch}"],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _write_lock(git_dir: Path, *, branch_at_start: str, head_at_start: str) -> None:
    lock = git_dir / ".claude.pid"
    lock.write_text(
        json.dumps(
            {
                "pid": 999,
                "branch_at_start": branch_at_start,
                "head_at_start": head_at_start,
            }
        ),
        encoding="utf-8",
    )


class TestHeadFlipGuard:
    def test_non_bash_tool_exits_0(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hook = _load_hook()
        event = _make_event(tool_name="Edit")
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(event)))
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 0

    def test_no_lock_file_exits_0(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        hook = _load_hook()
        _init_git(tmp_path)
        event = _make_event()
        monkeypatch.setattr(hook, "_git_dir", lambda: tmp_path / ".git")
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(event)))
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 0

    def test_corrupted_lock_exits_0(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        hook = _load_hook()
        _init_git(tmp_path)
        git_dir = tmp_path / ".git"
        (git_dir / ".claude.pid").write_text("not json", encoding="utf-8")
        event = _make_event()
        monkeypatch.setattr(hook, "_git_dir", lambda: git_dir)
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(event)))
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 0

    def test_missing_head_at_start_exits_0(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pre-Phase-1 lock files only had branch_at_start — fail-safe pass."""
        hook = _load_hook()
        _init_git(tmp_path)
        git_dir = tmp_path / ".git"
        (git_dir / ".claude.pid").write_text(json.dumps({"pid": 999, "branch_at_start": "main"}), encoding="utf-8")
        event = _make_event()
        monkeypatch.setattr(hook, "_git_dir", lambda: git_dir)
        monkeypatch.setattr(hook, "_current_branch", lambda: "main")
        monkeypatch.setattr(hook, "_current_head_sha", lambda: "a" * 40)
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(event)))
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 0

    def test_branch_flipped_defers_to_branch_guard(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """If branch name also changed, branch-flip-guard owns the warn —
        head-flip-guard exits 0 silently with no additionalContext emission."""
        hook = _load_hook()
        _init_git(tmp_path)
        git_dir = tmp_path / ".git"
        _write_lock(git_dir, branch_at_start="main", head_at_start="a" * 40)
        event = _make_event()
        monkeypatch.setattr(hook, "_git_dir", lambda: git_dir)
        monkeypatch.setattr(hook, "_current_branch", lambda: "feature")
        monkeypatch.setattr(hook, "_current_head_sha", lambda: "b" * 40)
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(event)))
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_head_stable_exits_0_silent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        hook = _load_hook()
        _init_git(tmp_path)
        git_dir = tmp_path / ".git"
        sha = "c" * 40
        _write_lock(git_dir, branch_at_start="main", head_at_start=sha)
        event = _make_event()
        monkeypatch.setattr(hook, "_git_dir", lambda: git_dir)
        monkeypatch.setattr(hook, "_current_branch", lambda: "main")
        monkeypatch.setattr(hook, "_current_head_sha", lambda: sha)
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(event)))
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 0
        assert capsys.readouterr().out == ""

    def test_head_moved_emits_additional_context(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """The load-bearing case: pull --rebase / reset --hard / amend.
        Exits 0 (advisory, not blocking) and emits additionalContext JSON
        on stdout carrying both SHAs and the re-resolve instruction."""
        hook = _load_hook()
        _init_git(tmp_path)
        git_dir = tmp_path / ".git"
        start_sha = "7" * 40
        new_sha = "6" * 40
        _write_lock(git_dir, branch_at_start="main", head_at_start=start_sha)
        event = _make_event(command="git pull --rebase origin main")
        monkeypatch.setattr(hook, "_git_dir", lambda: git_dir)
        monkeypatch.setattr(hook, "_current_branch", lambda: "main")
        monkeypatch.setattr(hook, "_current_head_sha", lambda: new_sha)
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(event)))
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 0

        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
        ctx = payload["hookSpecificOutput"]["additionalContext"]
        assert "HEAD SHA changed" in ctx
        assert start_sha[:8] in ctx
        assert new_sha[:8] in ctx
        assert "git rev-parse" in ctx
        assert "feedback_silent_mid_session_pull_rebase" in ctx
