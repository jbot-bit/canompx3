"""Tests for .claude/hooks/mcp-git-guard.py.

The Q3 mitigation hook — fires PostToolUse on `mcp__git__.*` tool calls
and BLOCKs branch-mutating ops on a flipped branch (since the pre-commit
backstop is bypassed by `repo.index.commit()` in upstream mcp-server-git).

Two layers of test coverage:

  1. **In-process** (importlib.util.module_from_spec + monkeypatch) —
     fast, mirrors the existing `test_branch_flip_guard.py` style. Most
     coverage lives here.

  2. **Subprocess** — invokes the real hook script with a real JSON event
     on stdin via `subprocess.run`. Exercises the actual contract Claude
     Code uses (stdin -> exit code + stderr). One subprocess test for
     each verdict path (pass / block / fail-safe) is enough — the
     in-process tests cover the matrix.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from io import StringIO
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOOK_PATH = PROJECT_ROOT / ".claude" / "hooks" / "mcp-git-guard.py"


def _load_hook() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mcp_git_guard", HOOK_PATH)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_event(tool_name: str, tool_input: dict | None = None) -> dict:
    return {"tool_name": tool_name, "tool_input": tool_input or {}}


def _init_git(repo: Path, branch: str = "main") -> None:
    subprocess.run(
        ["git", "init", "-q", f"--initial-branch={branch}"],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _write_lock(git_dir: Path, branch_at_start: str) -> None:
    (git_dir / ".claude.pid").write_text(
        json.dumps({"pid": 999, "branch_at_start": branch_at_start}),
        encoding="utf-8",
    )


def _patch_branch_state(
    monkeypatch: pytest.MonkeyPatch,
    hook: ModuleType,
    *,
    git_dir: Path | None,
    current: str | None,
) -> None:
    """Override the shared-module helpers on the hook's `_branch_state` ref."""
    monkeypatch.setattr(hook._branch_state, "git_dir", lambda: git_dir)
    monkeypatch.setattr(hook._branch_state, "current_branch", lambda: current)


class TestToolNameMatching:
    """The hook only fires on mcp__git__.* and only blocks on writes."""

    def test_bash_tool_exits_0(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hook = _load_hook()
        event = _make_event("Bash", {"command": "git checkout main"})
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(event)))
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 0

    def test_other_mcp_tool_exits_0(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hook = _load_hook()
        event = _make_event("mcp__sequential__thinking")
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(event)))
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 0

    def test_read_only_mcp_git_on_drift_exits_0(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Read-only MCP git tools never block, even on a flipped branch."""
        hook = _load_hook()
        _init_git(tmp_path)
        git_dir = tmp_path / ".git"
        _write_lock(git_dir, "main")
        _patch_branch_state(monkeypatch, hook, git_dir=git_dir, current="other")

        for tool in (
            "mcp__git__git_status",
            "mcp__git__git_diff",
            "mcp__git__git_diff_staged",
            "mcp__git__git_diff_unstaged",
            "mcp__git__git_log",
            "mcp__git__git_show",
        ):
            event = _make_event(tool)
            monkeypatch.setattr("sys.stdin", StringIO(json.dumps(event)))
            with pytest.raises(SystemExit) as exc:
                hook.main()
            assert exc.value.code == 0, f"read-only {tool} should exit 0 on drift"


class TestBlockOnDrift:
    """All write tools BLOCK on drift."""

    @pytest.mark.parametrize(
        "tool",
        [
            "mcp__git__git_commit",
            "mcp__git__git_checkout",
            "mcp__git__git_create_branch",
            "mcp__git__git_reset",
            "mcp__git__git_add",
            "mcp__git__git_branch",
        ],
    )
    def test_write_tool_blocks(
        self,
        tool: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        hook = _load_hook()
        _init_git(tmp_path)
        git_dir = tmp_path / ".git"
        _write_lock(git_dir, "main")
        _patch_branch_state(monkeypatch, hook, git_dir=git_dir, current="feature/x")

        event = _make_event(tool)
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(event)))
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 2
        captured = capsys.readouterr()
        assert "BLOCKED" in captured.err
        assert tool in captured.err
        assert "main" in captured.err
        assert "feature/x" in captured.err


class TestPassOnSameBranch:
    @pytest.mark.parametrize(
        "tool",
        [
            "mcp__git__git_commit",
            "mcp__git__git_checkout",
            "mcp__git__git_status",
        ],
    )
    def test_same_branch_exits_0(
        self,
        tool: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        hook = _load_hook()
        _init_git(tmp_path)
        git_dir = tmp_path / ".git"
        _write_lock(git_dir, "main")
        _patch_branch_state(monkeypatch, hook, git_dir=git_dir, current="main")

        event = _make_event(tool)
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(event)))
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 0


class TestFailSafe:
    def test_no_lock_file_exits_0(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        hook = _load_hook()
        _init_git(tmp_path)
        git_dir = tmp_path / ".git"
        _patch_branch_state(monkeypatch, hook, git_dir=git_dir, current="feature/x")

        event = _make_event("mcp__git__git_commit")
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(event)))
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 0

    def test_corrupt_lock_exits_0(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        hook = _load_hook()
        _init_git(tmp_path)
        git_dir = tmp_path / ".git"
        (git_dir / ".claude.pid").write_text("not json", encoding="utf-8")
        _patch_branch_state(monkeypatch, hook, git_dir=git_dir, current="feature/x")

        event = _make_event("mcp__git__git_commit")
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(event)))
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 0

    def test_not_in_repo_exits_0(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hook = _load_hook()
        _patch_branch_state(monkeypatch, hook, git_dir=None, current=None)

        event = _make_event("mcp__git__git_commit")
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(event)))
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 0

    def test_malformed_event_exits_0(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hook = _load_hook()
        monkeypatch.setattr("sys.stdin", StringIO("not json"))
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 0


class TestSubprocessContract:
    """Real subprocess invocation — proves the actual stdin->exit-code
    contract Claude Code uses. One canonical case per verdict path."""

    def _run(self, event: dict, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(HOOK_PATH)],
            input=json.dumps(event),
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=10,
        )

    def test_subprocess_block_on_write_drift(self, tmp_path: Path) -> None:
        _init_git(tmp_path)
        git_dir = tmp_path / ".git"
        _write_lock(git_dir, "main")
        # Flip the branch in the test repo so the hook sees a real drift.
        subprocess.run(
            ["git", "checkout", "-q", "-b", "feature/flipped"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        event = _make_event("mcp__git__git_commit")
        result = self._run(event, cwd=tmp_path)
        assert result.returncode == 2, result.stderr
        assert "BLOCKED" in result.stderr
        assert "mcp__git__git_commit" in result.stderr

    def test_subprocess_pass_on_read_only_drift(self, tmp_path: Path) -> None:
        _init_git(tmp_path)
        git_dir = tmp_path / ".git"
        _write_lock(git_dir, "main")
        subprocess.run(
            ["git", "checkout", "-q", "-b", "feature/flipped"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        event = _make_event("mcp__git__git_status")
        result = self._run(event, cwd=tmp_path)
        assert result.returncode == 0

    def test_subprocess_pass_on_no_lock(self, tmp_path: Path) -> None:
        _init_git(tmp_path)
        # No lock file written — fail-safe path.
        event = _make_event("mcp__git__git_commit")
        result = self._run(event, cwd=tmp_path)
        assert result.returncode == 0

    def test_subprocess_pass_on_wrong_matcher(self, tmp_path: Path) -> None:
        # Even with a flipped branch, a non-mcp__git__ tool must pass.
        _init_git(tmp_path)
        git_dir = tmp_path / ".git"
        _write_lock(git_dir, "main")
        subprocess.run(
            ["git", "checkout", "-q", "-b", "feature/flipped"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        event = _make_event("Bash", {"command": "git checkout main"})
        result = self._run(event, cwd=tmp_path)
        assert result.returncode == 0
