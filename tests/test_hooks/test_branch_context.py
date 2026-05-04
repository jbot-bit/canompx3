"""Tests for .claude/hooks/branch-context.py.

Covers:
- No-op when target file is outside canonical prefixes
- No-op when current branch is not research/* or session/*
- No-op when BRANCH_CONTEXT_OVERRIDE=1
- No-op when branch cannot be determined (fail-open)
- BLOCK when research/* branch edits pipeline/, trading_app/, or scripts/tools/
- ALLOW scripts/research/ from research/* (narrowed prefix)
"""

from __future__ import annotations

import importlib.util
import json
from io import StringIO
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOOK_PATH = PROJECT_ROOT / ".claude" / "hooks" / "branch-context.py"


def _load_hook() -> ModuleType:
    spec = importlib.util.spec_from_file_location("branch_context", HOOK_PATH)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _event(file_path: str, tool: str = "Edit") -> dict:
    return {"tool_name": tool, "tool_input": {"file_path": file_path}}


class TestBranchContext:
    def test_non_canonical_file_exits_0(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hook = _load_hook()
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(_event("docs/foo.md"))))
        monkeypatch.setattr(hook, "_current_branch", lambda: "research/edge-x")
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 0

    def test_non_research_branch_exits_0(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hook = _load_hook()
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(_event("pipeline/dst.py"))))
        monkeypatch.setattr(hook, "_current_branch", lambda: "feature/foo")
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 0

    def test_override_env_exits_0(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hook = _load_hook()
        monkeypatch.setenv("BRANCH_CONTEXT_OVERRIDE", "1")
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(_event("pipeline/dst.py"))))
        monkeypatch.setattr(hook, "_current_branch", lambda: "research/edge-x")
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 0

    def test_unknown_branch_fail_open(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hook = _load_hook()
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(_event("pipeline/dst.py"))))
        monkeypatch.setattr(hook, "_current_branch", lambda: None)
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 0

    @pytest.mark.parametrize("path", ["pipeline/dst.py", "trading_app/config.py", "scripts/tools/foo.py"])
    def test_research_branch_blocks_canonical(self, monkeypatch: pytest.MonkeyPatch, path: str) -> None:
        hook = _load_hook()
        monkeypatch.delenv("BRANCH_CONTEXT_OVERRIDE", raising=False)
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(_event(path))))
        monkeypatch.setattr(hook, "_current_branch", lambda: "research/edge-x")
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 2

    def test_research_branch_allows_scripts_research(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """scripts/research/ is legitimate research-branch work — must NOT block."""
        hook = _load_hook()
        monkeypatch.delenv("BRANCH_CONTEXT_OVERRIDE", raising=False)
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(_event("scripts/research/foo.py"))))
        monkeypatch.setattr(hook, "_current_branch", lambda: "research/edge-x")
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 0

    def test_session_branch_blocks_canonical(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hook = _load_hook()
        monkeypatch.delenv("BRANCH_CONTEXT_OVERRIDE", raising=False)
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(_event("trading_app/config.py"))))
        monkeypatch.setattr(hook, "_current_branch", lambda: "session/josh-foo")
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 2
