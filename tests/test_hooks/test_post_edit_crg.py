"""Tests for _crg_update() in .claude/hooks/post-edit-pipeline.py.

Covers (F3 contract):
- Calls subprocess for paths in declared prefix set (pipeline/, trading_app/,
  scripts/, research/, tests/)
- Skips for paths outside the prefix set (docs/, .claude/, etc.)
- Fail-silent on subprocess failure
- Bounded timeout (5s)
- Reachability bug regression: _crg_update must be called BEFORE the
  pipeline|trading_app early-exit so non-pipeline/non-trading_app declared
  prefixes (scripts/, research/, tests/) actually fire.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOOK_PATH = PROJECT_ROOT / ".claude" / "hooks" / "post-edit-pipeline.py"


def _load_hook() -> ModuleType:
    spec = importlib.util.spec_from_file_location("post_edit_pipeline", HOOK_PATH)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestCrgUpdate:
    @pytest.mark.parametrize(
        "path",
        [
            "pipeline/dst.py",
            "trading_app/config.py",
            "scripts/research/foo.py",
            "scripts/tools/foo.py",
            "research/edges/foo.py",
            "tests/test_foo.py",
        ],
    )
    def test_calls_subprocess_for_declared_prefixes(self, path: str) -> None:
        hook = _load_hook()
        with patch.object(subprocess, "run") as mock_run:
            hook._crg_update(path)
            assert mock_run.called
            args, kwargs = mock_run.call_args
            cmd = args[0]
            assert cmd[0] == "code-review-graph"
            assert "update" in cmd
            assert kwargs.get("timeout") == 5
            assert kwargs.get("check") is False

    @pytest.mark.parametrize(
        "path",
        ["docs/spec.md", ".claude/hooks/foo.py", "HANDOFF.md", "README.md"],
    )
    def test_skips_non_declared_prefixes(self, path: str) -> None:
        hook = _load_hook()
        with patch.object(subprocess, "run") as mock_run:
            hook._crg_update(path)
            assert not mock_run.called

    def test_fail_silent_on_subprocess_error(self) -> None:
        hook = _load_hook()
        with patch.object(subprocess, "run", side_effect=subprocess.TimeoutExpired("crg", 5)):
            # Must not raise
            hook._crg_update("pipeline/dst.py")

    def test_fail_silent_on_missing_binary(self) -> None:
        hook = _load_hook()
        with patch.object(subprocess, "run", side_effect=FileNotFoundError):
            hook._crg_update("pipeline/dst.py")

    def test_called_before_early_exit_for_scripts_path(self) -> None:
        """Regression: scripts/ files must trigger CRG update even though the
        post-edit pipeline early-exits for non-pipeline/non-trading_app files.
        """
        hook = _load_hook()
        import io
        import json
        import sys

        event = {"tool_input": {"file_path": "scripts/research/foo.py"}}
        with patch.object(subprocess, "run") as mock_run:
            with patch.object(sys, "stdin", io.StringIO(json.dumps(event))):
                with pytest.raises(SystemExit):
                    hook.main()
            assert mock_run.called, "CRG update did not fire for scripts/ — early-exit ordering regressed"
