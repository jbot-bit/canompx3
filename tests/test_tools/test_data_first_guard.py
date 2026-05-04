from __future__ import annotations

import importlib.util
import io
import sys
from contextlib import redirect_stderr
from datetime import UTC, datetime
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[2]
    path = root / ".claude" / "hooks" / "data-first-guard.py"
    spec = importlib.util.spec_from_file_location("data_first_guard", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules.pop("data_first_guard", None)
    spec.loader.exec_module(module)
    return module


def test_pretooluse_query_resets_consecutive_reads(tmp_path: Path) -> None:
    """A Bash query command must reset the read counter. Replaces the
    deleted UserPromptSubmit-branch tests — that path now lives in
    `.claude/hooks/tests/test_prompt_broker.py` (broker merge 2026-04-27).
    """
    module = _load_module()
    module.STATE_FILE = tmp_path / "data-first.json"
    module.save_state(
        {
            "investigation_mode": True,
            "consecutive_reads": 5,
            "last_updated": datetime.now(UTC).isoformat(),
        }
    )

    try:
        module.handle_pre_tool_use(
            {
                "tool_name": "Bash",
                "tool_input": {"command": "python -c 'print(1)'"},
            }
        )
    except SystemExit as exc:
        assert exc.code == 0

    state = module.load_state()
    assert state["consecutive_reads"] == 0


def test_pretooluse_blocks_after_threshold_in_investigation_mode(tmp_path: Path) -> None:
    module = _load_module()
    module.STATE_FILE = tmp_path / "data-first.json"
    module.save_state(
        {
            "investigation_mode": True,
            "consecutive_reads": module.BLOCK_THRESHOLD - 1,
            "last_updated": datetime.now(UTC).isoformat(),
        }
    )

    stderr = io.StringIO()
    try:
        with redirect_stderr(stderr):
            module.handle_pre_tool_use({"tool_name": "Read"})
    except SystemExit as exc:
        assert exc.code == 2  # BLOCK
    assert "DATA FIRST BLOCK" in stderr.getvalue()


def test_pretooluse_warns_at_warn_threshold(tmp_path: Path) -> None:
    module = _load_module()
    module.STATE_FILE = tmp_path / "data-first.json"
    module.save_state(
        {
            "investigation_mode": True,
            "consecutive_reads": module.WARN_THRESHOLD - 1,
            "last_updated": datetime.now(UTC).isoformat(),
        }
    )

    stderr = io.StringIO()
    try:
        with redirect_stderr(stderr):
            module.handle_pre_tool_use({"tool_name": "Read"})
    except SystemExit as exc:
        assert exc.code == 0  # WARN, allow
    assert "DATA FIRST WARNING" in stderr.getvalue()


def test_pretooluse_no_warn_outside_investigation_mode_below_block_plus_3(tmp_path: Path) -> None:
    module = _load_module()
    module.STATE_FILE = tmp_path / "data-first.json"
    module.save_state(
        {
            "investigation_mode": False,
            "consecutive_reads": 5,
            "last_updated": datetime.now(UTC).isoformat(),
        }
    )

    stderr = io.StringIO()
    try:
        with redirect_stderr(stderr):
            module.handle_pre_tool_use({"tool_name": "Read"})
    except SystemExit as exc:
        assert exc.code == 0
    assert stderr.getvalue() == ""
