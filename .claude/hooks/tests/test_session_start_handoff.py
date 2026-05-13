"""Tests for `_handoff_next_step_line()` in `.claude/hooks/session-start.py`.

Direct unit tests of the parser — drives the function in-process rather than
subprocess (faster, and avoids triggering the worktree-lock side effect).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(scope="module")
def session_start_module():
    """Load session-start.py as a module (filename has a hyphen → can't `import`)."""
    hook_path = Path(__file__).resolve().parents[1] / "session-start.py"
    spec = importlib.util.spec_from_file_location("session_start_hook", hook_path)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["session_start_hook"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_extracts_first_numbered_bullet(tmp_path, session_start_module):
    handoff = tmp_path / "HANDOFF.md"
    handoff.write_text(
        "# HANDOFF\n\n"
        "## Recent activity\n"
        "- something\n\n"
        "## Next Steps — Active\n"
        "1. **MGC LONDON_METALS — DO NOT RE-LITIGATE.** Verdict frozen at docs/audit/...\n"
        "2. Second item.\n\n"
        "## Other section\n",
        encoding="utf-8",
    )
    with patch.object(session_start_module, "PROJECT_ROOT", tmp_path):
        lines = session_start_module._handoff_next_step_line()
    assert len(lines) == 1
    assert "Resume → /next" in lines[0]
    assert "MGC LONDON_METALS" in lines[0]


def test_missing_handoff_returns_empty(tmp_path, session_start_module):
    with patch.object(session_start_module, "PROJECT_ROOT", tmp_path):
        lines = session_start_module._handoff_next_step_line()
    assert lines == []


def test_missing_next_steps_section_returns_empty(tmp_path, session_start_module):
    handoff = tmp_path / "HANDOFF.md"
    handoff.write_text("# HANDOFF\n\n## Recent activity\n- nothing\n", encoding="utf-8")
    with patch.object(session_start_module, "PROJECT_ROOT", tmp_path):
        lines = session_start_module._handoff_next_step_line()
    assert lines == []


def test_empty_next_steps_section_returns_empty(tmp_path, session_start_module):
    handoff = tmp_path / "HANDOFF.md"
    handoff.write_text(
        "# HANDOFF\n\n## Next Steps — Active\n\n## Next section\n",
        encoding="utf-8",
    )
    with patch.object(session_start_module, "PROJECT_ROOT", tmp_path):
        lines = session_start_module._handoff_next_step_line()
    assert lines == []


def test_truncates_long_bullets(tmp_path, session_start_module):
    handoff = tmp_path / "HANDOFF.md"
    long_bullet = "x" * 500
    handoff.write_text(
        f"## Next Steps — Active\n1. {long_bullet}\n",
        encoding="utf-8",
    )
    with patch.object(session_start_module, "PROJECT_ROOT", tmp_path):
        lines = session_start_module._handoff_next_step_line()
    assert len(lines) == 1
    # Output cue + "..." ellipsis; total line < 200 chars
    assert len(lines[0]) < 200
    assert "…" in lines[0]


def test_strips_bold_markers(tmp_path, session_start_module):
    handoff = tmp_path / "HANDOFF.md"
    handoff.write_text(
        "## Next Steps — Active\n1. **Highest-EV next is MNQ.** rank-3 candidate.\n",
        encoding="utf-8",
    )
    with patch.object(session_start_module, "PROJECT_ROOT", tmp_path):
        lines = session_start_module._handoff_next_step_line()
    assert len(lines) == 1
    assert "Highest-EV next is MNQ" in lines[0]
    assert "**" not in lines[0]


def test_fails_open_on_garbage(tmp_path, session_start_module):
    """A non-UTF8 / corrupted HANDOFF must not crash session start."""
    handoff = tmp_path / "HANDOFF.md"
    handoff.write_bytes(b"\xff\xfe\x00\x00 invalid utf-8 \xff\xff")
    with patch.object(session_start_module, "PROJECT_ROOT", tmp_path):
        lines = session_start_module._handoff_next_step_line()
    assert lines == []  # silent fail
