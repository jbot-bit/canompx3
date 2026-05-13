"""Tests for the pre-commit cue logic in `.claude/hooks/completion-notify.py`.

Drives the internal helpers in-process; the sound effect is intentionally
ignored (Notification/Stop branches both call winsound which we don't exercise
in tests — they're terminal-effect side-only).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(scope="module")
def cn_module():
    """Load completion-notify.py as a module."""
    hook_path = Path(__file__).resolve().parents[1] / "completion-notify.py"
    spec = importlib.util.spec_from_file_location("completion_notify_hook", hook_path)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["completion_notify_hook"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def state_file_redirect(tmp_path, cn_module):
    """Redirect the cue state file to a temp dir so tests don't pollute repo state."""
    fake_state = tmp_path / "completion-cue.json"
    with patch.object(cn_module, "CUE_STATE_FILE", fake_state):
        yield fake_state


def test_has_prod_edits_true(cn_module):
    fake_paths = ["pipeline/builder.py", "docs/HANDOFF.md"]
    with patch.object(cn_module, "_git_diff_names", return_value=fake_paths):
        assert cn_module._has_prod_edits() is True


def test_has_prod_edits_false_docs_only(cn_module):
    fake_paths = ["docs/HANDOFF.md", "memory/foo.md", "README.md"]
    with patch.object(cn_module, "_git_diff_names", return_value=fake_paths):
        assert cn_module._has_prod_edits() is False


def test_has_prod_edits_windows_paths(cn_module):
    """Git on Windows may emit backslash paths; helper must normalize."""
    fake_paths = ["trading_app\\strategy_engine.py"]
    with patch.object(cn_module, "_git_diff_names", return_value=fake_paths):
        assert cn_module._has_prod_edits() is True


def test_has_implementation_stage_true(tmp_path, cn_module):
    stages = tmp_path / "docs" / "runtime" / "stages"
    stages.mkdir(parents=True)
    (stages / "feat.md").write_text(
        "---\ntask: foo\nmode: IMPLEMENTATION\n---\n", encoding="utf-8"
    )
    with patch.object(cn_module, "PROJECT_ROOT", tmp_path):
        assert cn_module._has_implementation_stage() is True


def test_has_implementation_stage_false_design_only(tmp_path, cn_module):
    stages = tmp_path / "docs" / "runtime" / "stages"
    stages.mkdir(parents=True)
    (stages / "feat.md").write_text(
        "---\ntask: foo\nmode: DESIGN\n---\n", encoding="utf-8"
    )
    with patch.object(cn_module, "PROJECT_ROOT", tmp_path):
        assert cn_module._has_implementation_stage() is False


def test_has_implementation_stage_no_stages(tmp_path, cn_module):
    with patch.object(cn_module, "PROJECT_ROOT", tmp_path):
        assert cn_module._has_implementation_stage() is False


def test_has_implementation_stage_legacy_file(tmp_path, cn_module):
    runtime = tmp_path / "docs" / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "STAGE_STATE.md").write_text(
        "task: legacy\nmode: IMPLEMENTATION\n", encoding="utf-8"
    )
    with patch.object(cn_module, "PROJECT_ROOT", tmp_path):
        assert cn_module._has_implementation_stage() is True


def test_cue_emits_when_all_conditions_hold(cn_module, capsys, state_file_redirect):
    with patch.object(cn_module, "_has_prod_edits", return_value=True), patch.object(
        cn_module, "_has_implementation_stage", return_value=True
    ):
        cn_module._maybe_emit_cue()
    captured = capsys.readouterr()
    obj = json.loads(captured.out.strip())
    assert obj["hookSpecificOutput"]["hookEventName"] == "Stop"
    assert "/verify done" in obj["hookSpecificOutput"]["additionalContext"]
    assert "/code-review" in obj["hookSpecificOutput"]["additionalContext"]
    # Cooldown state must have been written.
    assert state_file_redirect.exists()


def test_cue_silent_when_no_prod_edits(cn_module, capsys, state_file_redirect):
    with patch.object(cn_module, "_has_prod_edits", return_value=False), patch.object(
        cn_module, "_has_implementation_stage", return_value=True
    ):
        cn_module._maybe_emit_cue()
    captured = capsys.readouterr()
    assert captured.out.strip() == ""


def test_cue_silent_when_not_implementation(cn_module, capsys, state_file_redirect):
    with patch.object(cn_module, "_has_prod_edits", return_value=True), patch.object(
        cn_module, "_has_implementation_stage", return_value=False
    ):
        cn_module._maybe_emit_cue()
    captured = capsys.readouterr()
    assert captured.out.strip() == ""


def test_cue_cooldown_silences_second_fire(cn_module, capsys, state_file_redirect):
    with patch.object(cn_module, "_has_prod_edits", return_value=True), patch.object(
        cn_module, "_has_implementation_stage", return_value=True
    ):
        cn_module._maybe_emit_cue()
        _ = capsys.readouterr()  # drain first emit
        cn_module._maybe_emit_cue()
    captured = capsys.readouterr()
    assert captured.out.strip() == ""  # cooldown blocks immediate re-fire
