"""Tests for read-budget-guard.py — Tier 3 discovery-loop hook.

Covers each command branch (increment / reset-on-edit / reset-session /
inject) and the cap thresholds.

Run: pytest .claude/hooks/tests/test_read_budget_guard.py -v
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

HOOK_PATH = Path(__file__).resolve().parents[1] / "read-budget-guard.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("read_budget_guard", HOOK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["read_budget_guard"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def hook(tmp_path, monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "STATE_FILE", tmp_path / "read-budget.json")
    return module


def _run(hook_module, cmd: str, stdin_payload: dict | None = None):
    argv = ["read-budget-guard.py", cmd]
    stdin_text = json.dumps(stdin_payload) if stdin_payload is not None else ""
    with patch.object(sys, "argv", argv), patch("sys.stdin", new=io.StringIO(stdin_text)):
        with pytest.raises(SystemExit) as exc:
            hook_module.main()
    return exc.value.code


# ─── increment ───

def test_increment_starts_from_zero(hook):
    code = _run(hook, "increment")
    assert code == 0
    state = json.loads(hook.STATE_FILE.read_text())
    assert state["reads"] == 1


def test_increment_accumulates(hook):
    for _ in range(5):
        _run(hook, "increment")
    state = json.loads(hook.STATE_FILE.read_text())
    assert state["reads"] == 5


# ─── reset-on-edit ───

def test_reset_on_prod_edit_clears_reads(hook):
    for _ in range(10):
        _run(hook, "increment")
    _run(hook, "reset-on-edit", {"tool_input": {"file_path": "pipeline/dst.py"}})
    state = json.loads(hook.STATE_FILE.read_text())
    assert state["reads"] == 0
    assert state["edits_to_prod"] == 1


def test_reset_on_trading_app_edit_clears(hook):
    for _ in range(8):
        _run(hook, "increment")
    _run(hook, "reset-on-edit", {"tool_input": {"file_path": "trading_app/strategy_x.py"}})
    state = json.loads(hook.STATE_FILE.read_text())
    assert state["reads"] == 0


def test_doc_edit_does_not_reset(hook):
    for _ in range(20):
        _run(hook, "increment")
    _run(hook, "reset-on-edit", {"tool_input": {"file_path": "docs/plans/x.md"}})
    state = json.loads(hook.STATE_FILE.read_text())
    assert state["reads"] == 20  # unchanged
    assert state["edits_to_prod"] == 0


def test_test_file_edit_does_not_reset(hook):
    for _ in range(15):
        _run(hook, "increment")
    _run(hook, "reset-on-edit", {"tool_input": {"file_path": "tests/test_x.py"}})
    state = json.loads(hook.STATE_FILE.read_text())
    assert state["reads"] == 15


# ─── reset-session ───

def test_reset_session_clears_state(hook):
    for _ in range(20):
        _run(hook, "increment")
    _run(hook, "reset-session")
    state = json.loads(hook.STATE_FILE.read_text())
    assert state == {"reads": 0, "edits_to_prod": 0, "last_warned_at": None}


# ─── inject ───

def test_inject_silent_below_soft_cap(hook, capsys):
    for _ in range(15):  # SOFT_CAP - 1
        _run(hook, "increment")
    _run(hook, "inject")
    out = capsys.readouterr().out
    assert out == ""


def test_inject_soft_warning_at_soft_cap(hook, capsys):
    for _ in range(16):  # SOFT_CAP
        _run(hook, "increment")
    _run(hook, "inject")
    out = capsys.readouterr().out
    payload = json.loads(out)
    msg = payload["hookSpecificOutput"]["additionalContext"]
    assert "soft" in msg.lower()
    assert "16 reads" in msg


def test_inject_hard_warning_at_hard_cap(hook, capsys):
    for _ in range(26):  # HARD_CAP
        _run(hook, "increment")
    _run(hook, "inject")
    out = capsys.readouterr().out
    payload = json.loads(out)
    msg = payload["hookSpecificOutput"]["additionalContext"]
    assert "hard" in msg.lower()
    assert "26 reads" in msg


def test_inject_soft_cooldown_silences_repeat(hook, capsys):
    for _ in range(16):
        _run(hook, "increment")
    _run(hook, "inject")
    capsys.readouterr()  # drain first warning
    _run(hook, "inject")
    out = capsys.readouterr().out
    assert out == ""  # still in cooldown


def test_inject_hard_warning_ignores_cooldown(hook, capsys):
    for _ in range(26):
        _run(hook, "increment")
    _run(hook, "inject")
    capsys.readouterr()
    _run(hook, "inject")
    out = capsys.readouterr().out
    # Hard warning fires every prompt — no cooldown.
    assert out != ""
    assert "hard" in out.lower()


def test_inject_after_reset_returns_to_silent(hook, capsys):
    for _ in range(20):
        _run(hook, "increment")
    _run(hook, "reset-on-edit", {"tool_input": {"file_path": "pipeline/dst.py"}})
    _run(hook, "inject")
    out = capsys.readouterr().out
    assert out == ""


# ─── unknown command (fail-open) ───

def test_unknown_command_exits_silently(hook, capsys):
    code = _run(hook, "wat")
    assert code == 0
    assert capsys.readouterr().out == ""
