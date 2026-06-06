"""Tests for the autopilot Stop-block path added to completion-notify.py.

The Stop hook normally only plays a sound + emits an advisory pre-commit cue.
Under an active autopilot run it must additionally emit a `decision:block` to
force the review/repair pass before the run can end — and ONLY when:
  AUTOPILOT_RUN=1 AND uncommitted prod edits exist AND no review pass has run.

Outside autopilot (no AUTOPILOT_RUN env) the behavior must be unchanged.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOOK = PROJECT_ROOT / ".claude" / "hooks" / "completion-notify.py"


@pytest.fixture
def cn():
    spec = importlib.util.spec_from_file_location("completion_notify_under_test", HOOK)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_blocks_when_autopilot_prod_edits_and_no_review(cn, monkeypatch, capsys):
    monkeypatch.setenv("AUTOPILOT_RUN", "1")
    monkeypatch.setenv("AUTOPILOT_RUN_ID", "unit-test-run")
    monkeypatch.setattr(cn, "_has_prod_edits", lambda: True)
    monkeypatch.setattr(cn, "_autopilot_review_done", lambda: False)
    assert cn._maybe_block_autopilot_stop() is True
    import json

    payload = json.loads(capsys.readouterr().out)
    assert payload["decision"] == "block"
    assert "review" in payload["reason"].lower()


def test_no_block_when_review_done(cn, monkeypatch):
    monkeypatch.setenv("AUTOPILOT_RUN", "1")
    monkeypatch.setenv("AUTOPILOT_RUN_ID", "unit-test-run")
    monkeypatch.setattr(cn, "_has_prod_edits", lambda: True)
    monkeypatch.setattr(cn, "_autopilot_review_done", lambda: True)
    assert cn._maybe_block_autopilot_stop() is False


def test_no_block_without_prod_edits(cn, monkeypatch):
    monkeypatch.setenv("AUTOPILOT_RUN", "1")
    monkeypatch.setattr(cn, "_has_prod_edits", lambda: False)
    monkeypatch.setattr(cn, "_autopilot_review_done", lambda: False)
    assert cn._maybe_block_autopilot_stop() is False


def test_inert_without_autopilot_env(cn, monkeypatch):
    monkeypatch.delenv("AUTOPILOT_RUN", raising=False)
    monkeypatch.setattr(cn, "_has_prod_edits", lambda: True)
    monkeypatch.setattr(cn, "_autopilot_review_done", lambda: False)
    assert cn._maybe_block_autopilot_stop() is False
