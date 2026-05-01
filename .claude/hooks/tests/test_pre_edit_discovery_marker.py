"""Tests for pre-edit-discovery-marker.py — Tier 2 discovery-loop hook.

Synthetic transcript fixtures cover each branch:
  - guarded path + REPRO: in user message → pass
  - guarded path + context_resolver.py invocation in tool_use → pass
  - guarded path + TRIVIAL: + small staged diff → pass
  - guarded path + TRIVIAL: + large staged diff → BLOCK
  - guarded path + no markers → BLOCK
  - guarded path + short transcript → pass (fail-open)
  - trivial path → pass (gate doesn't apply)
  - missing transcript → pass (fail-open)
  - missing session_id → pass (fail-open)
  - active marker file → pass

Run: pytest .claude/hooks/tests/test_pre_edit_discovery_marker.py -v
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

HOOK_PATH = Path(__file__).resolve().parents[1] / "pre-edit-discovery-marker.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("pre_edit_discovery_marker", HOOK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["pre_edit_discovery_marker"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def hook(tmp_path, monkeypatch):
    """Fresh module per test, with TRANSCRIPT_DIR + MARKER_FILE pinned to tmp."""
    module = _load_module()
    monkeypatch.setattr(module, "TRANSCRIPT_DIR", tmp_path / "transcripts")
    monkeypatch.setattr(module, "MARKER_FILE", tmp_path / "marker.json")
    (tmp_path / "transcripts").mkdir()
    return module


def _user_text(text: str) -> dict:
    return {
        "type": "user",
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }


def _assistant_bash(command: str) -> dict:
    return {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "name": "Bash",
                    "input": {"command": command, "description": "x"},
                }
            ],
        },
    }


def _write_transcript(hook_module, session_id: str, records: list[dict]) -> None:
    path = hook_module.TRANSCRIPT_DIR / f"{session_id}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        # Pad to >=5 records so the short-transcript fail-open doesn't trigger
        # except in the test that wants it.
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _run_hook(hook_module, capsys, file_path: str, session_id: str = "abc"):
    payload = {
        "session_id": session_id,
        "tool_input": {"file_path": file_path},
    }
    with patch("sys.stdin", new=__import__("io").StringIO(json.dumps(payload))):
        with pytest.raises(SystemExit) as exc:
            hook_module.main()
    captured = capsys.readouterr()
    return exc.value.code, captured


# ─────────── PASS branches ───────────

def test_repro_in_user_message_passes(hook, capsys):
    records = [_user_text("filler")] * 4 + [_user_text("REPRO: pytest fails on x")]
    _write_transcript(hook, "abc", records)
    code, captured = _run_hook(hook, capsys, "pipeline/dst.py")
    assert code == 0
    assert captured.err == ""


def test_context_resolver_in_tool_use_passes(hook, capsys):
    records = [_user_text("filler")] * 4 + [
        _assistant_bash("python scripts/tools/context_resolver.py --task 'fix x' --format markdown")
    ]
    _write_transcript(hook, "abc", records)
    code, _ = _run_hook(hook, capsys, "trading_app/strategy_x.py")
    assert code == 0


def test_trivial_with_small_diff_passes(hook, capsys, monkeypatch):
    records = [_user_text("filler")] * 4 + [_user_text("TRIVIAL: 2-line typo fix")]
    _write_transcript(hook, "abc", records)
    monkeypatch.setattr(hook, "_staged_diff_under_100", lambda: True)
    code, _ = _run_hook(hook, capsys, "pipeline/cost_model.py")
    assert code == 0


def test_short_transcript_fails_open(hook, capsys):
    records = [_user_text("filler")] * 3  # <5 records
    _write_transcript(hook, "abc", records)
    code, _ = _run_hook(hook, capsys, "pipeline/dst.py")
    assert code == 0


def test_trivial_path_skipped(hook, capsys):
    code, _ = _run_hook(hook, capsys, "docs/plans/x.md")
    assert code == 0


def test_missing_transcript_fails_open(hook, capsys):
    code, _ = _run_hook(hook, capsys, "pipeline/dst.py", session_id="nonexistent")
    assert code == 0


def test_missing_session_id_fails_open(hook):
    payload = {"tool_input": {"file_path": "pipeline/dst.py"}}
    with patch("sys.stdin", new=__import__("io").StringIO(json.dumps(payload))):
        with pytest.raises(SystemExit) as exc:
            hook.main()
    assert exc.value.code == 0


def test_active_marker_file_passes(hook, capsys):
    future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    hook.MARKER_FILE.write_text(json.dumps({"valid_until": future}))
    code, _ = _run_hook(hook, capsys, "pipeline/dst.py")
    assert code == 0


def test_expired_marker_file_does_not_pass(hook, capsys):
    past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    hook.MARKER_FILE.write_text(json.dumps({"valid_until": past}))
    records = [_user_text("doing stuff")] * 6
    _write_transcript(hook, "abc", records)
    code, out = _run_hook(hook, capsys, "pipeline/dst.py")
    assert code == 2
    assert "DISCOVERY-MARKER GUARD" in out.err


# ─────────── BLOCK branches ───────────

def test_no_markers_blocks(hook, capsys):
    records = [_user_text("just chatting about stuff")] * 6
    _write_transcript(hook, "abc", records)
    code, out = _run_hook(hook, capsys, "pipeline/dst.py")
    assert code == 2
    assert "DISCOVERY-MARKER GUARD" in out.err
    assert "pipeline/dst.py" in out.err


def test_trivial_with_large_diff_blocks(hook, capsys, monkeypatch):
    records = [_user_text("filler")] * 4 + [_user_text("TRIVIAL: small fix")]
    _write_transcript(hook, "abc", records)
    monkeypatch.setattr(hook, "_staged_diff_under_100", lambda: False)
    code, out = _run_hook(hook, capsys, "pipeline/cost_model.py")
    assert code == 2
    assert "TRIVIAL" in out.err or "no discovery-convergence" in out.err
