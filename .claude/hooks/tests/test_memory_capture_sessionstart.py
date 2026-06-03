"""Tests for the Claude-visible SessionStart cue hook.

Covers `memory-capture-sessionstart.py`: emits additionalContext when an
unconsumed/fresh/nonzero-signal breadcrumb exists, marks consumed (one-shot),
replays on clear AND compact, and stays silent on consumed/expired/zero/missing.
State redirected to tmp_path; no git needed (cue reads only the breadcrumb).
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

HOOKS_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mc():
    spec = importlib.util.spec_from_file_location(
        "_memory_capture", HOOKS_DIR / "_memory_capture.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_memory_capture"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def sstart():
    spec = importlib.util.spec_from_file_location(
        "memory_capture_sessionstart", HOOKS_DIR / "memory-capture-sessionstart.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["memory_capture_sessionstart"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def state(tmp_path, mc):
    with (
        patch.object(mc, "STATE_DIR", tmp_path),
        patch.object(mc, "PENDING_PATH", tmp_path / "pending.json"),
    ):
        yield tmp_path


def _write_crumb(mc, *, consumed=False, age_hours=1.0, counts=None):
    counts = counts if counts is not None else {"commits": 2, "files": 4, "stage_files": [], "doctrine_files": []}
    ts = (datetime.now(UTC) - timedelta(hours=age_hours)).isoformat()
    mc.PENDING_PATH.write_text(
        json.dumps({"session_id": "prev", "counts": counts, "ts": ts, "consumed": consumed}),
        encoding="utf-8",
    )


def _run(mod, payload: str) -> tuple[int, str]:
    out = io.StringIO()
    with patch.object(sys, "stdin", io.StringIO(payload)), patch.object(sys, "stdout", out):
        try:
            mod.main()
            code = 0
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 0
    return code, out.getvalue()


def test_emits_when_fresh_unconsumed_signal(state, mc, sstart):
    _write_crumb(mc)
    code, out = _run(sstart, json.dumps({"hook_event_name": "SessionStart", "source": "clear", "session_id": "new"}))
    assert code == 0
    payload = json.loads(out)
    ctx = payload["hookSpecificOutput"]["additionalContext"]
    assert payload["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "2 commit(s)" in ctx
    assert "memory" in ctx.lower()
    assert "taxonomy" in ctx
    # one-shot: breadcrumb flipped consumed
    assert mc.read_breadcrumb()["consumed"] is True


@pytest.mark.parametrize("source", ["clear", "compact"])
def test_replays_on_clear_and_compact(state, mc, sstart, source):
    _write_crumb(mc)
    code, out = _run(sstart, json.dumps({"hook_event_name": "SessionStart", "source": source, "session_id": "x"}))
    assert json.loads(out)["hookSpecificOutput"]["additionalContext"]


def test_silent_after_consume(state, mc, sstart):
    _write_crumb(mc)
    _run(sstart, json.dumps({"hook_event_name": "SessionStart", "source": "clear"}))
    code, out = _run(sstart, json.dumps({"hook_event_name": "SessionStart", "source": "clear"}))
    assert code == 0
    assert out.strip() == ""  # second start silent


def test_silent_when_expired(state, mc, sstart):
    _write_crumb(mc, age_hours=25.0)
    code, out = _run(sstart, json.dumps({"hook_event_name": "SessionStart", "source": "clear"}))
    assert code == 0
    assert out.strip() == ""


def test_silent_when_no_breadcrumb(state, mc, sstart):
    code, out = _run(sstart, json.dumps({"hook_event_name": "SessionStart", "source": "startup"}))
    assert code == 0
    assert out.strip() == ""


def test_silent_when_zero_signal(state, mc, sstart):
    _write_crumb(mc, counts={"commits": 0, "files": 0, "stage_files": [], "doctrine_files": []})
    code, out = _run(sstart, json.dumps({"hook_event_name": "SessionStart", "source": "clear"}))
    assert code == 0
    assert out.strip() == ""


def test_reads_session_type_variant(state, mc, sstart):
    # The cue gates only on the breadcrumb, not source/session_type — verify it
    # still fires when the build sends `session_type` instead of `source`.
    _write_crumb(mc)
    code, out = _run(sstart, json.dumps({"hook_event_name": "SessionStart", "session_type": "clear"}))
    assert json.loads(out)["hookSpecificOutput"]["additionalContext"]


def test_malformed_stdin_exits_zero(state, mc, sstart):
    _write_crumb(mc)
    code, out = _run(sstart, "garbage{{{")
    # Still reads breadcrumb regardless of stdin parse outcome -> emits.
    assert code == 0
    assert json.loads(out)["hookSpecificOutput"]["additionalContext"]


def test_empty_stdin_exits_zero(state, mc, sstart):
    code, out = _run(sstart, "")
    assert code == 0  # no breadcrumb -> silent


def test_corrupt_breadcrumb_silent(state, mc, sstart):
    mc.PENDING_PATH.write_text("{not json", encoding="utf-8")
    code, out = _run(sstart, json.dumps({"hook_event_name": "SessionStart", "source": "clear"}))
    assert code == 0
    assert out.strip() == ""
