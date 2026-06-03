"""Tests for the auto-memory-capture advisory hook + shared helper.

Covers `_memory_capture.py` (signals/threshold/breadcrumb/telemetry/dedup) and
`memory-capture-advisory.py` (PreCompact emit/silent/dedup, SessionEnd breadcrumb
+ telemetry, dispatch, fail-open). State paths are redirected to tmp_path so no
test pollutes repo state. Git is mocked at `mc._git` so signal logic is
deterministic and offline.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

HOOKS_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mc():
    """Load _memory_capture.py as an importable module."""
    spec = importlib.util.spec_from_file_location(
        "_memory_capture", HOOKS_DIR / "_memory_capture.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_memory_capture"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def advisory():
    """Load memory-capture-advisory.py as a module (after mc is importable)."""
    spec = importlib.util.spec_from_file_location(
        "memory_capture_advisory", HOOKS_DIR / "memory-capture-advisory.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["memory_capture_advisory"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def state(tmp_path, mc):
    """Redirect all state file paths into tmp_path."""
    with (
        patch.object(mc, "STATE_DIR", tmp_path),
        patch.object(mc, "PENDING_PATH", tmp_path / "pending.json"),
        patch.object(mc, "TELEMETRY_PATH", tmp_path / "telemetry.log"),
        patch.object(mc, "ADVISORY_PATH", tmp_path / "advisory.json"),
    ):
        yield tmp_path


def _run(mod, payload: str) -> tuple[int, str]:
    """Invoke a hook module's main() with stdin=payload; return (exit_code, stdout)."""
    out = io.StringIO()
    with patch.object(sys, "stdin", io.StringIO(payload)), patch.object(sys, "stdout", out):
        try:
            mod.main()
            code = 0
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 0
    return code, out.getvalue()


# --------------------------------------------------------------------------- #
# Threshold logic
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "sig,expected",
    [
        ({"commits": 1, "files": 0, "stage_files": [], "doctrine_files": []}, True),
        ({"commits": 0, "files": 3, "stage_files": [], "doctrine_files": []}, True),
        ({"commits": 0, "files": 2, "stage_files": [], "doctrine_files": []}, False),
        ({"commits": 0, "files": 0, "stage_files": ["docs/runtime/stages/x.md"], "doctrine_files": []}, True),
        ({"commits": 0, "files": 0, "stage_files": [], "doctrine_files": [".claude/rules/y.md"]}, True),
        ({"commits": 0, "files": 0, "stage_files": [], "doctrine_files": []}, False),
    ],
)
def test_signal_meets_threshold(mc, sig, expected):
    assert mc.signal_meets_threshold(sig) is expected


def test_stage_and_doctrine_classifiers(mc):
    assert mc._is_stage_file("docs/runtime/stages/2026-05-31-x.md")
    assert not mc._is_stage_file("docs/runtime/other.md")
    assert mc._is_doctrine_file(".claude/rules/foo.md")
    assert mc._is_doctrine_file("docs/some-doctrine.md")
    assert mc._is_doctrine_file("docs/the-rule.md")
    assert not mc._is_doctrine_file("docs/notes.md")


def test_classifiers_handle_windows_backslash(mc):
    # gather_signals normalizes, but classifiers receive normalized input;
    # verify the normalization happens end-to-end.
    with patch.object(mc, "_commit_count", return_value=0), patch.object(
        mc, "_diff_names", return_value=["docs\\runtime\\stages\\x.md", "a.py", "b.py"]
    ):
        sig = mc.gather_signals()
    assert sig["stage_files"] == ["docs/runtime/stages/x.md"]
    assert sig["files"] == 3


# --------------------------------------------------------------------------- #
# gather_signals via mocked git
# --------------------------------------------------------------------------- #
def test_gather_signals_commit_count_uses_head_at_start(mc):
    def fake_git(args):
        if args[:2] == ["rev-list", "--count"] and "abc..HEAD" in args[-1]:
            return 0, "2\n"
        if args == ["diff", "--name-only", "HEAD"]:
            return 0, "a.py\n"
        return 1, ""

    with patch.object(mc, "_head_at_start", return_value="abc"), patch.object(
        mc, "_git", side_effect=fake_git
    ):
        sig = mc.gather_signals()
    assert sig["commits"] == 2
    assert sig["files"] == 1


def test_commit_count_falls_back_to_origin_main(mc):
    def fake_git(args):
        if args == ["rev-list", "--count", "origin/main..HEAD"]:
            return 0, "5\n"
        return 1, ""

    with patch.object(mc, "_head_at_start", return_value=None), patch.object(
        mc, "_git", side_effect=fake_git
    ):
        assert mc._commit_count() == 5


def test_commit_count_zero_on_total_git_failure(mc):
    with patch.object(mc, "_head_at_start", return_value=None), patch.object(
        mc, "_git", return_value=(1, "")
    ):
        assert mc._commit_count() == 0


# --------------------------------------------------------------------------- #
# PreCompact behavior
# --------------------------------------------------------------------------- #
def test_precompact_emits_when_signal_met(state, mc, advisory):
    with patch.object(
        mc, "gather_signals", return_value={"commits": 2, "files": 4, "stage_files": [], "doctrine_files": []}
    ):
        code, out = _run(
            advisory,
            json.dumps({"hook_event_name": "PreCompact", "trigger": "manual", "session_id": "s1"}),
        )
    assert code == 0
    payload = json.loads(out)
    assert "systemMessage" in payload
    assert "2 commit(s)" in payload["systemMessage"]
    assert "taxonomy" in payload["systemMessage"]
    assert "dedup" in payload["systemMessage"].lower()
    # session recorded for dedup
    assert mc.already_advised("s1")


def test_precompact_silent_when_noop(state, mc, advisory):
    with patch.object(
        mc, "gather_signals", return_value={"commits": 0, "files": 0, "stage_files": [], "doctrine_files": []}
    ):
        code, out = _run(
            advisory,
            json.dumps({"hook_event_name": "PreCompact", "trigger": "manual", "session_id": "s2"}),
        )
    assert code == 0
    assert out.strip() == ""
    assert not mc.already_advised("s2")


def test_precompact_dedup_second_same_sid(state, mc, advisory):
    sig = {"commits": 3, "files": 0, "stage_files": [], "doctrine_files": []}
    with patch.object(mc, "gather_signals", return_value=sig):
        code1, out1 = _run(
            advisory, json.dumps({"hook_event_name": "PreCompact", "session_id": "dup"})
        )
        code2, out2 = _run(
            advisory, json.dumps({"hook_event_name": "PreCompact", "session_id": "dup"})
        )
    assert out1.strip() != ""
    assert out2.strip() == ""  # second time silent


def test_precompact_trigger_auto_same_as_manual(state, mc, advisory):
    sig = {"commits": 1, "files": 0, "stage_files": [], "doctrine_files": []}
    with patch.object(mc, "gather_signals", return_value=sig):
        code, out = _run(
            advisory, json.dumps({"hook_event_name": "PreCompact", "trigger": "auto", "session_id": "auto1"})
        )
    assert json.loads(out)["systemMessage"]


# --------------------------------------------------------------------------- #
# SessionEnd behavior
# --------------------------------------------------------------------------- #
def test_sessionend_writes_breadcrumb_when_signal(state, mc, advisory):
    sig = {"commits": 1, "files": 5, "stage_files": [], "doctrine_files": []}
    with patch.object(mc, "gather_signals", return_value=sig):
        code, out = _run(
            advisory, json.dumps({"hook_event_name": "SessionEnd", "reason": "clear", "session_id": "e1"})
        )
    assert code == 0
    assert out.strip() == ""  # SessionEnd has no channel
    crumb = mc.read_breadcrumb()
    assert crumb is not None
    assert crumb["session_id"] == "e1"
    assert crumb["consumed"] is False
    # telemetry appended
    assert mc.TELEMETRY_PATH.exists()
    line = json.loads(mc.TELEMETRY_PATH.read_text(encoding="utf-8").strip())
    assert line["event"] == "SessionEnd"
    assert line["signal_met"] is True


def test_sessionend_no_breadcrumb_when_noop_but_telemetry_always(state, mc, advisory):
    sig = {"commits": 0, "files": 0, "stage_files": [], "doctrine_files": []}
    with patch.object(mc, "gather_signals", return_value=sig):
        _run(advisory, json.dumps({"hook_event_name": "SessionEnd", "reason": "logout", "session_id": "e2"}))
    assert mc.read_breadcrumb() is None  # no breadcrumb on noop
    assert mc.TELEMETRY_PATH.exists()  # telemetry still appended
    line = json.loads(mc.TELEMETRY_PATH.read_text(encoding="utf-8").strip())
    assert line["signal_met"] is False


# --------------------------------------------------------------------------- #
# Dispatch + fail-open
# --------------------------------------------------------------------------- #
def test_dispatch_ignores_other_events(state, mc, advisory):
    code, out = _run(advisory, json.dumps({"hook_event_name": "Stop", "session_id": "x"}))
    assert code == 0
    assert out.strip() == ""
    assert mc.read_breadcrumb() is None


def test_malformed_stdin_exits_zero(state, mc, advisory):
    code, out = _run(advisory, "not json{{{")
    assert code == 0
    assert out.strip() == ""


def test_empty_stdin_exits_zero(state, mc, advisory):
    code, out = _run(advisory, "")
    assert code == 0


def test_non_dict_json_exits_zero(state, mc, advisory):
    code, out = _run(advisory, json.dumps(["a", "list"]))
    assert code == 0
    assert out.strip() == ""


def test_advised_cap_enforced(state, mc):
    for i in range(mc._ADVISED_CAP + 50):
        mc.record_advised(f"sid-{i}")
    data = json.loads(mc.ADVISORY_PATH.read_text(encoding="utf-8"))
    assert len(data["advised_sessions"]) == mc._ADVISED_CAP
    assert mc.already_advised(f"sid-{mc._ADVISED_CAP + 49}")  # most recent kept
    assert not mc.already_advised("sid-0")  # oldest evicted
