"""Tests for the live-peer heartbeat: writer (`session-heartbeat.py`) and
reader (`_live_heartbeat_lines` in `session-start.py`).

Closes the gap where two Claude sessions in the SAME git tree did not notice
each other (PID liveness unreliable on Windows; the mtime guard fires only at
start and only on tracked-file edits, so it misses idle/reading peers). The
heartbeat is a positive liveness FACT stamped on every tool call.

Reader tests drive `_live_heartbeat_lines` in-process with `_git_common_dir`
and `_git` patched to a tmp dir. Writer tests subprocess the real hook (its
contract is stdin->file side effect + exit 0).
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

HOOKS_DIR = Path(__file__).resolve().parents[1]
WARN_MARKER = "ANOTHER CLAUDE SESSION IS LIVE IN THIS WORKTREE"
SOFT_MARKER = "other live session"


@pytest.fixture(scope="module")
def ss_mod():
    hook_path = HOOKS_DIR / "session-start.py"
    spec = importlib.util.spec_from_file_location("session_start_hook_hb", hook_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["session_start_hook_hb"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_beat(beat_dir: Path, sid: str, cwd: str, branch: str = "main", ts: float | None = None) -> Path:
    beat_dir.mkdir(parents=True, exist_ok=True)
    p = beat_dir / f"{sid}.beat"
    p.write_text(
        json.dumps({"session_id": sid, "cwd": cwd, "branch": branch, "ts": ts or time.time()}),
        encoding="utf-8",
    )
    if ts is not None:
        os.utime(p, (ts, ts))
    return p


def _read(ss_mod, my_sid: str, common_dir: Path, tree: str):
    """Drive _live_heartbeat_lines with common-dir and show-toplevel patched."""
    def fake_git(args, timeout=5):
        if args[:1] == ["rev-parse"] and "--show-toplevel" in args:
            return 0, tree
        return 1, ""

    with patch.object(ss_mod, "_git_common_dir", return_value=common_dir), patch.object(
        ss_mod, "_git", side_effect=fake_git
    ):
        return ss_mod._live_heartbeat_lines(my_sid)


# ---------------------------------------------------------------- reader tests

def test_same_tree_peer_triggers_loud_warning(ss_mod, tmp_path):
    common = tmp_path / "gitdir"
    beats = common / ss_mod._HEARTBEAT_DIRNAME
    tree = str(tmp_path / "work")
    _write_beat(beats, "PEER", tree, branch="feature-x")
    lines = _read(ss_mod, "ME", common, tree)
    blob = "\n".join(lines)
    assert WARN_MARKER in blob
    assert "feature-x" in blob


def test_own_beat_never_self_alarms(ss_mod, tmp_path):
    common = tmp_path / "gitdir"
    beats = common / ss_mod._HEARTBEAT_DIRNAME
    tree = str(tmp_path / "work")
    _write_beat(beats, "ME", tree)
    lines = _read(ss_mod, "ME", common, tree)
    assert WARN_MARKER not in "\n".join(lines)


def test_stale_beat_ignored(ss_mod, tmp_path):
    common = tmp_path / "gitdir"
    beats = common / ss_mod._HEARTBEAT_DIRNAME
    tree = str(tmp_path / "work")
    old = time.time() - (ss_mod._HEARTBEAT_LIVE_WINDOW_SECS + 120)
    _write_beat(beats, "PEER", tree, ts=old)
    lines = _read(ss_mod, "ME", common, tree)
    assert WARN_MARKER not in "\n".join(lines)


def test_future_skew_beat_ignored(ss_mod, tmp_path):
    common = tmp_path / "gitdir"
    beats = common / ss_mod._HEARTBEAT_DIRNAME
    tree = str(tmp_path / "work")
    _write_beat(beats, "PEER", tree, ts=time.time() + 600)
    lines = _read(ss_mod, "ME", common, tree)
    assert WARN_MARKER not in "\n".join(lines)


def test_ancient_beat_pruned_on_read(ss_mod, tmp_path):
    common = tmp_path / "gitdir"
    beats = common / ss_mod._HEARTBEAT_DIRNAME
    tree = str(tmp_path / "work")
    ancient = time.time() - (ss_mod._HEARTBEAT_PRUNE_SECS + 600)
    p = _write_beat(beats, "DEAD", tree, ts=ancient)
    _read(ss_mod, "ME", common, tree)
    assert not p.exists()


def test_malformed_beat_ignored_no_crash(ss_mod, tmp_path):
    common = tmp_path / "gitdir"
    beats = common / ss_mod._HEARTBEAT_DIRNAME
    beats.mkdir(parents=True)
    (beats / "BAD.beat").write_text("{not valid json", encoding="utf-8")
    tree = str(tmp_path / "work")
    lines = _read(ss_mod, "ME", common, tree)  # must not raise
    assert WARN_MARKER not in "\n".join(lines)


def test_sibling_worktree_is_soft_note_not_loud(ss_mod, tmp_path):
    common = tmp_path / "gitdir"
    beats = common / ss_mod._HEARTBEAT_DIRNAME
    tree = str(tmp_path / "work")
    other = str(tmp_path / "other_tree")
    _write_beat(beats, "SIB", other, branch="parallel")
    lines = _read(ss_mod, "ME", common, tree)
    blob = "\n".join(lines)
    assert WARN_MARKER not in blob
    assert SOFT_MARKER in blob


def test_no_beat_dir_returns_empty(ss_mod, tmp_path):
    common = tmp_path / "gitdir"
    common.mkdir()
    tree = str(tmp_path / "work")
    assert _read(ss_mod, "ME", common, tree) == []


def test_no_common_dir_returns_empty(ss_mod):
    with patch.object(ss_mod, "_git_common_dir", return_value=None):
        assert ss_mod._live_heartbeat_lines("ME") == []


def test_same_tree_detected_despite_path_form_differences(ss_mod, tmp_path):
    """Bug-2 regression: writer cwd and reader show-toplevel may differ in
    slash/casing form; _canon_path must fold them so a SAME-tree peer still
    triggers the loud warning instead of being mis-classified as a sibling."""
    common = tmp_path / "gitdir"
    beats = common / ss_mod._HEARTBEAT_DIRNAME
    real_tree = tmp_path / "work"
    real_tree.mkdir()
    beat_cwd = str(real_tree).replace("\\", "/").lower()
    _write_beat(beats, "PEER", beat_cwd, branch="feat")
    lines = _read(ss_mod, "ME", common, str(real_tree))
    assert WARN_MARKER in "\n".join(lines)


def test_canon_path_handles_empty_and_garbage(ss_mod):
    """Bug-2 helper: _canon_path returns None on empty, never raises."""
    assert ss_mod._canon_path("") is None
    assert isinstance(ss_mod._canon_path("some/relative/path"), str)


def test_main_session_id_fallback_prefers_code_session_id(ss_mod):
    """Bug-1 regression: when stdin lacks session_id, main() must prefer
    CLAUDE_CODE_SESSION_ID (matching the writer) over CLAUDE_SESSION_ID, so the
    reader skips its OWN beat instead of false-alarming on /clear."""
    import inspect

    src = inspect.getsource(ss_mod.main)
    code_idx = src.find("CLAUDE_CODE_SESSION_ID")
    plain_idx = src.find('CLAUDE_SESSION_ID", ""')
    assert code_idx != -1, "CLAUDE_CODE_SESSION_ID fallback missing in main()"
    assert plain_idx == -1 or code_idx < plain_idx, "fallback order must match the writer"


# ---------------------------------------------------------------- writer tests

WRITER = HOOKS_DIR / "session-heartbeat.py"


def _run_writer(stdin: str, env: dict | None = None) -> int:
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.run(
        [sys.executable, str(WRITER)],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=10,
        env=e,
    ).returncode


@pytest.mark.parametrize("bad", ["", "{", "not json", "[]", '{"no_session":1}'])
def test_writer_fail_open_on_bad_stdin(bad):
    # No session_id resolvable -> exit 0, no crash. Scrub env so the harness's
    # own CLAUDE_CODE_SESSION_ID can't satisfy the id and write a real beat.
    assert _run_writer(bad, env={"CLAUDE_CODE_SESSION_ID": "", "CLAUDE_SESSION_ID": ""}) == 0


def test_writer_creates_beat_in_common_dir(tmp_path):
    # Real git repo so `git rev-parse --git-common-dir` resolves.
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, timeout=10)
    sid = "WRITER-TEST-SID"
    payload = json.dumps(
        {"session_id": sid, "cwd": str(tmp_path), "hook_event_name": "PostToolUse", "tool_name": "Bash"}
    )
    rc = subprocess.run(
        [sys.executable, str(WRITER)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=10,
    ).returncode
    assert rc == 0
    beat = tmp_path / ".git" / ".claude-heartbeats" / f"{sid}.beat"
    assert beat.exists()
    data = json.loads(beat.read_text(encoding="utf-8"))
    assert data["session_id"] == sid


def test_writer_throttles_within_window(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, timeout=10)
    sid = "THROTTLE-SID"
    payload = json.dumps({"session_id": sid, "cwd": str(tmp_path)})
    args = [sys.executable, str(WRITER)]
    subprocess.run(args, input=payload, capture_output=True, text=True, timeout=10)
    beat = tmp_path / ".git" / ".claude-heartbeats" / f"{sid}.beat"
    first = beat.stat().st_mtime
    subprocess.run(args, input=payload, capture_output=True, text=True, timeout=10)
    second = beat.stat().st_mtime
    assert first == second  # second write throttled
