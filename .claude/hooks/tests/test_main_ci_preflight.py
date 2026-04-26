"""Mock-based tests for `_main_ci_status_lines` in session-start.py.

Six scenarios per the institutional-grade design at
`docs/plans/2026-04-26-token-efficiency-design.md`:

1. Cache hit (fresh) → gh NOT invoked, cached line returned.
2. Cache miss + gh returns failure → RED line emitted, cache written.
3. Cache miss + gh returns success → green line emitted.
4. gh not installed (FileNotFoundError) → empty list, silent.
5. No completed runs (empty array) → empty list, silent.
6. Cache stale (>5min old) → gh re-called, fresh result returned.

Run: `pytest .claude/hooks/tests/test_main_ci_preflight.py -v`
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

HOOK_PATH = Path(__file__).resolve().parents[1] / "session-start.py"


def _load_hook_module():
    """Load session-start.py as a module despite the hyphen in its filename.
    Hyphenated filenames can't be imported with normal `import` syntax, so
    we use importlib's spec-from-file-location pathway.
    """
    spec = importlib.util.spec_from_file_location("session_start_hook", HOOK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["session_start_hook"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def hook(tmp_path, monkeypatch):
    """Fresh hook module per test, with `_git_common_dir` pinned to a temp dir
    so cache writes are isolated and deterministic.
    """
    module = _load_hook_module()
    monkeypatch.setattr(module, "_git_common_dir", lambda: tmp_path)
    return module


def _write_cache(common_dir: Path, payload: dict) -> Path:
    cache = common_dir / ".claude.main-ci-status"
    cache.write_text(json.dumps(payload), encoding="utf-8")
    return cache


def test_cache_hit_returns_cached_without_gh_call(hook, tmp_path):
    _write_cache(
        tmp_path,
        {
            "timestamp": int(time.time()),
            "conclusion": "success",
            "run_id": 999,
            "workflow": "tests",
        },
    )
    with patch.object(hook.subprocess, "run") as run_mock:
        lines = hook._main_ci_status_lines()
    assert lines == ["  Main CI: green"]
    run_mock.assert_not_called()


def test_cache_miss_red_emits_warning_and_writes_cache(hook, tmp_path):
    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = json.dumps(
        [{"conclusion": "failure", "databaseId": 18234567890, "name": "tests"}]
    )
    with patch.object(hook.subprocess, "run", return_value=fake_result):
        lines = hook._main_ci_status_lines()
    assert len(lines) == 1
    assert "Main CI: RED on run 18234567890 (tests)" in lines[0]
    assert "gh run view 18234567890" in lines[0]
    cache = tmp_path / ".claude.main-ci-status"
    assert cache.exists()
    payload = json.loads(cache.read_text(encoding="utf-8"))
    assert payload["conclusion"] == "failure"
    assert payload["run_id"] == 18234567890
    assert payload["workflow"] == "tests"


def test_cache_miss_green_emits_confirmation(hook):
    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = json.dumps(
        [{"conclusion": "success", "databaseId": 1, "name": "tests"}]
    )
    with patch.object(hook.subprocess, "run", return_value=fake_result):
        lines = hook._main_ci_status_lines()
    assert lines == ["  Main CI: green"]


def test_gh_not_installed_returns_empty_silently(hook, tmp_path):
    with patch.object(hook.subprocess, "run", side_effect=FileNotFoundError):
        lines = hook._main_ci_status_lines()
    assert lines == []
    assert not (tmp_path / ".claude.main-ci-status").exists()


def test_no_completed_runs_returns_empty_silently(hook):
    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = "[]"
    with patch.object(hook.subprocess, "run", return_value=fake_result):
        lines = hook._main_ci_status_lines()
    assert lines == []


def test_cache_stale_triggers_refresh(hook, tmp_path):
    stale_ts = int(time.time()) - hook._MAIN_CI_CACHE_TTL_SECS - 60
    _write_cache(
        tmp_path,
        {
            "timestamp": stale_ts,
            "conclusion": "failure",
            "run_id": 1,
            "workflow": "old",
        },
    )
    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = json.dumps(
        [{"conclusion": "success", "databaseId": 2, "name": "tests"}]
    )
    with patch.object(hook.subprocess, "run", return_value=fake_result) as run_mock:
        lines = hook._main_ci_status_lines()
    assert lines == ["  Main CI: green"]
    run_mock.assert_called_once()
    payload = json.loads((tmp_path / ".claude.main-ci-status").read_text(encoding="utf-8"))
    assert payload["run_id"] == 2


def test_gh_returncode_nonzero_returns_empty_silently(hook):
    """gh exits non-zero on auth failure, no remote, network error, etc.
    All such cases must be silent — never noise the session start.
    """
    fake_result = MagicMock()
    fake_result.returncode = 1
    fake_result.stdout = ""
    with patch.object(hook.subprocess, "run", return_value=fake_result):
        lines = hook._main_ci_status_lines()
    assert lines == []


def test_cancelled_conclusion_renders_informational_line(hook):
    """Conclusions other than success/failure (cancelled, skipped, startup_failure)
    are reported informationally — they're not strictly red but not green either.
    """
    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = json.dumps(
        [{"conclusion": "cancelled", "databaseId": 5, "name": "tests"}]
    )
    with patch.object(hook.subprocess, "run", return_value=fake_result):
        lines = hook._main_ci_status_lines()
    assert lines == ["  Main CI: last completed run = cancelled"]


def test_atomic_write_no_temp_file_left_behind(hook, tmp_path):
    """The atomic-write pattern (`tempfile + os.replace`) must leave no
    orphan `.tmp` files in the cache directory after a successful write.
    """
    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = json.dumps(
        [{"conclusion": "success", "databaseId": 1, "name": "tests"}]
    )
    with patch.object(hook.subprocess, "run", return_value=fake_result):
        hook._main_ci_status_lines()
    leftover_tmps = list(tmp_path.glob(".claude.main-ci-status.*.tmp"))
    assert leftover_tmps == [], f"Orphaned temp files: {leftover_tmps}"


def test_corrupted_cache_falls_through_to_fetch(hook, tmp_path):
    """Cache file with malformed JSON must be treated as a cache miss
    (parse-exception caught), NOT propagate the exception out.
    """
    cache = tmp_path / ".claude.main-ci-status"
    cache.write_text("{not-json", encoding="utf-8")
    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = json.dumps(
        [{"conclusion": "success", "databaseId": 1, "name": "tests"}]
    )
    with patch.object(hook.subprocess, "run", return_value=fake_result) as run_mock:
        lines = hook._main_ci_status_lines()
    assert lines == ["  Main CI: green"]
    run_mock.assert_called_once()
