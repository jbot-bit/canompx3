"""Unit tests for the stale-process reaper decision logic.

These exercise the PURE ``decide`` function against synthetic process tables —
no real process is ever enumerated or killed. The OS-boundary functions
(``_enumerate_*``, ``_kill``) are deliberately not invoked here; their
behavior is platform-specific and marked ``# pragma: no cover``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from scripts.tools.reap_stale_claude_processes import (
    ProcInfo,
    decide,
    read_session_start,
)

SESSION_START = datetime(2026, 5, 29, 3, 0, 0, tzinfo=UTC)
BEFORE = SESSION_START - timedelta(hours=2)
AFTER = SESSION_START + timedelta(minutes=10)


def _decisions_by_pid(procs, session_start: datetime | None = SESSION_START, self_pid=999, ancestry=frozenset()):
    return {d.proc.pid: d for d in decide(procs, session_start, self_pid, ancestry)}


def test_orphan_fork_worker_is_killed():
    procs = [
        ProcInfo(pid=100, ppid=27984, cmdline="python -c multiprocessing-fork spawn_main", started=AFTER),
        # parent 27984 is NOT in the live set -> orphan
    ]
    d = _decisions_by_pid(procs)[100]
    assert d.kill is True
    assert "orphan fork worker" in d.reason


def test_fork_worker_with_live_parent_is_kept():
    procs = [
        ProcInfo(pid=200, ppid=27984, cmdline="python -c multiprocessing-fork spawn_main", started=AFTER),
        ProcInfo(pid=27984, ppid=1, cmdline="python .claude/hooks/post-edit-pipeline.py", started=AFTER),
    ]
    d = _decisions_by_pid(procs)[200]
    assert d.kill is False
    assert "parent 27984 alive" in d.reason


def test_capital_path_process_never_killed_even_if_stale():
    # A live bot started long before the session must NEVER be a candidate.
    procs = [
        ProcInfo(pid=300, ppid=1, cmdline="python -m trading_app.live.webhook_server --demo", started=BEFORE),
    ]
    d = _decisions_by_pid(procs)[300]
    assert d.kill is False
    assert "capital-path" in d.reason


def test_prior_session_mcp_server_is_killed():
    procs = [
        ProcInfo(pid=400, ppid=1, cmdline="python scripts/tools/repo_state_mcp_server.py", started=BEFORE),
    ]
    d = _decisions_by_pid(procs)[400]
    assert d.kill is True
    assert "prior-session" in d.reason


def test_current_session_mcp_server_is_kept():
    procs = [
        ProcInfo(pid=500, ppid=1, cmdline="python scripts/tools/strategy_lab_mcp_server.py", started=AFTER),
    ]
    d = _decisions_by_pid(procs)[500]
    assert d.kill is False
    assert "current session" in d.reason


def test_self_and_ancestry_never_killed():
    procs = [
        ProcInfo(pid=999, ppid=888, cmdline="python scripts/tools/repo_state_mcp_server.py", started=BEFORE),
        ProcInfo(pid=888, ppid=1, cmdline="python scripts/tools/research_catalog_mcp_server.py", started=BEFORE),
    ]
    d = _decisions_by_pid(procs, self_pid=999, ancestry=frozenset({888}))
    assert d[999].kill is False and "self/ancestry" in d[999].reason
    assert d[888].kill is False and "self/ancestry" in d[888].reason


def test_non_project_process_ignored():
    procs = [
        ProcInfo(pid=600, ppid=1, cmdline="python some_random_user_script.py", started=BEFORE),
    ]
    d = _decisions_by_pid(procs)[600]
    assert d.kill is False
    assert "not a project process" in d.reason


def test_no_session_lock_disables_age_rule_but_not_orphan_rule():
    procs = [
        # stale MCP, but no lock -> cannot prove stale -> keep
        ProcInfo(pid=700, ppid=1, cmdline="python scripts/tools/repo_state_mcp_server.py", started=BEFORE),
        # orphan worker -> still killable without a lock
        ProcInfo(pid=701, ppid=42, cmdline="python -c multiprocessing-fork", started=BEFORE),
    ]
    d = _decisions_by_pid(procs, session_start=None)
    assert d[700].kill is False and "fail-closed" in d[700].reason
    assert d[701].kill is True and "orphan fork worker" in d[701].reason


def test_unknown_start_time_fails_closed():
    procs = [
        ProcInfo(pid=800, ppid=1, cmdline="python scripts/tools/strategy_lab_mcp_server.py", started=None),
    ]
    d = _decisions_by_pid(procs)[800]
    assert d.kill is False
    assert "cannot prove stale" in d.reason


def test_read_session_start_missing_file_returns_none(tmp_path):
    assert read_session_start(tmp_path / "nonexistent.pid") is None


def test_read_session_start_parses_iso(tmp_path):
    lock = tmp_path / ".claude.pid"
    lock.write_text('{"pid": 1, "iso_started": "2026-05-29T03:00:32.5+00:00"}', encoding="utf-8")
    ts = read_session_start(lock)
    assert ts is not None
    assert ts.year == 2026 and ts.hour == 3


def test_read_session_start_corrupt_returns_none(tmp_path):
    lock = tmp_path / ".claude.pid"
    lock.write_text("not json {{{", encoding="utf-8")
    assert read_session_start(lock) is None


def test_read_session_start_recovers_from_unescaped_windows_path(tmp_path):
    # Real-world lock: session-start.py writes the worktree path with unescaped
    # backslashes, making the blob invalid JSON. iso_started is still recoverable.
    lock = tmp_path / ".claude.pid"
    lock.write_text(
        '{"pid": 63904, "iso_started": "2026-05-24T03:00:32.5+00:00", '
        '"worktree": "C:\\Users\\joshd\\canompx3", "branch_at_start": "main"}',
        encoding="utf-8",
    )
    ts = read_session_start(lock)
    assert ts is not None
    assert ts.year == 2026 and ts.month == 5 and ts.day == 24
