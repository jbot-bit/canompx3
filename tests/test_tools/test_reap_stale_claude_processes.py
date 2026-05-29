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


def _decisions_by_pid(
    procs, session_start: datetime | None = SESSION_START, self_pid=999, ancestry=frozenset(), reap_duplicates=False
):
    return {d.proc.pid: d for d in decide(procs, session_start, self_pid, ancestry, reap_duplicates=reap_duplicates)}


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


def test_duplicate_generation_keeps_newest_pair_reaps_older():
    # 3 generations (6 procs) of the same MCP server, all postdating the session
    # (so Rule b keeps them) — Rule c must keep the newest pair, reap the 4 older.
    g1 = AFTER + timedelta(minutes=1)  # oldest
    g2 = AFTER + timedelta(minutes=2)
    g3 = AFTER + timedelta(minutes=3)  # newest
    procs = []
    for pid, start in [(11, g1), (12, g1), (13, g2), (14, g2), (15, g3), (16, g3)]:
        procs.append(ProcInfo(pid=pid, ppid=1, cmdline="python scripts/tools/repo_state_mcp_server.py", started=start))
    d = _decisions_by_pid(procs, reap_duplicates=True)
    # Newest pair kept
    assert d[15].kill is False and d[16].kill is False
    # Older two generations reaped as duplicates
    for pid in (11, 12, 13, 14):
        assert d[pid].kill is True, f"pid {pid} should be reaped"
        assert "duplicate generation" in d[pid].reason


def test_duplicate_generation_off_by_default():
    # Same 3 generations, but without the opt-in flag → nothing reaped as duplicate.
    g1, g2, g3 = AFTER + timedelta(minutes=1), AFTER + timedelta(minutes=2), AFTER + timedelta(minutes=3)
    procs = [
        ProcInfo(pid=pid, ppid=1, cmdline="python scripts/tools/repo_state_mcp_server.py", started=start)
        for pid, start in [(11, g1), (12, g1), (13, g2), (14, g2), (15, g3), (16, g3)]
    ]
    d = _decisions_by_pid(procs)  # reap_duplicates defaults False
    assert all(not d[pid].kill for pid in (11, 12, 13, 14, 15, 16))


def test_single_pair_never_reaped_as_duplicate():
    procs = [
        ProcInfo(pid=21, ppid=1, cmdline="python scripts/tools/strategy_lab_mcp_server.py", started=AFTER),
        ProcInfo(pid=22, ppid=1, cmdline="python scripts/tools/strategy_lab_mcp_server.py", started=AFTER),
    ]
    d = _decisions_by_pid(procs)
    assert d[21].kill is False and d[22].kill is False


def test_duplicate_rule_never_overrides_capital_path_exclusion():
    # Even with many generations, a capital-path proc stays excluded (Gate 1 wins).
    procs = [
        ProcInfo(pid=31, ppid=1, cmdline="python -m trading_app.live.webhook_server --live", started=AFTER),
        ProcInfo(pid=32, ppid=1, cmdline="python -m trading_app.live.webhook_server --live", started=AFTER),
        ProcInfo(pid=33, ppid=1, cmdline="python -m trading_app.live.webhook_server --live", started=AFTER),
    ]
    d = _decisions_by_pid(procs, reap_duplicates=True)
    for pid in (31, 32, 33):
        assert d[pid].kill is False
        assert "capital-path" in d[pid].reason


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
