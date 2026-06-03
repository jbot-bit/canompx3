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
    procs,
    session_start: datetime | None = SESSION_START,
    self_pid=999,
    ancestry=frozenset(),
    reap_duplicates=False,
    current_launcher_pid=None,
    launcher_of=None,
    launcher_started=None,
):
    return {
        d.proc.pid: d
        for d in decide(
            procs,
            session_start,
            self_pid,
            ancestry,
            reap_duplicates=reap_duplicates,
            current_launcher_pid=current_launcher_pid,
            launcher_of=launcher_of,
            launcher_started=launcher_started,
        )
    }


# A prior-session launcher PID -> a start time well before SESSION_START.
STALE_LAUNCHER_STARTED = {42: SESSION_START - timedelta(hours=10)}


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


# --- Rule (d): MCP server parented by a stale (prior-session) Claude launcher ---


def test_mcp_under_prior_session_launcher_is_killed():
    # An MCP server whose ancestral claude.exe launcher is a DIFFERENT (prior)
    # session than the current one is a leftover — killable in the auto path,
    # independent of the duplicate flag. The launcher itself is alive but stale.
    procs = [
        ProcInfo(pid=900, ppid=42, cmdline="python scripts/tools/repo_state_mcp_server.py", started=AFTER),
    ]
    d = _decisions_by_pid(
        procs,
        current_launcher_pid=7777,  # the live session's claude.exe
        launcher_of={900: 42},  # pid 900's ancestral launcher is the OLD claude.exe 42
        launcher_started=STALE_LAUNCHER_STARTED,  # launcher 42 predates the session
    )[900]
    assert d.kill is True
    assert "prior-session launcher" in d.reason


def test_mcp_under_current_launcher_is_kept():
    # The live session's own MCP children must never be reaped by Rule (d).
    procs = [
        ProcInfo(pid=901, ppid=7777, cmdline="python scripts/tools/strategy_lab_mcp_server.py", started=AFTER),
    ]
    d = _decisions_by_pid(
        procs,
        current_launcher_pid=7777,
        launcher_of={901: 7777},  # belongs to the live launcher
    )[901]
    assert d.kill is False
    assert "current session" in d.reason


def test_rule_d_disabled_when_current_launcher_unknown():
    # If we cannot identify the live session's launcher, Rule (d) is fail-closed:
    # we must not kill an MCP server we can't prove belongs to a prior session.
    procs = [
        ProcInfo(pid=902, ppid=42, cmdline="python scripts/tools/repo_state_mcp_server.py", started=AFTER),
    ]
    d = _decisions_by_pid(
        procs,
        current_launcher_pid=None,  # unknown
        launcher_of={902: 42},
    )[902]
    # Without a current launcher AND postdating the session lock, falls through to
    # Rule (b)'s "started after session" keep — NOT killed by Rule (d).
    assert d.kill is False
    assert "prior-session launcher" not in d.reason


def test_rule_d_never_overrides_capital_path():
    # Even under a stale launcher, a capital-path proc stays excluded (Gate 1 wins).
    procs = [
        ProcInfo(pid=903, ppid=42, cmdline="python -m trading_app.live.webhook_server --live", started=AFTER),
    ]
    d = _decisions_by_pid(
        procs,
        current_launcher_pid=7777,
        launcher_of={903: 42},
    )[903]
    assert d.kill is False
    assert "capital-path" in d.reason


def test_rule_d_only_applies_to_project_processes():
    # A non-project process under a stale launcher is still ignored, not killed.
    procs = [
        ProcInfo(pid=904, ppid=42, cmdline="python some_random_user_script.py", started=AFTER),
    ]
    d = _decisions_by_pid(
        procs,
        current_launcher_pid=7777,
        launcher_of={904: 42},
    )[904]
    assert d.kill is False
    assert "not a project process" in d.reason


def test_claude_launcher_pid_resolution():
    # _claude_launcher_pid walks parent links to the nearest claude.exe ancestor.
    from scripts.tools.reap_stale_claude_processes import _claude_launcher_pid

    procs = [
        ProcInfo(pid=10, ppid=20, cmdline="python scripts/tools/repo_state_mcp_server.py", started=AFTER),
        ProcInfo(pid=20, ppid=30, cmdline="python scripts/tools/repo_state_mcp_server.py", started=AFTER),  # stub
        ProcInfo(pid=30, ppid=1, cmdline="C:\\Users\\joshd\\.local\\bin\\claude.exe", started=AFTER),
    ]
    assert _claude_launcher_pid(10, procs) == 30
    assert _claude_launcher_pid(20, procs) == 30
    # A process with no claude.exe ancestor resolves to None.
    orphan = [ProcInfo(pid=40, ppid=1, cmdline="python scripts/tools/strategy_lab_mcp_server.py", started=AFTER)]
    assert _claude_launcher_pid(40, orphan) is None


# --- Stress / edge-case hardening for Rule (d) + launcher resolution ---


def test_is_claude_launcher_matcher():
    # Cross-platform launcher detection with false-positive guards.
    from scripts.tools.reap_stale_claude_processes import _is_claude_launcher

    assert _is_claude_launcher(r"C:\Users\joshd\.local\bin\claude.exe") is True
    assert _is_claude_launcher("/home/u/.local/bin/claude") is True
    assert _is_claude_launcher("claude") is True
    assert _is_claude_launcher("claude --resume") is True
    assert _is_claude_launcher("/opt/claude/bin/claude") is True
    # Must NOT match unrelated words / paths.
    assert _is_claude_launcher("python claudette_tool.py") is False
    assert _is_claude_launcher("python scripts/tools/repo_state_mcp_server.py") is False
    # A claude DIRECTORY component is not the launcher token (avoids marking an
    # MCP server living under a claude/ folder as a launcher).
    assert _is_claude_launcher("node /usr/lib/claude/cli.js") is False


def test_launcher_pid_resolution_is_cycle_safe():
    # PID reuse can create a parent cycle (A->B->A). The walk must terminate.
    from scripts.tools.reap_stale_claude_processes import _claude_launcher_pid

    procs = [
        ProcInfo(pid=1, ppid=2, cmdline="python scripts/tools/repo_state_mcp_server.py", started=AFTER),
        ProcInfo(pid=2, ppid=1, cmdline="python scripts/tools/repo_state_mcp_server.py", started=AFTER),
    ]
    # No claude.exe in the cycle -> None, and crucially it returns (no hang).
    assert _claude_launcher_pid(1, procs) is None


def test_rule_d_ownership_beats_age_for_current_session():
    # A process owned by OUR launcher must be kept even if it (paradoxically)
    # predates the session lock — ownership is stronger evidence than a timestamp.
    procs = [
        ProcInfo(pid=950, ppid=7777, cmdline="python scripts/tools/repo_state_mcp_server.py", started=BEFORE),
    ]
    d = _decisions_by_pid(
        procs,
        session_start=SESSION_START,
        current_launcher_pid=7777,
        launcher_of={950: 7777},
    )[950]
    assert d.kill is False
    assert "current session launcher" in d.reason


def test_rule_d_kills_prior_launcher_even_when_proc_postdates_lock():
    # The dominant real bug: an MCP server STARTED AFTER our lock but owned by a
    # DIFFERENT (still-alive) launcher. Rule (b) alone keeps it ("started after
    # session"); Rule (d) correctly reaps it via ownership.
    procs = [
        ProcInfo(pid=951, ppid=42, cmdline="python scripts/tools/research_catalog_mcp_server.py", started=AFTER),
    ]
    d = _decisions_by_pid(
        procs,
        session_start=SESSION_START,
        current_launcher_pid=7777,
        launcher_of={951: 42},
        launcher_started=STALE_LAUNCHER_STARTED,
    )[951]
    assert d.kill is True
    assert "prior-session launcher" in d.reason


def test_rule_d_keeps_concurrent_live_peer_launcher():
    # A DIFFERENT launcher that started AFTER our session lock is a concurrent
    # live peer (another active session / worktree) — its MCP servers must NEVER
    # be reaped. This is the multi-session safety the lease-incident class demands.
    peer_started = SESSION_START + timedelta(minutes=5)
    procs = [
        ProcInfo(pid=952, ppid=55, cmdline="python scripts/tools/repo_state_mcp_server.py", started=AFTER),
    ]
    d = _decisions_by_pid(
        procs,
        session_start=SESSION_START,
        current_launcher_pid=7777,
        launcher_of={952: 55},
        launcher_started={55: peer_started},  # peer launcher is NEWER than our lock
    )[952]
    assert d.kill is False
    assert "concurrent live-peer launcher" in d.reason


def test_rule_d_unknown_launcher_start_falls_through_to_rule_b():
    # If the owning launcher's start time is unknown, we cannot prove it stale —
    # Rule (d) must not kill on a guess; it falls through to lock-based Rule (b).
    procs = [
        # postdates lock -> Rule b keeps it (no kill on unprovable peer)
        ProcInfo(pid=953, ppid=55, cmdline="python scripts/tools/repo_state_mcp_server.py", started=AFTER),
    ]
    d = _decisions_by_pid(
        procs,
        session_start=SESSION_START,
        current_launcher_pid=7777,
        launcher_of={953: 55},
        launcher_started={},  # owner start unknown
    )[953]
    assert d.kill is False
    assert "prior-session launcher" not in d.reason


def test_rule_d_partial_launcher_map_mixed_outcomes():
    # Three MCP servers: one ours, one prior-launcher, one with unresolved owner
    # (falls through to Rule b). All must land on the correct verdict.
    procs = [
        ProcInfo(pid=961, ppid=7777, cmdline="python scripts/tools/strategy_lab_mcp_server.py", started=AFTER),  # ours
        ProcInfo(
            pid=962, ppid=42, cmdline="python scripts/tools/strategy_lab_mcp_server.py", started=AFTER
        ),  # prior launcher
        ProcInfo(
            pid=963, ppid=99, cmdline="python scripts/tools/strategy_lab_mcp_server.py", started=BEFORE
        ),  # unresolved -> Rule b
    ]
    d = _decisions_by_pid(
        procs,
        session_start=SESSION_START,
        current_launcher_pid=7777,
        launcher_of={961: 7777, 962: 42},  # 963 absent -> owner None
        launcher_started=STALE_LAUNCHER_STARTED,  # launcher 42 predates session
    )
    assert d[961].kill is False and "current session launcher" in d[961].reason
    assert d[962].kill is True and "prior-session launcher" in d[962].reason
    assert d[963].kill is True and "prior-session MCP" in d[963].reason  # Rule b caught it


def test_self_and_capital_path_still_win_over_rule_d():
    # Gate 0 (self) and Gate 1 (capital) precede Rule (d) even with a stale owner.
    procs = [
        ProcInfo(pid=999, ppid=42, cmdline="python scripts/tools/repo_state_mcp_server.py", started=AFTER),  # self
        ProcInfo(pid=970, ppid=42, cmdline="python -m trading_app.live.session_orchestrator", started=AFTER),  # capital
    ]
    d = _decisions_by_pid(
        procs,
        self_pid=999,
        current_launcher_pid=7777,
        launcher_of={999: 42, 970: 42},
    )
    assert d[999].kill is False and "self/ancestry" in d[999].reason
    assert d[970].kill is False and "capital-path" in d[970].reason


def test_build_launcher_map_excludes_non_project_and_finds_self(monkeypatch):
    # _build_launcher_map: current launcher = claude.exe ancestor of os.getpid();
    # launcher_of only contains PROJECT-signature processes.
    from scripts.tools import reap_stale_claude_processes as mod

    procs = [
        ProcInfo(pid=500, ppid=400, cmdline="python scripts/tools/repo_state_mcp_server.py", started=AFTER),
        ProcInfo(pid=400, ppid=1, cmdline="C:\\Users\\x\\.local\\bin\\claude.exe", started=AFTER),  # launcher A
        ProcInfo(pid=600, ppid=700, cmdline="python scripts/tools/strategy_lab_mcp_server.py", started=AFTER),
        ProcInfo(pid=700, ppid=1, cmdline="C:\\Users\\x\\.local\\bin\\claude.exe", started=AFTER),  # launcher B
        ProcInfo(pid=800, ppid=400, cmdline="python some_random_user_script.py", started=AFTER),  # non-project
        ProcInfo(pid=42, ppid=400, cmdline="python -m pytest", started=AFTER),  # this 'self'
    ]
    monkeypatch.setattr(mod.os, "getpid", lambda: 42)
    current, launcher_of, launcher_started = mod._build_launcher_map(procs)
    assert current == 400  # self (42) -> claude.exe 400
    assert launcher_of == {500: 400, 600: 700}  # project procs only; non-project 800 excluded
    # launcher_started carries each owning launcher's start time (both AFTER here).
    assert launcher_started == {400: AFTER, 700: AFTER}


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
