"""Stage 4 Part B — orphan-dashboard reap tests.

Covers _reap_orphan_dashboard: the self-healing belt for a dashboard subprocess
orphaned by a window-[X] close (CTRL_CLOSE_EVENT skips the orchestrator's
teardown finally, so the dashboard child is never terminated and keeps holding
port 8080). The next launch must reclaim that orphan before spawning a fresh one.

Fail-open contract: any error leaves the launch to proceed (nuisance, not capital
— never touches positions/brokers).
"""

import os
from unittest.mock import MagicMock, patch

import trading_app.live.bot_dashboard as dash


def _pid_file(tmp_path):
    p = tmp_path / "bot_dashboard.pid"
    return p


def test_reap_no_pid_file_is_noop(tmp_path):
    with patch.object(dash, "_DASHBOARD_PID_FILE", _pid_file(tmp_path)):
        # No file → must not raise, must not call any kill.
        with patch("subprocess.run") as run:
            dash._reap_orphan_dashboard()
            run.assert_not_called()


def test_reap_dead_pid_removes_stale_file_no_kill(tmp_path):
    pf = _pid_file(tmp_path)
    pf.write_text("99999", encoding="utf-8")
    with patch.object(dash, "_DASHBOARD_PID_FILE", pf):
        with patch.object(dash, "_canonical_pid_is_alive", return_value=False):
            with patch("subprocess.run") as run:
                dash._reap_orphan_dashboard()
                run.assert_not_called()
    assert not pf.exists(), "stale pid file for a dead process must be removed"


def test_reap_live_orphan_is_killed_and_file_cleared(tmp_path):
    pf = _pid_file(tmp_path)
    pf.write_text("4242:123456", encoding="utf-8")  # pid:create_time format
    with patch.object(dash, "_DASHBOARD_PID_FILE", pf):
        with patch.object(dash, "_canonical_pid_is_alive", return_value=True):
            with patch.object(dash.sys, "platform", "win32"):
                with patch("subprocess.run") as run:
                    dash._reap_orphan_dashboard()
                    run.assert_called_once()
                    args = run.call_args[0][0]
                    assert "taskkill" in args and "4242" in args
    assert not pf.exists(), "pid file must be cleared after reaping the orphan"


def test_reap_passes_create_time_to_oracle(tmp_path):
    """The recorded create_time must reach the canonical oracle (reuse defense)."""
    pf = _pid_file(tmp_path)
    pf.write_text("4242:987654", encoding="utf-8")
    with patch.object(dash, "_DASHBOARD_PID_FILE", pf):
        with patch.object(dash, "_canonical_pid_is_alive", return_value=False) as alive:
            with patch("subprocess.run") as run:
                dash._reap_orphan_dashboard()
                alive.assert_called_once_with(4242, 987654)
                run.assert_not_called()


def test_reap_skips_own_pid(tmp_path):
    """A pid file naming THIS process must never be self-killed."""
    pf = _pid_file(tmp_path)
    pf.write_text(str(os.getpid()), encoding="utf-8")
    with patch.object(dash, "_DASHBOARD_PID_FILE", pf):
        with patch("subprocess.run") as run:
            dash._reap_orphan_dashboard()
            run.assert_not_called()
    assert not pf.exists()


def test_reap_empty_pid_file_removed(tmp_path):
    pf = _pid_file(tmp_path)
    pf.write_text("   ", encoding="utf-8")
    with patch.object(dash, "_DASHBOARD_PID_FILE", pf):
        with patch("subprocess.run") as run:
            dash._reap_orphan_dashboard()
            run.assert_not_called()
    assert not pf.exists()


def test_reap_invalid_pid_text_removed(tmp_path):
    pf = _pid_file(tmp_path)
    pf.write_text("not-a-pid", encoding="utf-8")
    with patch.object(dash, "_DASHBOARD_PID_FILE", pf):
        with patch("subprocess.run") as run:
            dash._reap_orphan_dashboard()
            run.assert_not_called()
    assert not pf.exists()


def test_reap_is_fail_open_on_error(tmp_path):
    """Any internal error must be swallowed — launch proceeds regardless."""
    pf = _pid_file(tmp_path)
    pf.write_text("4242:1", encoding="utf-8")
    with patch.object(dash, "_DASHBOARD_PID_FILE", pf):
        with patch.object(
            dash,
            "_canonical_pid_is_alive",
            side_effect=RuntimeError("boom"),
        ):
            # Must NOT raise.
            dash._reap_orphan_dashboard()


def test_canonical_oracle_fails_closed_when_unavailable():
    """If worktree_guard is unavailable, treat the PID as NOT our orphan (False)
    so we never taskkill an innocent recycled PID on a guess."""
    with patch.dict("sys.modules", {"scripts.tools.worktree_guard": None}):
        assert dash._canonical_pid_is_alive(4242, 123) is False


def test_launch_records_pid_file_and_reaps_first(tmp_path):
    """launch_dashboard_background must reap THEN record the new pid."""
    pf = _pid_file(tmp_path)
    fake_proc = MagicMock()
    fake_proc.pid = 7777
    with patch.object(dash, "_DASHBOARD_PID_FILE", pf):
        with patch.object(dash, "_reap_orphan_dashboard") as reap:
            with patch("subprocess.Popen", return_value=fake_proc):
                proc = dash.launch_dashboard_background(port=8080)
                reap.assert_called_once()
                assert proc is fake_proc
    # Format is "pid" or "pid:create_time" — assert the PID part regardless.
    assert pf.read_text(encoding="utf-8").strip().split(":")[0] == "7777"
