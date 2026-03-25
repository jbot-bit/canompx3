"""Tests for persistent dollar-based HWM tracker."""

import json
from pathlib import Path

import pytest

from trading_app.account_hwm_tracker import AccountHWMTracker, HWMState


@pytest.fixture
def state_dir(tmp_path):
    return tmp_path / "state"


@pytest.fixture
def tracker(state_dir):
    return AccountHWMTracker("ACC001", "topstep", dd_limit_dollars=2000.0, state_dir=state_dir)


class TestInitialisation:
    def test_fresh_start_no_state_file(self, tracker, state_dir):
        assert tracker._hwm == 0.0
        assert not tracker._halt

    def test_invalid_dd_limit_raises(self, state_dir):
        with pytest.raises(ValueError, match="positive"):
            AccountHWMTracker("ACC001", "topstep", dd_limit_dollars=0, state_dir=state_dir)

    def test_first_equity_sets_hwm(self, tracker):
        state = tracker.update_equity(50000.0)
        assert state.hwm_dollars == 50000.0
        assert state.dd_used_dollars == 0.0
        assert state.is_safe

    def test_state_persists_across_instances(self, state_dir):
        t1 = AccountHWMTracker("ACC001", "topstep", dd_limit_dollars=2000.0, state_dir=state_dir)
        t1.update_equity(51000.0)

        t2 = AccountHWMTracker("ACC001", "topstep", dd_limit_dollars=2000.0, state_dir=state_dir)
        assert t2._hwm == 51000.0
        assert t2._last_equity == 51000.0


class TestHWMTracking:
    def test_hwm_rises_with_equity(self, tracker):
        tracker.update_equity(50000.0)
        tracker.update_equity(51000.0)
        tracker.update_equity(51500.0)
        state = tracker.update_equity(51200.0)  # drops but HWM stays
        assert state.hwm_dollars == 51500.0
        assert state.last_equity == 51200.0
        assert state.dd_used_dollars == 300.0

    def test_hwm_never_decreases(self, tracker):
        tracker.update_equity(50000.0)
        tracker.update_equity(52000.0)
        tracker.update_equity(49000.0)
        assert tracker._hwm == 52000.0

    def test_the_account_killer_scenario(self, tracker):
        """The exact scenario from the audit: 4 days of slow bleed past DD limit."""
        # Day 1: +$1,200
        tracker.update_equity(50000.0)
        tracker.update_equity(51200.0)
        assert tracker._hwm == 51200.0

        # Day 2: -$900 (total DD from HWM = $900)
        tracker.update_equity(50300.0)
        halted, reason = tracker.check_halt()
        assert not halted
        assert "45%" in reason  # $900 / $2000 = 45%

        # Day 3: -$800 more (total DD from HWM = $1,700)
        tracker.update_equity(49500.0)
        halted, reason = tracker.check_halt()
        assert not halted
        assert "WARNING_75" in reason  # $1700 / $2000 = 85% > 75% threshold

        # Day 4: -$400 more (total DD from HWM = $2,100 >= $2,000 limit)
        state = tracker.update_equity(49100.0)
        assert state.halt_triggered
        halted, reason = tracker.check_halt()
        assert halted
        assert "HWM_HALT" in reason


class TestHaltBehavior:
    def test_halt_triggers_at_limit(self, tracker):
        tracker.update_equity(50000.0)
        tracker.update_equity(52000.0)  # HWM = 52000
        tracker.update_equity(50000.0)  # DD = 2000 = limit
        halted, _ = tracker.check_halt()
        assert halted

    def test_halt_persists_across_restart(self, state_dir):
        t1 = AccountHWMTracker("ACC001", "topstep", dd_limit_dollars=2000.0, state_dir=state_dir)
        t1.update_equity(52000.0)
        t1.update_equity(49999.0)  # DD > 2000

        t2 = AccountHWMTracker("ACC001", "topstep", dd_limit_dollars=2000.0, state_dir=state_dir)
        halted, _ = t2.check_halt()
        assert halted

    def test_manual_reset_clears_halt(self, tracker):
        tracker.update_equity(52000.0)
        tracker.update_equity(49000.0)
        assert tracker._halt
        tracker.reset_halt()
        halted, _ = tracker.check_halt()
        assert not halted

    def test_warn_at_75_pct(self, tracker):
        tracker.update_equity(50000.0)
        tracker.update_equity(51500.0)  # HWM
        tracker.update_equity(50000.0)  # DD = 1500 = 75%
        halted, reason = tracker.check_halt()
        assert not halted
        assert "WARNING_75" in reason

    def test_warn_at_50_pct(self, tracker):
        tracker.update_equity(50000.0)
        tracker.update_equity(51000.0)  # HWM
        tracker.update_equity(50000.0)  # DD = 1000 = 50%
        halted, reason = tracker.check_halt()
        assert not halted
        assert "WARNING_50" in reason

    def test_warning_level_in_state(self, tracker):
        tracker.update_equity(50000.0)
        tracker.update_equity(51500.0)  # HWM
        state = tracker.update_equity(50000.0)  # DD = 1500 = 75%
        assert state.warning_level == "WARNING_75"

    def test_warning_level_clear(self, tracker):
        state = tracker.update_equity(50000.0)
        assert state.warning_level == "CLEAR"

    def test_warning_level_halt(self, tracker):
        tracker.update_equity(52000.0)
        state = tracker.update_equity(49000.0)  # DD = 3000 > limit
        assert state.warning_level == "HALT"


class TestPollFailures:
    def test_none_equity_increments_failure_count(self, tracker):
        tracker.update_equity(50000.0)
        tracker.update_equity(None)
        assert tracker._consecutive_poll_failures == 1
        assert not tracker._halt

    def test_three_consecutive_none_triggers_halt(self, tracker):
        tracker.update_equity(50000.0)
        tracker.update_equity(None)
        tracker.update_equity(None)
        tracker.update_equity(None)
        assert tracker._halt

    def test_successful_poll_resets_failure_count(self, tracker):
        tracker.update_equity(50000.0)
        tracker.update_equity(None)
        tracker.update_equity(None)
        tracker.update_equity(50000.0)
        assert tracker._consecutive_poll_failures == 0
        assert not tracker._halt


class TestSessionLog:
    def test_session_logged(self, tracker):
        tracker.update_equity(50000.0)
        tracker.record_session_start(50000.0)
        tracker.update_equity(50500.0)
        tracker.record_session_end(50200.0)
        assert len(tracker._session_log) == 1
        entry = tracker._session_log[0]
        assert entry["start_equity"] == 50000.0
        assert entry["end_equity"] == 50200.0
        assert entry["peak_intraday"] == 50500.0

    def test_session_log_capped_at_30(self, tracker):
        tracker.update_equity(50000.0)
        for i in range(35):
            tracker.record_session_start(50000.0)
            tracker.record_session_end(50000.0)
        assert len(tracker._session_log) == 30


class TestCorruptState:
    def test_corrupt_json_recovers(self, state_dir):
        state_dir.mkdir(parents=True, exist_ok=True)
        corrupt_file = state_dir / "account_hwm_ACC002.json"
        corrupt_file.write_text("{invalid json")

        t = AccountHWMTracker("ACC002", "topstep", dd_limit_dollars=2000.0, state_dir=state_dir)
        assert t._hwm == 0.0  # reinitialised
        # Backup should exist
        backups = list(state_dir.glob("account_hwm_ACC002_CORRUPT_*.json"))
        assert len(backups) == 1


class TestMultipleAccounts:
    def test_separate_state_files(self, state_dir):
        t1 = AccountHWMTracker("ACC001", "topstep", dd_limit_dollars=2000.0, state_dir=state_dir)
        t2 = AccountHWMTracker("ACC002", "tradeify", dd_limit_dollars=2000.0, state_dir=state_dir)

        t1.update_equity(50000.0)
        t2.update_equity(60000.0)

        assert t1._hwm == 50000.0
        assert t2._hwm == 60000.0

        files = list(state_dir.glob("account_hwm_*.json"))
        assert len(files) == 2


class TestStatusSummary:
    def test_summary_fields(self, tracker):
        tracker.update_equity(51000.0)
        tracker.update_equity(50500.0)
        s = tracker.get_status_summary()
        assert s["account_id"] == "ACC001"
        assert s["firm"] == "topstep"
        assert s["hwm_dollars"] == 51000.0
        assert s["dd_used_dollars"] == 500.0
        assert s["dd_remaining_dollars"] == 1500.0
        assert s["dd_pct_used"] == 25.0
        assert not s["halt_triggered"]
