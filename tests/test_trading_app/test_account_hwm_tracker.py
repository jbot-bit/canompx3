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
    """Intraday-trailing tracker (legacy behavior — HWM advances on every update)."""
    return AccountHWMTracker(
        "ACC001",
        "topstep",
        dd_limit_dollars=2000.0,
        state_dir=state_dir,
        dd_type="intraday_trailing",
    )


@pytest.fixture
def eod_tracker(state_dir):
    """EOD-trailing tracker — HWM only advances on record_session_end."""
    return AccountHWMTracker(
        "ACC001",
        "topstep",
        dd_limit_dollars=2000.0,
        state_dir=state_dir,
        dd_type="eod_trailing",
    )


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
        t1 = AccountHWMTracker(
            "ACC001",
            "topstep",
            dd_limit_dollars=2000.0,
            state_dir=state_dir,
            dd_type="intraday_trailing",
        )
        t1.update_equity(51000.0)

        t2 = AccountHWMTracker(
            "ACC001",
            "topstep",
            dd_limit_dollars=2000.0,
            state_dir=state_dir,
            dd_type="intraday_trailing",
        )
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
        for _i in range(35):
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


class TestEODTrailing:
    """EOD trailing: HWM only advances at session close, not intraday."""

    def test_intraday_does_not_advance_hwm(self, eod_tracker):
        eod_tracker.update_equity(50000.0)
        eod_tracker.update_equity(51500.0)  # intraday peak
        eod_tracker.update_equity(50800.0)  # drops back
        # HWM should still be at initialisation value (50000), not intraday peak
        assert eod_tracker._hwm == 50000.0

    def test_session_end_advances_hwm(self, eod_tracker):
        eod_tracker.update_equity(50000.0)
        eod_tracker.update_equity(51500.0)
        eod_tracker.record_session_end(51200.0)  # EOD close at 51200
        # HWM should advance to 51200 (the EOD close), not 51500 (intraday peak)
        assert eod_tracker._hwm == 51200.0

    def test_halt_still_works_intraday(self, eod_tracker):
        eod_tracker.update_equity(50000.0)
        eod_tracker.update_equity(47500.0)  # DD = $2500 > $2000 limit
        halted, reason = eod_tracker.check_halt()
        assert halted
        assert "HALT" in reason

    def test_dd_computed_from_frozen_hwm(self, eod_tracker):
        eod_tracker.update_equity(50000.0)
        eod_tracker.record_session_end(51000.0)  # HWM -> 51000
        eod_tracker.update_equity(50500.0)  # intraday drop
        state = eod_tracker.update_equity(50200.0)  # DD = 51000 - 50200 = 800
        assert state.dd_used_dollars == 800.0
        assert state.hwm_dollars == 51000.0  # not moved by intraday


class TestHWMFreeze:
    """Freeze: HWM stops trailing when it reaches freeze_at_balance."""

    def test_freeze_at_balance(self, state_dir):
        t = AccountHWMTracker(
            "ACC001",
            "topstep",
            dd_limit_dollars=2000.0,
            state_dir=state_dir,
            dd_type="intraday_trailing",
            freeze_at_balance=52100.0,
        )
        t.update_equity(50000.0)
        t.update_equity(52200.0)  # past freeze point
        assert t._hwm_frozen is True
        assert t._hwm == 52200.0

        # Further equity increases should NOT advance HWM
        t.update_equity(53000.0)
        assert t._hwm == 52200.0  # frozen

    def test_freeze_persists_across_restart(self, state_dir):
        t1 = AccountHWMTracker(
            "ACC001",
            "topstep",
            dd_limit_dollars=2000.0,
            state_dir=state_dir,
            dd_type="intraday_trailing",
            freeze_at_balance=52100.0,
        )
        t1.update_equity(50000.0)
        t1.update_equity(52500.0)
        assert t1._hwm_frozen is True

        t2 = AccountHWMTracker(
            "ACC001",
            "topstep",
            dd_limit_dollars=2000.0,
            state_dir=state_dir,
            dd_type="intraday_trailing",
            freeze_at_balance=52100.0,
        )
        assert t2._hwm_frozen is True
        assert t2._hwm == 52500.0

    def test_eod_freeze_on_session_end(self, state_dir):
        t = AccountHWMTracker(
            "ACC001",
            "topstep",
            dd_limit_dollars=2000.0,
            state_dir=state_dir,
            dd_type="eod_trailing",
            freeze_at_balance=52100.0,
        )
        t.update_equity(50000.0)
        t.update_equity(52500.0)  # intraday peak — does NOT advance HWM (EOD mode)
        assert t._hwm == 50000.0
        assert t._hwm_frozen is False

        t.record_session_end(52300.0)  # EOD close above freeze point
        assert t._hwm == 52300.0
        assert t._hwm_frozen is True

        # Next session: HWM should NOT advance
        t.record_session_end(53000.0)
        assert t._hwm == 52300.0  # frozen

    def test_no_freeze_when_none(self, state_dir):
        t = AccountHWMTracker(
            "ACC001",
            "topstep",
            dd_limit_dollars=2000.0,
            state_dir=state_dir,
            dd_type="intraday_trailing",  # no freeze_at_balance
        )
        t.update_equity(50000.0)
        t.update_equity(55000.0)
        assert t._hwm_frozen is False
        assert t._hwm == 55000.0


class TestDDBudgetValidation:
    """prop_profiles.validate_dd_budget()"""

    def test_validation_runs_without_error(self):
        from trading_app.prop_profiles import validate_dd_budget

        violations = validate_dd_budget()
        # Should return a list (may be empty if all profiles pass)
        assert isinstance(violations, list)

    def test_validation_for_topstep_under_budget(self):
        from trading_app.prop_profiles import validate_dd_budget

        violations = validate_dd_budget("topstep_50k")
        # TopStep has 1 MGC lane — well under budget
        assert len(violations) == 0

    def test_validation_catches_overcommit(self):
        """Synthetic profile that is deliberately over budget."""
        from trading_app.prop_profiles import (
            ACCOUNT_PROFILES,
            ACCOUNT_TIERS,
            AccountProfile,
            DailyLaneSpec,
            validate_dd_budget,
        )

        # Temporarily inject a bad profile that exceeds MFFU Core 50K DD=$2000.
        # 12 MNQ lanes with no cap = 12 * 120 * 0.75 * 2 = $2160 > $2000
        bad = AccountProfile(
            profile_id="_test_overcommit",
            firm="mffu",
            account_size=50_000,
            active=True,
            daily_lanes=(
                DailyLaneSpec("MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER_O15", "MNQ", "NYSE_OPEN"),
                DailyLaneSpec("MNQ_NYSE_CLOSE_E2_RR1.0_CB1_NO_FILTER_O15", "MNQ", "NYSE_CLOSE"),
                DailyLaneSpec("MNQ_COMEX_SETTLE_E2_RR1.0_CB1_NO_FILTER", "MNQ", "COMEX_SETTLE"),
                DailyLaneSpec("MNQ_SINGAPORE_OPEN_E2_RR1.0_CB1_NO_FILTER_O15", "MNQ", "SINGAPORE_OPEN"),
                DailyLaneSpec("MNQ_US_DATA_1000_E2_RR1.0_CB1_NO_FILTER", "MNQ", "US_DATA_1000"),
                DailyLaneSpec("MNQ_TOKYO_OPEN_E2_RR1.0_CB1_NO_FILTER", "MNQ", "TOKYO_OPEN"),
                DailyLaneSpec("MNQ_CME_PRECLOSE_E2_RR1.0_CB1_NO_FILTER", "MNQ", "CME_PRECLOSE"),
                DailyLaneSpec("MNQ_EUROPE_FLOW_E2_RR1.0_CB1_NO_FILTER", "MNQ", "EUROPE_FLOW"),
                DailyLaneSpec("MNQ_LONDON_METALS_E2_RR1.0_CB1_NO_FILTER", "MNQ", "LONDON_METALS"),
                DailyLaneSpec("MNQ_US_DATA_830_E2_RR1.0_CB1_NO_FILTER", "MNQ", "US_DATA_830"),
                DailyLaneSpec("MNQ_CME_REOPEN_E2_RR1.0_CB1_NO_FILTER", "MNQ", "CME_REOPEN"),
                DailyLaneSpec("MNQ_BRISBANE_1025_E2_RR1.0_CB1_NO_FILTER", "MNQ", "BRISBANE_1025"),
            ),
        )
        ACCOUNT_PROFILES["_test_overcommit"] = bad
        try:
            violations = validate_dd_budget("_test_overcommit")
            assert len(violations) == 1
            assert "worst-case" in violations[0]
        finally:
            del ACCOUNT_PROFILES["_test_overcommit"]
