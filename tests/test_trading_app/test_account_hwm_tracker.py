"""Tests for persistent dollar-based HWM tracker."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from trading_app.account_hwm_tracker import AccountHWMTracker, HWMState, _BRISBANE


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
        assert "DD_TRAILING" in reason


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
        halted, reason = tracker.check_halt()
        assert halted
        assert "POLL_FAILURE" in reason

    def test_successful_poll_resets_failure_count(self, tracker):
        tracker.update_equity(50000.0)
        tracker.update_equity(None)
        tracker.update_equity(None)
        tracker.update_equity(50000.0)
        assert tracker._consecutive_poll_failures == 0
        assert not tracker._halt

    def test_poll_failure_counter_persisted_before_threshold(self, tracker, state_dir):
        """Sub-threshold poll failure counter must survive a process restart so that
        a crashing process cannot reset the counter and allow indefinite poll failures
        without ever triggering the halt (fail-open via restart)."""
        tracker.update_equity(50000.0)
        tracker.update_equity(None)  # failure 1 of 3 — below threshold
        assert tracker._consecutive_poll_failures == 1
        assert not tracker._halt
        # Simulate process restart by reloading from the same state file
        reloaded = AccountHWMTracker(
            "ACC001",
            "topstep",
            dd_limit_dollars=2000.0,
            state_dir=state_dir,
            dd_type="intraday_trailing",
        )
        # Counter must have been written and restored — restart does NOT reset it
        assert reloaded._consecutive_poll_failures == 1


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
        assert "DD_TRAILING" in reason

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


def _mock_brisbane_time(hour, weekday=1, day_offset=0):
    """Create a UTC datetime that maps to the given Brisbane hour and weekday.

    Brisbane = UTC+10. weekday: 0=Mon, 1=Tue, ...
    day_offset shifts the date forward/backward by N days.
    """
    # Start from a known Monday: 2026-04-06 is a Monday
    base = datetime(2026, 4, 6, tzinfo=UTC)
    # Shift to desired weekday
    target_bris = base + timedelta(days=weekday + day_offset)
    # Set Brisbane hour, then convert back to UTC
    target_bris = target_bris.replace(hour=hour, minute=0, second=0)
    target_utc = target_bris - timedelta(hours=10)  # Brisbane is UTC+10
    return target_utc


class TestDailyLossLimit:
    """Daily loss limit: non-ratcheting, resets at 09:00 Brisbane."""

    def test_daily_limit_hit_triggers_halt(self, state_dir):
        """Acceptance: daily limit hit → check_halt returns DAILY_LOSS."""
        # Tuesday 10:00 Brisbane
        t0 = _mock_brisbane_time(10, weekday=1)
        with patch("trading_app.account_hwm_tracker.datetime") as mock_dt:
            mock_dt.now.return_value = t0
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            t = AccountHWMTracker(
                "SELF001",
                "self_funded",
                dd_limit_dollars=3000.0,
                state_dir=state_dir,
                dd_type="none",
                daily_loss_limit=600.0,
            )
            t.update_equity(30000.0)
            # Lose $600 (exactly at limit)
            t.update_equity(29400.0)
            halted, reason = t.check_halt()
            assert halted
            assert "DAILY_LOSS" in reason

    def test_daily_limit_resets_next_day(self, state_dir):
        """Acceptance: daily limit resets next day → check_halt returns NONE."""
        # Tuesday 10:00 Brisbane
        t0 = _mock_brisbane_time(10, weekday=1)
        # Wednesday 10:00 Brisbane
        t1 = _mock_brisbane_time(10, weekday=2)

        with patch("trading_app.account_hwm_tracker.datetime") as mock_dt:
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            # Day 1: hit the limit
            mock_dt.now.return_value = t0
            t = AccountHWMTracker(
                "SELF001",
                "self_funded",
                dd_limit_dollars=3000.0,
                state_dir=state_dir,
                dd_type="none",
                daily_loss_limit=600.0,
            )
            t.update_equity(30000.0)
            t.update_equity(29400.0)
            halted, _ = t.check_halt()
            assert halted

            # Day 2: equity stayed at 29400, but daily counter resets
            mock_dt.now.return_value = t1
            t.update_equity(29400.0)
            halted, reason = t.check_halt()
            assert not halted
            assert "DAILY_LOSS" not in reason

    def test_below_daily_limit_no_halt(self, state_dir):
        """Loss below daily limit does NOT halt."""
        t0 = _mock_brisbane_time(10, weekday=1)
        with patch("trading_app.account_hwm_tracker.datetime") as mock_dt:
            mock_dt.now.return_value = t0
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            t = AccountHWMTracker(
                "SELF001",
                "self_funded",
                dd_limit_dollars=3000.0,
                state_dir=state_dir,
                dd_type="none",
                daily_loss_limit=600.0,
            )
            t.update_equity(30000.0)
            t.update_equity(29500.0)  # $500 loss < $600 limit
            halted, _ = t.check_halt()
            assert not halted


class TestWeeklyLossLimit:
    """Weekly loss limit: non-ratcheting, resets Monday 09:00 Brisbane."""

    def test_weekly_limit_hit_triggers_halt(self, state_dir):
        """Acceptance: weekly limit hit → check_halt returns WEEKLY_LOSS."""
        t0 = _mock_brisbane_time(10, weekday=2)  # Wednesday
        with patch("trading_app.account_hwm_tracker.datetime") as mock_dt:
            mock_dt.now.return_value = t0
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            t = AccountHWMTracker(
                "SELF001",
                "self_funded",
                dd_limit_dollars=3000.0,
                state_dir=state_dir,
                dd_type="none",
                weekly_loss_limit=1500.0,
            )
            t.update_equity(30000.0)
            t.update_equity(28500.0)  # $1500 loss = limit
            halted, reason = t.check_halt()
            assert halted
            assert "WEEKLY_LOSS" in reason

    def test_weekly_limit_resets_monday(self, state_dir):
        """Acceptance: weekly limit resets Monday → check_halt returns NONE."""
        # Friday 10:00
        t0 = _mock_brisbane_time(10, weekday=4)
        # Next Monday 10:00
        t1 = _mock_brisbane_time(10, weekday=0, day_offset=7)

        with patch("trading_app.account_hwm_tracker.datetime") as mock_dt:
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            # Week 1: hit the weekly limit
            mock_dt.now.return_value = t0
            t = AccountHWMTracker(
                "SELF001",
                "self_funded",
                dd_limit_dollars=3000.0,
                state_dir=state_dir,
                dd_type="none",
                weekly_loss_limit=1500.0,
            )
            t.update_equity(30000.0)
            t.update_equity(28500.0)
            halted, _ = t.check_halt()
            assert halted

            # Next Monday: weekly counter resets
            mock_dt.now.return_value = t1
            t.update_equity(28500.0)
            halted, reason = t.check_halt()
            assert not halted
            assert "WEEKLY_LOSS" not in reason


class TestPropProfileUnchanged:
    """Acceptance: prop profile (no daily/weekly) → unchanged behavior."""

    def test_no_daily_weekly_limits_behaves_as_before(self, state_dir):
        """Prop tracker without daily/weekly limits works identically."""
        t = AccountHWMTracker(
            "PROP001",
            "topstep",
            dd_limit_dollars=2000.0,
            state_dir=state_dir,
            dd_type="intraday_trailing",
            # No daily_loss_limit or weekly_loss_limit
        )
        t.update_equity(50000.0)
        t.update_equity(51000.0)
        t.update_equity(50500.0)
        halted, reason = t.check_halt()
        assert not halted
        assert "HWM_OK" in reason or "WARNING" in reason

        # Hit trailing DD limit
        t.update_equity(49000.0)
        halted, reason = t.check_halt()
        assert halted
        assert "DD_TRAILING" in reason

    def test_daily_weekly_none_skips_period_checks(self, state_dir):
        """With None limits, period state stays None."""
        t = AccountHWMTracker(
            "PROP001",
            "topstep",
            dd_limit_dollars=2000.0,
            state_dir=state_dir,
            dd_type="intraday_trailing",
        )
        t.update_equity(50000.0)
        assert t._daily_start_equity is None
        assert t._weekly_start_equity is None


class TestDailyWeeklyPersistence:
    """Period state persists across restarts."""

    def test_daily_state_survives_restart(self, state_dir):
        t0 = _mock_brisbane_time(10, weekday=1)
        with patch("trading_app.account_hwm_tracker.datetime") as mock_dt:
            mock_dt.now.return_value = t0
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            t1 = AccountHWMTracker(
                "SELF001",
                "self_funded",
                dd_limit_dollars=3000.0,
                state_dir=state_dir,
                dd_type="none",
                daily_loss_limit=600.0,
            )
            t1.update_equity(30000.0)
            saved_date = t1._daily_start_date

        # Reload from disk
        t2 = AccountHWMTracker(
            "SELF001",
            "self_funded",
            dd_limit_dollars=3000.0,
            state_dir=state_dir,
            dd_type="none",
            daily_loss_limit=600.0,
        )
        assert t2._daily_start_equity == 30000.0
        assert t2._daily_start_date == saved_date


class TestDailyWeeklyValidation:
    """Constructor validation for period limits."""

    def test_negative_daily_limit_raises(self, state_dir):
        with pytest.raises(ValueError, match="daily_loss_limit must be positive"):
            AccountHWMTracker(
                "X",
                "x",
                dd_limit_dollars=1000.0,
                state_dir=state_dir,
                daily_loss_limit=-100.0,
            )

    def test_negative_weekly_limit_raises(self, state_dir):
        with pytest.raises(ValueError, match="weekly_loss_limit must be positive"):
            AccountHWMTracker(
                "X",
                "x",
                dd_limit_dollars=1000.0,
                state_dir=state_dir,
                weekly_loss_limit=0,
            )


class TestDDTrailingTakesPrecedence:
    """Trailing DD halt takes priority over daily/weekly."""

    def test_trailing_dd_fires_before_daily(self, state_dir):
        t0 = _mock_brisbane_time(10, weekday=1)
        with patch("trading_app.account_hwm_tracker.datetime") as mock_dt:
            mock_dt.now.return_value = t0
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            t = AccountHWMTracker(
                "SELF001",
                "self_funded",
                dd_limit_dollars=500.0,  # Very tight DD limit
                state_dir=state_dir,
                dd_type="none",
                daily_loss_limit=600.0,
            )
            t.update_equity(30000.0)
            # Lose $500 — hits DD limit (500) before daily (600)
            t.update_equity(29500.0)
            halted, reason = t.check_halt()
            assert halted
            assert "DD_TRAILING" in reason


# ──────────────────────────────────────────────────────────────────────
# Stage 2 — HWM persistence integrity hardening (design v3 § 5)
#
# Tests in this section pin behavior introduced by stage doc
# `docs/runtime/stages/hwm-stage2-tracker-integrity.md` and parent design
# `docs/plans/2026-04-25-hwm-persistence-integrity-hardening-design.md`.
#
# Mutation-proof — each test asserts a specific code path with substring
# tokens that survive cosmetic refactor but fail when the path is removed.
# ──────────────────────────────────────────────────────────────────────


def _write_state_file_with_age(state_dir: Path, account_id: str, age_seconds: float) -> Path:
    """Write a minimal valid HWM state file with last_equity_timestamp set
    to (now - age_seconds). Returns the file path.

    Uses real timestamps (no datetime patching) so state_file_age_days()
    reads a genuinely-old file. Avoids the patch interaction with the
    pure helper.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    last_ts = (datetime.now(UTC) - timedelta(seconds=age_seconds)).isoformat()
    data = {
        "account_id": account_id,
        "firm": "topstep",
        "hwm_dollars": 50000.0,
        "hwm_timestamp": last_ts,
        "last_equity": 50000.0,
        "last_equity_timestamp": last_ts,
        "halt_triggered": False,
        "halt_timestamp": None,
        "halt_reason": "",
        "consecutive_poll_failures": 0,
        "hwm_frozen": False,
        "session_log": [],
    }
    path = state_dir / f"account_hwm_{account_id}.json"
    path.write_text(json.dumps(data))
    return path


class TestStaleStateBoundaries:
    """B2/B3 — stale-state warn (24h) and fail-closed raise (30 days)."""

    def test_load_stale_30_days_plus_1s_raises_with_stale_state_fail_token(self, state_dir):
        path = _write_state_file_with_age(state_dir, "STALE001", age_seconds=30 * 86400 + 1)
        with pytest.raises(RuntimeError, match=r"STALE_STATE_FAIL"):
            AccountHWMTracker("STALE001", "topstep", dd_limit_dollars=2000.0, state_dir=state_dir)
        # Verify message also names file path and the age figure
        try:
            AccountHWMTracker("STALE001", "topstep", dd_limit_dollars=2000.0, state_dir=state_dir)
        except RuntimeError as exc:
            msg = str(exc)
            assert str(path) in msg or path.name in msg, f"file path missing from raise: {msg!r}"
            assert "30" in msg, f"30-day threshold figure missing from raise: {msg!r}"
            assert "archive" in msg.lower() or "delete" in msg.lower(), f"repair recipe missing from raise: {msg!r}"

    def test_load_stale_30_days_minus_1s_warns_does_not_raise(self, state_dir, caplog):
        import logging

        _write_state_file_with_age(state_dir, "STALE002", age_seconds=30 * 86400 - 1)
        with caplog.at_level(logging.WARNING, logger="trading_app.account_hwm_tracker"):
            t = AccountHWMTracker("STALE002", "topstep", dd_limit_dollars=2000.0, state_dir=state_dir)
        assert t._hwm == 50000.0  # loaded successfully
        warns = [r for r in caplog.records if "old" in r.message.lower()]
        assert warns, f"Expected stale-state log.warning; got: {[r.message for r in caplog.records]!r}"

    def test_load_stale_29_days_logs_warning_continues(self, state_dir, caplog):
        import logging

        _write_state_file_with_age(state_dir, "STALE003", age_seconds=29 * 86400)
        with caplog.at_level(logging.WARNING, logger="trading_app.account_hwm_tracker"):
            t = AccountHWMTracker("STALE003", "topstep", dd_limit_dollars=2000.0, state_dir=state_dir)
        assert t._hwm == 50000.0
        warns = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("29" in r.message or "days old" in r.message for r in warns), (
            f"Expected age in warning message; got: {[r.message for r in warns]!r}"
        )

    def test_load_warn_24h_plus_1s_emits_log_warning(self, state_dir, caplog):
        import logging

        _write_state_file_with_age(state_dir, "STALE004", age_seconds=86400 + 1)
        with caplog.at_level(logging.WARNING, logger="trading_app.account_hwm_tracker"):
            AccountHWMTracker("STALE004", "topstep", dd_limit_dollars=2000.0, state_dir=state_dir)
        warns = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warns, f"24h+1s must warn; got no WARNING records"

    def test_load_warn_24h_minus_1s_silent(self, state_dir, caplog):
        import logging

        _write_state_file_with_age(state_dir, "STALE005", age_seconds=86400 - 1)
        with caplog.at_level(logging.WARNING, logger="trading_app.account_hwm_tracker"):
            AccountHWMTracker("STALE005", "topstep", dd_limit_dollars=2000.0, state_dir=state_dir)
        # Filter for stale-related WARNINGs only (other unrelated logs may exist)
        stale_warns = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and ("days old" in r.message or "old (account" in r.message)
        ]
        assert not stale_warns, f"Under-24h must be silent; got: {[r.message for r in stale_warns]!r}"

    def test_load_recent_silent(self, state_dir, caplog):
        import logging

        _write_state_file_with_age(state_dir, "STALE006", age_seconds=3600)  # 1 hour
        with caplog.at_level(logging.WARNING, logger="trading_app.account_hwm_tracker"):
            AccountHWMTracker("STALE006", "topstep", dd_limit_dollars=2000.0, state_dir=state_dir)
        stale_warns = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and ("days old" in r.message or "old (account" in r.message)
        ]
        assert not stale_warns

    def test_load_stale_message_does_not_claim_canonical_grounding(self, state_dir):
        """Mutation guard: design v3 § 2 explicitly disclaims canonical grounding
        for the 30-day figure. The raised message MUST NOT contain @canonical-source
        or topstep_xfa_parameters.txt:349 — those would falsely promote the
        operational heuristic to a literature-grounded rule."""
        _write_state_file_with_age(state_dir, "STALE007", age_seconds=30 * 86400 + 60)
        with pytest.raises(RuntimeError) as exc_info:
            AccountHWMTracker("STALE007", "topstep", dd_limit_dollars=2000.0, state_dir=state_dir)
        msg = str(exc_info.value)
        assert "@canonical-source" not in msg, f"Forbidden token in raise: {msg!r}"
        assert "topstep_xfa_parameters.txt:349" not in msg, f"Forbidden citation in raise: {msg!r}"


class TestNotifyCallback:
    """B4 / B5 / B6 — notify_callback dispatch on integrity events."""

    def test_load_corrupt_invokes_notify_callback_when_provided(self, state_dir):
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "account_hwm_NOTIFY001.json").write_text("{garbage")
        calls: list[str] = []
        AccountHWMTracker(
            "NOTIFY001",
            "topstep",
            dd_limit_dollars=2000.0,
            state_dir=state_dir,
            notify_callback=calls.append,
        )
        assert len(calls) == 1, f"Expected exactly one corrupt-state notify; got {len(calls)}: {calls!r}"
        assert "CORRUPT" in calls[0], f"Notify must mention CORRUPT; got: {calls[0]!r}"

    def test_load_corrupt_no_callback_preserves_existing_behavior(self, state_dir):
        """Backwards compat — log.error, backup, reinit. No exception."""
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "account_hwm_NOTIFY002.json").write_text("{garbage")
        t = AccountHWMTracker("NOTIFY002", "topstep", dd_limit_dollars=2000.0, state_dir=state_dir)
        assert t._hwm == 0.0  # reinit
        assert list(state_dir.glob("account_hwm_NOTIFY002_CORRUPT_*.json"))  # backup exists

    def test_load_corrupt_callback_raises_does_not_break_construction(self, state_dir, caplog):
        import logging

        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "account_hwm_NOTIFY003.json").write_text("{garbage")

        def _bad_callback(_msg: str) -> None:
            raise RuntimeError("simulated callback failure")

        with caplog.at_level(logging.ERROR, logger="trading_app.account_hwm_tracker"):
            t = AccountHWMTracker(
                "NOTIFY003",
                "topstep",
                dd_limit_dollars=2000.0,
                state_dir=state_dir,
                notify_callback=_bad_callback,
            )
        assert t._hwm == 0.0  # construction succeeded despite callback raise
        dispatch_failures = [r for r in caplog.records if "notify_callback dispatch failed" in r.message]
        assert dispatch_failures, "Expected log.error from _safe_notify dispatch failure"

    def test_poll_recovery_from_one_dispatches_notify_with_prior_count(self, state_dir):
        calls: list[str] = []
        t = AccountHWMTracker(
            "POLL001",
            "topstep",
            dd_limit_dollars=2000.0,
            state_dir=state_dir,
            notify_callback=calls.append,
        )
        t.update_equity(None)  # 1 failure
        t.update_equity(50000.0)  # recovery
        recovery_msgs = [c for c in calls if "RECOVERY" in c]
        assert len(recovery_msgs) == 1, f"Expected one recovery notify; got: {calls!r}"
        assert "1 consecutive failure" in recovery_msgs[0], (
            f"Expected '1 consecutive failure' in message; got: {recovery_msgs[0]!r}"
        )

    def test_poll_recovery_from_two_dispatches_notify_with_prior_count(self, state_dir):
        """Mutation guard: verify the count is the actual prior count, not a constant."""
        calls: list[str] = []
        t = AccountHWMTracker(
            "POLL002",
            "topstep",
            dd_limit_dollars=2000.0,
            state_dir=state_dir,
            notify_callback=calls.append,
        )
        t.update_equity(None)  # 1
        t.update_equity(None)  # 2
        t.update_equity(50000.0)  # recovery from 2
        recovery_msgs = [c for c in calls if "RECOVERY" in c]
        assert len(recovery_msgs) == 1
        assert "2 consecutive failure" in recovery_msgs[0], (
            f"Expected '2 consecutive failure' (not 1); got: {recovery_msgs[0]!r}"
        )

    def test_poll_recovery_from_zero_does_not_dispatch(self, state_dir):
        """Mutation guard against spam — steady-state successes are silent."""
        calls: list[str] = []
        t = AccountHWMTracker(
            "POLL003",
            "topstep",
            dd_limit_dollars=2000.0,
            state_dir=state_dir,
            notify_callback=calls.append,
        )
        t.update_equity(50000.0)
        t.update_equity(50100.0)
        t.update_equity(50200.0)
        recovery_msgs = [c for c in calls if "RECOVERY" in c]
        assert recovery_msgs == [], f"Steady-state must be silent; got: {recovery_msgs!r}"

    def test_save_state_oserror_dispatches_notify_and_reraises_with_persist_fail_token(self, state_dir):
        calls: list[str] = []
        t = AccountHWMTracker(
            "PERSIST001",
            "topstep",
            dd_limit_dollars=2000.0,
            state_dir=state_dir,
            notify_callback=calls.append,
        )
        t.update_equity(50000.0)  # primes state
        # Inject OSError on the next write_text call
        original_write = Path.write_text

        def _raising_write(self, *args, **kwargs):
            if "tmp" in self.name:
                raise OSError("disk full simulated")
            return original_write(self, *args, **kwargs)

        with patch.object(Path, "write_text", _raising_write):
            with pytest.raises(OSError, match="disk full simulated"):
                t.update_equity(50100.0)
        persist_msgs = [c for c in calls if "STATE_PERSIST_FAIL" in c]
        assert len(persist_msgs) == 1, f"Expected one persist-fail notify; got: {calls!r}"

    def test_save_state_oserror_does_not_increment_poll_failure_counter(self, state_dir):
        """Mutation guard — persistence and broker-poll failure modes stay separate."""
        t = AccountHWMTracker(
            "PERSIST002",
            "topstep",
            dd_limit_dollars=2000.0,
            state_dir=state_dir,
            notify_callback=lambda _msg: None,
        )
        t.update_equity(50000.0)
        original_write = Path.write_text

        def _raising_write(self, *args, **kwargs):
            if "tmp" in self.name:
                raise OSError("simulated")
            return original_write(self, *args, **kwargs)

        prior = t._consecutive_poll_failures
        with patch.object(Path, "write_text", _raising_write):
            with pytest.raises(OSError):
                t.update_equity(50100.0)
        assert t._consecutive_poll_failures == prior, (
            f"OSError on persist must not increment broker-poll failure counter "
            f"(was {prior}, now {t._consecutive_poll_failures})"
        )


class TestAnnotationDiscipline:
    """B7 / B8 — UNGROUNDED+Rationale on 6 constants; @canonical-source on class docstring."""

    def test_eod_trailing_class_docstring_has_canonical_source_annotation_block(self):
        doc = AccountHWMTracker.__doc__ or ""
        assert "@canonical-source" in doc, "Class docstring missing @canonical-source token"
        assert "@verbatim" in doc, "Class docstring missing @verbatim token"
        assert "@audit-finding" in doc, "Class docstring missing @audit-finding token"
        assert "topstep_xfa_parameters.txt:289" in doc, (
            "Class docstring must cite topstep_xfa_parameters.txt:289 (rule-breach grounding)"
        )

    def test_ungrounded_constants_have_explicit_label_and_rationale_within_5_lines_above(self):
        """Positional check (not file-wide grep): for each named constant, both
        UNGROUNDED token AND a Rationale: block must appear within the 5 lines
        immediately above the assignment. Pass Three drift-check forward-compat
        + institutional-rigor.md § 7 honesty requirement.
        """
        import inspect
        from trading_app import account_hwm_tracker as mod

        source_lines = inspect.getsource(mod).splitlines()
        target_constants = [
            "_MAX_SESSION_LOG = 30",
            "_MAX_CONSECUTIVE_POLL_FAILURES = 3",
            "_STATE_STALENESS_FAIL_DAYS = 30",
            "_STATE_STALENESS_WARN_DAYS = 1",
        ]
        for needle in target_constants:
            idx = next((i for i, line in enumerate(source_lines) if line.strip() == needle), None)
            assert idx is not None, f"Constant {needle!r} not found in module source"
            window = "\n".join(source_lines[max(0, idx - 5) : idx])
            assert "UNGROUNDED" in window, f"{needle}: missing UNGROUNDED token within 5 lines above. Window:\n{window}"
            assert "Rationale:" in window, (
                f"{needle}: missing 'Rationale:' block within 5 lines above. Window:\n{window}"
            )


class TestStateFileAgeDays:
    """B9 — shared pure helper for age computation."""

    def test_state_file_age_days_pure_function_no_logging(self, state_dir, caplog, tmp_path):
        from trading_app.account_hwm_tracker import state_file_age_days

        path = _write_state_file_with_age(state_dir, "AGE001", age_seconds=3600)
        before_files = set(state_dir.iterdir())
        with caplog.at_level("DEBUG", logger="trading_app.account_hwm_tracker"):
            age = state_file_age_days(path)
        assert age is not None
        assert 0.04 <= age <= 0.05, f"Expected ~1h ({3600 / 86400:.4f} days); got {age}"
        assert caplog.records == [], f"Helper must be silent; got: {[r.message for r in caplog.records]!r}"
        after_files = set(state_dir.iterdir())
        assert before_files == after_files, "Helper must not create files"

    def test_state_file_age_days_returns_none_on_missing_file(self, tmp_path):
        from trading_app.account_hwm_tracker import state_file_age_days

        assert state_file_age_days(tmp_path / "does_not_exist.json") is None

    def test_state_file_age_days_corrupt_json_falls_back_to_mtime(self, state_dir):
        """SG1 fix (Stage 2 audit-gate): corrupt JSON must NOT return None,
        because that bypasses the fail-closed stale gate. Falls back to file
        mtime — strictly fresher than last_equity_timestamp would be."""
        import os
        from trading_app.account_hwm_tracker import state_file_age_days

        state_dir.mkdir(parents=True, exist_ok=True)
        path = state_dir / "corrupt.json"
        path.write_text("{not json")
        # Force mtime to 40 days ago — proves the fallback is used
        forty_days_ago = datetime.now(UTC).timestamp() - (40 * 86400)
        os.utime(path, (forty_days_ago, forty_days_ago))
        age = state_file_age_days(path)
        assert age is not None, "Corrupt JSON must NOT return None — that bypasses the stale gate"
        assert 39.5 < age < 40.5, f"Expected ~40 day mtime-based age; got {age}"

    def test_state_file_age_days_missing_timestamp_falls_back_to_mtime(self, state_dir):
        """SG1 fix: missing/null last_equity_timestamp must NOT return None."""
        import os
        from trading_app.account_hwm_tracker import state_file_age_days

        state_dir.mkdir(parents=True, exist_ok=True)
        path = state_dir / "no_ts.json"
        path.write_text(json.dumps({"hwm_dollars": 50000.0, "last_equity_timestamp": None}))
        forty_days_ago = datetime.now(UTC).timestamp() - (40 * 86400)
        os.utime(path, (forty_days_ago, forty_days_ago))
        age = state_file_age_days(path)
        assert age is not None
        assert 39.5 < age < 40.5

    def test_state_file_age_days_returns_none_only_when_file_missing(self, tmp_path):
        """Helper returns None ONLY when the file does not exist. All other
        paths fall back to mtime so the stale gate cannot be bypassed."""
        from trading_app.account_hwm_tracker import state_file_age_days

        assert state_file_age_days(tmp_path / "nope.json") is None

    def test_load_stale_gate_fires_on_null_timestamp_old_mtime(self, state_dir):
        """SG1 — fail-closed gate must fire when timestamp is null but file
        is genuinely stale by mtime. Pre-fix this bypassed both gates and
        loaded successfully with stale balance."""
        import os

        state_dir.mkdir(parents=True, exist_ok=True)
        path = state_dir / "account_hwm_SG1NULL.json"
        path.write_text(
            json.dumps(
                {
                    "account_id": "SG1NULL",
                    "firm": "topstep",
                    "hwm_dollars": 50000.0,
                    "last_equity": 50000.0,
                    "last_equity_timestamp": None,  # the bug condition
                }
            )
        )
        forty_days_ago = datetime.now(UTC).timestamp() - (40 * 86400)
        os.utime(path, (forty_days_ago, forty_days_ago))
        with pytest.raises(RuntimeError, match=r"STALE_STATE_FAIL"):
            AccountHWMTracker("SG1NULL", "topstep", dd_limit_dollars=2000.0, state_dir=state_dir)

    def test_load_stale_gate_fires_on_corrupt_json_old_mtime(self, state_dir):
        """SG1 — fail-closed gate must fire when JSON is corrupt but file is
        old. Pre-fix the corrupt path would run instead, silently reinitialising
        from broker without any age signal."""
        import os

        state_dir.mkdir(parents=True, exist_ok=True)
        path = state_dir / "account_hwm_SG1CORRUPT.json"
        path.write_text("{not json")
        forty_days_ago = datetime.now(UTC).timestamp() - (40 * 86400)
        os.utime(path, (forty_days_ago, forty_days_ago))
        with pytest.raises(RuntimeError, match=r"STALE_STATE_FAIL"):
            AccountHWMTracker("SG1CORRUPT", "topstep", dd_limit_dollars=2000.0, state_dir=state_dir)


class TestPostHaltRecoveryQualifier:
    """SG3 fix — post-halt POLL_FAILURE recovery notify includes
    REMAINS HALTED qualifier so operator does not misread RECOVERY as 'safe'."""

    def test_post_halt_recovery_notify_includes_remains_halted_qualifier(self, state_dir):
        calls: list[str] = []
        t = AccountHWMTracker(
            "SG3HALT",
            "topstep",
            dd_limit_dollars=2000.0,
            state_dir=state_dir,
            notify_callback=calls.append,
        )
        t.update_equity(50000.0)
        # 3 consecutive None polls → POLL_FAILURE halt
        t.update_equity(None)
        t.update_equity(None)
        t.update_equity(None)
        assert t._halt is True
        assert t._halt_reason == "POLL_FAILURE"
        # Recovery while still halted
        t.update_equity(50100.0)
        recovery = [c for c in calls if "RECOVERY" in c]
        assert len(recovery) == 1
        assert "REMAINS HALTED" in recovery[0], (
            f"Post-halt recovery notify must mention REMAINS HALTED; got: {recovery[0]!r}"
        )
        assert "reset_halt" in recovery[0], f"Notify must name the operator action; got: {recovery[0]!r}"

    def test_pre_halt_recovery_notify_excludes_remains_halted_qualifier(self, state_dir):
        """Mutation guard: the qualifier must NOT fire when recovery happens
        before the halt threshold is hit. (1 failure → recovery: silent halt
        reason, no qualifier needed.)"""
        calls: list[str] = []
        t = AccountHWMTracker(
            "SG3OK",
            "topstep",
            dd_limit_dollars=2000.0,
            state_dir=state_dir,
            notify_callback=calls.append,
        )
        t.update_equity(50000.0)
        t.update_equity(None)  # 1 failure, below threshold
        assert t._halt is False
        t.update_equity(50100.0)
        recovery = [c for c in calls if "RECOVERY" in c]
        assert len(recovery) == 1
        assert "REMAINS HALTED" not in recovery[0], (
            f"Pre-halt recovery must NOT mention REMAINS HALTED; got: {recovery[0]!r}"
        )


class TestNonFiniteEquity:
    """SG4 fix — NaN/Inf equity from broker routes through poll-failure path
    so the 3-strike halt mechanism actually engages instead of NaN silently
    bypassing every comparison."""

    def test_nan_equity_treated_as_poll_failure(self, state_dir):
        t = AccountHWMTracker("SG4NAN", "topstep", dd_limit_dollars=2000.0, state_dir=state_dir)
        t.update_equity(50000.0)  # primes a real value
        prior = t._consecutive_poll_failures
        t.update_equity(float("nan"))
        assert t._consecutive_poll_failures == prior + 1, (
            f"NaN must increment poll-failure counter; was {prior}, now {t._consecutive_poll_failures}"
        )
        # Equity field MUST NOT have absorbed the NaN
        assert t._last_equity == 50000.0, f"NaN must not propagate into _last_equity; got {t._last_equity}"

    def test_positive_inf_equity_treated_as_poll_failure(self, state_dir):
        t = AccountHWMTracker("SG4INF", "topstep", dd_limit_dollars=2000.0, state_dir=state_dir)
        t.update_equity(50000.0)
        t.update_equity(float("inf"))
        assert t._consecutive_poll_failures == 1
        assert t._last_equity == 50000.0

    def test_three_consecutive_nan_polls_fire_halt(self, state_dir):
        """Mutation guard for SG4: NaN must walk to halt just like None."""
        t = AccountHWMTracker("SG4HALT", "topstep", dd_limit_dollars=2000.0, state_dir=state_dir)
        t.update_equity(50000.0)
        t.update_equity(float("nan"))
        t.update_equity(float("nan"))
        t.update_equity(float("nan"))
        assert t._halt is True
        assert t._halt_reason == "POLL_FAILURE"

    def test_bool_equity_treated_as_poll_failure(self, state_dir):
        """SG-NEW-1 (audit-gate follow-up to SG4): a buggy adapter returning
        True/False must NOT be silently recorded as equity=1.0 / equity=0.0.
        Python `isinstance(True, int)` is True by language design — without
        an explicit bool guard, _is_finite_equity(True) would pass and the
        kill-switch would never engage on this caller-contract violation."""
        t = AccountHWMTracker("SGNEWBOOL", "topstep", dd_limit_dollars=2000.0, state_dir=state_dir)
        t.update_equity(50000.0)
        prior = t._consecutive_poll_failures
        t.update_equity(True)  # type: ignore[arg-type]
        assert t._consecutive_poll_failures == prior + 1, (
            f"True must route through poll-failure path; was {prior}, now {t._consecutive_poll_failures}"
        )
        assert t._last_equity == 50000.0, f"True must not propagate into _last_equity; got {t._last_equity}"
        # And False
        t.update_equity(False)  # type: ignore[arg-type]
        assert t._consecutive_poll_failures == prior + 2

    def test_daily_loss_halt_active_at_recovery_does_not_get_poll_failure_qualifier(self, state_dir):
        """SG-NEW-2 (audit-gate documentation): the REMAINS HALTED qualifier
        is intentionally scoped to halt_reason == "POLL_FAILURE" only. A halt
        from DAILY_LOSS / WEEKLY_LOSS / DD_TRAILING is unrelated to poll
        recovery — the operator action required is different (review the
        actual DD breach, not retry the broker connection). This test pins
        that intentional design so a future Stage 3 refactor does not
        accidentally widen the qualifier to all halt reasons."""
        calls: list[str] = []
        t = AccountHWMTracker(
            "SGNEW2",
            "topstep",
            dd_limit_dollars=2000.0,
            state_dir=state_dir,
            daily_loss_limit=500.0,
            dd_type="intraday_trailing",
            notify_callback=calls.append,
        )
        t.update_equity(50000.0)
        # Trigger DAILY_LOSS halt
        t.update_equity(49400.0)  # daily loss = 600 > 500 limit
        assert t._halt is True
        assert t._halt_reason == "DAILY_LOSS"
        # Now a poll failure followed by recovery (still halted by DAILY_LOSS)
        t.update_equity(None)
        t.update_equity(49500.0)
        recovery = [c for c in calls if "RECOVERY" in c]
        assert len(recovery) == 1
        assert "REMAINS HALTED" not in recovery[0], (
            f"Non-POLL_FAILURE halt must not get the qualifier; got: {recovery[0]!r}"
        )


class TestEmptyStateFile:
    """SG2 — empty state file routes through corrupt path with notify dispatch."""

    def test_empty_file_with_callback_dispatches_corrupt_notify(self, state_dir):
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "account_hwm_SG2EMPTY.json").write_text("")
        calls: list[str] = []
        t = AccountHWMTracker(
            "SG2EMPTY",
            "topstep",
            dd_limit_dollars=2000.0,
            state_dir=state_dir,
            notify_callback=calls.append,
        )
        # Empty file routes through corrupt path → reinit
        assert t._hwm == 0.0
        # Backup created
        assert list(state_dir.glob("account_hwm_SG2EMPTY_CORRUPT_*.json"))
        # Notify dispatched
        corrupt_notifies = [c for c in calls if "CORRUPT" in c]
        assert len(corrupt_notifies) == 1, f"Expected one CORRUPT notify; got {calls!r}"


class TestReadStateFileSharedHelper:
    """Stage 3 — shared `read_state_file` helper for external consumers
    (pre_session_check, weekly_review, scripts). Returns dict on success,
    None on missing/empty/JSON-error/non-dict-top-level/OSError; emits
    log.warning with file path + granular reason on every None-return path
    (institutional-rigor.md § 6 — no silent failures).
    """

    def test_read_state_file_returns_dict_on_valid_state(self, tmp_path):
        from trading_app.account_hwm_tracker import read_state_file

        f = tmp_path / "account_hwm_OK.json"
        f.write_text(json.dumps({"account_id": "OK", "hwm_dollars": 50000.0}))
        data = read_state_file(f)
        assert data is not None
        assert data["account_id"] == "OK"
        assert data["hwm_dollars"] == 50000.0

    def test_read_state_file_returns_none_on_missing_file(self, tmp_path, caplog):
        from trading_app.account_hwm_tracker import read_state_file

        missing = tmp_path / "account_hwm_MISSING.json"
        with caplog.at_level("WARNING", logger="trading_app.account_hwm_tracker"):
            assert read_state_file(missing) is None
        assert any("does not exist" in r.message for r in caplog.records), (
            f"Expected 'does not exist' in log; got {[r.message for r in caplog.records]!r}"
        )

    def test_read_state_file_returns_none_on_empty_file(self, tmp_path, caplog):
        from trading_app.account_hwm_tracker import read_state_file

        f = tmp_path / "account_hwm_EMPTY.json"
        f.write_text("")
        with caplog.at_level("WARNING", logger="trading_app.account_hwm_tracker"):
            assert read_state_file(f) is None
        assert any("is empty" in r.message for r in caplog.records)

    def test_read_state_file_returns_none_on_corrupt_json(self, tmp_path, caplog):
        from trading_app.account_hwm_tracker import read_state_file

        f = tmp_path / "account_hwm_CORRUPT.json"
        f.write_text("{not valid json")
        with caplog.at_level("WARNING", logger="trading_app.account_hwm_tracker"):
            assert read_state_file(f) is None
        assert any("JSON parse failed" in r.message for r in caplog.records)

    def test_read_state_file_returns_none_on_non_dict_top_level(self, tmp_path, caplog):
        from trading_app.account_hwm_tracker import read_state_file

        f = tmp_path / "account_hwm_LIST.json"
        f.write_text(json.dumps([1, 2, 3]))
        with caplog.at_level("WARNING", logger="trading_app.account_hwm_tracker"):
            assert read_state_file(f) is None
        assert any("not a dict" in r.message for r in caplog.records)

    def test_read_state_file_returns_none_on_oserror(self, tmp_path, caplog, monkeypatch):
        """Mutation guard: silently swallowing the OSError (no log.warning) flips this test."""
        from trading_app.account_hwm_tracker import read_state_file

        f = tmp_path / "account_hwm_OSERROR.json"
        f.write_text(json.dumps({"account_id": "X"}))

        # Patch read_text on Path to raise OSError
        from pathlib import Path as _Path

        original_read_text = _Path.read_text

        def boom(self, *args, **kwargs):
            if str(self).endswith("account_hwm_OSERROR.json"):
                raise OSError("simulated permission denied")
            return original_read_text(self, *args, **kwargs)

        monkeypatch.setattr(_Path, "read_text", boom)
        with caplog.at_level("WARNING", logger="trading_app.account_hwm_tracker"):
            assert read_state_file(f) is None
        assert any("OSError" in r.message for r in caplog.records), (
            f"Expected granular OSError reason in log (no silent swallow); got {[r.message for r in caplog.records]!r}"
        )
