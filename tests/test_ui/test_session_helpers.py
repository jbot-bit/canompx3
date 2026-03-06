# tests/test_ui/test_session_helpers.py
"""Tests for DST-safe session scheduling and filter translation."""

import pytest
from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo
from unittest.mock import patch

BRISBANE = ZoneInfo("Australia/Brisbane")


# ── get_upcoming_sessions ────────────────────────────────────────────────────


class TestGetUpcomingSessions:
    """Chronological session ordering — no trading-day boundary logic."""

    def test_morning_returns_remaining_sessions(self):
        """At 9:30 AM, CME_REOPEN (9:00) is past, TOKYO_OPEN (10:00) is next."""
        from ui.session_helpers import get_upcoming_sessions

        now = datetime(2026, 3, 6, 9, 30, tzinfo=BRISBANE)
        upcoming = get_upcoming_sessions(now)
        names = [name for name, _ in upcoming]

        assert "CME_REOPEN" not in names, "9:00 session should be past at 9:30"
        assert names[0] == "TOKYO_OPEN", "10:00 session should be next"

    def test_late_night_returns_overnight_sessions(self):
        """At 11:45 PM, NYSE_OPEN (00:30 tomorrow) should be next."""
        from ui.session_helpers import get_upcoming_sessions

        now = datetime(2026, 3, 6, 23, 45, tzinfo=BRISBANE)
        upcoming = get_upcoming_sessions(now)
        names = [name for name, _ in upcoming]

        assert names[0] == "NYSE_OPEN", "00:30 tomorrow should be next after 23:45"

    def test_no_duplicate_sessions(self):
        """Each session name appears at most once."""
        from ui.session_helpers import get_upcoming_sessions

        now = datetime(2026, 3, 6, 8, 0, tzinfo=BRISBANE)
        upcoming = get_upcoming_sessions(now)
        names = [name for name, _ in upcoming]

        assert len(names) == len(set(names)), f"Duplicates found: {names}"

    def test_dst_transition_cme_reopen_at_0800(self):
        """During US DST, CME_REOPEN is at 08:00 Brisbane — must still be found."""
        from ui.session_helpers import get_upcoming_sessions

        # Mar 8 2026 = US spring forward day, CME_REOPEN at 08:00 Brisbane
        now = datetime(2026, 3, 8, 7, 50, tzinfo=BRISBANE)
        upcoming = get_upcoming_sessions(now)
        names = [name for name, _ in upcoming]

        assert "CME_REOPEN" in names, "CME_REOPEN at 08:00 must appear"
        # It should be the first upcoming session
        assert names[0] == "CME_REOPEN"

    def test_dst_transition_nyse_open_at_2330(self):
        """During US DST, NYSE_OPEN is at 23:30 same calendar day (not 00:30 next)."""
        from ui.session_helpers import get_upcoming_sessions

        # Mar 8 2026 = US spring forward, NYSE_OPEN at 23:30 Brisbane
        now = datetime(2026, 3, 8, 23, 0, tzinfo=BRISBANE)
        upcoming = get_upcoming_sessions(now)
        names = [name for name, _ in upcoming]

        assert names[0] == "NYSE_OPEN", "NYSE_OPEN at 23:30 should be next"

    def test_all_sessions_are_real_datetimes(self):
        """Every returned session has a timezone-aware Brisbane datetime."""
        from ui.session_helpers import get_upcoming_sessions

        now = datetime(2026, 3, 6, 8, 0, tzinfo=BRISBANE)
        upcoming = get_upcoming_sessions(now)

        for name, dt in upcoming:
            assert dt.tzinfo is not None, f"{name} has naive datetime"
            assert dt > now, f"{name} at {dt} is not in the future"


# ── current_trading_day ──────────────────────────────────────────────────────


class TestCurrentTradingDay:
    """Uses pipeline's compute_trading_day — just a wrapper."""

    def test_morning_after_9am(self):
        from ui.session_helpers import current_trading_day

        td = current_trading_day(datetime(2026, 3, 6, 10, 0, tzinfo=BRISBANE))
        assert td == date(2026, 3, 6)

    def test_overnight_before_9am(self):
        from ui.session_helpers import current_trading_day

        td = current_trading_day(datetime(2026, 3, 7, 2, 0, tzinfo=BRISBANE))
        assert td == date(2026, 3, 6), "2 AM Saturday = Friday trading day"

    def test_exactly_9am_is_new_day(self):
        from ui.session_helpers import current_trading_day

        td = current_trading_day(datetime(2026, 3, 6, 9, 0, tzinfo=BRISBANE))
        assert td == date(2026, 3, 6)


# ── is_weekend ───────────────────────────────────────────────────────────────


class TestIsWeekend:
    def test_friday_morning_not_weekend(self):
        from ui.session_helpers import is_weekend

        now = datetime(2026, 3, 6, 10, 0, tzinfo=BRISBANE)  # Friday
        assert is_weekend(now) is False

    def test_saturday_morning_after_sessions_done(self):
        from ui.session_helpers import is_weekend

        # Saturday 10 AM — all Friday overnight sessions complete, no new ones
        now = datetime(2026, 3, 7, 10, 0, tzinfo=BRISBANE)
        assert is_weekend(now) is True

    def test_saturday_early_morning_not_weekend(self):
        from ui.session_helpers import is_weekend

        # Saturday 3 AM — Friday overnight sessions might still be active
        now = datetime(2026, 3, 7, 3, 0, tzinfo=BRISBANE)
        # This is tricky: CME is closed, but our function should still return True
        # because there are no real sessions on a Saturday (CME closes Fri 5 PM CT)
        # Actually Friday's last session is NYSE_CLOSE at 7 AM Sat.
        # At 3 AM Sat, COMEX_SETTLE (4:30 AM) and CME_PRECLOSE (6:45 AM) are still upcoming
        # BUT — these are only valid on trading days, and Saturday is not a trading day
        # So is_weekend should check: is the calendar day a weekend?
        # Friday overnight sessions are FRIDAY's trading day — they already happened
        # The dashboard at 3 AM Saturday should show weekend state
        assert is_weekend(now) is True

    def test_sunday_afternoon(self):
        from ui.session_helpers import is_weekend

        now = datetime(2026, 3, 8, 14, 0, tzinfo=BRISBANE)  # Sunday
        assert is_weekend(now) is True

    def test_monday_morning(self):
        from ui.session_helpers import is_weekend

        now = datetime(2026, 3, 9, 8, 50, tzinfo=BRISBANE)  # Monday
        assert is_weekend(now) is False


# ── filter_to_english ────────────────────────────────────────────────────────


class TestFilterToEnglish:
    def test_g_filter_basic(self):
        from ui.session_helpers import filter_to_english

        assert filter_to_english("ORB_G4") == "ORB >= 4pts"

    def test_g_filter_with_cont(self):
        from ui.session_helpers import filter_to_english

        result = filter_to_english("ORB_G5_CONT")
        assert "5pts" in result
        assert "continuation" in result.lower()

    def test_g_filter_with_fast(self):
        from ui.session_helpers import filter_to_english

        result = filter_to_english("ORB_G4_FAST10")
        assert "4pts" in result
        assert "fast" in result.lower()

    def test_g_filter_with_nomon(self):
        from ui.session_helpers import filter_to_english

        result = filter_to_english("ORB_G6_NOMON")
        assert "6pts" in result
        assert "monday" in result.lower()

    def test_vol_filter(self):
        from ui.session_helpers import filter_to_english

        result = filter_to_english("VOL_RV12_N20")
        assert "vol" in result.lower()

    def test_dir_long(self):
        from ui.session_helpers import filter_to_english

        result = filter_to_english("DIR_LONG")
        assert "long" in result.lower()

    def test_unknown_filter_returns_itself(self):
        from ui.session_helpers import filter_to_english

        assert filter_to_english("SOME_NEW_FILTER") == "SOME_NEW_FILTER"


# ── build_session_briefings ──────────────────────────────────────────────────


class TestBuildSessionBriefings:
    """Merge multiple strategies per session+instrument into one briefing."""

    def test_merges_multiple_filters_same_instrument(self):
        from ui.session_helpers import build_session_briefings

        briefings = build_session_briefings()
        # CME_REOPEN has MGC with 3 filters (G5, VOL_RV12_N20, G4_FAST10)
        cme_mgc = [b for b in briefings if b.session == "CME_REOPEN" and b.instrument == "MGC"]
        assert len(cme_mgc) == 1, "Should merge to single briefing per session+instrument"
        assert len(cme_mgc[0].conditions) >= 2, "Should have multiple conditions"

    def test_briefing_has_rr_target(self):
        from ui.session_helpers import build_session_briefings

        briefings = build_session_briefings()
        for b in briefings:
            assert b.rr_target is not None and b.rr_target > 0, f"{b.session} {b.instrument} missing rr_target"

    def test_briefing_has_session_time(self):
        from ui.session_helpers import build_session_briefings

        briefings = build_session_briefings()
        for b in briefings:
            assert b.session_hour is not None, f"{b.session} missing session time"


# ── get_app_state ────────────────────────────────────────────────────────────


class TestGetAppState:
    def test_returns_valid_state_name(self):
        from ui.session_helpers import get_app_state

        now = datetime(2026, 3, 6, 10, 30, tzinfo=BRISBANE)
        state = get_app_state(now)
        assert state.name in (
            "WEEKEND",
            "IDLE",
            "APPROACHING",
            "ALERT",
            "LIVE",
            "POST",
            "OVERNIGHT",
        )

    def test_weekend_on_saturday(self):
        from ui.session_helpers import get_app_state

        now = datetime(2026, 3, 7, 12, 0, tzinfo=BRISBANE)
        state = get_app_state(now)
        assert state.name == "WEEKEND"

    def test_alert_close_to_session(self):
        from ui.session_helpers import get_app_state

        # 5 min before TOKYO_OPEN (10:00)
        now = datetime(2026, 3, 6, 9, 55, tzinfo=BRISBANE)
        state = get_app_state(now)
        assert state.name == "ALERT"

    def test_approaching_30min_before(self):
        from ui.session_helpers import get_app_state

        # 30 min before TOKYO_OPEN (10:00)
        now = datetime(2026, 3, 6, 9, 30, tzinfo=BRISBANE)
        state = get_app_state(now)
        assert state.name == "APPROACHING"

    def test_idle_long_gap(self):
        from ui.session_helpers import get_app_state

        # 12:00 PM — 6 hours until LONDON_METALS at 18:00
        now = datetime(2026, 3, 6, 12, 0, tzinfo=BRISBANE)
        state = get_app_state(now)
        assert state.name == "IDLE"

    def test_overnight_late_evening(self):
        from ui.session_helpers import get_app_state

        # 9:30 PM — next session is US_DATA_830 at 11:30 PM (outside awake hours)
        now = datetime(2026, 3, 6, 21, 30, tzinfo=BRISBANE)
        state = get_app_state(now)
        assert state.name == "OVERNIGHT"

    def test_state_has_next_session_info(self):
        from ui.session_helpers import get_app_state

        now = datetime(2026, 3, 6, 9, 30, tzinfo=BRISBANE)
        state = get_app_state(now)
        assert state.next_session is not None
        assert state.next_session_dt is not None
        assert state.minutes_to_next is not None


# ── get_refresh_seconds ──────────────────────────────────────────────────────


class TestGetRefreshSeconds:
    def test_weekend_slow_refresh(self):
        from ui.session_helpers import get_refresh_seconds

        assert get_refresh_seconds(minutes_to_next=2000, is_weekend=True) >= 60

    def test_alert_fast_refresh(self):
        from ui.session_helpers import get_refresh_seconds

        assert get_refresh_seconds(minutes_to_next=5, is_weekend=False) <= 10

    def test_idle_medium_refresh(self):
        from ui.session_helpers import get_refresh_seconds

        assert 15 <= get_refresh_seconds(minutes_to_next=120, is_weekend=False) <= 60
