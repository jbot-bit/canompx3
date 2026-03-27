# tests/test_ui/test_session_helpers.py
"""Tests for DST-safe session scheduling and filter translation."""

from datetime import date, datetime, time, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

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


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestBuildSessionBriefings:
    """Merge multiple strategies per session+instrument into one briefing."""

    def test_merges_multiple_filters_same_instrument(self):
        from ui.session_helpers import build_session_briefings

        briefings = build_session_briefings()
        # Find ANY session+instrument with 2+ conditions to verify merge behavior.
        # (CME_REOPEN+MGC is seasonal-dependent — FAST10 gated off outside Nov-Feb.)
        multi_condition = [b for b in briefings if len(b.conditions) >= 2]
        assert len(multi_condition) >= 1, "At least one session+instrument should have multiple merged conditions"
        # Verify merge produces exactly 1 briefing per (session, instrument)
        seen = set()
        for b in briefings:
            key = (b.session, b.instrument)
            assert key not in seen, f"Duplicate briefing for {key}"
            seen.add(key)

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


# ── filter_to_english — new modifiers ────────────────────────────────────────


class TestFilterToEnglishNewModifiers:
    """Tests for filter modifiers added in DST awareness update."""

    def test_fast5(self):
        from ui.session_helpers import filter_to_english

        result = filter_to_english("ORB_G4_FAST5")
        assert "4pts" in result
        assert "fast" in result.lower()
        assert "5 bars" in result

    def test_notue(self):
        from ui.session_helpers import filter_to_english

        result = filter_to_english("ORB_G6_NOTUE")
        assert "6pts" in result
        assert "tuesday" in result.lower()

    def test_nofri(self):
        from ui.session_helpers import filter_to_english

        result = filter_to_english("ORB_G5_NOFRI")
        assert "5pts" in result
        assert "friday" in result.lower()

    def test_no_filter(self):
        from ui.session_helpers import filter_to_english

        result = filter_to_english("NO_FILTER")
        assert "all days" in result.lower()


# ── DST transition detection ─────────────────────────────────────────────────


class TestDSTSessionChanges:
    """Tests for DST transition detection functions."""

    def test_us_spring_forward_detects_7_sessions(self):
        """US DST starts Mar 8 2026. Monday Mar 9 should detect 7 shifted sessions."""
        from ui.session_helpers import get_dst_session_changes

        changes = get_dst_session_changes(date(2026, 3, 9))
        assert len(changes) == 7, f"Expected 7 US-linked sessions, got {len(changes)}"

    def test_us_spring_forward_all_minus_60(self):
        """All 7 sessions should shift -60 minutes (earlier) on spring forward."""
        from ui.session_helpers import get_dst_session_changes

        changes = get_dst_session_changes(date(2026, 3, 9))
        for c in changes:
            assert c.shift_minutes == -60, f"{c.session} shifted {c.shift_minutes}min, expected -60"

    def test_us_spring_forward_includes_cme_reopen(self):
        from ui.session_helpers import get_dst_session_changes

        changes = get_dst_session_changes(date(2026, 3, 9))
        names = {c.session for c in changes}
        assert "CME_REOPEN" in names

    def test_us_spring_forward_excludes_asia(self):
        """TOKYO_OPEN and SINGAPORE_OPEN should NOT shift (no DST in Asia)."""
        from ui.session_helpers import get_dst_session_changes

        changes = get_dst_session_changes(date(2026, 3, 9))
        names = {c.session for c in changes}
        assert "TOKYO_OPEN" not in names
        assert "SINGAPORE_OPEN" not in names
        assert "BRISBANE_1025" not in names

    def test_nyse_open_midnight_wrap(self):
        """NYSE_OPEN wraps from 00:30 to 23:30 — should be -60min, not +1380."""
        from ui.session_helpers import get_dst_session_changes

        changes = get_dst_session_changes(date(2026, 3, 9))
        nyse = [c for c in changes if c.session == "NYSE_OPEN"]
        assert len(nyse) == 1
        assert nyse[0].shift_minutes == -60
        assert nyse[0].old_hour == 0 and nyse[0].old_minute == 30
        assert nyse[0].new_hour == 23 and nyse[0].new_minute == 30

    def test_normal_day_no_changes(self):
        """A day with no DST transition should return empty list."""
        from ui.session_helpers import get_dst_session_changes

        # Mar 11 2026 (Wed) — DST already happened on Mar 8, no change today
        changes = get_dst_session_changes(date(2026, 3, 11))
        assert len(changes) == 0

    def test_us_fall_back_detects_plus_60(self):
        """US DST ends Nov 1 2026. Monday Nov 2 should detect +60 shifts."""
        from ui.session_helpers import get_dst_session_changes

        changes = get_dst_session_changes(date(2026, 11, 2))
        assert len(changes) >= 6  # at least 6 US-linked sessions
        for c in changes:
            assert c.shift_minutes == 60, f"{c.session} shifted {c.shift_minutes}min, expected +60"


class TestRecentDSTChanges:
    """Tests for get_recent_dst_changes — lookback window."""

    def test_detects_changes_within_lookback(self):
        from ui.session_helpers import get_recent_dst_changes

        # Day of transition
        recent = get_recent_dst_changes(date(2026, 3, 9), lookback_days=3)
        assert len(recent) == 7

    def test_still_detects_2_days_after(self):
        from ui.session_helpers import get_recent_dst_changes

        # 2 days after DST transition (Wed Mar 11)
        recent = get_recent_dst_changes(date(2026, 3, 11), lookback_days=3)
        assert len(recent) == 7, "Should still show within 3-day window"

    def test_gone_after_lookback_expires(self):
        from ui.session_helpers import get_recent_dst_changes

        # 5 days after (Fri Mar 13) with 3-day lookback
        recent = get_recent_dst_changes(date(2026, 3, 13), lookback_days=3)
        assert len(recent) == 0, "Should expire after lookback window"

    def test_deduplicates_by_session(self):
        from ui.session_helpers import get_recent_dst_changes

        recent = get_recent_dst_changes(date(2026, 3, 9), lookback_days=3)
        names = [c.session for c in recent]
        assert len(names) == len(set(names)), "Should deduplicate"


class TestUpcomingDSTTransitions:
    """Tests for get_upcoming_dst_transitions."""

    def test_detects_uk_dst_from_march(self):
        """From Mar 9, UK DST on Mar 29 should be detected within 30 days."""
        from ui.session_helpers import get_upcoming_dst_transitions

        upcoming = get_upcoming_dst_transitions(date(2026, 3, 9), lookahead_days=30)
        uk = [t for t in upcoming if t.region == "UK"]
        assert len(uk) == 1
        assert uk[0].direction == "start"
        assert uk[0].transition_date == date(2026, 3, 29)
        assert uk[0].days_away == 20

    def test_no_us_transition_right_after_spring_forward(self):
        """Right after US DST starts, next US transition is in November — too far."""
        from ui.session_helpers import get_upcoming_dst_transitions

        upcoming = get_upcoming_dst_transitions(date(2026, 3, 9), lookahead_days=30)
        us = [t for t in upcoming if t.region == "US"]
        assert len(us) == 0, "Nov transition is >30 days away"

    def test_returns_empty_when_no_transitions(self):
        """Mid-summer — no transitions within 30 days."""
        from ui.session_helpers import get_upcoming_dst_transitions

        upcoming = get_upcoming_dst_transitions(date(2026, 7, 1), lookahead_days=30)
        assert len(upcoming) == 0
