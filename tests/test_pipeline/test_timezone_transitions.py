"""Tests for timezone transitions and DST handling (T10).

Brisbane (UTC+10) has NO daylight saving, but US markets do.
These tests verify that trading day boundaries and UTC conversions
are correct across US DST transitions.
"""
import pytest
from datetime import datetime, date, timedelta, timezone
from zoneinfo import ZoneInfo

from pipeline.dst import (
    cme_open_brisbane, us_equity_open_brisbane, us_data_open_brisbane,
    london_open_brisbane, us_post_equity_brisbane, cme_close_brisbane,
)
from pipeline.build_daily_features import compute_trading_day_utc_range

BRISBANE = ZoneInfo("Australia/Brisbane")
UTC = timezone.utc


class TestBrisbaneNoDST:
    """Brisbane never changes UTC offset. Verify this assumption."""

    def test_summer_offset(self):
        # January (Australian summer)
        dt = datetime(2025, 1, 15, 12, 0, tzinfo=BRISBANE)
        assert dt.utcoffset() == timedelta(hours=10)

    def test_winter_offset(self):
        # July (Australian winter)
        dt = datetime(2025, 7, 15, 12, 0, tzinfo=BRISBANE)
        assert dt.utcoffset() == timedelta(hours=10)

    def test_us_dst_transition_day_march(self):
        # US springs forward 2nd Sunday of March
        dt = datetime(2025, 3, 9, 12, 0, tzinfo=BRISBANE)
        assert dt.utcoffset() == timedelta(hours=10)

    def test_us_dst_transition_day_november(self):
        # US falls back 1st Sunday of November
        dt = datetime(2025, 11, 2, 12, 0, tzinfo=BRISBANE)
        assert dt.utcoffset() == timedelta(hours=10)


class TestTradingDayBoundary:
    """Trading day = 09:00 Brisbane to next 09:00 Brisbane = 23:00 UTC to 23:00 UTC."""

    def test_trading_day_start_utc(self):
        td = date(2025, 6, 15)
        # 09:00 Brisbane on June 15 = 23:00 UTC on June 14
        start_brisbane = datetime(td.year, td.month, td.day, 9, 0, tzinfo=BRISBANE)
        start_utc = start_brisbane.astimezone(UTC)
        assert start_utc.hour == 23
        assert start_utc.day == 14

    def test_trading_day_end_utc(self):
        td = date(2025, 6, 15)
        # Next 09:00 Brisbane = 23:00 UTC on June 15
        end_brisbane = datetime(td.year, td.month, td.day, 9, 0, tzinfo=BRISBANE) + timedelta(hours=24)
        end_utc = end_brisbane.astimezone(UTC)
        assert end_utc.hour == 23
        assert end_utc.day == 15

    def test_bar_before_0900_belongs_to_previous_day(self):
        # Bar at 08:59 Brisbane = 22:59 UTC on June 14
        bar_brisbane = datetime(2025, 6, 15, 8, 59, tzinfo=BRISBANE)
        bar_utc = bar_brisbane.astimezone(UTC)
        assert bar_utc.hour == 22
        assert bar_utc.day == 14
        # Trading day June 15 starts at 23:00 UTC June 14
        # 22:59 UTC is BEFORE that, so it belongs to trading day June 14
        td_june14_start = datetime(2025, 6, 13, 23, 0, tzinfo=UTC)
        td_june14_end = datetime(2025, 6, 14, 23, 0, tzinfo=UTC)
        assert td_june14_start <= bar_utc < td_june14_end

    def test_bar_at_0900_belongs_to_current_day(self):
        # Bar at 09:00 Brisbane = 23:00 UTC previous day
        bar_brisbane = datetime(2025, 6, 15, 9, 0, tzinfo=BRISBANE)
        bar_utc = bar_brisbane.astimezone(UTC)
        td_start_utc = datetime(2025, 6, 14, 23, 0, tzinfo=UTC)
        td_end_utc = datetime(2025, 6, 15, 23, 0, tzinfo=UTC)
        assert td_start_utc <= bar_utc < td_end_utc


class TestUTCConversionConsistency:
    """Verify that our 23:00 UTC formula matches proper timezone conversion."""

    def test_formula_matches_zoneinfo(self):
        for td in [date(2025, 1, 15), date(2025, 6, 15), date(2025, 3, 9), date(2025, 11, 2)]:
            # Formula: previous day 23:00 UTC
            prev_day = td - timedelta(days=1)
            formula_start = datetime(prev_day.year, prev_day.month, prev_day.day,
                                     23, 0, tzinfo=UTC)
            # Proper: 09:00 Brisbane on trading day
            proper_start = datetime(td.year, td.month, td.day, 9, 0,
                                    tzinfo=BRISBANE).astimezone(UTC)
            assert formula_start == proper_start, f"Mismatch on {td}"

    def test_year_boundary(self):
        td = date(2025, 1, 1)
        prev_day = td - timedelta(days=1)
        formula_start = datetime(prev_day.year, prev_day.month, prev_day.day,
                                 23, 0, tzinfo=UTC)
        proper_start = datetime(td.year, td.month, td.day, 9, 0,
                                tzinfo=BRISBANE).astimezone(UTC)
        assert formula_start == proper_start
        assert formula_start.year == 2024
        assert formula_start.month == 12
        assert formula_start.day == 31


# All 6 resolvers keyed by name for parametrized tests
ALL_RESOLVERS = {
    "CME_OPEN": cme_open_brisbane,
    "US_EQUITY_OPEN": us_equity_open_brisbane,
    "US_DATA_OPEN": us_data_open_brisbane,
    "LONDON_OPEN": london_open_brisbane,
    "US_POST_EQUITY": us_post_equity_brisbane,
    "CME_CLOSE": cme_close_brisbane,
}

# First trading day (Monday) after each 2025 DST transition
# US spring forward: Sun Mar 9 → Mon Mar 10
# UK spring forward: Sun Mar 30 → Mon Mar 31
# UK fall back:      Sun Oct 26 → Mon Oct 27
# US fall back:      Sun Nov 2  → Mon Nov 3
TRANSITION_MONDAYS = [
    date(2025, 3, 10),   # US spring forward
    date(2025, 3, 31),   # UK spring forward
    date(2025, 10, 27),  # UK fall back
    date(2025, 11, 3),   # US fall back
]


class TestResolverOutputOnTransitionDays:
    """Verify resolver outputs on the first trading day after DST transitions.

    Gap identified in DST audit: test_timezone_transitions.py tested trading
    day boundaries but never verified resolver output values near transitions.
    """

    @pytest.mark.parametrize("td", TRANSITION_MONDAYS,
                             ids=["US_spring", "UK_spring", "UK_fall", "US_fall"])
    @pytest.mark.parametrize("name", ALL_RESOLVERS.keys())
    def test_resolver_within_trading_day(self, name, td):
        """Every resolver output must fall within the trading day UTC window."""
        resolver = ALL_RESOLVERS[name]
        hour, minute = resolver(td)

        # Convert to UTC
        if hour < 9:
            cal_date = td + timedelta(days=1)
        else:
            cal_date = td
        local_dt = datetime(cal_date.year, cal_date.month, cal_date.day,
                            hour, minute, tzinfo=BRISBANE)
        utc_dt = local_dt.astimezone(UTC)

        td_start, td_end = compute_trading_day_utc_range(td)
        assert td_start <= utc_dt < td_end, (
            f"{name} on {td}: resolved to {hour:02d}:{minute:02d} Brisbane = "
            f"{utc_dt} UTC, outside [{td_start}, {td_end})"
        )

    @pytest.mark.parametrize("td", TRANSITION_MONDAYS,
                             ids=["US_spring", "UK_spring", "UK_fall", "US_fall"])
    def test_resolver_returns_valid_time(self, td):
        """All resolvers return (hour, minute) with valid ranges."""
        for name, resolver in ALL_RESOLVERS.items():
            hour, minute = resolver(td)
            assert 0 <= hour <= 23, f"{name}: hour={hour}"
            assert 0 <= minute <= 59, f"{name}: minute={minute}"

    def test_cme_open_shifts_after_us_spring(self):
        """CME_OPEN should be (8,0) on first Monday after US spring forward."""
        assert cme_open_brisbane(date(2025, 3, 10)) == (8, 0)

    def test_cme_open_shifts_after_us_fall(self):
        """CME_OPEN should be (9,0) on first Monday after US fall back."""
        assert cme_open_brisbane(date(2025, 11, 3)) == (9, 0)

    def test_london_open_shifts_after_uk_spring(self):
        """LONDON_OPEN should be (17,0) on first Monday after UK spring forward."""
        assert london_open_brisbane(date(2025, 3, 31)) == (17, 0)

    def test_london_open_shifts_after_uk_fall(self):
        """LONDON_OPEN should be (18,0) on first Monday after UK fall back."""
        assert london_open_brisbane(date(2025, 10, 27)) == (18, 0)

    def test_us_equity_shifts_after_us_spring(self):
        """US_EQUITY_OPEN should be (23,30) after US spring forward."""
        assert us_equity_open_brisbane(date(2025, 3, 10)) == (23, 30)

    def test_us_equity_shifts_after_us_fall(self):
        """US_EQUITY_OPEN should be (0,30) after US fall back."""
        assert us_equity_open_brisbane(date(2025, 11, 3)) == (0, 30)
