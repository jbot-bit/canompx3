"""Tests for timezone transitions and DST handling (T10).

Brisbane (UTC+10) has NO daylight saving, but US markets do.
These tests verify that trading day boundaries and UTC conversions
are correct across US DST transitions.
"""
import pytest
from datetime import datetime, date, timedelta, timezone
from zoneinfo import ZoneInfo

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
