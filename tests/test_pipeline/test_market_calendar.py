"""Tests for pipeline.market_calendar — CME holiday and early close awareness."""

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pytest

from pipeline.market_calendar import (
    is_cme_holiday,
    is_early_close,
    is_market_open_at,
    session_close_utc,
    effective_close_et,
)

UTC = ZoneInfo("UTC")
ET = ZoneInfo("America/New_York")


class TestIsCmeHoliday:
    """Full market closures."""

    def test_new_years_day(self):
        assert is_cme_holiday(date(2025, 1, 1)) is True

    def test_good_friday_2025(self):
        assert is_cme_holiday(date(2025, 4, 18)) is True

    def test_christmas_2025(self):
        assert is_cme_holiday(date(2025, 12, 25)) is True

    def test_good_friday_2026(self):
        assert is_cme_holiday(date(2026, 4, 3)) is True

    def test_carter_funeral_special_closure(self):
        assert is_cme_holiday(date(2025, 1, 9)) is True

    def test_normal_tuesday_not_holiday(self):
        assert is_cme_holiday(date(2025, 7, 15)) is False

    def test_mlk_day_is_NOT_full_holiday(self):
        """MLK Day is early close, not full closure."""
        assert is_cme_holiday(date(2025, 1, 20)) is False

    def test_thanksgiving_is_NOT_full_holiday(self):
        """Thanksgiving is early close, not full closure."""
        assert is_cme_holiday(date(2025, 11, 27)) is False

    def test_weekend_saturday(self):
        assert is_cme_holiday(date(2025, 7, 19)) is True

    def test_weekend_sunday(self):
        assert is_cme_holiday(date(2025, 7, 20)) is True


class TestIsEarlyClose:
    """12:00 PM CT early close days."""

    def test_mlk_day(self):
        assert is_early_close(date(2025, 1, 20)) is True

    def test_presidents_day(self):
        assert is_early_close(date(2025, 2, 17)) is True

    def test_memorial_day(self):
        assert is_early_close(date(2025, 5, 26)) is True

    def test_july_4th(self):
        assert is_early_close(date(2025, 7, 4)) is True

    def test_labor_day(self):
        assert is_early_close(date(2025, 9, 1)) is True

    def test_thanksgiving(self):
        assert is_early_close(date(2025, 11, 27)) is True

    def test_black_friday(self):
        assert is_early_close(date(2025, 11, 28)) is True

    def test_christmas_eve(self):
        assert is_early_close(date(2025, 12, 24)) is True

    def test_normal_day_not_early(self):
        assert is_early_close(date(2025, 7, 15)) is False

    def test_full_holiday_not_early(self):
        """A full holiday is not an early close — it's closed entirely."""
        assert is_early_close(date(2025, 12, 25)) is False

    def test_2026_early_closes(self):
        assert is_early_close(date(2026, 1, 19)) is True  # MLK
        assert is_early_close(date(2026, 11, 26)) is True  # Thanksgiving
        assert is_early_close(date(2026, 12, 24)) is True  # Christmas Eve


class TestSessionCloseUtc:
    """Actual exchange close time."""

    def test_normal_day_edt(self):
        close = session_close_utc(date(2025, 7, 15))
        assert close is not None
        # Normal close in EDT = 5:00 PM CT = 22:00 UTC
        assert close.hour == 22

    def test_early_close_memorial_day(self):
        close = session_close_utc(date(2025, 5, 26))
        assert close is not None
        # Early close in EDT = 12:00 PM CT = 17:00 UTC
        assert close.hour == 17

    def test_early_close_christmas_eve_est(self):
        close = session_close_utc(date(2025, 12, 24))
        assert close is not None
        # Early close in EST = 12:00 PM CT = 18:00 UTC
        assert close.hour == 18

    def test_holiday_returns_none(self):
        assert session_close_utc(date(2025, 12, 25)) is None


class TestIsMarketOpenAt:
    """Per-minute market open checks."""

    def test_normal_morning_open(self):
        # 9:00 AM CT on normal day = 14:00 UTC
        utc_time = datetime(2025, 7, 15, 14, 0, tzinfo=UTC)
        assert is_market_open_at(utc_time) is True

    def test_morning_on_early_close_day(self):
        # 9:00 AM CT on Memorial Day = 14:00 UTC
        utc_time = datetime(2025, 5, 26, 14, 0, tzinfo=UTC)
        assert is_market_open_at(utc_time) is True

    def test_afternoon_on_early_close_day_blocked(self):
        # 2:45 PM CT on Memorial Day = 19:45 UTC — AFTER 12:00 CT close
        utc_time = datetime(2025, 5, 26, 19, 45, tzinfo=UTC)
        assert is_market_open_at(utc_time) is False

    def test_evening_before_early_close_open(self):
        # 7:00 PM CT Sunday before Memorial Day = overnight session
        utc_time = datetime(2025, 5, 26, 0, 0, tzinfo=UTC)
        assert is_market_open_at(utc_time) is True

    def test_holiday_closed(self):
        utc_time = datetime(2025, 12, 25, 14, 0, tzinfo=UTC)
        assert is_market_open_at(utc_time) is False

    def test_weekend_closed(self):
        utc_time = datetime(2025, 7, 19, 14, 0, tzinfo=UTC)
        assert is_market_open_at(utc_time) is False


class TestEffectiveCloseEt:
    """min(firm_close, exchange_close) logic."""

    def test_normal_day_firm_close_wins(self):
        # TopStep close 16:10 ET, exchange close 17:00 ET (normal)
        result = effective_close_et(date(2025, 7, 15), firm_close_et=time(16, 10))
        assert result == time(16, 10)  # Firm is earlier

    def test_early_close_day_exchange_wins(self):
        # TopStep close 16:10 ET, exchange close 13:00 ET (early close)
        result = effective_close_et(date(2025, 5, 26), firm_close_et=time(16, 10))
        assert result == time(13, 0)  # Exchange is earlier

    def test_early_close_tradeify_exchange_wins(self):
        # Tradeify close 16:59 ET, exchange close 13:00 ET (early close)
        result = effective_close_et(date(2025, 11, 27), firm_close_et=time(16, 59))
        assert result == time(13, 0)  # Exchange is earlier

    def test_holiday_returns_none(self):
        result = effective_close_et(date(2025, 12, 25), firm_close_et=time(16, 10))
        assert result is None  # No trading at all
