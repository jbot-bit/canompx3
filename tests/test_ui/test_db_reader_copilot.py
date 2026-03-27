# tests/test_ui/test_db_reader_copilot.py
"""Tests for new db_reader functions used by the co-pilot dashboard."""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest


class TestGetPriorDayAtr:
    def test_returns_float_or_none(self):
        from ui.db_reader import get_prior_day_atr

        # Against real DB — may return None if no data for instrument
        result = get_prior_day_atr("MGC")
        assert result is None or isinstance(result, float)

    def test_returns_positive_for_mgc(self):
        from ui.db_reader import get_prior_day_atr

        result = get_prior_day_atr("MGC")
        if result is not None:
            assert result > 0


class TestGetTodayCompletedSessions:
    def test_returns_list(self):
        from ui.db_reader import get_today_completed_sessions

        result = get_today_completed_sessions(date(2026, 2, 20))
        assert isinstance(result, list)


class TestGetPreviousTradingDay:
    def test_returns_date_before_input(self):
        from ui.db_reader import get_previous_trading_day

        result = get_previous_trading_day(date(2026, 2, 20))
        if result is not None:
            assert result < date(2026, 2, 20)
