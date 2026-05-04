"""Tests for trading_app.calendar_overlay."""

from datetime import date
from unittest.mock import patch

from trading_app.calendar_overlay import (
    CALENDAR_RULES,
    CalendarAction,
    _get_active_signals,
    get_calendar_action,
)

# =========================================================================
# CalendarAction enum
# =========================================================================


class TestCalendarActionEnum:
    def test_skip_value(self):
        assert CalendarAction.SKIP.value == 0.0

    def test_half_size_value(self):
        assert CalendarAction.HALF_SIZE.value == 0.5

    def test_neutral_value(self):
        assert CalendarAction.NEUTRAL.value == 1.0

    def test_skip_is_most_restrictive(self):
        assert CalendarAction.SKIP.value < CalendarAction.HALF_SIZE.value
        assert CalendarAction.HALF_SIZE.value < CalendarAction.NEUTRAL.value


# =========================================================================
# Empty rules → NEUTRAL
# =========================================================================


class TestEmptyRules:
    def test_empty_rules_returns_neutral(self):
        """With empty CALENDAR_RULES, get_calendar_action returns NEUTRAL for any input."""
        with patch("trading_app.calendar_overlay.CALENDAR_RULES", {}):
            result = get_calendar_action("MGC", "TOKYO_OPEN", date(2025, 1, 6))
            assert result == CalendarAction.NEUTRAL

    def test_empty_rules_any_instrument(self):
        with patch("trading_app.calendar_overlay.CALENDAR_RULES", {}):
            result = get_calendar_action("DOESNT_EXIST", "FAKE_SESSION", date(2025, 6, 15))
            assert result == CalendarAction.NEUTRAL


# =========================================================================
# Known rule lookup
# =========================================================================


class TestRuleLookup:
    def test_skip_rule_on_matching_day(self):
        """Mock a SKIP rule for NFP + MGC TOKYO_OPEN, verify on an NFP day."""
        rules = {("MGC", "TOKYO_OPEN", "NFP"): CalendarAction.SKIP}
        # 2025-01-03 is the first Friday of January 2025 (NFP day)
        nfp_day = date(2025, 1, 3)
        with patch("trading_app.calendar_overlay.CALENDAR_RULES", rules):
            result = get_calendar_action("MGC", "TOKYO_OPEN", nfp_day)
            assert result == CalendarAction.SKIP

    def test_half_size_rule_on_matching_day(self):
        """Mock a HALF_SIZE rule for OPEX + MNQ NYSE_OPEN, verify on OPEX day."""
        rules = {("MNQ", "NYSE_OPEN", "OPEX"): CalendarAction.HALF_SIZE}
        # 2025-01-17 is the third Friday of January 2025 (OPEX day)
        opex_day = date(2025, 1, 17)
        with patch("trading_app.calendar_overlay.CALENDAR_RULES", rules):
            result = get_calendar_action("MNQ", "NYSE_OPEN", opex_day)
            assert result == CalendarAction.HALF_SIZE

    def test_rule_does_not_fire_on_non_matching_day(self):
        """NFP rule should not fire on a non-NFP day."""
        rules = {("MGC", "TOKYO_OPEN", "NFP"): CalendarAction.SKIP}
        # 2025-01-06 is a Monday, not NFP
        non_nfp = date(2025, 1, 6)
        with patch("trading_app.calendar_overlay.CALENDAR_RULES", rules):
            result = get_calendar_action("MGC", "TOKYO_OPEN", non_nfp)
            # Monday signal fires but no rule for Monday, so NEUTRAL
            assert result == CalendarAction.NEUTRAL


# =========================================================================
# Most-restrictive logic
# =========================================================================


class TestMostRestrictive:
    def test_skip_beats_half_size(self):
        """When two signals fire (one HALF_SIZE, one SKIP), result is SKIP."""
        # 2025-01-03 is NFP (first Friday) and also a Friday
        nfp_friday = date(2025, 1, 3)
        rules = {
            ("MGC", "TOKYO_OPEN", "Friday"): CalendarAction.HALF_SIZE,
            ("MGC", "TOKYO_OPEN", "NFP"): CalendarAction.SKIP,
        }
        with patch("trading_app.calendar_overlay.CALENDAR_RULES", rules):
            result = get_calendar_action("MGC", "TOKYO_OPEN", nfp_friday)
            assert result == CalendarAction.SKIP

    def test_half_size_beats_neutral(self):
        """HALF_SIZE is more restrictive than NEUTRAL (which is default)."""
        # 2025-01-17 is OPEX Friday — both Friday and OPEX signals fire
        opex_friday = date(2025, 1, 17)
        rules = {
            ("MES", "NYSE_OPEN", "OPEX"): CalendarAction.HALF_SIZE,
            # No Friday rule → NEUTRAL for Friday
        }
        with patch("trading_app.calendar_overlay.CALENDAR_RULES", rules):
            result = get_calendar_action("MES", "NYSE_OPEN", opex_friday)
            assert result == CalendarAction.HALF_SIZE

    def test_multiple_neutral_stays_neutral(self):
        """When all matching rules are NEUTRAL, result is NEUTRAL."""
        rules = {
            ("MGC", "TOKYO_OPEN", "Friday"): CalendarAction.NEUTRAL,
            ("MGC", "TOKYO_OPEN", "NFP"): CalendarAction.NEUTRAL,
        }
        nfp_friday = date(2025, 1, 3)
        with patch("trading_app.calendar_overlay.CALENDAR_RULES", rules):
            result = get_calendar_action("MGC", "TOKYO_OPEN", nfp_friday)
            assert result == CalendarAction.NEUTRAL


# =========================================================================
# Unknown instrument / session → NEUTRAL
# =========================================================================


class TestUnknownKeys:
    def test_unknown_instrument_returns_neutral(self):
        """Instrument not in rules returns NEUTRAL."""
        rules = {("MGC", "TOKYO_OPEN", "NFP"): CalendarAction.SKIP}
        nfp_day = date(2025, 1, 3)
        with patch("trading_app.calendar_overlay.CALENDAR_RULES", rules):
            result = get_calendar_action("UNKNOWN_INST", "TOKYO_OPEN", nfp_day)
            assert result == CalendarAction.NEUTRAL

    def test_unknown_session_returns_neutral(self):
        """Session not in rules returns NEUTRAL."""
        rules = {("MGC", "TOKYO_OPEN", "NFP"): CalendarAction.SKIP}
        nfp_day = date(2025, 1, 3)
        with patch("trading_app.calendar_overlay.CALENDAR_RULES", rules):
            result = get_calendar_action("MGC", "UNKNOWN_SESSION", nfp_day)
            assert result == CalendarAction.NEUTRAL


# =========================================================================
# Calendar signal detection
# =========================================================================


class TestNfpDetection:
    def test_first_friday_january_2025(self):
        """2025-01-03 is the first Friday of January — NFP day."""
        assert date(2025, 1, 3).weekday() == 4  # Friday
        assert date(2025, 1, 3).day <= 7
        signals = _get_active_signals(date(2025, 1, 3))
        assert "NFP" in signals

    def test_third_friday_is_not_nfp(self):
        """Third Friday (OPEX) should not be NFP."""
        signals = _get_active_signals(date(2025, 1, 17))
        assert "NFP" not in signals

    def test_non_friday_early_month_is_not_nfp(self):
        """2025-01-02 is a Thursday — not NFP even though day <= 7."""
        signals = _get_active_signals(date(2025, 1, 2))
        assert "NFP" not in signals


class TestOpexDetection:
    def test_third_friday_january_2025(self):
        """2025-01-17 is the third Friday of January — OPEX day."""
        assert date(2025, 1, 17).weekday() == 4  # Friday
        assert 15 <= date(2025, 1, 17).day <= 21
        signals = _get_active_signals(date(2025, 1, 17))
        assert "OPEX" in signals

    def test_first_friday_is_not_opex(self):
        """First Friday should not be OPEX."""
        signals = _get_active_signals(date(2025, 1, 3))
        assert "OPEX" not in signals


# =========================================================================
# _get_active_signals comprehensive
# =========================================================================


class TestGetActiveSignals:
    def test_monday_signal(self):
        """2025-01-06 is a Monday."""
        assert date(2025, 1, 6).weekday() == 0
        signals = _get_active_signals(date(2025, 1, 6))
        assert "Monday" in signals
        assert "Friday" not in signals

    def test_friday_signal(self):
        """2025-01-10 is a Friday (second Friday, not NFP or OPEX)."""
        assert date(2025, 1, 10).weekday() == 4
        signals = _get_active_signals(date(2025, 1, 10))
        assert "Friday" in signals
        assert "NFP" not in signals
        assert "OPEX" not in signals

    def test_nfp_friday_has_both(self):
        """NFP day (first Friday) should have both NFP and Friday signals."""
        signals = _get_active_signals(date(2025, 1, 3))
        assert "NFP" in signals
        assert "Friday" in signals

    def test_opex_friday_has_both(self):
        """OPEX day (third Friday) should have both OPEX and Friday signals."""
        signals = _get_active_signals(date(2025, 1, 17))
        assert "OPEX" in signals
        assert "Friday" in signals

    def test_fomc_day_detected(self):
        """2025-01-29 is a known FOMC date."""
        signals = _get_active_signals(date(2025, 1, 29))
        assert "FOMC" in signals

    def test_fomc_day_after_detected(self):
        """Day after FOMC is also flagged (announcement + next day)."""
        signals = _get_active_signals(date(2025, 1, 30))
        assert "FOMC" in signals

    def test_cpi_day_detected(self):
        """2025-01-15 is a known CPI release date."""
        signals = _get_active_signals(date(2025, 1, 15))
        assert "CPI" in signals

    def test_opex_week_detected(self):
        """2025-01-13 (Monday of OPEX week — OPEX is 2025-01-17)."""
        signals = _get_active_signals(date(2025, 1, 13))
        assert "OPEX_WEEK" in signals

    def test_dow_signal_every_weekday(self):
        """Every weekday should produce exactly one DOW signal."""
        # 2025-01-06 Mon, 07 Tue, 08 Wed, 09 Thu, 10 Fri
        expected = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        for i, dow_name in enumerate(expected):
            d = date(2025, 1, 6 + i)
            signals = _get_active_signals(d)
            assert dow_name in signals
            # No other DOW signal
            other_dows = [n for n in expected if n != dow_name]
            for other in other_dows:
                assert other not in signals


# =========================================================================
# Error handling — fail-open-to-trade
# =========================================================================


class TestErrorHandling:
    def test_exception_returns_skip(self):
        """If _get_active_signals raises, get_calendar_action returns SKIP (fail-closed)."""
        with patch(
            "trading_app.calendar_overlay._get_active_signals",
            side_effect=ValueError("boom"),
        ):
            # Need non-empty rules to reach the signal check
            with patch(
                "trading_app.calendar_overlay.CALENDAR_RULES",
                {("MGC", "TOKYO_OPEN", "NFP"): CalendarAction.SKIP},
            ):
                result = get_calendar_action("MGC", "TOKYO_OPEN", date(2025, 1, 3))
                assert result == CalendarAction.SKIP


# =========================================================================
# Current state — rules are empty (cascade found zero actionable signals)
# =========================================================================


class TestCurrentState:
    def test_current_rules_are_empty(self):
        """The cascade scanner found zero actionable rules. Verify the loaded state."""
        assert CALENDAR_RULES == {}

    def test_current_state_always_neutral(self):
        """With current empty rules, everything is NEUTRAL."""
        assert get_calendar_action("MGC", "TOKYO_OPEN", date(2025, 1, 3)) == CalendarAction.NEUTRAL
        assert get_calendar_action("MNQ", "NYSE_OPEN", date(2025, 1, 17)) == CalendarAction.NEUTRAL
        assert get_calendar_action("MES", "CME_REOPEN", date(2025, 3, 19)) == CalendarAction.NEUTRAL
