"""
Tests for trading_app.entry_rules confirm bars detection and entry models.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta, timezone

from trading_app.entry_rules import (
    detect_entry_with_confirm_bars,
    detect_confirm,
    resolve_entry,
    EntrySignal,
    ConfirmResult,
)


def _make_bars(closes: list[float], start_ts: datetime, freq_seconds: int = 60) -> pd.DataFrame:
    """Create a bars DataFrame from a list of close prices."""
    timestamps = [
        pd.Timestamp(start_ts).tz_localize("UTC") + pd.Timedelta(seconds=i * freq_seconds)
        if start_ts.tzinfo is None
        else pd.Timestamp(start_ts) + pd.Timedelta(seconds=i * freq_seconds)
        for i in range(len(closes))
    ]
    return pd.DataFrame({
        "ts_utc": timestamps,
        "open": closes,
        "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes],
        "close": closes,
        "volume": [100] * len(closes),
    })


# ORB range: 2340 (low) to 2350 (high)
ORB_HIGH = 2350.0
ORB_LOW = 2340.0
BREAK_TS = datetime(2024, 1, 15, 14, 5, tzinfo=timezone.utc)
WINDOW_END = datetime(2024, 1, 15, 23, 0, tzinfo=timezone.utc)


class TestConfirmBars1:
    """confirm_bars=1: Confirm on first close outside ORB."""

    def test_long_confirm_immediate(self):
        bars = _make_bars([2351.0], BREAK_TS)
        result = detect_confirm(bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long", 1, WINDOW_END)
        assert result.confirmed is True
        assert result.confirm_bar_close == 2351.0

    def test_short_confirm_immediate(self):
        bars = _make_bars([2339.0], BREAK_TS)
        result = detect_confirm(bars, BREAK_TS, ORB_HIGH, ORB_LOW, "short", 1, WINDOW_END)
        assert result.confirmed is True
        assert result.confirm_bar_close == 2339.0

    def test_no_break_in_window(self):
        bars = _make_bars([2345.0, 2346.0, 2344.0], BREAK_TS)
        result = detect_confirm(bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long", 1, WINDOW_END)
        assert result.confirmed is False

    def test_e2_entry_at_confirm_close(self):
        """E2 wrapper: entry_price = confirm bar close, stop = opposite ORB."""
        bars = _make_bars([2351.0], BREAK_TS)
        signal = detect_entry_with_confirm_bars(
            bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long", 1, WINDOW_END,
            entry_model="E2",
        )
        assert signal.triggered is True
        assert signal.entry_price == 2351.0  # confirm bar close
        assert signal.stop_price == ORB_LOW

    def test_e2_short_entry(self):
        """E2 short: entry at confirm close, stop at orb_high."""
        bars = _make_bars([2339.0], BREAK_TS)
        signal = detect_entry_with_confirm_bars(
            bars, BREAK_TS, ORB_HIGH, ORB_LOW, "short", 1, WINDOW_END,
            entry_model="E2",
        )
        assert signal.triggered is True
        assert signal.entry_price == 2339.0
        assert signal.stop_price == ORB_HIGH


class TestConfirmBars2:
    """confirm_bars=2: Require 2 consecutive closes outside ORB."""

    def test_two_consecutive_long(self):
        bars = _make_bars([2351.0, 2352.0], BREAK_TS)
        result = detect_confirm(bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long", 2, WINDOW_END)
        assert result.confirmed is True

    def test_reset_after_inside_close(self):
        bars = _make_bars([2351.0, 2345.0, 2352.0], BREAK_TS)
        result = detect_confirm(bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long", 2, WINDOW_END)
        assert result.confirmed is False

    def test_reset_then_two_consecutive(self):
        bars = _make_bars([2351.0, 2345.0, 2352.0, 2353.0], BREAK_TS)
        result = detect_confirm(bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long", 2, WINDOW_END)
        assert result.confirmed is True


class TestConfirmBars3:
    """confirm_bars=3: Require 3 consecutive closes outside ORB."""

    def test_three_consecutive_short(self):
        bars = _make_bars([2339.0, 2338.0, 2337.0], BREAK_TS)
        result = detect_confirm(bars, BREAK_TS, ORB_HIGH, ORB_LOW, "short", 3, WINDOW_END)
        assert result.confirmed is True

    def test_two_of_three_not_enough(self):
        bars = _make_bars([2339.0, 2338.0], BREAK_TS)
        result = detect_confirm(bars, BREAK_TS, ORB_HIGH, ORB_LOW, "short", 3, WINDOW_END)
        assert result.confirmed is False

    def test_reset_midway(self):
        bars = _make_bars([2339.0, 2338.0, 2345.0, 2339.0, 2338.0], BREAK_TS)
        result = detect_confirm(bars, BREAK_TS, ORB_HIGH, ORB_LOW, "short", 3, WINDOW_END)
        assert result.confirmed is False

    def test_reset_then_three(self):
        bars = _make_bars([2339.0, 2338.0, 2345.0, 2339.0, 2338.0, 2337.0], BREAK_TS)
        result = detect_confirm(bars, BREAK_TS, ORB_HIGH, ORB_LOW, "short", 3, WINDOW_END)
        assert result.confirmed is True


class TestEdgeCases:
    """Edge cases and validation."""

    def test_invalid_confirm_bars(self):
        bars = _make_bars([2351.0], BREAK_TS)
        with pytest.raises(ValueError, match="confirm_bars must be 1-10"):
            detect_entry_with_confirm_bars(
                bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long", 0, WINDOW_END
            )

    def test_invalid_break_dir(self):
        bars = _make_bars([2351.0], BREAK_TS)
        with pytest.raises(ValueError, match="break_dir must be"):
            detect_entry_with_confirm_bars(
                bars, BREAK_TS, ORB_HIGH, ORB_LOW, "up", 1, WINDOW_END
            )

    def test_empty_bars(self):
        bars = _make_bars([], BREAK_TS)
        signal = detect_entry_with_confirm_bars(
            bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long", 1, WINDOW_END
        )
        assert signal.triggered is False

    def test_window_boundary(self):
        """Bars exactly at window end should be excluded."""
        window_end = datetime(2024, 1, 15, 14, 7, tzinfo=timezone.utc)
        bars = _make_bars([2351.0, 2352.0, 2353.0], BREAK_TS)
        signal = detect_entry_with_confirm_bars(
            bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long", 3, window_end
        )
        assert signal.triggered is False

    def test_entry_ts_default_e1(self):
        """Default entry model is E1 — entry_ts is bar AFTER confirm."""
        bars = _make_bars([2351.0, 2352.0, 2353.0], BREAK_TS)
        signal = detect_entry_with_confirm_bars(
            bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long", 2, WINDOW_END
        )
        assert signal.triggered is True
        assert signal.entry_model == "E1"
        # E1: entry is bar AFTER 2nd confirm bar (index 2 = minute 7)
        expected_ts = BREAK_TS.replace(second=0, minute=7)
        assert signal.entry_ts.replace(tzinfo=timezone.utc) == expected_ts


# ============================================================================
# detect_confirm tests
# ============================================================================

class TestDetectConfirm:
    """Tests for the separated confirm detection logic."""

    def test_confirm_returns_bar_info(self):
        bars = _make_bars([2351.0, 2352.0], BREAK_TS)
        result = detect_confirm(bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long", 1, WINDOW_END)
        assert result.confirmed is True
        assert result.confirm_bar_idx == 0
        assert result.confirm_bar_close == 2351.0
        assert result.confirm_bar_ts is not None

    def test_no_confirm(self):
        bars = _make_bars([2345.0], BREAK_TS)
        result = detect_confirm(bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long", 1, WINDOW_END)
        assert result.confirmed is False
        assert result.confirm_bar_idx is None

    def test_confirm_preserves_orb_info(self):
        bars = _make_bars([2351.0], BREAK_TS)
        result = detect_confirm(bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long", 1, WINDOW_END)
        assert result.orb_high == ORB_HIGH
        assert result.orb_low == ORB_LOW
        assert result.break_dir == "long"


# ============================================================================
# resolve_entry E1/E2/E3 tests
# ============================================================================

class TestResolveEntryE1:
    """E1: Market-On-Confirm. Next bar OPEN after confirm."""

    def test_e1_long_uses_next_bar_open(self):
        bars = _make_bars([2351.0, 2352.0, 2354.0], BREAK_TS)
        confirm = detect_confirm(bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long", 1, WINDOW_END)
        signal = resolve_entry(bars, confirm, "E1", WINDOW_END)
        assert signal.triggered is True
        assert signal.entry_model == "E1"
        # E1 entry = open of bar after confirm bar (bar index 1)
        assert signal.entry_price == 2352.0  # open of bar 1
        assert signal.entry_ts > confirm.confirm_bar_ts

    def test_e1_no_next_bar(self):
        """If there's no bar after confirm, E1 doesn't trigger."""
        bars = _make_bars([2351.0], BREAK_TS)
        confirm = detect_confirm(bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long", 1, WINDOW_END)
        signal = resolve_entry(bars, confirm, "E1", WINDOW_END)
        assert signal.triggered is False

    def test_e1_short(self):
        bars = _make_bars([2339.0, 2338.0, 2336.0], BREAK_TS)
        confirm = detect_confirm(bars, BREAK_TS, ORB_HIGH, ORB_LOW, "short", 1, WINDOW_END)
        signal = resolve_entry(bars, confirm, "E1", WINDOW_END)
        assert signal.triggered is True
        assert signal.entry_price == 2338.0  # open of next bar
        assert signal.stop_price == ORB_HIGH


class TestResolveEntryE2:
    """E2: Market-On-Confirm-Close. Confirm bar CLOSE."""

    def test_e2_long_uses_confirm_close(self):
        bars = _make_bars([2351.0, 2352.0], BREAK_TS)
        confirm = detect_confirm(bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long", 1, WINDOW_END)
        signal = resolve_entry(bars, confirm, "E2", WINDOW_END)
        assert signal.triggered is True
        assert signal.entry_model == "E2"
        assert signal.entry_price == 2351.0  # confirm bar close
        assert signal.entry_ts == confirm.confirm_bar_ts

    def test_e2_always_fills(self):
        """E2 always fills if confirm happens (even with just 1 bar)."""
        bars = _make_bars([2351.0], BREAK_TS)
        confirm = detect_confirm(bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long", 1, WINDOW_END)
        signal = resolve_entry(bars, confirm, "E2", WINDOW_END)
        assert signal.triggered is True
        assert signal.entry_price == 2351.0

    def test_e2_short(self):
        bars = _make_bars([2339.0], BREAK_TS)
        confirm = detect_confirm(bars, BREAK_TS, ORB_HIGH, ORB_LOW, "short", 1, WINDOW_END)
        signal = resolve_entry(bars, confirm, "E2", WINDOW_END)
        assert signal.triggered is True
        assert signal.entry_price == 2339.0
        assert signal.stop_price == ORB_HIGH


class TestResolveEntryE3:
    """E3: Limit-At-ORB. Retrace to ORB level after confirm."""

    def test_e3_long_retrace(self):
        """Long: bar low <= orb_high means retrace fills limit buy."""
        # Bar 0: confirm (close > orb_high). Bar 1: low dips to orb_high.
        bars = _make_bars([2351.0, 2352.0, 2349.0], BREAK_TS)
        # Override bar 2 low to touch orb_high
        bars.loc[2, "low"] = 2349.5  # still > ORB_HIGH, won't retrace
        bars.loc[2, "low"] = ORB_HIGH - 0.5  # now retraces to ORB level

        confirm = detect_confirm(bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long", 1, WINDOW_END)
        signal = resolve_entry(bars, confirm, "E3", WINDOW_END)
        assert signal.triggered is True
        assert signal.entry_model == "E3"
        assert signal.entry_price == ORB_HIGH  # limit fill at ORB level
        assert signal.entry_ts > confirm.confirm_bar_ts

    def test_e3_no_retrace(self):
        """If price never retraces, E3 doesn't fill."""
        # All bars after confirm stay well above ORB high
        bars = _make_bars([2351.0, 2355.0, 2358.0], BREAK_TS)
        # Override lows so none touch orb_high
        bars.loc[1, "low"] = 2354.0
        bars.loc[2, "low"] = 2357.0

        confirm = detect_confirm(bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long", 1, WINDOW_END)
        signal = resolve_entry(bars, confirm, "E3", WINDOW_END)
        assert signal.triggered is False

    def test_e3_short_retrace(self):
        """Short: bar high >= orb_low means retrace fills limit sell."""
        bars = _make_bars([2339.0, 2338.0, 2341.0], BREAK_TS)
        bars.loc[2, "high"] = ORB_LOW + 0.5  # retraces up to ORB low

        confirm = detect_confirm(bars, BREAK_TS, ORB_HIGH, ORB_LOW, "short", 1, WINDOW_END)
        signal = resolve_entry(bars, confirm, "E3", WINDOW_END)
        assert signal.triggered is True
        assert signal.entry_price == ORB_LOW
        assert signal.stop_price == ORB_HIGH

    def test_e3_not_confirmed_no_fill(self):
        """No confirm = no E3 attempt."""
        bars = _make_bars([2345.0], BREAK_TS)
        confirm = detect_confirm(bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long", 1, WINDOW_END)
        signal = resolve_entry(bars, confirm, "E3", WINDOW_END)
        assert signal.triggered is False


class TestDefaultEntryModel:
    """Default entry model is E1."""

    def test_default_is_e1(self):
        """Default entry_model is E1 (market at next bar open)."""
        bars = _make_bars([2351.0, 2352.0], BREAK_TS)
        signal = detect_entry_with_confirm_bars(
            bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long", 1, WINDOW_END,
        )
        assert signal.entry_model == "E1"
        # E1 entry is the OPEN of the next bar, not the ORB level
        assert signal.entry_price == 2352.0  # open == close in test helper

    def test_invalid_entry_model(self):
        bars = _make_bars([2351.0], BREAK_TS)
        with pytest.raises(ValueError, match="Unknown entry_model"):
            detect_entry_with_confirm_bars(
                bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long", 1, WINDOW_END,
                entry_model="E99",
            )


class TestE3StopBeforeFill:
    """E3 must not fill if stop is breached before or on the retrace bar."""

    def test_stop_breached_before_retrace_no_fill(self):
        """Stop hit on bar before retrace — no E3 fill."""
        # Confirm at bar 0: close=2351 (above orb_high=2350)
        # Bar 1: crashes through stop (low=2339 < orb_low=2340) but no retrace
        # Bar 2: retraces to 2350 (low=2349.5 <= orb_high=2350)
        # E3 should NOT fill because stop was hit on bar 1
        timestamps = [
            BREAK_TS,
            BREAK_TS + timedelta(minutes=1),
            BREAK_TS + timedelta(minutes=2),
        ]
        bars = pd.DataFrame({
            "ts_utc": timestamps,
            "open": [2351.0, 2345.0, 2349.0],
            "high": [2352.0, 2346.0, 2351.0],
            "low": [2350.5, 2339.0, 2349.0],
            "close": [2351.0, 2345.0, 2350.0],
            "volume": [100, 100, 100],
        })
        confirm = detect_confirm(bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long", 1, WINDOW_END)
        assert confirm.confirmed is True
        signal = resolve_entry(bars, confirm, "E3", WINDOW_END)
        assert signal.triggered is False

    def test_stop_hit_on_retrace_bar_no_fill(self):
        """Same bar touches both ORB level and stop — no E3 fill."""
        # Confirm at bar 0: close=2351 (above orb_high=2350)
        # Bar 1: big range bar: low=2339 (stop), also low<=2350 (retrace)
        # E3 should NOT fill — can't enter a stopped-out trade
        timestamps = [
            BREAK_TS,
            BREAK_TS + timedelta(minutes=1),
        ]
        bars = pd.DataFrame({
            "ts_utc": timestamps,
            "open": [2351.0, 2352.0],
            "high": [2352.0, 2355.0],
            "low": [2350.5, 2339.0],
            "close": [2351.0, 2340.0],
            "volume": [100, 100],
        })
        confirm = detect_confirm(bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long", 1, WINDOW_END)
        assert confirm.confirmed is True
        signal = resolve_entry(bars, confirm, "E3", WINDOW_END)
        assert signal.triggered is False

    def test_clean_retrace_fills(self):
        """Normal retrace without stop breach — E3 fills correctly."""
        # Confirm at bar 0: close=2351 (above orb_high=2350)
        # Bar 1: price drifts up to 2355 (no retrace, no stop)
        # Bar 2: retraces to 2350 (low=2349.5), stop NOT hit (low > 2340)
        timestamps = [
            BREAK_TS,
            BREAK_TS + timedelta(minutes=1),
            BREAK_TS + timedelta(minutes=2),
        ]
        bars = pd.DataFrame({
            "ts_utc": timestamps,
            "open": [2351.0, 2353.0, 2352.0],
            "high": [2352.0, 2355.0, 2353.0],
            "low": [2350.5, 2352.0, 2349.5],
            "close": [2351.0, 2354.0, 2350.5],
            "volume": [100, 100, 100],
        })
        confirm = detect_confirm(bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long", 1, WINDOW_END)
        assert confirm.confirmed is True
        signal = resolve_entry(bars, confirm, "E3", WINDOW_END)
        assert signal.triggered is True
        assert signal.entry_price == ORB_HIGH  # E3 fills at ORB level

    def test_short_e3_stop_before_retrace(self):
        """Short E3: stop hit before retrace — no fill."""
        # Short break: close=2339 (below orb_low=2340)
        # Bar 1: spikes through stop (high=2351 >= orb_high=2350)
        # Bar 2: retraces to orb_low (high=2340.5 >= 2340)
        # E3 should NOT fill — stop was breached first
        timestamps = [
            BREAK_TS,
            BREAK_TS + timedelta(minutes=1),
            BREAK_TS + timedelta(minutes=2),
        ]
        bars = pd.DataFrame({
            "ts_utc": timestamps,
            "open": [2339.0, 2342.0, 2339.0],
            "high": [2340.0, 2351.0, 2340.5],
            "low": [2338.0, 2341.0, 2338.0],
            "close": [2339.0, 2343.0, 2339.5],
            "volume": [100, 100, 100],
        })
        confirm = detect_confirm(bars, BREAK_TS, ORB_HIGH, ORB_LOW, "short", 1, WINDOW_END)
        assert confirm.confirmed is True
        signal = resolve_entry(bars, confirm, "E3", WINDOW_END)
        assert signal.triggered is False
