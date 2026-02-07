"""
Tests for trading_app.entry_rules confirm bars detection.
"""

import pytest
import pandas as pd
from datetime import datetime, timezone

from trading_app.entry_rules import detect_entry_with_confirm_bars, EntrySignal


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
    """confirm_bars=1: Enter on first close outside ORB."""

    def test_long_break_immediate(self):
        bars = _make_bars([2351.0], BREAK_TS)
        signal = detect_entry_with_confirm_bars(
            bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long", 1, WINDOW_END
        )
        assert signal.triggered is True
        assert signal.entry_price == ORB_HIGH
        assert signal.stop_price == ORB_LOW

    def test_short_break_immediate(self):
        bars = _make_bars([2339.0], BREAK_TS)
        signal = detect_entry_with_confirm_bars(
            bars, BREAK_TS, ORB_HIGH, ORB_LOW, "short", 1, WINDOW_END
        )
        assert signal.triggered is True
        assert signal.entry_price == ORB_LOW
        assert signal.stop_price == ORB_HIGH

    def test_no_break_in_window(self):
        bars = _make_bars([2345.0, 2346.0, 2344.0], BREAK_TS)
        signal = detect_entry_with_confirm_bars(
            bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long", 1, WINDOW_END
        )
        assert signal.triggered is False
        assert signal.entry_ts is None


class TestConfirmBars2:
    """confirm_bars=2: Require 2 consecutive closes outside ORB."""

    def test_two_consecutive_long(self):
        bars = _make_bars([2351.0, 2352.0], BREAK_TS)
        signal = detect_entry_with_confirm_bars(
            bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long", 2, WINDOW_END
        )
        assert signal.triggered is True
        assert signal.entry_price == ORB_HIGH

    def test_reset_after_inside_close(self):
        # 1 outside, 1 inside (reset), 1 outside — should NOT trigger
        bars = _make_bars([2351.0, 2345.0, 2352.0], BREAK_TS)
        signal = detect_entry_with_confirm_bars(
            bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long", 2, WINDOW_END
        )
        assert signal.triggered is False

    def test_reset_then_two_consecutive(self):
        # 1 outside, 1 inside (reset), 2 outside — triggers
        bars = _make_bars([2351.0, 2345.0, 2352.0, 2353.0], BREAK_TS)
        signal = detect_entry_with_confirm_bars(
            bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long", 2, WINDOW_END
        )
        assert signal.triggered is True


class TestConfirmBars3:
    """confirm_bars=3: Require 3 consecutive closes outside ORB."""

    def test_three_consecutive_short(self):
        bars = _make_bars([2339.0, 2338.0, 2337.0], BREAK_TS)
        signal = detect_entry_with_confirm_bars(
            bars, BREAK_TS, ORB_HIGH, ORB_LOW, "short", 3, WINDOW_END
        )
        assert signal.triggered is True
        assert signal.entry_price == ORB_LOW
        assert signal.stop_price == ORB_HIGH

    def test_two_of_three_not_enough(self):
        bars = _make_bars([2339.0, 2338.0], BREAK_TS)
        signal = detect_entry_with_confirm_bars(
            bars, BREAK_TS, ORB_HIGH, ORB_LOW, "short", 3, WINDOW_END
        )
        assert signal.triggered is False

    def test_reset_midway(self):
        # 2 outside, 1 inside, 2 outside — not enough
        bars = _make_bars([2339.0, 2338.0, 2345.0, 2339.0, 2338.0], BREAK_TS)
        signal = detect_entry_with_confirm_bars(
            bars, BREAK_TS, ORB_HIGH, ORB_LOW, "short", 3, WINDOW_END
        )
        assert signal.triggered is False

    def test_reset_then_three(self):
        # 2 outside, 1 inside, 3 outside — triggers
        bars = _make_bars([2339.0, 2338.0, 2345.0, 2339.0, 2338.0, 2337.0], BREAK_TS)
        signal = detect_entry_with_confirm_bars(
            bars, BREAK_TS, ORB_HIGH, ORB_LOW, "short", 3, WINDOW_END
        )
        assert signal.triggered is True


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
        # 3 bars: break_ts, +1m, +2m (= window_end, excluded)
        bars = _make_bars([2351.0, 2352.0, 2353.0], BREAK_TS)
        signal = detect_entry_with_confirm_bars(
            bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long", 3, window_end
        )
        # Only 2 bars in window (14:05, 14:06), need 3 — should fail
        assert signal.triggered is False

    def test_entry_ts_is_confirming_bar(self):
        """entry_ts should be the timestamp of the Nth confirming bar."""
        bars = _make_bars([2351.0, 2352.0], BREAK_TS)
        signal = detect_entry_with_confirm_bars(
            bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long", 2, WINDOW_END
        )
        assert signal.triggered is True
        # Second bar is the confirming bar (index 1, ts = BREAK_TS + 60s)
        expected_ts = BREAK_TS.replace(second=0, minute=6)
        assert signal.entry_ts.replace(tzinfo=timezone.utc) == expected_ts
