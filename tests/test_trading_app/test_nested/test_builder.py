"""
Tests for trading_app.nested.builder — resample_to_5m, E3 sub-bar verification,
and outcome computation with 5m bars.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from trading_app.nested.builder import resample_to_5m, _verify_e3_sub_bar_fill
from trading_app.outcome_builder import compute_single_outcome
from pipeline.cost_model import get_cost_spec


# ============================================================================
# HELPERS
# ============================================================================

def _cost():
    return get_cost_spec("MGC")


def _make_bars(start_ts, prices, interval_minutes=1):
    """Create bars_df from list of (open, high, low, close, volume) tuples."""
    rows = []
    ts = start_ts
    for o, h, l, c, v in prices:
        rows.append({
            "ts_utc": ts,
            "open": float(o),
            "high": float(h),
            "low": float(l),
            "close": float(c),
            "volume": int(v),
        })
        ts = ts + timedelta(minutes=interval_minutes)
    return pd.DataFrame(rows)


# ============================================================================
# resample_to_5m tests
# ============================================================================

class TestResampleTo5m:
    """Tests for the 1m -> 5m resampling function."""

    def test_five_bars_to_one(self):
        """5 consecutive 1m bars should produce exactly 1 5m bar."""
        # All bars at :00, :01, :02, :03, :04 -> bucket at :00
        start = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        bars = _make_bars(start, [
            (100, 105, 99, 102, 10),  # :00
            (102, 107, 101, 104, 20),  # :01
            (104, 110, 103, 108, 30),  # :02
            (108, 112, 106, 109, 15),  # :03
            (109, 111, 107, 110, 25),  # :04
        ])

        # after_ts before all bars
        before = datetime(2023, 12, 31, 23, 59, tzinfo=timezone.utc)
        result = resample_to_5m(bars, before)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["open"] == 100.0   # first open
        assert row["high"] == 112.0   # max high
        assert row["low"] == 99.0     # min low
        assert row["close"] == 110.0  # last close
        assert row["volume"] == 100   # sum volume

    def test_ten_bars_to_two(self):
        """10 consecutive 1m bars across 2 buckets."""
        start = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        prices = [(100 + i, 105 + i, 98 + i, 101 + i, 10) for i in range(10)]
        bars = _make_bars(start, prices)

        before = datetime(2023, 12, 31, 23, 59, tzinfo=timezone.utc)
        result = resample_to_5m(bars, before)

        assert len(result) == 2

    def test_preserves_timestamps_alignment(self):
        """Output timestamps should be floored to 5-minute boundaries."""
        # Bars at 00:03, 00:04, 00:05, 00:06, 00:07, 00:08
        start = datetime(2024, 1, 5, 0, 3, tzinfo=timezone.utc)
        prices = [(100, 105, 99, 102, 10) for _ in range(6)]
        bars = _make_bars(start, prices)

        before = datetime(2024, 1, 5, 0, 2, tzinfo=timezone.utc)
        result = resample_to_5m(bars, before)

        # :03 and :04 -> bucket :00, :05-:08 -> bucket :05
        assert len(result) == 2
        ts0 = result.iloc[0]["ts_utc"]
        ts1 = result.iloc[1]["ts_utc"]
        assert ts0.minute == 0  # floored to :00
        assert ts1.minute == 5  # floored to :05

    def test_after_ts_filters_correctly(self):
        """Bars at or before after_ts should be excluded."""
        start = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        prices = [(100, 105, 99, 102, 10) for _ in range(10)]
        bars = _make_bars(start, prices)

        # after_ts = :04 -> bars :05 through :09 included (5 bars, 1 bucket)
        after = datetime(2024, 1, 5, 0, 4, tzinfo=timezone.utc)
        result = resample_to_5m(bars, after)

        assert len(result) == 1  # :05-:09 -> single bucket at :05

    def test_empty_when_no_bars_after(self):
        """Returns empty DataFrame if no bars after after_ts."""
        start = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        bars = _make_bars(start, [(100, 105, 99, 102, 10)])

        after = datetime(2024, 1, 5, 1, 0, tzinfo=timezone.utc)
        result = resample_to_5m(bars, after)

        assert result.empty

    def test_single_bar_produces_single_5m(self):
        """A single 1m bar after after_ts should produce one 5m bar."""
        start = datetime(2024, 1, 5, 0, 7, tzinfo=timezone.utc)
        bars = _make_bars(start, [(100, 105, 99, 102, 50)])

        after = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        result = resample_to_5m(bars, after)

        assert len(result) == 1
        assert result.iloc[0]["open"] == 100
        assert result.iloc[0]["volume"] == 50


# ============================================================================
# E3 sub-bar fill verification tests
# ============================================================================

class TestE3SubBarFillVerification:
    """Tests for _verify_e3_sub_bar_fill()."""

    def test_long_fill_confirmed_by_1m_low(self):
        """1m bar low touches the limit price -> fill confirmed."""
        entry_ts = datetime(2024, 1, 5, 0, 5, tzinfo=timezone.utc)
        # 1m bars within the 5m candle [0:05, 0:10)
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 5, tzinfo=timezone.utc),
            [
                (2700, 2705, 2698, 2702, 10),  # low=2698 <= 2700 -> fill
                (2702, 2710, 2701, 2708, 20),
            ],
        )
        assert _verify_e3_sub_bar_fill(bars, entry_ts, 2700.0, "long") is True

    def test_long_fill_rejected_by_1m_data(self):
        """1m bars don't actually reach the limit price -> fill rejected."""
        entry_ts = datetime(2024, 1, 5, 0, 5, tzinfo=timezone.utc)
        # 1m bars within the 5m candle -- low never touches 2700
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 5, tzinfo=timezone.utc),
            [
                (2702, 2705, 2701, 2704, 10),  # low=2701 > 2700
                (2704, 2710, 2703, 2708, 20),   # low=2703 > 2700
            ],
        )
        assert _verify_e3_sub_bar_fill(bars, entry_ts, 2700.0, "long") is False

    def test_short_fill_confirmed_by_1m_high(self):
        """Short E3: 1m bar high reaches the limit price."""
        entry_ts = datetime(2024, 1, 5, 0, 5, tzinfo=timezone.utc)
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 5, tzinfo=timezone.utc),
            [
                (2690, 2692, 2685, 2688, 10),  # high=2692 >= 2690
            ],
        )
        assert _verify_e3_sub_bar_fill(bars, entry_ts, 2690.0, "short") is True

    def test_short_fill_rejected_by_1m_data(self):
        """Short E3: 1m bars don't reach the limit price."""
        entry_ts = datetime(2024, 1, 5, 0, 5, tzinfo=timezone.utc)
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 5, tzinfo=timezone.utc),
            [
                (2685, 2689, 2682, 2686, 10),  # high=2689 < 2690
            ],
        )
        assert _verify_e3_sub_bar_fill(bars, entry_ts, 2690.0, "short") is False

    def test_no_bars_in_candle(self):
        """No 1m bars in the 5m candle window -> fill rejected."""
        entry_ts = datetime(2024, 1, 5, 0, 5, tzinfo=timezone.utc)
        # Bars are at 0:00, outside the 5m candle [0:05, 0:10)
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc),
            [(2700, 2705, 2698, 2702, 10)],
        )
        assert _verify_e3_sub_bar_fill(bars, entry_ts, 2700.0, "long") is False


# ============================================================================
# 5m outcome differs from 1m outcome
# ============================================================================

class TestOutcomeResolutionDifference:
    """Verify that same setup with 1m vs 5m bars can produce different results."""

    def test_5m_outcome_can_differ_from_1m(self):
        """Same ORB setup: 1m bars show win (fast spike), 5m bars show loss.

        With 1m bars and E1: confirm at :00, entry at :01 open, target hit at :07.
        With 5m bars and E1: confirm at 5m candle 0, entry at 5m candle 1 open,
        fill bar high < target (no fill-bar exit), then candle 2 has both
        target+stop -> ambiguous -> loss.
        """
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=timezone.utc)

        # 15 bars = 3 x 5m candles after resample
        # E1 entry = 2701 (bar :01 open). Risk=11. Target=2723.
        # Key: 5m candle 1 high must be < 2723 so fill-bar exit doesn't trigger.
        # 1m bars :05-:09 highs kept <= 2722.
        bars_1m = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc),
            [
                # 5m candle 0 (:00-:04): confirm bar
                (2698, 2702, 2695, 2701, 100),  # :00 close=2701 > 2700 -> confirm
                (2701, 2702, 2700, 2701, 100),  # :01 E1 entry at open=2701
                (2701, 2702, 2700, 2701, 100),  # :02
                (2701, 2702, 2700, 2701, 100),  # :03
                (2701, 2702, 2700, 2701, 100),  # :04
                # 5m candle 1 (:05-:09): gradual rise, no 1m bar hits target
                (2701, 2710, 2700, 2710, 100),  # :05
                (2710, 2716, 2709, 2715, 100),  # :06
                (2715, 2722, 2714, 2720, 100),  # :07 high=2722 < target=2723
                (2720, 2722, 2719, 2720, 100),  # :08
                (2720, 2721, 2719, 2720, 100),  # :09
                # 5m candle 2 (:10-:14): spike + crash -> 1m :10 hits target alone
                (2720, 2724, 2718, 2722, 100),  # :10 high=2724 >= 2723 -> 1m WIN
                (2722, 2723, 2685, 2686, 100),  # :11 crash (hits stop on 1m)
                (2686, 2688, 2684, 2685, 100),  # :12
                (2685, 2686, 2683, 2684, 100),  # :13
                (2684, 2685, 2681, 2682, 100),  # :14
            ],
        )

        # 1m path: E1 entry at :01 open = 2701
        # Risk = 2701 - 2690 = 11. Target = 2701 + 22 = 2723
        # Bar :10 high = 2724 >= 2723 -> WIN (target hit before stop on 1m)
        result_1m = compute_single_outcome(
            bars_df=bars_1m, break_ts=break_ts,
            orb_high=orb_high, orb_low=orb_low, break_dir="long",
            rr_target=2.0, confirm_bars=1, trading_day_end=td_end,
            cost_spec=_cost(), entry_model="E1",
        )
        assert result_1m["outcome"] == "win"

        # 5m path: 3 candles
        # Candle 0 (:00): open=2698, high=2702, low=2695, close=2701 -> confirm
        # Candle 1 (:05): E1 entry at open=2701, high=2722 < 2723 -> no fill-bar exit
        # Candle 2 (:10): open=2720, high=2724, low=2681, close=2682
        #   high=2724 >= 2723 -> target hit
        #   low=2681 <= 2690 -> stop hit
        #   BOTH on same bar -> ambiguous -> conservative LOSS
        bars_5m = resample_to_5m(bars_1m, datetime(2024, 1, 4, 23, 59, tzinfo=timezone.utc))
        assert len(bars_5m) == 3  # verify 3 candles

        result_5m = compute_single_outcome(
            bars_df=bars_5m, break_ts=break_ts,
            orb_high=orb_high, orb_low=orb_low, break_dir="long",
            rr_target=2.0, confirm_bars=1, trading_day_end=td_end,
            cost_spec=_cost(), entry_model="E1",
        )
        assert result_5m["outcome"] == "loss"

        # Different outcomes prove resolution matters
        assert result_1m["outcome"] != result_5m["outcome"]


# ============================================================================
# Confirm bars on 5m bars
# ============================================================================

class TestConfirmBarsOn5m:
    """Verify confirm bar logic works correctly with 5m bars."""

    def test_cb1_on_5m_is_structural_hold(self):
        """CB1 on 5m = single 5-minute close outside ORB (structural acceptance)."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=timezone.utc)

        # 5m bar that closes above ORB high
        bars_5m = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc),
            [
                (2698, 2705, 2695, 2702, 500),  # close 2702 > 2700 -> confirm
                (2702, 2730, 2701, 2725, 600),   # E1 entry at open = 2702
                # target at RR2.0: 2702 + (2702-2690)*2 = 2702+24 = 2726
                (2725, 2730, 2720, 2728, 400),   # high 2730 > 2726 -> win
            ],
            interval_minutes=5,
        )

        result = compute_single_outcome(
            bars_df=bars_5m, break_ts=break_ts,
            orb_high=orb_high, orb_low=orb_low, break_dir="long",
            rr_target=2.0, confirm_bars=1, trading_day_end=td_end,
            cost_spec=_cost(), entry_model="E1",
        )
        assert result["outcome"] == "win"
        assert result["entry_price"] == 2702.0  # next bar open after confirm
