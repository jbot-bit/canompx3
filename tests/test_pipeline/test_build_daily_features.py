"""
Tests for pipeline/build_daily_features.py — staged daily features builder.

Tests organized by module:
  1. Trading day assignment
  2. ORB ranges
  3. Break detection
  4. Session stats
  5. RSI (Wilder's)
  6. Outcome at RR=1.0
  7. Orchestrator integration
"""

import pytest
import duckdb
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from pipeline.build_daily_features import (
    compute_trading_day,
    compute_trading_day_utc_range,
    compute_orb_range,
    detect_break,
    compute_session_stats,
    compute_overnight_stats,
    classify_day_type,
    compute_garch_forecast,
    _wilders_rsi,
    compute_outcome,
    build_features_for_day,
    build_daily_features,
    verify_daily_features,
    _orb_utc_window,
    _session_utc_window,
    ORB_LABELS,
    BRISBANE_TZ,
    UTC_TZ,
)


# =============================================================================
# HELPERS
# =============================================================================

def _ts(year, month, day, hour, minute=0) -> pd.Timestamp:
    """Create a UTC-aware Timestamp."""
    return pd.Timestamp(year=year, month=month, day=day,
                        hour=hour, minute=minute, tz="UTC")


def _make_bars(timestamps, opens, highs, lows, closes, volumes=None):
    """Build a bars DataFrame from lists."""
    n = len(timestamps)
    if volumes is None:
        volumes = [100] * n
    return pd.DataFrame({
        "ts_utc": timestamps,
        "open": np.array(opens, dtype=float),
        "high": np.array(highs, dtype=float),
        "low": np.array(lows, dtype=float),
        "close": np.array(closes, dtype=float),
        "volume": np.array(volumes, dtype=int),
        "source_symbol": ["MGCM4"] * n,
    })


# =============================================================================
# MODULE 1: TRADING DAY ASSIGNMENT
# =============================================================================

class TestTradingDay:

    def test_bar_at_2300_utc_belongs_to_next_trading_day(self):
        """23:00 UTC = 09:00 Brisbane -> new trading day."""
        ts = pd.Timestamp("2024-01-04 23:00:00", tz="UTC")
        td = compute_trading_day(ts)
        assert td == date(2024, 1, 5)

    def test_bar_at_2259_utc_belongs_to_current_trading_day(self):
        """22:59 UTC = 08:59 Brisbane -> still previous trading day."""
        ts = pd.Timestamp("2024-01-04 22:59:00", tz="UTC")
        td = compute_trading_day(ts)
        assert td == date(2024, 1, 4)

    def test_bar_at_0000_utc_belongs_to_trading_day(self):
        """00:00 UTC = 10:00 Brisbane -> same trading day as Brisbane date."""
        ts = pd.Timestamp("2024-01-05 00:00:00", tz="UTC")
        td = compute_trading_day(ts)
        assert td == date(2024, 1, 5)

    def test_bar_at_1430_utc_belongs_to_trading_day(self):
        """14:30 UTC = 00:30 Brisbane (next day) -> same trading day."""
        ts = pd.Timestamp("2024-01-05 14:30:00", tz="UTC")
        td = compute_trading_day(ts)
        assert td == date(2024, 1, 5)

    def test_trading_day_utc_range(self):
        """Trading day 2024-01-05: [2024-01-04 23:00 UTC, 2024-01-05 23:00 UTC)."""
        start, end = compute_trading_day_utc_range(date(2024, 1, 5))
        assert start == datetime(2024, 1, 4, 23, 0, 0, tzinfo=UTC_TZ)
        assert end == datetime(2024, 1, 5, 23, 0, 0, tzinfo=UTC_TZ)

    def test_trading_day_utc_range_24h(self):
        """Range is exactly 24 hours."""
        start, end = compute_trading_day_utc_range(date(2024, 6, 15))
        assert (end - start) == timedelta(hours=24)


# =============================================================================
# MODULE 2: ORB RANGES
# =============================================================================

class TestOrbRanges:

    def test_0900_orb_window_5min(self):
        """0900 ORB on 2024-01-05: 23:00-23:05 UTC on 2024-01-04."""
        start, end = _orb_utc_window(date(2024, 1, 5), "0900", 5)
        assert start == datetime(2024, 1, 4, 23, 0, 0, tzinfo=UTC_TZ)
        assert end == datetime(2024, 1, 4, 23, 5, 0, tzinfo=UTC_TZ)

    def test_0900_orb_window_15min(self):
        """0900 ORB with 15min duration."""
        start, end = _orb_utc_window(date(2024, 1, 5), "0900", 15)
        assert start == datetime(2024, 1, 4, 23, 0, 0, tzinfo=UTC_TZ)
        assert end == datetime(2024, 1, 4, 23, 15, 0, tzinfo=UTC_TZ)

    def test_0900_orb_window_30min(self):
        """0900 ORB with 30min duration."""
        start, end = _orb_utc_window(date(2024, 1, 5), "0900", 30)
        assert start == datetime(2024, 1, 4, 23, 0, 0, tzinfo=UTC_TZ)
        assert end == datetime(2024, 1, 4, 23, 30, 0, tzinfo=UTC_TZ)

    def test_1000_orb_window(self):
        """1000 ORB: 00:00-00:05 UTC on trading day."""
        start, end = _orb_utc_window(date(2024, 1, 5), "1000", 5)
        assert start == datetime(2024, 1, 5, 0, 0, 0, tzinfo=UTC_TZ)
        assert end == datetime(2024, 1, 5, 0, 5, 0, tzinfo=UTC_TZ)

    def test_0030_orb_window(self):
        """0030 ORB: crosses midnight Brisbane, same trading day.
        00:30 Brisbane = 14:30 UTC. But 0030 is after midnight Brisbane
        so calendar day is trading_day + 1."""
        start, end = _orb_utc_window(date(2024, 1, 5), "0030", 5)
        assert start == datetime(2024, 1, 5, 14, 30, 0, tzinfo=UTC_TZ)
        assert end == datetime(2024, 1, 5, 14, 35, 0, tzinfo=UTC_TZ)

    def test_orb_range_basic(self):
        """ORB range from 5 bars within window."""
        bars = _make_bars(
            timestamps=[_ts(2024, 1, 4, 23, m) for m in range(5)],
            opens=[2350, 2351, 2349, 2350, 2352],
            highs=[2355, 2353, 2351, 2353, 2356],
            lows=[2348, 2349, 2347, 2349, 2351],
            closes=[2351, 2350, 2349, 2352, 2354],
        )
        result = compute_orb_range(bars, date(2024, 1, 5), "0900", 5)
        assert result["high"] == 2356.0
        assert result["low"] == 2347.0
        assert result["size"] == 9.0

    def test_orb_range_no_data(self):
        """ORB returns None when no bars in window."""
        bars = _make_bars(
            timestamps=[_ts(2024, 1, 5, 1, 0)],  # outside 0900 window
            opens=[2350], highs=[2352], lows=[2349], closes=[2351],
        )
        result = compute_orb_range(bars, date(2024, 1, 5), "0900", 5)
        assert result["high"] is None
        assert result["low"] is None
        assert result["size"] is None

    def test_orb_range_partial_bars(self):
        """ORB with fewer than orb_minutes bars still computes."""
        bars = _make_bars(
            timestamps=[_ts(2024, 1, 4, 23, 0), _ts(2024, 1, 4, 23, 2)],
            opens=[2350, 2351],
            highs=[2355, 2353],
            lows=[2348, 2350],
            closes=[2351, 2352],
        )
        result = compute_orb_range(bars, date(2024, 1, 5), "0900", 5)
        assert result["high"] == 2355.0
        assert result["low"] == 2348.0
        assert result["size"] == 7.0


# =============================================================================
# MODULE 3: BREAK DETECTION
# =============================================================================

class TestBreakDetection:

    def test_long_break(self):
        """First close above ORB high = long break."""
        # ORB: high=2355, low=2348
        # Break detection window after ORB (0900 5min ends at 23:05 UTC)
        bars = _make_bars(
            timestamps=[
                _ts(2024, 1, 4, 23, 5),  # first bar after ORB
                _ts(2024, 1, 4, 23, 6),
                _ts(2024, 1, 4, 23, 7),
            ],
            opens=[2354, 2355, 2356],
            highs=[2355, 2357, 2358],
            lows=[2353, 2354, 2355],
            closes=[2354, 2356, 2357],  # 2356 > 2355 = break long
        )
        result = detect_break(bars, date(2024, 1, 5), "0900", 5, 2355.0, 2348.0)
        assert result["break_dir"] == "long"
        assert result["break_ts"] == _ts(2024, 1, 4, 23, 6).to_pydatetime()

    def test_short_break(self):
        """First close below ORB low = short break."""
        bars = _make_bars(
            timestamps=[
                _ts(2024, 1, 4, 23, 5),
                _ts(2024, 1, 4, 23, 6),
            ],
            opens=[2349, 2347],
            highs=[2350, 2348],
            lows=[2347, 2345],
            closes=[2349, 2347],  # 2347 < 2348 = break short
        )
        result = detect_break(bars, date(2024, 1, 5), "0900", 5, 2355.0, 2348.0)
        assert result["break_dir"] == "short"

    def test_no_break(self):
        """No close outside ORB range = no break."""
        bars = _make_bars(
            timestamps=[
                _ts(2024, 1, 4, 23, 5),
                _ts(2024, 1, 4, 23, 6),
            ],
            opens=[2350, 2351],
            highs=[2354, 2354],
            lows=[2349, 2349],
            closes=[2352, 2353],  # both within [2348, 2355]
        )
        result = detect_break(bars, date(2024, 1, 5), "0900", 5, 2355.0, 2348.0)
        assert result["break_dir"] is None
        assert result["break_ts"] is None

    def test_break_uses_close_not_high(self):
        """Break requires CLOSE outside range, not just high/low touch."""
        bars = _make_bars(
            timestamps=[_ts(2024, 1, 4, 23, 5)],
            opens=[2354],
            highs=[2358],  # touches above ORB high
            lows=[2353],
            closes=[2354],  # but closes inside range
        )
        result = detect_break(bars, date(2024, 1, 5), "0900", 5, 2355.0, 2348.0)
        assert result["break_dir"] is None

    def test_break_with_no_orb(self):
        """No break if ORB range is None."""
        bars = _make_bars(
            timestamps=[_ts(2024, 1, 4, 23, 5)],
            opens=[2354], highs=[2358], lows=[2353], closes=[2356],
        )
        result = detect_break(bars, date(2024, 1, 5), "0900", 5, None, None)
        assert result["break_dir"] is None


# =============================================================================
# MODULE 4: SESSION STATS
# =============================================================================

class TestSessionStats:

    def test_asia_session_window(self):
        """Fixed Asia stat window: 09:00-17:00 Brisbane (not DST-aware)."""
        start, end = _session_utc_window(date(2024, 1, 5), "asia")
        # 09:00 Brisbane = 23:00 UTC Jan 4
        assert start == datetime(2024, 1, 4, 23, 0, 0, tzinfo=UTC_TZ)
        # 17:00 Brisbane = 07:00 UTC Jan 5
        assert end == datetime(2024, 1, 5, 7, 0, 0, tzinfo=UTC_TZ)

    def test_london_session_window(self):
        """Fixed London stat window: 18:00-23:00 Brisbane (not DST-aware)."""
        start, end = _session_utc_window(date(2024, 1, 5), "london")
        assert start == datetime(2024, 1, 5, 8, 0, 0, tzinfo=UTC_TZ)
        assert end == datetime(2024, 1, 5, 13, 0, 0, tzinfo=UTC_TZ)

    def test_ny_session_window(self):
        """Fixed NY stat window: 23:00-02:00 Brisbane (not DST-aware)."""
        start, end = _session_utc_window(date(2024, 1, 5), "ny")
        assert start == datetime(2024, 1, 5, 13, 0, 0, tzinfo=UTC_TZ)
        assert end == datetime(2024, 1, 5, 16, 0, 0, tzinfo=UTC_TZ)

    def test_session_stats_basic(self):
        """Session stats computed from bars within session windows."""
        bars = _make_bars(
            timestamps=[
                # Asia session (23:00-07:00 UTC)
                _ts(2024, 1, 4, 23, 0),
                _ts(2024, 1, 5, 3, 0),
                # London session (08:00-13:00 UTC)
                _ts(2024, 1, 5, 9, 0),
                _ts(2024, 1, 5, 10, 0),
                # NY session (13:00-16:00 UTC)
                _ts(2024, 1, 5, 14, 0),
            ],
            opens=[2350, 2355, 2360, 2365, 2370],
            highs=[2352, 2358, 2362, 2368, 2375],
            lows=[2348, 2353, 2358, 2363, 2368],
            closes=[2351, 2356, 2361, 2366, 2372],
        )
        result = compute_session_stats(bars, date(2024, 1, 5))
        assert result["session_asia_high"] == 2358.0
        assert result["session_asia_low"] == 2348.0
        assert result["session_london_high"] == 2368.0
        assert result["session_london_low"] == 2358.0
        assert result["session_ny_high"] == 2375.0
        assert result["session_ny_low"] == 2368.0

    def test_session_stats_no_data(self):
        """Session returns None when no bars in window."""
        bars = _make_bars(
            timestamps=[_ts(2024, 1, 4, 23, 0)],  # Only Asia
            opens=[2350], highs=[2352], lows=[2348], closes=[2351],
        )
        result = compute_session_stats(bars, date(2024, 1, 5))
        assert result["session_asia_high"] == 2352.0
        assert result["session_london_high"] is None
        assert result["session_ny_high"] is None


# =============================================================================
# MODULE 4b: Overnight Stats + Day Type
# =============================================================================

class TestOvernightStats:

    def test_overnight_keys_present(self):
        """compute_overnight_stats returns all required keys."""
        result = compute_overnight_stats(pd.DataFrame(), date(2024, 1, 5))
        assert set(result.keys()) == {
            "overnight_high", "overnight_low", "overnight_range",
            "pre_1000_high", "pre_1000_low",
        }

    def test_overnight_all_none_on_empty_bars(self):
        """All None when no bars."""
        result = compute_overnight_stats(pd.DataFrame(), date(2024, 1, 5))
        assert result["overnight_high"] is None
        assert result["overnight_low"] is None
        assert result["overnight_range"] is None
        assert result["pre_1000_high"] is None
        assert result["pre_1000_low"] is None

    def test_overnight_range_from_asia_bars(self):
        """overnight_high/low computed from Asia session window bars only."""
        # Asia window: 23:00 UTC Jan 4 to 07:00 UTC Jan 5 (09:00-17:00 Brisbane)
        bars = _make_bars(
            timestamps=[
                _ts(2024, 1, 4, 23, 30),  # Asia
                _ts(2024, 1, 5, 3, 0),    # Asia
                _ts(2024, 1, 5, 10, 0),   # NY session (outside Asia)
            ],
            opens=[2350, 2355, 2380],
            highs=[2355, 2360, 2390],
            lows=[2348, 2352, 2378],
            closes=[2352, 2358, 2385],
        )
        result = compute_overnight_stats(bars, date(2024, 1, 5))
        assert result["overnight_high"] == 2360.0
        assert result["overnight_low"] == 2348.0
        assert result["overnight_range"] == 2360.0 - 2348.0

    def test_pre_1000_excludes_1000_bars(self):
        """pre_1000_* uses bars before 10:00 Brisbane (00:00 UTC Jan 5)."""
        # 1000 Brisbane = 00:00 UTC (Brisbane is UTC+10)
        # pre_1000 = bars in [trading_day_start, 00:00 UTC Jan 5)
        bars = _make_bars(
            timestamps=[
                _ts(2024, 1, 4, 23, 0),   # 09:00 Brisbane — included
                _ts(2024, 1, 4, 23, 30),  # 09:30 Brisbane — included
                _ts(2024, 1, 5, 0, 0),    # 10:00 Brisbane — excluded (start of ORB)
            ],
            opens=[2350, 2355, 2370],
            highs=[2356, 2360, 2380],
            lows=[2348, 2353, 2368],
            closes=[2354, 2358, 2375],
        )
        result = compute_overnight_stats(bars, date(2024, 1, 5))
        # pre_1000 should only see the first two bars
        assert result["pre_1000_high"] == 2360.0
        assert result["pre_1000_low"] == 2348.0


class TestClassifyDayType:

    def test_trend_up(self):
        """Wide range, closed in top 30% = TREND_UP."""
        # range=5, atr=4 → range_pct=1.25. close at 4.8/5=0.96 → TREND_UP
        assert classify_day_type(100.0, 105.0, 100.0, 104.8, 4.0) == "TREND_UP"

    def test_trend_down(self):
        """Wide range, closed in bottom 30% = TREND_DOWN."""
        # range=5, atr=4 → range_pct=1.25. close at 0.2/5=0.04 → TREND_DOWN
        assert classify_day_type(105.0, 105.0, 100.0, 100.2, 4.0) == "TREND_DOWN"

    def test_non_trend(self):
        """Range < 50% of ATR = NON_TREND."""
        # range=1.5, atr=4 → range_pct=0.375 < 0.5 → NON_TREND
        assert classify_day_type(100.0, 101.0, 99.5, 100.2, 4.0) == "NON_TREND"

    def test_balanced(self):
        """Medium range, closed in middle = BALANCED."""
        # range=4, atr=4 → range_pct=1.0. close at 2/4=0.5 → not TREND, not reversal
        assert classify_day_type(100.0, 104.0, 100.0, 102.0, 4.0) == "BALANCED"

    def test_reversal_up(self):
        """Opened low, closed high (medium range) = REVERSAL_UP."""
        # range=4, atr=4 → range_pct=1.0
        # close_pct = 2.5/4 = 0.625 (in 0.3-0.7 range, avoids TREND checks)
        # open=100.5 < lower_40pct=101.6, close=102.5 > upper_60pct=102.4
        assert classify_day_type(100.5, 104.0, 100.0, 102.5, 4.0) == "REVERSAL_UP"

    def test_reversal_down(self):
        """Opened high, closed low (medium range) = REVERSAL_DOWN."""
        # close_pct = 1.5/4 = 0.375 (in 0.3-0.7 range, avoids TREND checks)
        # open=103.5 > upper_60pct=102.4, close=101.5 < lower_40pct=101.6
        assert classify_day_type(103.5, 104.0, 100.0, 101.5, 4.0) == "REVERSAL_DOWN"

    def test_none_on_any_missing_input(self):
        """Returns None when any parameter is None."""
        assert classify_day_type(None, 105.0, 100.0, 104.0, 4.0) is None
        assert classify_day_type(100.0, None, 100.0, 104.0, 4.0) is None
        assert classify_day_type(100.0, 105.0, None, 104.0, 4.0) is None
        assert classify_day_type(100.0, 105.0, 100.0, None, 4.0) is None
        assert classify_day_type(100.0, 105.0, 100.0, 104.0, None) is None

    def test_none_on_zero_atr(self):
        """Returns None when atr_20 is zero."""
        assert classify_day_type(100.0, 105.0, 100.0, 104.0, 0.0) is None

    def test_none_on_zero_range(self):
        """Returns None when high == low."""
        assert classify_day_type(100.0, 100.0, 100.0, 100.0, 4.0) is None


# =============================================================================
# MODULE 4c: GARCH Forecast
# =============================================================================

class TestGarchForecast:

    def test_garch_returns_none_on_insufficient_data(self):
        """GARCH returns None with fewer than 252 closes."""
        closes = [2350.0 + i for i in range(100)]
        result = compute_garch_forecast(closes)
        assert result is None

    def test_garch_returns_none_on_empty(self):
        """GARCH returns None on empty input."""
        assert compute_garch_forecast([]) is None

    def test_garch_returns_none_on_constant_prices(self):
        """GARCH returns None when all prices are identical (zero variance)."""
        closes = [2350.0] * 300
        result = compute_garch_forecast(closes)
        assert result is None


# =============================================================================
# MODULE 5: RSI
# =============================================================================

class TestRSI:

    def test_wilders_rsi_all_up(self):
        """All gains -> RSI = 100."""
        closes = np.array([float(i) for i in range(20)])  # 0,1,2,...,19
        rsi = _wilders_rsi(closes, period=14)
        assert rsi == 100.0

    def test_wilders_rsi_all_down(self):
        """All losses -> RSI = 0."""
        closes = np.array([float(20 - i) for i in range(20)])  # 20,19,...,1
        rsi = _wilders_rsi(closes, period=14)
        assert rsi == 0.0

    def test_wilders_rsi_insufficient_data(self):
        """Too few bars -> None."""
        closes = np.array([1.0, 2.0, 3.0])
        rsi = _wilders_rsi(closes, period=14)
        assert rsi is None

    def test_wilders_rsi_range(self):
        """RSI should be between 0 and 100."""
        np.random.seed(42)
        closes = np.cumsum(np.random.randn(100)) + 2350
        rsi = _wilders_rsi(closes, period=14)
        assert rsi is not None
        assert 0 <= rsi <= 100

    def test_wilders_rsi_known_value(self):
        """Test with a known sequence to verify correctness."""
        # Simple alternating: +1, -0.5, +1, -0.5, ...
        closes = [100.0]
        for i in range(30):
            if i % 2 == 0:
                closes.append(closes[-1] + 1.0)
            else:
                closes.append(closes[-1] - 0.5)
        closes = np.array(closes)
        rsi = _wilders_rsi(closes, period=14)
        assert rsi is not None
        # With +1/-0.5 pattern, gains dominate -> RSI > 50
        assert rsi > 50


# =============================================================================
# MODULE 6: OUTCOME
# =============================================================================

class TestOutcome:

    def _bars_after_break(self, highs, lows):
        """Helper to create bars after a break timestamp."""
        n = len(highs)
        base_ts = _ts(2024, 1, 4, 23, 10)
        timestamps = [base_ts + timedelta(minutes=i+1) for i in range(n)]
        return _make_bars(
            timestamps=timestamps,
            opens=[h - 1 for h in highs],
            highs=highs,
            lows=lows,
            closes=[(h + l) / 2 for h, l in zip(highs, lows)],
        )

    def test_long_win(self):
        """Long break, price reaches target (entry + ORB size) before stop."""
        # ORB: high=2350, low=2340 -> size=10
        # Entry=2350, target=2360, stop=2340
        bars = _make_bars(
            timestamps=[
                _ts(2024, 1, 4, 23, 10),  # break bar (not used for outcome)
                _ts(2024, 1, 4, 23, 11),
                _ts(2024, 1, 4, 23, 12),
            ],
            opens=[2351, 2355, 2358],
            highs=[2354, 2358, 2361],  # 2361 >= 2360 target
            lows=[2350, 2354, 2357],
            closes=[2353, 2357, 2360],
        )
        result = compute_outcome(
            bars, date(2024, 1, 5), "0900", 5,
            orb_high=2350.0, orb_low=2340.0,
            break_dir="long",
            break_ts=_ts(2024, 1, 4, 23, 10).to_pydatetime(),
        )
        assert result["outcome"] == "win"

    def test_long_loss(self):
        """Long break, price hits stop before target."""
        # ORB: high=2350, low=2340 -> size=10
        bars = _make_bars(
            timestamps=[
                _ts(2024, 1, 4, 23, 10),
                _ts(2024, 1, 4, 23, 11),
            ],
            opens=[2351, 2345],
            highs=[2352, 2346],
            lows=[2348, 2339],  # 2339 <= 2340 stop
            closes=[2349, 2341],
        )
        result = compute_outcome(
            bars, date(2024, 1, 5), "0900", 5,
            orb_high=2350.0, orb_low=2340.0,
            break_dir="long",
            break_ts=_ts(2024, 1, 4, 23, 10).to_pydatetime(),
        )
        assert result["outcome"] == "loss"

    def test_short_win(self):
        """Short break, price reaches target before stop."""
        # ORB: high=2350, low=2340 -> size=10
        # Entry=2340, target=2330, stop=2350
        bars = _make_bars(
            timestamps=[
                _ts(2024, 1, 4, 23, 10),
                _ts(2024, 1, 4, 23, 11),
            ],
            opens=[2339, 2335],
            highs=[2340, 2336],
            lows=[2335, 2329],  # 2329 <= 2330 target
            closes=[2336, 2330],
        )
        result = compute_outcome(
            bars, date(2024, 1, 5), "0900", 5,
            orb_high=2350.0, orb_low=2340.0,
            break_dir="short",
            break_ts=_ts(2024, 1, 4, 23, 10).to_pydatetime(),
        )
        assert result["outcome"] == "win"

    def test_scratch(self):
        """Neither target nor stop hit by end of day."""
        bars = _make_bars(
            timestamps=[
                _ts(2024, 1, 4, 23, 10),
                _ts(2024, 1, 4, 23, 11),
            ],
            opens=[2351, 2352],
            highs=[2354, 2355],  # doesn't reach 2360 target
            lows=[2349, 2350],   # doesn't reach 2340 stop
            closes=[2352, 2353],
        )
        result = compute_outcome(
            bars, date(2024, 1, 5), "0900", 5,
            orb_high=2350.0, orb_low=2340.0,
            break_dir="long",
            break_ts=_ts(2024, 1, 4, 23, 10).to_pydatetime(),
        )
        assert result["outcome"] == "scratch"

    def test_both_hit_same_bar_is_loss(self):
        """If target AND stop hit in same bar -> conservative = loss."""
        bars = _make_bars(
            timestamps=[
                _ts(2024, 1, 4, 23, 10),
                _ts(2024, 1, 4, 23, 11),
            ],
            opens=[2351, 2345],
            highs=[2352, 2361],  # hits target 2360
            lows=[2349, 2339],   # hits stop 2340
            closes=[2350, 2345],
        )
        result = compute_outcome(
            bars, date(2024, 1, 5), "0900", 5,
            orb_high=2350.0, orb_low=2340.0,
            break_dir="long",
            break_ts=_ts(2024, 1, 4, 23, 10).to_pydatetime(),
        )
        assert result["outcome"] == "loss"

    def test_no_break_no_outcome(self):
        """No break -> outcome is None."""
        bars = _make_bars(
            timestamps=[_ts(2024, 1, 4, 23, 10)],
            opens=[2350], highs=[2352], lows=[2348], closes=[2351],
        )
        result = compute_outcome(
            bars, date(2024, 1, 5), "0900", 5,
            orb_high=2350.0, orb_low=2340.0,
            break_dir=None, break_ts=None,
        )
        assert result["outcome"] is None

    def test_no_same_bar_execution(self):
        """Outcome starts evaluating AFTER break bar, not at it."""
        # Break bar at 23:10. Only bar is at 23:10 (break bar itself).
        # No bars after -> scratch
        bars = _make_bars(
            timestamps=[_ts(2024, 1, 4, 23, 10)],
            opens=[2351],
            highs=[2361],  # would hit target if used
            lows=[2339],   # would hit stop if used
            closes=[2351],
        )
        result = compute_outcome(
            bars, date(2024, 1, 5), "0900", 5,
            orb_high=2350.0, orb_low=2340.0,
            break_dir="long",
            break_ts=_ts(2024, 1, 4, 23, 10).to_pydatetime(),
        )
        assert result["outcome"] == "scratch"

    def test_mae_mfe_null(self):
        """MAE/MFE are NULL until cost model (Phase 2)."""
        bars = _make_bars(
            timestamps=[_ts(2024, 1, 4, 23, 10), _ts(2024, 1, 4, 23, 11)],
            opens=[2351, 2355],
            highs=[2354, 2361],
            lows=[2350, 2354],
            closes=[2353, 2360],
        )
        result = compute_outcome(
            bars, date(2024, 1, 5), "0900", 5,
            orb_high=2350.0, orb_low=2340.0,
            break_dir="long",
            break_ts=_ts(2024, 1, 4, 23, 10).to_pydatetime(),
        )
        assert result["mae_r"] is None
        assert result["mfe_r"] is None


# =============================================================================
# INTEGRATION: FULL BUILD WITH DB
# =============================================================================

class TestBuildIntegration:

    @pytest.fixture
    def feature_db(self, tmp_path):
        """Create a DB with bars_1m, bars_5m, daily_features and sample data."""
        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))

        # Create schema
        from pipeline.init_db import BARS_1M_SCHEMA, BARS_5M_SCHEMA, DAILY_FEATURES_SCHEMA
        con.execute(BARS_1M_SCHEMA)
        con.execute(BARS_5M_SCHEMA)
        con.execute(DAILY_FEATURES_SCHEMA)

        # Insert sample bars_1m for trading day 2024-01-05
        # Trading day starts at 23:00 UTC Jan 4
        # Insert bars covering the full 0900 ORB window + some after
        bars = []
        base = datetime(2024, 1, 4, 23, 0, 0, tzinfo=UTC_TZ)
        for i in range(60):  # 1 hour of 1m bars
            ts = base + timedelta(minutes=i)
            price = 2350.0 + i * 0.1
            bars.append((
                ts.isoformat(), 'MGC', 'MGCM4',
                price, price + 2.0, price - 1.0, price + 0.5,
                100 + i
            ))

        for b in bars:
            con.execute("""
                INSERT INTO bars_1m VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, list(b))

        # Insert some bars_5m for RSI computation (need 15+ bars)
        base_5m = datetime(2024, 1, 4, 20, 0, 0, tzinfo=UTC_TZ)
        for i in range(20):
            ts = base_5m + timedelta(minutes=i * 5)
            price = 2340.0 + i * 0.5
            con.execute("""
                INSERT INTO bars_5m VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [ts.isoformat(), 'MGC', 'MGCM4',
                  price, price + 1.0, price - 1.0, price + 0.3, 500])

        yield con
        con.close()

    def test_build_features_for_day(self, feature_db):
        """Full build for one trading day produces valid row."""
        row = build_features_for_day(feature_db, 'MGC', date(2024, 1, 5), 5)

        assert row["trading_day"] == date(2024, 1, 5)
        assert row["symbol"] == "MGC"
        assert row["bar_count_1m"] == 60

        # ORB 0900 should have data (bars start at 23:00 UTC = 09:00 Brisbane)
        assert row["orb_0900_high"] is not None
        assert row["orb_0900_low"] is not None
        assert row["orb_0900_size"] is not None
        assert row["orb_0900_size"] >= 0

        # Session Asia should have data
        assert row["session_asia_high"] is not None
        assert row["session_asia_low"] is not None

    def test_build_and_verify(self, feature_db):
        """Build and verify round-trip."""
        count = build_daily_features(
            feature_db, 'MGC', date(2024, 1, 5), date(2024, 1, 5), 5, False
        )
        assert count == 1

        ok, failures = verify_daily_features(
            feature_db, 'MGC', date(2024, 1, 5), date(2024, 1, 5)
        )
        assert ok, f"Verification failed: {failures}"

    def test_idempotent(self, feature_db):
        """Running build twice produces same result."""
        build_daily_features(
            feature_db, 'MGC', date(2024, 1, 5), date(2024, 1, 5), 5, False
        )
        build_daily_features(
            feature_db, 'MGC', date(2024, 1, 5), date(2024, 1, 5), 5, False
        )

        count = feature_db.execute("""
            SELECT COUNT(*) FROM daily_features
            WHERE symbol = 'MGC' AND trading_day = '2024-01-05'
        """).fetchone()[0]
        assert count == 1

    def test_dry_run_no_writes(self, feature_db):
        """Dry run doesn't write to database."""
        build_daily_features(
            feature_db, 'MGC', date(2024, 1, 5), date(2024, 1, 5), 5, True
        )

        count = feature_db.execute(
            "SELECT COUNT(*) FROM daily_features"
        ).fetchone()[0]
        assert count == 0

    def test_different_orb_minutes(self, feature_db):
        """Can build with different ORB durations."""
        row_5 = build_features_for_day(feature_db, 'MGC', date(2024, 1, 5), 5)
        row_15 = build_features_for_day(feature_db, 'MGC', date(2024, 1, 5), 15)

        # orb_minutes is stored in the row
        assert row_5["orb_minutes"] == 5
        assert row_15["orb_minutes"] == 15

        # 15-min ORB should have >= range as 5-min (more bars included)
        if row_5["orb_0900_size"] is not None and row_15["orb_0900_size"] is not None:
            assert row_15["orb_0900_size"] >= row_5["orb_0900_size"]

    def test_different_orb_minutes_coexist_in_db(self, feature_db):
        """5m and 15m builds can coexist for the same day."""
        build_daily_features(
            feature_db, 'MGC', date(2024, 1, 5), date(2024, 1, 5), 5, False
        )
        build_daily_features(
            feature_db, 'MGC', date(2024, 1, 5), date(2024, 1, 5), 15, False
        )

        count = feature_db.execute("""
            SELECT COUNT(*) FROM daily_features
            WHERE symbol = 'MGC' AND trading_day = '2024-01-05'
        """).fetchone()[0]
        assert count == 2  # one row for 5m, one for 15m

        orb_vals = feature_db.execute("""
            SELECT orb_minutes FROM daily_features
            WHERE symbol = 'MGC' AND trading_day = '2024-01-05'
            ORDER BY orb_minutes
        """).fetchall()
        assert [v[0] for v in orb_vals] == [5, 15]
