"""
Tests for LiveORBBuilder.

CME_REOPEN for trading_day 2026-03-03 (March, winter/CST):
  cme_open_brisbane → (9, 0) Brisbane
  Brisbane 09:00 → UTC 23:00 of the PREVIOUS calendar day (2026-03-02 23:00 UTC)

Bars must be at 2026-03-02 23:xx UTC to fall inside this window.
"""
from datetime import date, datetime, timezone

import pytest

from trading_app.live.bar_aggregator import Bar
from trading_app.live.live_market_state import LiveORBBuilder


def _bar(cal_day: int, hour_utc: int, minute_utc: int,
         high: float, low: float, symbol="MGC") -> Bar:
    """Create a Bar. cal_day = day of month in March 2026."""
    ts = datetime(2026, 3, cal_day, hour_utc, minute_utc, 0, tzinfo=timezone.utc)
    mid = (high + low) / 2
    return Bar(ts_utc=ts, open=mid, high=high, low=low, close=mid, volume=10, symbol=symbol)


TRADING_DAY = date(2026, 3, 3)


def test_orb_incomplete_before_enough_bars():
    """With 4 bars we don't yet have a complete 5-min ORB."""
    # CME_REOPEN for 2026-03-03 → session starts 2026-03-02 23:00 UTC
    builder = LiveORBBuilder(instrument="MGC", trading_day=TRADING_DAY)
    for m in range(4):
        builder.on_bar(_bar(2, 23, m, high=2001.0 + m, low=1999.0 - m))
    orb = builder.get_orb("CME_REOPEN", orb_minutes=5)
    assert orb is None


def test_orb_complete_after_five_bars():
    """With 5 bars (minutes 0-4), the 5-min ORB is complete."""
    builder = LiveORBBuilder(instrument="MGC", trading_day=TRADING_DAY)
    for m in range(5):
        builder.on_bar(_bar(2, 23, m, high=2001.0 + m, low=1999.0 - m))
    orb = builder.get_orb("CME_REOPEN", orb_minutes=5)
    assert orb is not None
    assert orb.high == 2005.0   # max(2001,2002,2003,2004,2005)
    assert orb.low == 1995.0    # min(1999,1998,1997,1996,1995)
    assert orb.complete is True
    assert orb.bars_count == 5


def test_orb_size_computed():
    """ORB size = high - low."""
    builder = LiveORBBuilder(instrument="MGC", trading_day=TRADING_DAY)
    for m in range(5):
        builder.on_bar(_bar(2, 23, m, high=2010.0, low=2000.0))
    orb = builder.get_orb("CME_REOPEN", orb_minutes=5)
    assert orb is not None
    assert abs(orb.size - 10.0) < 0.001


def test_bars_outside_window_excluded():
    """Bars before the session window don't count toward the ORB."""
    builder = LiveORBBuilder(instrument="MGC", trading_day=TRADING_DAY)
    # Add bar at 22:00 UTC (before CME_REOPEN window at 23:00 UTC)
    builder.on_bar(_bar(2, 22, 0, high=9999.0, low=1.0))
    # Add 5 bars in the window
    for m in range(5):
        builder.on_bar(_bar(2, 23, m, high=2001.0, low=1999.0))
    orb = builder.get_orb("CME_REOPEN", orb_minutes=5)
    assert orb is not None
    assert orb.high == 2001.0   # 9999 excluded
    assert orb.low == 1999.0


def test_unknown_session_returns_none():
    builder = LiveORBBuilder(instrument="MGC", trading_day=TRADING_DAY)
    builder.on_bar(_bar(2, 23, 0, high=2000.0, low=1990.0))
    assert builder.get_orb("NONEXISTENT_SESSION", orb_minutes=5) is None


def test_get_bars_since_filters_correctly():
    """get_bars_since returns only bars at or after session start."""
    builder = LiveORBBuilder(instrument="MGC", trading_day=TRADING_DAY)
    # Before session
    builder.on_bar(_bar(2, 22, 0, high=9999.0, low=1.0))
    # In session
    for m in range(3):
        builder.on_bar(_bar(2, 23, m, high=2001.0, low=1999.0))
    bars = builder.get_bars_since("CME_REOPEN")
    assert len(bars) == 3
