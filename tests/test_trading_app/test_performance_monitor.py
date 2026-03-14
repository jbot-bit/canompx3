"""
Tests for PerformanceMonitor (in-memory CUSUM tracking).

Trade persistence is handled by TradeJournal — PerformanceMonitor is in-memory only.
"""

from datetime import date
from unittest.mock import MagicMock

from trading_app.live.performance_monitor import PerformanceMonitor, TradeRecord


def _make_strategy():
    s = MagicMock()
    s.strategy_id = "MGC_TEST_E1"
    s.expectancy_r = 0.1
    s.win_rate = 0.55
    s.rr_target = 2.0
    return s


def test_record_trade_updates_daily_r():
    monitor = PerformanceMonitor([_make_strategy()])
    monitor.record_trade(TradeRecord(
        strategy_id="MGC_TEST_E1", trading_day=date(2026, 3, 15),
        direction="long", entry_price=3000.0, exit_price=3004.0,
        actual_r=0.8, expected_r=0.1,
    ))
    summary = monitor.daily_summary()
    assert summary["n_trades"] == 1
    assert summary["total_r"] == 0.8


def test_cusum_tracks_trades():
    monitor = PerformanceMonitor([_make_strategy()])
    monitor.record_trade(TradeRecord(
        strategy_id="MGC_TEST_E1", trading_day=date(2026, 3, 15),
        direction="long", entry_price=3000.0, exit_price=3004.0,
        actual_r=0.8, expected_r=0.1,
    ))
    cusum = monitor.get_cusum("MGC_TEST_E1")
    assert cusum is not None and cusum.n_trades == 1


def test_reset_daily_clears_state():
    monitor = PerformanceMonitor([_make_strategy()])
    monitor.record_trade(TradeRecord(
        strategy_id="MGC_TEST_E1", trading_day=date(2026, 3, 15),
        direction="long", entry_price=3000.0, exit_price=3004.0,
        actual_r=0.8, expected_r=0.1,
    ))
    monitor.reset_daily()
    assert monitor.trade_count == 0
    summary = monitor.daily_summary()
    assert summary["n_trades"] == 0
