"""
Tests for PerformanceMonitor DB persistence.
"""

import duckdb
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


def test_record_trade_writes_to_db(tmp_path):
    db = tmp_path / "test.db"
    con = duckdb.connect(str(db))
    con.execute("""CREATE TABLE live_trades (
        id INTEGER, strategy_id TEXT, trading_day DATE, direction TEXT,
        entry_price DOUBLE, exit_price DOUBLE, actual_r DOUBLE,
        expected_r DOUBLE, slippage_pts DOUBLE, recorded_at TIMESTAMPTZ
    )""")
    con.close()

    monitor = PerformanceMonitor([_make_strategy()], db_path=db)
    monitor.record_trade(TradeRecord(
        strategy_id="MGC_TEST_E1", trading_day=date(2026, 3, 15),
        direction="long", entry_price=3000.0, exit_price=3004.0,
        actual_r=0.8, expected_r=0.1,
    ))

    con = duckdb.connect(str(db), read_only=True)
    rows = con.execute("SELECT strategy_id FROM live_trades").fetchall()
    con.close()
    assert len(rows) == 1
    assert rows[0][0] == "MGC_TEST_E1"


def test_record_trade_no_db_doesnt_crash():
    monitor = PerformanceMonitor([_make_strategy()], db_path=None)
    monitor.record_trade(TradeRecord(
        strategy_id="MGC_TEST_E1", trading_day=date(2026, 3, 15),
        direction="long", entry_price=3000.0, exit_price=3004.0,
        actual_r=0.8, expected_r=0.1,
    ))
