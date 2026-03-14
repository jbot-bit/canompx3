"""
Tests for PerformanceMonitor DB persistence.
"""

import duckdb
from datetime import date
from unittest.mock import MagicMock

from pipeline.init_db import init_db
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
    init_db(db, force=False)

    monitor = PerformanceMonitor([_make_strategy()], db_path=db)
    monitor.record_trade(TradeRecord(
        strategy_id="MGC_TEST_E1", trading_day=date(2026, 3, 15),
        direction="long", entry_price=3000.0, exit_price=3004.0,
        actual_r=0.8, expected_r=0.1,
    ))

    con = duckdb.connect(str(db), read_only=True)
    rows = con.execute("SELECT strategy_id FROM pm_trade_log").fetchall()
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


def test_record_trade_db_write_failure_doesnt_block_cusum(tmp_path):
    """Write failure must not prevent CUSUM update or crash the caller."""
    db = tmp_path / "test.db"
    # Intentionally do NOT create the pm_trade_log table — write will fail
    duckdb.connect(str(db)).close()

    monitor = PerformanceMonitor([_make_strategy()], db_path=db)
    # Should not raise — fail-open
    monitor.record_trade(TradeRecord(
        strategy_id="MGC_TEST_E1", trading_day=date(2026, 3, 15),
        direction="long", entry_price=3000.0, exit_price=3004.0,
        actual_r=0.8, expected_r=0.1,
    ))
    # CUSUM still ran — monitor is functional
    cusum = monitor.get_cusum("MGC_TEST_E1")
    assert cusum is not None and cusum.n_trades == 1
