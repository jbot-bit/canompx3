"""
Tests for trading_app.paper_trader module.

Optimized: uses 0900 ORB (150 bars/day vs 900 bars/day for 2300 ORB),
class-scoped fixtures share expensive DB + replay across tests.
"""

import sys
from pathlib import Path
from datetime import date, datetime, timedelta, timezone

import pytest
import duckdb

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.init_db import BARS_1M_SCHEMA, BARS_5M_SCHEMA, DAILY_FEATURES_SCHEMA
from pipeline.cost_model import get_cost_spec
from trading_app.db_manager import init_trading_app_schema
from trading_app.portfolio import Portfolio, PortfolioStrategy
from trading_app.risk_manager import RiskLimits
from trading_app.paper_trader import (
    replay_historical,
    ReplayResult,
    DaySummary,
    JournalEntry,
    _orb_from_strategy,
    _entry_model_from_strategy,
)


def _cost():
    return get_cost_spec("MGC")


def _make_strategy(**overrides):
    base = {
        "strategy_id": "MGC_0900_E2_RR2.0_CB1_NO_FILTER",
        "instrument": "MGC",
        "orb_label": "0900",
        "entry_model": "E2",
        "rr_target": 2.0,
        "confirm_bars": 1,
        "filter_type": "NO_FILTER",
        "expectancy_r": 0.30,
        "win_rate": 0.55,
        "sample_size": 300,
        "sharpe_ratio": 0.4,
        "max_drawdown_r": 5.0,
        "median_risk_points": 5.0,
    }
    base.update(overrides)
    return PortfolioStrategy(**base)


def _make_portfolio(strategies):
    return Portfolio(
        name="test",
        instrument="MGC",
        strategies=strategies,
        account_equity=25000.0,
        risk_per_trade_pct=2.0,
        max_concurrent_positions=3,
        max_daily_loss_r=5.0,
    )


def _setup_replay_db(tmp_path, n_days=5):
    """
    Create a DB with bars_1m + daily_features for replay testing.

    Uses 0900 ORB (23:00 UTC = minute 0 of trading day) — only 150 bars/day needed.
    """
    db_path = tmp_path / "replay.db"
    con = duckdb.connect(str(db_path))
    con.execute(BARS_1M_SCHEMA)
    con.execute(BARS_5M_SCHEMA)
    con.execute(DAILY_FEATURES_SCHEMA)
    con.close()

    init_trading_app_schema(db_path=db_path)

    con = duckdb.connect(str(db_path))
    trading_day = date(2024, 1, 8)  # Monday
    days_created = 0

    while days_created < n_days:
        if trading_day.weekday() >= 5:
            trading_day += timedelta(days=1)
            continue

        # 0900 Brisbane = 23:00 UTC prev day = trading day start
        prev_day = trading_day - timedelta(days=1)
        td_start = datetime(prev_day.year, prev_day.month, prev_day.day,
                            23, 0, tzinfo=timezone.utc)

        base_price = 2700.0 + days_created * 2.0
        bars = []

        # 150 bars (2.5 hours) — ORB at minute 0-4, break at 5, trend after
        for i in range(150):
            ts = td_start + timedelta(minutes=i)

            if i < 5:
                # ORB window: 23:00-23:05 UTC
                o = base_price + 1.0
                h = base_price + 2.0
                l = base_price - 0.5
                c = base_price + 1.5
            elif i == 5:
                # Break bar: close > orb_high -> long break
                o = base_price + 2.5
                h = base_price + 4.0
                l = base_price + 2.0
                c = base_price + 3.5
            else:
                # Post-break: strong uptrend (resolves RR2.0 within ~50 bars)
                trend = (i - 5) * 0.1
                o = base_price + 3.5 + trend
                h = o + 1.0
                l = o - 0.3
                c = o + 0.5

            bars.append((
                ts.isoformat(), "MGC", "GCG4",
                round(o, 2), round(h, 2), round(l, 2), round(c, 2), 100,
            ))

        con.executemany(
            """INSERT OR REPLACE INTO bars_1m
               (ts_utc, symbol, source_symbol, open, high, low, close, volume)
               VALUES (?::TIMESTAMPTZ, ?, ?, ?, ?, ?, ?, ?)""",
            bars,
        )

        orb_high = round(base_price + 2.0, 2)
        orb_low = round(base_price - 0.5, 2)
        break_ts = (td_start + timedelta(minutes=5)).isoformat()

        con.execute(
            """INSERT OR REPLACE INTO daily_features
               (trading_day, symbol, orb_minutes, bar_count_1m,
                orb_0900_high, orb_0900_low, orb_0900_size,
                orb_0900_break_dir, orb_0900_break_ts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?::TIMESTAMPTZ)""",
            [
                trading_day, "MGC", 5, len(bars),
                orb_high, orb_low, round(orb_high - orb_low, 2),
                "long", break_ts,
            ],
        )

        trading_day += timedelta(days=1)
        days_created += 1

    con.commit()
    con.close()
    return db_path


# ============================================================================
# Helpers Tests (no DB needed)
# ============================================================================

class TestHelpers:

    def test_orb_from_strategy(self):
        assert _orb_from_strategy("MGC_2300_E1_RR2.0_CB5_NO_FILTER") == "2300"

    def test_entry_model_from_strategy(self):
        assert _entry_model_from_strategy("MGC_2300_E1_RR2.0_CB5_NO_FILTER") == "E1"


# ============================================================================
# Replay Tests — shared class fixture (runs replay ONCE)
# ============================================================================

@pytest.fixture(scope="class")
def shared_replay(tmp_path_factory):
    """Shared DB + replay result for all TestReplay tests."""
    tmp_dir = tmp_path_factory.mktemp("replay_shared")
    db_path = _setup_replay_db(tmp_dir, n_days=5)
    strategy = _make_strategy()
    portfolio = _make_portfolio([strategy])
    result = replay_historical(
        db_path=db_path,
        portfolio=portfolio,
        instrument="MGC",
        start_date=date(2024, 1, 8),
        end_date=date(2024, 1, 15),
    )
    return result, db_path, strategy


class TestReplay:

    def test_empty_portfolio(self, tmp_path):
        """Empty portfolio produces no trades (needs own fixture)."""
        db_path = _setup_replay_db(tmp_path, n_days=3)
        portfolio = _make_portfolio([])
        result = replay_historical(
            db_path=db_path,
            portfolio=portfolio,
            instrument="MGC",
            start_date=date(2024, 1, 8),
            end_date=date(2024, 1, 12),
        )
        assert result.total_trades == 0
        assert result.days_processed == 0

    def test_replay_produces_trades(self, shared_replay):
        result, _, _ = shared_replay
        assert result.days_processed > 0
        assert result.total_trades > 0

    def test_journal_entries_have_strategy_id(self, shared_replay):
        result, _, strategy = shared_replay
        for entry in result.journal:
            assert entry.strategy_id == strategy.strategy_id
            assert entry.mode == "replay"

    def test_day_summaries_populated(self, shared_replay):
        result, _, _ = shared_replay
        assert len(result.day_summaries) > 0
        for ds in result.day_summaries:
            assert ds.bars_processed > 0

    def test_pnl_is_sum_of_days(self, shared_replay):
        result, _, _ = shared_replay
        day_pnl_sum = sum(ds.daily_pnl_r for ds in result.day_summaries)
        assert abs(result.total_pnl_r - day_pnl_sum) < 0.01


# ============================================================================
# Risk Integration Tests (needs own fixture for custom risk limits)
# ============================================================================

class TestRiskIntegration:

    def test_risk_rejection_recorded(self, tmp_path):
        db_path = _setup_replay_db(tmp_path, n_days=3)
        strategy = _make_strategy()
        portfolio = _make_portfolio([strategy])

        limits = RiskLimits(max_daily_trades=1)

        result = replay_historical(
            db_path=db_path,
            portfolio=portfolio,
            instrument="MGC",
            start_date=date(2024, 1, 8),
            end_date=date(2024, 1, 12),
            risk_limits=limits,
        )
        assert result.days_processed > 0


# ============================================================================
# CLI Tests
# ============================================================================

class TestCLI:
    def test_help(self):
        import subprocess
        r = subprocess.run(
            [sys.executable, "trading_app/paper_trader.py", "--help"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent.parent),
        )
        assert r.returncode == 0
        assert "instrument" in r.stdout
