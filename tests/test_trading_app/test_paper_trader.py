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
    _print_header,
    _print_drawdown,
    _print_strategy_summary,
    _print_session_summary,
    _print_risk_rejections,
    _print_daily_equity,
    _export_csv,
)


def _cost():
    return get_cost_spec("MGC")


def _make_strategy(**overrides):
    base = {
        "strategy_id": "MGC_CME_REOPEN_E1_RR2.0_CB1_NO_FILTER",
        "instrument": "MGC",
        "orb_label": "CME_REOPEN",
        "entry_model": "E1",
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
        td_start = datetime(prev_day.year, prev_day.month, prev_day.day, 23, 0, tzinfo=timezone.utc)

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

            bars.append(
                (
                    ts.isoformat(),
                    "MGC",
                    "GCG4",
                    round(o, 2),
                    round(h, 2),
                    round(l, 2),
                    round(c, 2),
                    100,
                )
            )

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
                orb_CME_REOPEN_high, orb_CME_REOPEN_low, orb_CME_REOPEN_size,
                orb_CME_REOPEN_break_dir, orb_CME_REOPEN_break_ts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?::TIMESTAMPTZ)""",
            [
                trading_day,
                "MGC",
                5,
                len(bars),
                orb_high,
                orb_low,
                round(orb_high - orb_low, 2),
                "long",
                break_ts,
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
        assert _orb_from_strategy("MGC_US_DATA_830_E1_RR2.0_CB5_NO_FILTER") == "US_DATA_830"
        assert _orb_from_strategy("MGC_CME_REOPEN_E2_RR2.0_CB1_ORB_G5") == "CME_REOPEN"
        assert _orb_from_strategy("MGC_TOKYO_OPEN_E1_RR2.5_CB2_NO_FILTER") == "TOKYO_OPEN"

    def test_entry_model_from_strategy(self):
        assert _entry_model_from_strategy("MGC_US_DATA_830_E1_RR2.0_CB5_NO_FILTER") == "E1"
        assert _entry_model_from_strategy("MGC_CME_REOPEN_E2_RR2.0_CB1_ORB_G5") == "E2"


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
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent.parent),
        )
        assert r.returncode == 0
        assert "instrument" in r.stdout

    def test_calendar_filter_cli_arg_removed(self, tmp_path):
        """--calendar-filter CLI arg was removed; calendar overlay is now automatic."""
        import os
        import subprocess

        # Run paper_trader with the old --calendar-filter flag — should be rejected
        result = subprocess.run(
            [
                sys.executable,
                "trading_app/paper_trader.py",
                "--instrument",
                "MGC",
                "--start",
                "2024-01-08",
                "--end",
                "2024-01-09",
                "--calendar-filter",
                "NFP",
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent.parent),
            env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent.parent)},
        )
        # The flag no longer exists — argparse should reject it
        assert "unrecognized arguments" in result.stderr

    def test_calendar_overlay_importable(self):
        """get_calendar_action is importable and returns NEUTRAL with empty rules."""
        from datetime import date

        from trading_app.calendar_overlay import CalendarAction, get_calendar_action

        result = get_calendar_action("MGC", "TOKYO_OPEN", date(2025, 1, 6))
        assert result == CalendarAction.NEUTRAL

    def test_cli_has_output_and_quiet_flags(self):
        """New --output and --quiet flags are recognized."""
        import subprocess

        r = subprocess.run(
            [sys.executable, "trading_app/paper_trader.py", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent.parent),
        )
        assert r.returncode == 0
        assert "--output" in r.stdout
        assert "--quiet" in r.stdout


# ============================================================================
# Output Helper Tests
# ============================================================================


def _make_test_result():
    """Build a ReplayResult with known data for output testing."""
    journal = [
        JournalEntry(
            mode="replay",
            trading_day=date(2024, 1, 8),
            strategy_id="MGC_TOKYO_OPEN_E1_RR2.5_CB4_ORB_G5",
            entry_model="E1",
            direction="long",
            entry_ts=datetime(2024, 1, 7, 23, 10, tzinfo=timezone.utc),
            entry_price=2705.0,
            stop_price=2700.0,
            target_price=2712.5,
            contracts=2,
            exit_ts=datetime(2024, 1, 7, 23, 45, tzinfo=timezone.utc),
            exit_price=2712.5,
            outcome="win",
            pnl_r=2.5,
        ),
        JournalEntry(
            mode="replay",
            trading_day=date(2024, 1, 9),
            strategy_id="MGC_CME_REOPEN_E2_RR1.0_CB1_ORB_G5",
            entry_model="E2",
            direction="short",
            entry_ts=datetime(2024, 1, 9, 14, 35, tzinfo=timezone.utc),
            entry_price=2710.0,
            stop_price=2715.0,
            target_price=2705.0,
            contracts=3,
            exit_ts=datetime(2024, 1, 9, 15, 10, tzinfo=timezone.utc),
            exit_price=2715.0,
            outcome="loss",
            pnl_r=-1.0,
        ),
        JournalEntry(
            mode="replay",
            trading_day=date(2024, 1, 9),
            strategy_id="MGC_TOKYO_OPEN_E1_RR2.5_CB4_ORB_G5",
            entry_model="E1",
            direction="long",
            risk_rejected=True,
            risk_reason="max_per_orb: 1 positions on TOKYO_OPEN",
        ),
    ]
    day_summaries = [
        DaySummary(trading_day=date(2024, 1, 8), bars_processed=150, trades_entered=1, wins=1, daily_pnl_r=2.5),
        DaySummary(
            trading_day=date(2024, 1, 9),
            bars_processed=150,
            trades_entered=1,
            losses=1,
            daily_pnl_r=-1.0,
            risk_rejections=1,
        ),
    ]
    return ReplayResult(
        start_date=date(2024, 1, 8),
        end_date=date(2024, 1, 9),
        days_processed=2,
        total_trades=2,
        total_wins=1,
        total_losses=1,
        total_pnl_r=1.5,
        total_risk_rejections=1,
        journal=journal,
        day_summaries=day_summaries,
    )


class TestOutputHelpers:
    def test_print_header(self, capsys):
        result = _make_test_result()
        _print_header(result, "MGC")
        output = capsys.readouterr().out
        assert "PAPER TRADER REPLAY: MGC" in output
        assert "Trades: 2" in output
        assert "Win Rate: 50.0%" in output
        assert "+1.50R" in output

    def test_print_drawdown(self, capsys):
        result = _make_test_result()
        _print_drawdown(result)
        output = capsys.readouterr().out
        assert "Max Drawdown" in output
        assert "High Water" in output

    def test_print_strategy_summary(self, capsys):
        result = _make_test_result()
        _print_strategy_summary(result)
        output = capsys.readouterr().out
        assert "STRATEGY BREAKDOWN" in output
        assert "TOKYO_OPEN" in output
        assert "CME_REOPEN" in output

    def test_print_session_summary(self, capsys):
        result = _make_test_result()
        _print_session_summary(result)
        output = capsys.readouterr().out
        assert "SESSION BREAKDOWN" in output
        assert "TOKYO_OPEN" in output
        assert "CME_REOPEN" in output

    def test_print_risk_rejections(self, capsys):
        result = _make_test_result()
        _print_risk_rejections(result)
        output = capsys.readouterr().out
        assert "RISK REJECTIONS: 1" in output
        assert "max_per_orb" in output

    def test_print_daily_equity(self, capsys):
        result = _make_test_result()
        _print_daily_equity(result, quiet=False)
        output = capsys.readouterr().out
        assert "DAILY EQUITY" in output
        assert "2024-01-08" in output
        assert "2024-01-09" in output

    def test_print_daily_equity_quiet(self, capsys):
        result = _make_test_result()
        _print_daily_equity(result, quiet=True)
        output = capsys.readouterr().out
        assert output == ""

    def test_export_csv(self, tmp_path, capsys):
        result = _make_test_result()
        csv_path = str(tmp_path / "test_journal.csv")
        _export_csv(result, csv_path)
        output = capsys.readouterr().out
        assert "Journal exported" in output
        assert "3 rows" in output

        import csv

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3
        assert rows[0]["strategy_id"] == "MGC_TOKYO_OPEN_E1_RR2.5_CB4_ORB_G5"
        assert rows[0]["outcome"] == "win"
        assert rows[0]["pnl_r"] == "2.5"
        assert rows[2]["risk_rejected"] == "True"

    def test_drawdown_computation(self, capsys):
        """Drawdown is correctly computed from daily summaries."""
        result = ReplayResult(
            start_date=date(2024, 1, 8),
            end_date=date(2024, 1, 12),
            days_processed=5,
            day_summaries=[
                DaySummary(trading_day=date(2024, 1, 8), daily_pnl_r=3.0),
                DaySummary(trading_day=date(2024, 1, 9), daily_pnl_r=-5.0),
                DaySummary(trading_day=date(2024, 1, 10), daily_pnl_r=1.0),
                DaySummary(trading_day=date(2024, 1, 11), daily_pnl_r=-2.0),
                DaySummary(trading_day=date(2024, 1, 12), daily_pnl_r=4.0),
            ],
        )
        # Cumulative: 3, -2, -1, -3, 1
        # High water: 3, 3, 3, 3, 3
        # Drawdown: 0, -5, -4, -6, -2
        # Max drawdown = -6.0 on Jan 11
        _print_drawdown(result)
        output = capsys.readouterr().out
        assert "-6.00R" in output
        assert "2024-01-11" in output
        assert "High Water" in output
        assert "+3.00R" in output

    def test_drawdown_no_drawdown(self, capsys):
        """When equity only goes up, drawdown shows 'none'."""
        result = ReplayResult(
            start_date=date(2024, 1, 8),
            end_date=date(2024, 1, 9),
            days_processed=2,
            day_summaries=[
                DaySummary(trading_day=date(2024, 1, 8), daily_pnl_r=2.0),
                DaySummary(trading_day=date(2024, 1, 9), daily_pnl_r=1.0),
            ],
        )
        _print_drawdown(result)
        output = capsys.readouterr().out
        assert "0.00R (none)" in output
