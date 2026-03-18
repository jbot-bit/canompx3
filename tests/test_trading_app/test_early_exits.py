"""
Tests for timed early exit logic (kill losers).

Research: CME_REOPEN at 38m, TOKYO_OPEN at 39m, SINGAPORE_OPEN at 31m,
LONDON_METALS at 36m, CME_PRECLOSE at 16m. Other sessions: no early exit.
Rule: At N minutes after fill, if bar close vs entry is negative, exit at bar close.
"""

import sys
from pathlib import Path
from datetime import date, datetime, timezone, timedelta

import pytest
import numpy as np
import pandas as pd

from trading_app.config import EARLY_EXIT_MINUTES
from trading_app.outcome_builder import compute_single_outcome
from trading_app.execution_engine import (
    ExecutionEngine,
    ActiveTrade,
    TradeState,
)
from trading_app.portfolio import Portfolio, PortfolioStrategy
from pipeline.cost_model import get_cost_spec

# ============================================================================
# Helpers
# ============================================================================


def _cost():
    return get_cost_spec("MGC")


def _make_bars(closes, start_ts, freq_seconds=60):
    """Build bars DataFrame with realistic OHLCV from close prices."""
    timestamps = [pd.Timestamp(start_ts, tz="UTC") + pd.Timedelta(seconds=i * freq_seconds) for i in range(len(closes))]
    return pd.DataFrame(
        {
            "ts_utc": timestamps,
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [100] * len(closes),
        }
    )


def _make_bars_ohlc(ohlc_rows, start_ts, freq_seconds=60):
    """Build bars from explicit (open, high, low, close) tuples."""
    base = pd.Timestamp(start_ts) if start_ts.tzinfo else pd.Timestamp(start_ts, tz="UTC")
    timestamps = [base + pd.Timedelta(seconds=i * freq_seconds) for i in range(len(ohlc_rows))]
    return pd.DataFrame(
        {
            "ts_utc": timestamps,
            "open": [r[0] for r in ohlc_rows],
            "high": [r[1] for r in ohlc_rows],
            "low": [r[2] for r in ohlc_rows],
            "close": [r[3] for r in ohlc_rows],
            "volume": [100] * len(ohlc_rows),
        }
    )


ORB_HIGH = 2350.0
ORB_LOW = 2340.0
BREAK_TS = datetime(2024, 1, 15, 23, 5, tzinfo=timezone.utc)  # CME_REOPEN session
DAY_END = datetime(2024, 1, 16, 23, 0, tzinfo=timezone.utc)

# ============================================================================
# Config Tests
# ============================================================================


class TestEarlyExitConfig:
    def test_all_t80_disabled(self):
        """T80 disabled 2026-03-18: OOS validation NO-GO. All values must be None."""
        for session, value in EARLY_EXIT_MINUTES.items():
            assert value is None, f"EARLY_EXIT_MINUTES['{session}'] = {value}, expected None (T80 disabled)"

    def test_all_active_sessions_present(self):
        expected = {
            "CME_REOPEN",
            "TOKYO_OPEN",
            "SINGAPORE_OPEN",
            "LONDON_METALS",
            "EUROPE_FLOW",
            "US_DATA_830",
            "NYSE_OPEN",
            "US_DATA_1000",
            "COMEX_SETTLE",
            "CME_PRECLOSE",
            "NYSE_CLOSE",
            "BRISBANE_1025",
        }
        assert set(EARLY_EXIT_MINUTES.keys()) == expected


# ============================================================================
# outcome_builder Tests
# ============================================================================


class TestOutcomeBuilderEarlyExit:
    def test_cme_reopen_losing_at_15min_no_early_exit(self):
        """CME_REOPEN long trade, losing at 15 min -> resolves normally (no early_exit)."""
        bars = []
        bars.append((2351.0, 2351.5, 2350.5, 2351.0))
        bars.append((2351.0, 2351.5, 2350.5, 2351.0))
        for i in range(21):
            price = 2351.0 - i * 0.3
            bars.append((price + 0.2, price + 0.5, price - 0.5, price))

        bars_df = _make_bars_ohlc(bars, BREAK_TS)
        cost = _cost()

        result = compute_single_outcome(
            bars_df=bars_df,
            break_ts=BREAK_TS,
            orb_high=ORB_HIGH,
            orb_low=ORB_LOW,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=DAY_END,
            cost_spec=cost,
            entry_model="E1",
            orb_label="CME_REOPEN",
        )

        assert result["outcome"] != "early_exit"
        assert result["outcome"] in ("scratch", "loss", "win")

    def test_cme_reopen_winning_at_15min_no_early_exit(self):
        """CME_REOPEN long trade, winning at 15 min -> normal outcome, NOT early_exit."""
        bars = []
        bars.append((2351.0, 2351.5, 2350.5, 2351.0))
        bars.append((2351.0, 2351.5, 2350.5, 2351.0))
        for i in range(21):
            price = 2351.0 + i * 0.3
            bars.append((price - 0.2, price + 0.5, price - 0.5, price))

        bars_df = _make_bars_ohlc(bars, BREAK_TS)
        cost = _cost()

        result = compute_single_outcome(
            bars_df=bars_df,
            break_ts=BREAK_TS,
            orb_high=ORB_HIGH,
            orb_low=ORB_LOW,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=DAY_END,
            cost_spec=cost,
            entry_model="E1",
            orb_label="CME_REOPEN",
        )

        assert result["outcome"] != "early_exit"

    def test_tokyo_open_losing_at_30min_no_early_exit(self):
        """TOKYO_OPEN long trade, losing at 30 min -> resolves normally (no early_exit)."""
        break_ts_tokyo = datetime(2024, 1, 16, 0, 5, tzinfo=timezone.utc)
        bars = []
        bars.append((2351.0, 2351.5, 2350.5, 2351.0))
        bars.append((2351.0, 2351.5, 2350.5, 2351.0))
        for i in range(36):
            price = 2351.0 - i * 0.15
            bars.append((price + 0.1, price + 0.3, price - 0.3, price))

        bars_df = _make_bars_ohlc(bars, break_ts_tokyo)
        cost = _cost()

        result = compute_single_outcome(
            bars_df=bars_df,
            break_ts=break_ts_tokyo,
            orb_high=ORB_HIGH,
            orb_low=ORB_LOW,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=DAY_END,
            cost_spec=cost,
            entry_model="E1",
            orb_label="TOKYO_OPEN",
        )

        assert result["outcome"] != "early_exit"
        assert result["outcome"] in ("scratch", "loss", "win")

    def test_london_metals_no_early_exit(self):
        """LONDON_METALS session: outcome_builder does not apply early exit."""
        break_ts_london = datetime(2024, 1, 16, 8, 5, tzinfo=timezone.utc)
        bars = []
        bars.append((2351.0, 2351.5, 2350.5, 2351.0))
        bars.append((2351.0, 2351.5, 2350.5, 2351.0))
        for i in range(21):
            price = 2351.0 - i * 0.3
            bars.append((price + 0.2, price + 0.5, price - 0.5, price))

        bars_df = _make_bars_ohlc(bars, break_ts_london)
        cost = _cost()

        result = compute_single_outcome(
            bars_df=bars_df,
            break_ts=break_ts_london,
            orb_high=ORB_HIGH,
            orb_low=ORB_LOW,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=DAY_END,
            cost_spec=cost,
            entry_model="E1",
            orb_label="LONDON_METALS",
        )

        assert result["outcome"] != "early_exit"

    def test_stop_hit_before_threshold_normal_loss(self):
        """If stop is hit before the T80 threshold, normal loss NOT early_exit."""
        bars = []
        bars.append((2351.0, 2351.5, 2350.5, 2351.0))
        bars.append((2351.0, 2351.5, 2350.5, 2351.0))
        bars.append((2351.0, 2351.5, 2350.5, 2351.0))
        bars.append((2345.0, 2345.5, 2339.0, 2339.5))  # low 2339 < stop 2340

        bars_df = _make_bars_ohlc(bars, BREAK_TS)
        cost = _cost()

        result = compute_single_outcome(
            bars_df=bars_df,
            break_ts=BREAK_TS,
            orb_high=ORB_HIGH,
            orb_low=ORB_LOW,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=DAY_END,
            cost_spec=cost,
            entry_model="E1",
            orb_label="CME_REOPEN",
        )

        assert result["outcome"] == "loss"
        assert result["pnl_r"] == -1.0

    def test_no_orb_label_no_early_exit(self):
        """If orb_label is None, no early exit logic runs."""
        bars = []
        bars.append((2351.0, 2351.5, 2350.5, 2351.0))
        bars.append((2351.0, 2351.5, 2350.5, 2351.0))
        for i in range(21):
            price = 2351.0 - i * 0.3
            bars.append((price + 0.2, price + 0.5, price - 0.5, price))

        bars_df = _make_bars_ohlc(bars, BREAK_TS)
        cost = _cost()

        result = compute_single_outcome(
            bars_df=bars_df,
            break_ts=BREAK_TS,
            orb_high=ORB_HIGH,
            orb_low=ORB_LOW,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=DAY_END,
            cost_spec=cost,
            entry_model="E1",
            orb_label=None,
        )

        assert result["outcome"] != "early_exit"

    def test_outcome_builder_never_produces_early_exit(self):
        """Outcome builder should never produce early_exit outcomes."""
        bars = []
        bars.append((2351.0, 2351.5, 2350.5, 2351.0))
        bars.append((2351.0, 2351.5, 2350.5, 2351.0))
        for i in range(21):
            price = 2351.0 - i * 0.3
            bars.append((price + 0.2, price + 0.5, price - 0.5, price))

        bars_df = _make_bars_ohlc(bars, BREAK_TS)
        cost = _cost()

        result = compute_single_outcome(
            bars_df=bars_df,
            break_ts=BREAK_TS,
            orb_high=ORB_HIGH,
            orb_low=ORB_LOW,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=DAY_END,
            cost_spec=cost,
            entry_model="E1",
            orb_label="CME_REOPEN",
        )

        assert result["outcome"] != "early_exit"
        assert result["mae_r"] is not None
        assert result["mfe_r"] is not None
        assert result["mae_r"] >= 0
        assert result["mfe_r"] >= 0


# ============================================================================
# execution_engine Tests
# ============================================================================


def _make_strategy(**overrides):
    base = dict(
        strategy_id="MGC_CME_REOPEN_E1_RR2.0_CB1_ORB_G4",
        instrument="MGC",
        orb_label="CME_REOPEN",
        entry_model="E1",
        rr_target=2.0,
        confirm_bars=1,
        filter_type="NO_FILTER",
        expectancy_r=0.30,
        win_rate=0.55,
        sample_size=300,
        sharpe_ratio=0.4,
        max_drawdown_r=5.0,
        median_risk_points=10.0,
    )
    base.update(overrides)
    return PortfolioStrategy(**base)


def _make_portfolio(strategies=None, **overrides):
    if strategies is None:
        strategies = [_make_strategy()]
    defaults = dict(
        name="test",
        instrument="MGC",
        strategies=strategies,
        account_equity=25000.0,
        risk_per_trade_pct=2.0,
        max_concurrent_positions=3,
        max_daily_loss_r=5.0,
    )
    defaults.update(overrides)
    return Portfolio(**defaults)


def _bar(ts, o, h, l, c, v=100):
    return {"ts_utc": ts, "open": float(o), "high": float(h), "low": float(l), "close": float(c), "volume": int(v)}


class TestExecutionEngineEarlyExit:
    def test_no_early_exit_when_t80_disabled(self):
        """T80 disabled: losing trade at 45m should NOT trigger early_exit."""
        strategy = _make_strategy(orb_label="CME_REOPEN")
        portfolio = _make_portfolio(strategies=[strategy])
        engine = ExecutionEngine(portfolio, _cost(), live_session_costs=False)
        engine.on_trading_day_start(date(2024, 1, 5))

        base = datetime(2024, 1, 4, 23, 0, tzinfo=timezone.utc)

        for i in range(5):
            ts = base + timedelta(minutes=i)
            engine.on_bar(_bar(ts, 2345, 2350, 2340, 2345))

        break_ts = base + timedelta(minutes=5)
        engine.on_bar(_bar(break_ts, 2350, 2355, 2349, 2352))

        fill_ts = base + timedelta(minutes=6)
        engine.on_bar(_bar(fill_ts, 2352, 2353, 2351, 2352))

        # Feed bars for 45 minutes, gradually losing — no early exit should fire
        early_exits = []
        for i in range(1, 45):
            ts = fill_ts + timedelta(minutes=i)
            price = 2352.0 - i * 0.1
            events = engine.on_bar(_bar(ts, price + 0.1, price + 0.3, price - 0.3, price))
            early_exits.extend(e for e in events if e.reason == "early_exit_timed")

        assert len(early_exits) == 0, "T80 is disabled — no early exit should fire"

    def test_no_early_exit_when_winning(self):
        """CME_REOPEN E1 trade winning at 15 min -> no early exit."""
        strategy = _make_strategy(orb_label="CME_REOPEN")
        portfolio = _make_portfolio(strategies=[strategy])
        engine = ExecutionEngine(portfolio, _cost(), live_session_costs=False)
        engine.on_trading_day_start(date(2024, 1, 5))

        base = datetime(2024, 1, 4, 23, 0, tzinfo=timezone.utc)

        # Build ORB
        for i in range(5):
            ts = base + timedelta(minutes=i)
            engine.on_bar(_bar(ts, 2345, 2350, 2340, 2345))

        # Break bar
        break_ts = base + timedelta(minutes=5)
        engine.on_bar(_bar(break_ts, 2350, 2355, 2349, 2352))

        # Fill bar
        fill_ts = base + timedelta(minutes=6)
        engine.on_bar(_bar(fill_ts, 2352, 2353, 2351, 2352))

        # Feed bars going up for 20 minutes
        early_exits = []
        for i in range(1, 20):
            ts = fill_ts + timedelta(minutes=i)
            price = 2352.0 + i * 0.2  # drifting up
            events = engine.on_bar(_bar(ts, price - 0.1, price + 0.3, price - 0.3, price))
            early_exits.extend(e for e in events if e.reason == "early_exit_timed")

        assert len(early_exits) == 0

    def test_no_early_exit_for_london_metals(self):
        """LONDON_METALS session has early exit threshold (36m) but engine test."""
        strategy = _make_strategy(
            strategy_id="MGC_LONDON_METALS_E1_RR2.0_CB1_NO_FILTER",
            orb_label="LONDON_METALS",
        )
        portfolio = _make_portfolio(strategies=[strategy])
        engine = ExecutionEngine(portfolio, _cost(), live_session_costs=False)
        engine.on_trading_day_start(date(2024, 1, 5))

        # LONDON_METALS = 18:00 Brisbane = 08:00 UTC (winter). ORB window 08:00-08:05
        base = datetime(2024, 1, 5, 8, 0, tzinfo=timezone.utc)

        for i in range(5):
            ts = base + timedelta(minutes=i)
            engine.on_bar(_bar(ts, 2345, 2350, 2340, 2345))

        break_ts = base + timedelta(minutes=5)
        engine.on_bar(_bar(break_ts, 2350, 2355, 2349, 2352))

        fill_ts = base + timedelta(minutes=6)
        engine.on_bar(_bar(fill_ts, 2352, 2353, 2351, 2352))

        # Losing for 20 bars — before T80=36m, no early exit should fire
        early_exits = []
        for i in range(1, 20):
            ts = fill_ts + timedelta(minutes=i)
            price = 2352.0 - i * 0.3
            events = engine.on_bar(_bar(ts, price + 0.1, price + 0.3, price - 0.3, price))
            early_exits.extend(e for e in events if e.reason == "early_exit_timed")

        assert len(early_exits) == 0

    def test_no_partial_loss_when_t80_disabled(self):
        """T80 disabled: losing trade drifts but no early exit — trade stays open."""
        strategy = _make_strategy(orb_label="CME_REOPEN")
        portfolio = _make_portfolio(strategies=[strategy])
        engine = ExecutionEngine(portfolio, _cost(), live_session_costs=False)
        engine.on_trading_day_start(date(2024, 1, 5))

        base = datetime(2024, 1, 4, 23, 0, tzinfo=timezone.utc)

        for i in range(5):
            ts = base + timedelta(minutes=i)
            engine.on_bar(_bar(ts, 2345, 2350, 2340, 2345))

        break_ts = base + timedelta(minutes=5)
        engine.on_bar(_bar(break_ts, 2350, 2355, 2349, 2352))

        fill_ts = base + timedelta(minutes=6)
        engine.on_bar(_bar(fill_ts, 2352, 2353, 2351, 2352))

        # Feed 45 bars drifting down slowly (stays above stop)
        for i in range(1, 45):
            ts = fill_ts + timedelta(minutes=i)
            price = 2352.0 - i * 0.1
            engine.on_bar(_bar(ts, price + 0.1, price + 0.3, price - 0.3, price))

        # Trade should still be open — no early exit, no stop hit
        exited = [t for t in engine.completed_trades if t.pnl_r is not None]
        assert len(exited) == 0, "T80 disabled — trade should remain open (no stop hit)"
