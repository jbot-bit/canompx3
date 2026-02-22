"""
Tests for timed early exit logic (kill losers).

Research: 0900 at 15 min, 1000 at 30 min. Other sessions: no early exit.
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
    timestamps = [
        pd.Timestamp(start_ts, tz="UTC") + pd.Timedelta(seconds=i * freq_seconds)
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

def _make_bars_ohlc(ohlc_rows, start_ts, freq_seconds=60):
    """Build bars from explicit (open, high, low, close) tuples."""
    base = pd.Timestamp(start_ts) if start_ts.tzinfo else pd.Timestamp(start_ts, tz="UTC")
    timestamps = [
        base + pd.Timedelta(seconds=i * freq_seconds)
        for i in range(len(ohlc_rows))
    ]
    return pd.DataFrame({
        "ts_utc": timestamps,
        "open": [r[0] for r in ohlc_rows],
        "high": [r[1] for r in ohlc_rows],
        "low": [r[2] for r in ohlc_rows],
        "close": [r[3] for r in ohlc_rows],
        "volume": [100] * len(ohlc_rows),
    })

ORB_HIGH = 2350.0
ORB_LOW = 2340.0
BREAK_TS = datetime(2024, 1, 15, 23, 5, tzinfo=timezone.utc)  # 0900 session
DAY_END = datetime(2024, 1, 16, 23, 0, tzinfo=timezone.utc)

# ============================================================================
# Config Tests
# ============================================================================

class TestEarlyExitConfig:

    def test_0900_is_15_minutes(self):
        assert EARLY_EXIT_MINUTES["0900"] == 15

    def test_1000_is_30_minutes(self):
        assert EARLY_EXIT_MINUTES["1000"] == 30

    def test_1800_is_30_minutes(self):
        """P5b: T80=36m, avg_r_after=-0.339R (worst dead-chop penalty)."""
        assert EARLY_EXIT_MINUTES["1800"] == 30

    def test_other_sessions_none(self):
        for label in ["2300", "0030"]:
            assert EARLY_EXIT_MINUTES[label] is None, f"{label} should be None"

    def test_1100_has_no_early_exit(self):
        """1100 has no early exit configured (no research basis yet)."""
        assert EARLY_EXIT_MINUTES["1100"] is None

    def test_1130_has_no_early_exit(self):
        """1130 has no early exit configured."""
        assert EARLY_EXIT_MINUTES["1130"] is None

    def test_cme_open_has_no_early_exit(self):
        """CME_OPEN has no early exit configured."""
        assert EARLY_EXIT_MINUTES["CME_OPEN"] is None

    def test_all_active_sessions_present(self):
        expected = {
            "0900", "1000", "1100", "1130", "1800", "2300", "0030",
            "CME_OPEN", "US_EQUITY_OPEN", "US_DATA_OPEN", "LONDON_OPEN",
            "US_POST_EQUITY", "CME_CLOSE",
        }
        assert set(EARLY_EXIT_MINUTES.keys()) == expected

# ============================================================================
# outcome_builder Tests
# ============================================================================

class TestOutcomeBuilderEarlyExit:

    def test_0900_losing_at_15min_no_early_exit(self):
        """0900 long trade, losing at 15 min -> resolves normally (no early_exit)."""
        # Break bar + 1 confirm bar, then entry on next bar
        # Bar 0 = break (23:05), Bar 1 = confirm (23:06), Bar 2 = entry (23:07)
        # Price drifts down but never hits stop (2340) or target (2371)
        bars = []
        # Break bar (23:05): close above ORB high
        bars.append((2351.0, 2351.5, 2350.5, 2351.0))
        # Confirm bar (23:06): close above ORB high
        bars.append((2351.0, 2351.5, 2350.5, 2351.0))
        # Entry bar (23:07) and 20 more bars, gradually drifting down
        for i in range(21):
            price = 2351.0 - i * 0.3  # slowly losing
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
            orb_label="0900",
        )

        # Without early_exit, trade resolves as scratch (price never hits stop or target)
        assert result["outcome"] != "early_exit"
        assert result["outcome"] in ("scratch", "loss", "win")

    def test_0900_winning_at_15min_no_early_exit(self):
        """0900 long trade, winning at 15 min -> normal outcome, NOT early_exit."""
        bars = []
        # Break bar
        bars.append((2351.0, 2351.5, 2350.5, 2351.0))
        # Confirm bar
        bars.append((2351.0, 2351.5, 2350.5, 2351.0))
        # Entry bar and 20 more bars, gradually going up
        for i in range(21):
            price = 2351.0 + i * 0.3  # slowly winning
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
            orb_label="0900",
        )

        assert result["outcome"] != "early_exit"

    def test_1000_losing_at_30min_no_early_exit(self):
        """1000 long trade, losing at 30 min -> resolves normally (no early_exit)."""
        break_ts_1000 = datetime(2024, 1, 16, 0, 5, tzinfo=timezone.utc)
        bars = []
        # Break bar
        bars.append((2351.0, 2351.5, 2350.5, 2351.0))
        # Confirm bar
        bars.append((2351.0, 2351.5, 2350.5, 2351.0))
        # Entry bar + 35 more bars drifting down (need >= 30 min post-entry)
        for i in range(36):
            price = 2351.0 - i * 0.15  # slowly losing
            bars.append((price + 0.1, price + 0.3, price - 0.3, price))

        bars_df = _make_bars_ohlc(bars, break_ts_1000)
        cost = _cost()

        result = compute_single_outcome(
            bars_df=bars_df,
            break_ts=break_ts_1000,
            orb_high=ORB_HIGH,
            orb_low=ORB_LOW,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=DAY_END,
            cost_spec=cost,
            entry_model="E1",
            orb_label="1000",
        )

        # Without early_exit, trade resolves as scratch (price never hits stop or target)
        assert result["outcome"] != "early_exit"
        assert result["outcome"] in ("scratch", "loss", "win")

    def test_1800_no_early_exit(self):
        """1800 session has no early exit threshold."""
        break_ts_1800 = datetime(2024, 1, 16, 8, 5, tzinfo=timezone.utc)
        bars = []
        bars.append((2351.0, 2351.5, 2350.5, 2351.0))
        bars.append((2351.0, 2351.5, 2350.5, 2351.0))
        # 20 bars drifting down — should NOT trigger early exit
        for i in range(21):
            price = 2351.0 - i * 0.3
            bars.append((price + 0.2, price + 0.5, price - 0.5, price))

        bars_df = _make_bars_ohlc(bars, break_ts_1800)
        cost = _cost()

        result = compute_single_outcome(
            bars_df=bars_df,
            break_ts=break_ts_1800,
            orb_high=ORB_HIGH,
            orb_low=ORB_LOW,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=DAY_END,
            cost_spec=cost,
            entry_model="E1",
            orb_label="1800",
        )

        assert result["outcome"] != "early_exit"

    def test_stop_hit_before_threshold_normal_loss(self):
        """If stop is hit before the 15min threshold, normal loss NOT early_exit."""
        bars = []
        # Break bar
        bars.append((2351.0, 2351.5, 2350.5, 2351.0))
        # Confirm bar
        bars.append((2351.0, 2351.5, 2350.5, 2351.0))
        # Entry bar at 2351 open, stop at 2340 (ORB low)
        bars.append((2351.0, 2351.5, 2350.5, 2351.0))
        # Bar that crashes through stop (within 15 min)
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
            orb_label="0900",
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
            orb_label="0900",
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
        strategy_id="MGC_0900_E1_RR2.0_CB1_ORB_G4",
        instrument="MGC",
        orb_label="0900",
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
    return {"ts_utc": ts, "open": float(o), "high": float(h),
            "low": float(l), "close": float(c), "volume": int(v)}

class TestExecutionEngineEarlyExit:

    def test_early_exit_fires_at_15min_0900(self):
        """0900 E1 trade losing at 15 min -> early_exit event."""
        strategy = _make_strategy(orb_label="0900")
        portfolio = _make_portfolio(strategies=[strategy])
        engine = ExecutionEngine(portfolio, _cost(), live_session_costs=False)
        engine.on_trading_day_start(date(2024, 1, 5))

        # 0900 = 23:00 UTC prev day. ORB window 23:00-23:05
        base = datetime(2024, 1, 4, 23, 0, tzinfo=timezone.utc)

        # Build ORB: 5 bars within window
        for i in range(5):
            ts = base + timedelta(minutes=i)
            engine.on_bar(_bar(ts, 2345, 2350, 2340, 2345))

        # Break bar at 23:05 — close above ORB high
        break_ts = base + timedelta(minutes=5)
        events = engine.on_bar(_bar(break_ts, 2350, 2355, 2349, 2352))

        # E1 ARMED: next bar fills
        fill_ts = base + timedelta(minutes=6)
        events = engine.on_bar(_bar(fill_ts, 2352, 2353, 2351, 2352))
        assert any(e.event_type == "ENTRY" for e in events)

        # Feed bars for 15 minutes, gradually losing
        for i in range(1, 20):
            ts = fill_ts + timedelta(minutes=i)
            price = 2352.0 - i * 0.3  # drifting down
            events = engine.on_bar(_bar(ts, price + 0.1, price + 0.3, price - 0.3, price))
            if any(e.event_type == "EXIT" and e.reason == "early_exit_timed" for e in events):
                break
        else:
            pytest.fail("Early exit event not fired within 20 bars")

        exit_event = [e for e in events if e.reason == "early_exit_timed"][0]
        assert exit_event.event_type == "EXIT"

    def test_no_early_exit_when_winning(self):
        """0900 E1 trade winning at 15 min -> no early exit."""
        strategy = _make_strategy(orb_label="0900")
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

    def test_no_early_exit_for_1800(self):
        """1800 session has no early exit threshold."""
        strategy = _make_strategy(
            strategy_id="MGC_1800_E1_RR2.0_CB1_NO_FILTER",
            orb_label="1800",
        )
        portfolio = _make_portfolio(strategies=[strategy])
        engine = ExecutionEngine(portfolio, _cost(), live_session_costs=False)
        engine.on_trading_day_start(date(2024, 1, 5))

        # 1800 = 08:00 UTC. ORB window 08:00-08:05
        base = datetime(2024, 1, 5, 8, 0, tzinfo=timezone.utc)

        for i in range(5):
            ts = base + timedelta(minutes=i)
            engine.on_bar(_bar(ts, 2345, 2350, 2340, 2345))

        break_ts = base + timedelta(minutes=5)
        engine.on_bar(_bar(break_ts, 2350, 2355, 2349, 2352))

        fill_ts = base + timedelta(minutes=6)
        engine.on_bar(_bar(fill_ts, 2352, 2353, 2351, 2352))

        # Losing for 20 bars — no early exit should fire
        early_exits = []
        for i in range(1, 20):
            ts = fill_ts + timedelta(minutes=i)
            price = 2352.0 - i * 0.3
            events = engine.on_bar(_bar(ts, price + 0.1, price + 0.3, price - 0.3, price))
            early_exits.extend(e for e in events if e.reason == "early_exit_timed")

        assert len(early_exits) == 0

    def test_early_exit_pnl_is_partial_loss(self):
        """Early exit PnL should be between -1R and 0."""
        strategy = _make_strategy(orb_label="0900")
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

        for i in range(1, 20):
            ts = fill_ts + timedelta(minutes=i)
            price = 2352.0 - i * 0.3
            engine.on_bar(_bar(ts, price + 0.1, price + 0.3, price - 0.3, price))

        # Check PnL on the completed trade
        exited = [t for t in engine.completed_trades if t.pnl_r is not None]
        assert len(exited) == 1
        assert -1.0 < exited[0].pnl_r < 0
