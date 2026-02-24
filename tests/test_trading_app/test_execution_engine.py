"""
Tests for trading_app.execution_engine module.
"""

import sys
from pathlib import Path
from datetime import date, datetime, timezone, timedelta

import pytest

from trading_app.execution_engine import (
    ExecutionEngine,
    LiveORB,
    ActiveTrade,
    TradeEvent,
    TradeState,
    ORB_WINDOWS_UTC,
)
from trading_app.portfolio import Portfolio, PortfolioStrategy
from pipeline.cost_model import get_cost_spec

def _cost():
    return get_cost_spec("MGC")

def _make_strategy(**overrides):
    base = dict(
        strategy_id="MGC_US_DATA_830_E1_RR2.0_CB1_NO_FILTER",
        instrument="MGC",
        orb_label="US_DATA_830",
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

# ============================================================================
# ORB Detection Tests
# ============================================================================

class TestORBDetection:

    def test_orb_range_built_from_bars(self):
        """ORB high/low computed from bars within window."""
        engine = ExecutionEngine(_make_portfolio(), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))

        # US_DATA_830 ORB window: 13:30-13:35 UTC (winter) on trading day
        ts_base = datetime(2024, 1, 5, 13, 30, tzinfo=timezone.utc)
        engine.on_bar(_bar(ts_base, 2700, 2705, 2695, 2702))
        engine.on_bar(_bar(ts_base + timedelta(minutes=1), 2702, 2710, 2698, 2708))

        orb = engine.orbs["US_DATA_830"]
        assert orb.high == 2710.0
        assert orb.low == 2695.0
        assert not orb.complete  # Still within window

    def test_orb_completes_after_window(self):
        """ORB marked complete when bar timestamp >= window end."""
        engine = ExecutionEngine(_make_portfolio(), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))

        ts_base = datetime(2024, 1, 5, 13, 30, tzinfo=timezone.utc)
        for i in range(5):
            engine.on_bar(_bar(ts_base + timedelta(minutes=i), 2700, 2705, 2695, 2702))

        # Bar at 13:35 — window is over
        engine.on_bar(_bar(ts_base + timedelta(minutes=5), 2702, 2703, 2701, 2703))
        assert engine.orbs["US_DATA_830"].complete

    def test_break_detection_long(self):
        """Close above ORB high triggers long break."""
        engine = ExecutionEngine(_make_portfolio(), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))

        # Build ORB
        ts_base = datetime(2024, 1, 5, 13, 30, tzinfo=timezone.utc)
        for i in range(5):
            engine.on_bar(_bar(ts_base + timedelta(minutes=i), 2700, 2705, 2695, 2702))

        # Bar after window with close > orb_high (2705)
        engine.on_bar(_bar(ts_base + timedelta(minutes=5), 2704, 2710, 2703, 2706))
        assert engine.orbs["US_DATA_830"].break_dir == "long"

    def test_no_break_inside_range(self):
        """Close within ORB range = no break."""
        engine = ExecutionEngine(_make_portfolio(), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))

        ts_base = datetime(2024, 1, 5, 13, 30, tzinfo=timezone.utc)
        for i in range(5):
            engine.on_bar(_bar(ts_base + timedelta(minutes=i), 2700, 2705, 2695, 2702))

        engine.on_bar(_bar(ts_base + timedelta(minutes=5), 2702, 2704, 2696, 2700))
        assert engine.orbs["US_DATA_830"].break_dir is None

# ============================================================================
# Entry Tests
# ============================================================================

class TestEntry:

    def _run_to_break(self, engine, orb_high=2705.0):
        """Helper: build ORB and trigger a long break."""
        ts_base = datetime(2024, 1, 5, 13, 30, tzinfo=timezone.utc)
        for i in range(5):
            engine.on_bar(_bar(ts_base + timedelta(minutes=i), 2700, 2705, 2695, 2702))
        # Break bar
        events = engine.on_bar(
            _bar(ts_base + timedelta(minutes=5), 2704, 2710, 2703, 2706)
        )
        return ts_base, events

    def test_e1_enters_next_bar(self):
        """E1 with CB1: enters at next bar's open."""
        strategy = _make_strategy(entry_model="E1", confirm_bars=1,
                                  strategy_id="MGC_US_DATA_830_E1_RR2.0_CB1_NO_FILTER")
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))

        ts_base, events = self._run_to_break(engine)
        # E1: should be ARMED, not yet entered
        assert len([e for e in events if e.event_type == "ENTRY"]) == 0

        # Next bar
        next_events = engine.on_bar(
            _bar(ts_base + timedelta(minutes=6), 2708, 2715, 2707, 2712)
        )
        entry_events = [e for e in next_events if e.event_type == "ENTRY"]
        assert len(entry_events) == 1
        assert entry_events[0].price == 2708.0  # Next bar open

    def test_e3_enters_on_retrace(self):
        """E3: enters at ORB level when price retraces."""
        strategy = _make_strategy(entry_model="E3", confirm_bars=1,
                                  strategy_id="MGC_US_DATA_830_E3_RR2.0_CB1_NO_FILTER")
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))

        ts_base, events = self._run_to_break(engine)
        assert len([e for e in events if e.event_type == "ENTRY"]) == 0

        # Bar that retraces: low <= orb_high (2705)
        retrace_events = engine.on_bar(
            _bar(ts_base + timedelta(minutes=6), 2708, 2712, 2704, 2710)
        )
        entry_events = [e for e in retrace_events if e.event_type == "ENTRY"]
        assert len(entry_events) == 1
        assert entry_events[0].price == 2705.0  # ORB high

    def test_e3_no_retrace_no_fill(self):
        """E3: no entry if price doesn't retrace."""
        strategy = _make_strategy(entry_model="E3", confirm_bars=1,
                                  strategy_id="MGC_US_DATA_830_E3_RR2.0_CB1_NO_FILTER")
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))

        ts_base, events = self._run_to_break(engine)

        # Bar that doesn't retrace: low > orb_high (2705)
        no_retrace = engine.on_bar(
            _bar(ts_base + timedelta(minutes=6), 2708, 2715, 2706, 2712)
        )
        assert len([e for e in no_retrace if e.event_type == "ENTRY"]) == 0

# ============================================================================
# Exit Tests
# ============================================================================

class TestExit:

    def test_target_hit(self):
        """Trade exits at target price."""
        strategy = _make_strategy(entry_model="E1", confirm_bars=1, rr_target=2.0)
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))

        ts_base = datetime(2024, 1, 5, 13, 30, tzinfo=timezone.utc)
        # Build ORB: high=2705, low=2695
        for i in range(5):
            engine.on_bar(_bar(ts_base + timedelta(minutes=i), 2700, 2705, 2695, 2702))

        # Break bar: confirms + arms E1
        engine.on_bar(_bar(ts_base + timedelta(minutes=5), 2704, 2710, 2703, 2706))

        # E1 fill bar: entry at open=2708, stop=2695, risk=13, target=2708+26=2734
        engine.on_bar(_bar(ts_base + timedelta(minutes=6), 2708, 2712, 2707, 2710))

        # Target hit bar
        events = engine.on_bar(
            _bar(ts_base + timedelta(minutes=7), 2710, 2740, 2709, 2735)
        )
        exit_events = [e for e in events if e.event_type == "EXIT"]
        assert len(exit_events) == 1
        assert "win" in exit_events[0].reason

    def test_stop_hit(self):
        """Trade exits at stop price."""
        strategy = _make_strategy(entry_model="E1", confirm_bars=1, rr_target=2.0)
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))

        ts_base = datetime(2024, 1, 5, 13, 30, tzinfo=timezone.utc)
        for i in range(5):
            engine.on_bar(_bar(ts_base + timedelta(minutes=i), 2700, 2705, 2695, 2702))

        # Break bar
        engine.on_bar(_bar(ts_base + timedelta(minutes=5), 2704, 2710, 2703, 2706))

        # E1 fill bar: entry at open=2708, no stop/target on this bar
        engine.on_bar(_bar(ts_base + timedelta(minutes=6), 2708, 2712, 2707, 2710))

        # Stop hit: low <= 2695
        events = engine.on_bar(
            _bar(ts_base + timedelta(minutes=7), 2704, 2706, 2694, 2695)
        )
        exit_events = [e for e in events if e.event_type == "EXIT"]
        assert len(exit_events) == 1
        assert "loss" in exit_events[0].reason

    def test_ambiguous_bar_is_loss(self):
        """Bar hitting both target and stop = conservative loss."""
        strategy = _make_strategy(entry_model="E1", confirm_bars=1, rr_target=2.0)
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))

        ts_base = datetime(2024, 1, 5, 13, 30, tzinfo=timezone.utc)
        for i in range(5):
            engine.on_bar(_bar(ts_base + timedelta(minutes=i), 2700, 2705, 2695, 2702))
        engine.on_bar(_bar(ts_base + timedelta(minutes=5), 2704, 2710, 2703, 2706))

        # E1 fill bar: entry at open=2708, no exit on this bar
        engine.on_bar(_bar(ts_base + timedelta(minutes=6), 2708, 2712, 2707, 2710))

        # Huge bar: hits both target (2708+26=2734) and stop (2695)
        events = engine.on_bar(
            _bar(ts_base + timedelta(minutes=7), 2710, 2740, 2690, 2720)
        )
        exit_events = [e for e in events if e.event_type == "EXIT"]
        assert len(exit_events) == 1
        assert "loss" in exit_events[0].reason

    def test_session_end_scratch(self):
        """Open position at session end = scratch."""
        strategy = _make_strategy(entry_model="E1", confirm_bars=1)
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))

        ts_base = datetime(2024, 1, 5, 13, 30, tzinfo=timezone.utc)
        for i in range(5):
            engine.on_bar(_bar(ts_base + timedelta(minutes=i), 2700, 2705, 2695, 2702))
        engine.on_bar(_bar(ts_base + timedelta(minutes=5), 2704, 2710, 2703, 2706))
        # E1 fill bar
        engine.on_bar(_bar(ts_base + timedelta(minutes=6), 2708, 2712, 2707, 2710))

        # No target/stop hit — session ends
        events = engine.on_trading_day_end()
        scratch_events = [e for e in events if e.event_type == "SCRATCH"]
        assert len(scratch_events) == 1

# ============================================================================
# PnL Tests
# ============================================================================

class TestPnL:

    def test_win_pnl_uses_to_r_multiple(self):
        """Win PnL must use to_r_multiple (friction deducted from PnL)."""
        strategy = _make_strategy(entry_model="E1", confirm_bars=1, rr_target=2.0)
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))

        ts_base = datetime(2024, 1, 5, 13, 30, tzinfo=timezone.utc)
        for i in range(5):
            engine.on_bar(_bar(ts_base + timedelta(minutes=i), 2700, 2705, 2695, 2702))
        # Break bar
        engine.on_bar(_bar(ts_base + timedelta(minutes=5), 2704, 2710, 2703, 2706))
        # E1 fill bar: entry=2708, stop=2695, risk=13, target=2708+26=2734
        engine.on_bar(_bar(ts_base + timedelta(minutes=6), 2708, 2712, 2707, 2710))
        # Target hit
        engine.on_bar(_bar(ts_base + timedelta(minutes=7), 2710, 2740, 2709, 2735))

        wins = [t for t in engine.completed_trades if t.pnl_r is not None and t.pnl_r > 0]
        assert len(wins) == 1
        assert wins[0].pnl_r < 2.0  # Costs reduce wins below RR target

    def test_loss_pnl_is_minus_one(self):
        """Loss PnL = exactly -1.0R."""
        strategy = _make_strategy(entry_model="E1", confirm_bars=1, rr_target=2.0)
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))

        ts_base = datetime(2024, 1, 5, 13, 30, tzinfo=timezone.utc)
        for i in range(5):
            engine.on_bar(_bar(ts_base + timedelta(minutes=i), 2700, 2705, 2695, 2702))
        # Break bar
        engine.on_bar(_bar(ts_base + timedelta(minutes=5), 2704, 2710, 2703, 2706))
        # E1 fill bar: entry=2708, no exit on this bar
        engine.on_bar(_bar(ts_base + timedelta(minutes=6), 2708, 2712, 2707, 2710))
        # Stop hit
        engine.on_bar(_bar(ts_base + timedelta(minutes=7), 2704, 2706, 2694, 2695))

        losses = [t for t in engine.completed_trades if t.pnl_r is not None and t.pnl_r < 0]
        assert len(losses) == 1
        assert losses[0].pnl_r == -1.0

    def test_daily_pnl_accumulates(self):
        """Daily PnL tracks cumulative R."""
        strategy = _make_strategy(entry_model="E1", confirm_bars=1, rr_target=2.0)
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))

        ts_base = datetime(2024, 1, 5, 13, 30, tzinfo=timezone.utc)
        for i in range(5):
            engine.on_bar(_bar(ts_base + timedelta(minutes=i), 2700, 2705, 2695, 2702))
        engine.on_bar(_bar(ts_base + timedelta(minutes=5), 2704, 2710, 2703, 2706))
        # E1 fill bar
        engine.on_bar(_bar(ts_base + timedelta(minutes=6), 2708, 2712, 2707, 2710))
        # Stop hit
        engine.on_bar(_bar(ts_base + timedelta(minutes=7), 2704, 2706, 2694, 2695))

        summary = engine.get_daily_summary()
        assert summary["daily_pnl_r"] == -1.0

# ============================================================================
# Daily Summary Tests
# ============================================================================

class TestDailySummary:

    def test_summary_counts(self):
        strategy = _make_strategy(entry_model="E1", confirm_bars=1, rr_target=2.0)
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))

        summary = engine.get_daily_summary()
        assert summary["trading_day"] == date(2024, 1, 5)
        assert summary["trades_entered"] == 0
        assert summary["wins"] == 0
        assert summary["losses"] == 0

    def test_reset_on_new_day(self):
        engine = ExecutionEngine(_make_portfolio(), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))
        engine.daily_pnl_r = -3.0

        engine.on_trading_day_start(date(2024, 1, 6))
        assert engine.daily_pnl_r == 0.0
        assert engine.daily_trade_count == 0

# ============================================================================
# Filter Tests
# ============================================================================

class TestFilters:

    def test_orb_size_filter_rejects(self):
        """Strategy with G4 filter rejects ORBs < 4pt."""
        strategy = _make_strategy(
            filter_type="ORB_G4",
            strategy_id="MGC_US_DATA_830_E1_RR2.0_CB1_ORB_G4",
        )
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))

        # ORB with small range: high=2702, low=2700 = 2pt < 4pt threshold
        ts_base = datetime(2024, 1, 5, 13, 30, tzinfo=timezone.utc)
        for i in range(5):
            engine.on_bar(_bar(ts_base + timedelta(minutes=i), 2700, 2702, 2700, 2701))

        # Break attempt
        events = engine.on_bar(
            _bar(ts_base + timedelta(minutes=5), 2702, 2705, 2701, 2703)
        )
        entry_events = [e for e in events if e.event_type == "ENTRY"]
        assert len(entry_events) == 0  # Filtered out

# ============================================================================
# Confirm Count Direction Tests
# ============================================================================

class TestConfirmCountDirection:
    """Confirm count initialization must be directional.

    The break bar's confirm_count should only be 1 if the close is beyond
    the ORB edge in the break direction. This test locks in the invariant
    so that a non-directional check (close > high OR close < low) can never
    regress.

    Note: For CB≥2, the break bar is processed TWICE on the same on_bar()
    call — once in _arm_strategies (sets confirm_count=1) and again in
    _process_confirming (increments to 2). This is by design: the break bar
    IS a confirming bar. For CB1, the trade is immediately entered and
    never reaches _process_confirming.
    """

    def test_long_break_cb3_needs_two_more_bars(self):
        """Long break with CB3: break bar counts as 2, needs 1 more confirm bar."""
        strategy = _make_strategy(
            entry_model="E1", confirm_bars=3,
            strategy_id="MGC_US_DATA_830_E1_RR2.0_CB3_NO_FILTER",
        )
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))

        # Build ORB: high=2705, low=2695
        ts_base = datetime(2024, 1, 5, 13, 30, tzinfo=timezone.utc)
        for i in range(5):
            engine.on_bar(_bar(ts_base + timedelta(minutes=i), 2700, 2705, 2695, 2702))

        # Break bar: close=2706 > orb_high=2705 → long break
        # _arm_strategies sets confirm_count=1, _process_confirming increments to 2
        engine.on_bar(_bar(ts_base + timedelta(minutes=5), 2704, 2710, 2703, 2706))

        # With CB3, trade should still be CONFIRMING (2 < 3)
        assert len(engine.active_trades) == 1
        trade = engine.active_trades[0]
        assert trade.direction == "long"
        assert trade.confirm_count == 2

        # One more confirming bar should complete it (2+1=3 >= CB3)
        engine.on_bar(_bar(ts_base + timedelta(minutes=6), 2708, 2712, 2706, 2709))
        # Trade should now be ARMED (confirm complete)
        assert len([t for t in engine.active_trades if t.state == TradeState.CONFIRMING]) == 0

    def test_long_break_reversal_bar_resets_confirm(self):
        """If bar after break closes inside ORB, confirm_count resets to 0."""
        strategy = _make_strategy(
            entry_model="E1", confirm_bars=3,
            strategy_id="MGC_US_DATA_830_E1_RR2.0_CB3_NO_FILTER",
        )
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))

        # Build ORB: high=2705, low=2695
        ts_base = datetime(2024, 1, 5, 13, 30, tzinfo=timezone.utc)
        for i in range(5):
            engine.on_bar(_bar(ts_base + timedelta(minutes=i), 2700, 2705, 2695, 2702))

        # Break bar: close=2706 > orb_high=2705 → long, confirm_count=2
        engine.on_bar(_bar(ts_base + timedelta(minutes=5), 2704, 2710, 2703, 2706))

        # Next bar closes INSIDE ORB range → resets confirm_count to 0
        engine.on_bar(_bar(ts_base + timedelta(minutes=6), 2704, 2706, 2700, 2703))

        assert len(engine.active_trades) == 1
        trade = engine.active_trades[0]
        assert trade.confirm_count == 0, "Reversal bar should reset confirm_count"

    def test_short_break_confirm_count_directional(self):
        """Short break: confirm counted only when close < orb.low."""
        strategy = _make_strategy(
            entry_model="E1", confirm_bars=3,
            strategy_id="MGC_US_DATA_830_E1_RR2.0_CB3_NO_FILTER",
        )
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))

        # Build ORB: high=2705, low=2695
        ts_base = datetime(2024, 1, 5, 13, 30, tzinfo=timezone.utc)
        for i in range(5):
            engine.on_bar(_bar(ts_base + timedelta(minutes=i), 2700, 2705, 2695, 2702))

        # Break bar: close=2694 < orb_low=2695 → short break, confirm_count=2
        engine.on_bar(_bar(ts_base + timedelta(minutes=5), 2696, 2698, 2692, 2694))

        assert len(engine.active_trades) == 1
        trade = engine.active_trades[0]
        assert trade.direction == "short"
        assert trade.confirm_count == 2


# ============================================================================
# CLI Tests
# ============================================================================

class TestArmedAtBarGuard:
    """The armed_at_bar guard prevents look-ahead bias.

    When E1 transitions to ARMED on bar N, the trade must NOT fill on
    bar N (the confirm bar).  It must wait for bar N+1.  Without this
    guard, the confirm bar and the fill bar would be the same bar,
    which is look-ahead bias.
    """

    def _build_orb_and_break(self, engine, ts_base):
        """Build US_DATA_830 ORB (high=2705, low=2695) and trigger a long break.

        Returns (break_bar_ts, break_events).
        """
        # 5 bars inside the ORB window (13:30-13:34 UTC)
        for i in range(5):
            engine.on_bar(_bar(ts_base + timedelta(minutes=i), 2700, 2705, 2695, 2702))

        # Bar at 13:35 — outside the window, close > orb_high => long break
        break_ts = ts_base + timedelta(minutes=5)
        events = engine.on_bar(
            _bar(break_ts, 2704, 2710, 2703, 2706)
        )
        return break_ts, events

    def test_e1_armed_at_bar_no_fill_same_bar(self):
        """E1 ARMED on bar N must NOT produce an ENTRY on bar N."""
        strategy = _make_strategy(
            entry_model="E1",
            confirm_bars=1,
            strategy_id="MGC_US_DATA_830_E1_RR2.0_CB1_NO_FILTER",
        )
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))

        ts_base = datetime(2024, 1, 5, 13, 30, tzinfo=timezone.utc)
        break_ts, break_events = self._build_orb_and_break(engine, ts_base)

        # --- Bar N (break bar) assertions ---
        # 1) No ENTRY event on the break bar
        entry_events = [e for e in break_events if e.event_type == "ENTRY"]
        assert len(entry_events) == 0, "E1 must NOT fill on the break/confirm bar"

        # 2) Trade exists and is ARMED
        armed = [t for t in engine.active_trades if t.state == TradeState.ARMED]
        assert len(armed) == 1, "Exactly one trade should be ARMED"

        # 3) armed_at_bar matches current bar count (the guard value)
        trade = armed[0]
        assert trade.armed_at_bar == engine._bar_count, (
            "armed_at_bar must equal _bar_count on the bar where ARMED was set"
        )

    def test_e1_armed_fills_on_next_bar(self):
        """E1 ARMED on bar N fills on bar N+1 with that bar's open price."""
        strategy = _make_strategy(
            entry_model="E1",
            confirm_bars=1,
            strategy_id="MGC_US_DATA_830_E1_RR2.0_CB1_NO_FILTER",
        )
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))

        ts_base = datetime(2024, 1, 5, 13, 30, tzinfo=timezone.utc)
        _break_ts, break_events = self._build_orb_and_break(engine, ts_base)

        # Confirm ARMED, no ENTRY on bar N
        assert len([e for e in break_events if e.event_type == "ENTRY"]) == 0

        # --- Bar N+1 ---
        next_bar_ts = ts_base + timedelta(minutes=6)
        next_bar = _bar(next_bar_ts, 2708, 2715, 2707, 2712)
        next_events = engine.on_bar(next_bar)

        entry_events = [e for e in next_events if e.event_type == "ENTRY"]
        assert len(entry_events) == 1, "E1 must fill on the bar after ARMED"
        assert entry_events[0].price == 2708.0, "E1 fills at next bar's open"
        assert entry_events[0].timestamp == next_bar_ts

        # Trade should now be ENTERED
        entered = [t for t in engine.active_trades if t.state == TradeState.ENTERED]
        assert len(entered) == 1

class TestFillBarExitEngine:
    """Fill-bar exit must be checked for E1 and E3 (matches outcome_builder)."""

    def _build_orb(self, engine, ts_base):
        """Build ORB: high=2705, low=2695."""
        for i in range(5):
            engine.on_bar(_bar(ts_base + timedelta(minutes=i), 2700, 2705, 2695, 2702))

    def test_e1_fill_bar_target_hit(self):
        """E1: if fill bar hits target, trade should exit as win on same bar."""
        strategy = _make_strategy(
            entry_model="E1", confirm_bars=1, rr_target=1.0,
            strategy_id="MGC_US_DATA_830_E1_RR1.0_CB1_NO_FILTER",
        )
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))

        ts_base = datetime(2024, 1, 5, 13, 30, tzinfo=timezone.utc)
        self._build_orb(engine, ts_base)

        # Break bar (long): close > 2705
        engine.on_bar(_bar(ts_base + timedelta(minutes=5), 2704, 2710, 2703, 2706))
        # E1 now ARMED, will enter on next bar's open

        # Fill bar: open=2708, entry=2708, stop=2695, risk=13, target=2708+13=2721
        # bar high=2725 >= 2721 -> target hit on fill bar
        events = engine.on_bar(
            _bar(ts_base + timedelta(minutes=6), 2708, 2725, 2707, 2720)
        )
        entry_events = [e for e in events if e.event_type == "ENTRY"]
        exit_events = [e for e in events if e.event_type == "EXIT"]
        assert len(entry_events) == 1
        assert len(exit_events) == 1, "Fill bar should detect target hit"
        assert "win" in exit_events[0].reason

    def test_e1_fill_bar_stop_hit(self):
        """E1: if fill bar hits stop, trade should exit as loss on same bar."""
        strategy = _make_strategy(
            entry_model="E1", confirm_bars=1, rr_target=2.0,
            strategy_id="MGC_US_DATA_830_E1_RR2.0_CB1_NO_FILTER",
        )
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))

        ts_base = datetime(2024, 1, 5, 13, 30, tzinfo=timezone.utc)
        self._build_orb(engine, ts_base)

        # Break bar (long): close > 2705
        engine.on_bar(_bar(ts_base + timedelta(minutes=5), 2704, 2710, 2703, 2706))

        # Fill bar: open=2708, stop=2695, bar low=2694 <= 2695 -> stop hit
        events = engine.on_bar(
            _bar(ts_base + timedelta(minutes=6), 2708, 2710, 2694, 2696)
        )
        entry_events = [e for e in events if e.event_type == "ENTRY"]
        exit_events = [e for e in events if e.event_type == "EXIT"]
        assert len(entry_events) == 1
        assert len(exit_events) == 1, "Fill bar should detect stop hit"
        assert "loss" in exit_events[0].reason

    def test_e1_fill_bar_ambiguous_is_loss(self):
        """E1: if fill bar hits BOTH stop and target, resolve as loss."""
        strategy = _make_strategy(
            entry_model="E1", confirm_bars=1, rr_target=1.0,
            strategy_id="MGC_US_DATA_830_E1_RR1.0_CB1_NO_FILTER",
        )
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))

        ts_base = datetime(2024, 1, 5, 13, 30, tzinfo=timezone.utc)
        self._build_orb(engine, ts_base)

        # Break bar (long)
        engine.on_bar(_bar(ts_base + timedelta(minutes=5), 2704, 2710, 2703, 2706))

        # Fill bar: open=2708, stop=2695, target=2708+13=2721
        # high=2725 >= 2721 AND low=2694 <= 2695 -> both hit -> LOSS
        events = engine.on_bar(
            _bar(ts_base + timedelta(minutes=6), 2708, 2725, 2694, 2700)
        )
        exit_events = [e for e in events if e.event_type == "EXIT"]
        assert len(exit_events) == 1, "Ambiguous fill bar should exit as loss"
        assert "loss" in exit_events[0].reason

    def test_e3_fill_bar_target_hit(self):
        """E3: if retrace bar also hits target, trade exits as win."""
        strategy = _make_strategy(
            entry_model="E3", confirm_bars=1, rr_target=1.0,
            strategy_id="MGC_US_DATA_830_E3_RR1.0_CB1_NO_FILTER",
        )
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))

        ts_base = datetime(2024, 1, 5, 13, 30, tzinfo=timezone.utc)
        self._build_orb(engine, ts_base)

        # Break bar (long)
        engine.on_bar(_bar(ts_base + timedelta(minutes=5), 2704, 2710, 2703, 2706))

        # E3 retrace bar: low=2704 <= orb_high(2705) -> fills at 2705
        # entry=2705, stop=2695, risk=10, target=2705+10=2715
        # high=2716 >= 2715 -> target hit on fill bar
        events = engine.on_bar(
            _bar(ts_base + timedelta(minutes=6), 2708, 2716, 2704, 2712)
        )
        entry_events = [e for e in events if e.event_type == "ENTRY"]
        exit_events = [e for e in events if e.event_type == "EXIT"]
        assert len(entry_events) == 1
        assert len(exit_events) == 1, "E3 fill bar should detect target hit"
        assert "win" in exit_events[0].reason

class TestCLI:
    def test_import(self):
        """Module imports without error."""
        import trading_app.execution_engine
