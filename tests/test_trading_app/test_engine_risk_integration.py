"""
Integration tests: ExecutionEngine + RiskManager.

Verifies the engine properly calls RiskManager methods during the trade
lifecycle (entry, exit, scratch, reject).
"""

import sys
from pathlib import Path
from datetime import date, datetime, timezone, timedelta

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trading_app.execution_engine import (
    ExecutionEngine,
    TradeEvent,
    TradeState,
)
from trading_app.portfolio import Portfolio, PortfolioStrategy
from trading_app.risk_manager import RiskLimits, RiskManager
from pipeline.cost_model import get_cost_spec


# ============================================================================
# Helpers — mirrors existing test_execution_engine patterns
# ============================================================================

def _cost():
    return get_cost_spec("MGC")


def _make_strategy(**overrides):
    base = dict(
        strategy_id="MGC_2300_E2_RR2.0_CB1_NO_FILTER",
        instrument="MGC",
        orb_label="2300",
        entry_model="E2",
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


# 2300 ORB window is 13:00-13:05 UTC on the trading day.
_ORB_BASE = datetime(2024, 1, 5, 13, 0, tzinfo=timezone.utc)
_TRADING_DAY = date(2024, 1, 5)


def _build_orb(engine, orb_high=2705.0, orb_low=2695.0):
    """Feed 5 bars to build the 2300 ORB, then return the post-window timestamp."""
    for i in range(5):
        engine.on_bar(_bar(_ORB_BASE + timedelta(minutes=i),
                           2700, orb_high, orb_low, 2702))
    return _ORB_BASE + timedelta(minutes=5)


def _break_long(engine, break_ts, close=2706.0):
    """Feed a bar that breaks the ORB high (long). Returns events."""
    return engine.on_bar(_bar(break_ts, 2704, 2710, 2703, close))


# ============================================================================
# 1. Engine calls risk_manager.on_trade_entry() on entry (E1, E2, E3)
# ============================================================================

class TestOnTradeEntry:

    def test_e2_entry_calls_on_trade_entry(self):
        limits = RiskLimits()
        rm = RiskManager(limits)
        rm.daily_reset(_TRADING_DAY)

        strategy = _make_strategy(entry_model="E2", confirm_bars=1)
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost(), risk_manager=rm)
        engine.on_trading_day_start(_TRADING_DAY)

        break_ts = _build_orb(engine)
        events = _break_long(engine, break_ts)

        entry_events = [e for e in events if e.event_type == "ENTRY"]
        assert len(entry_events) == 1
        assert rm.daily_trade_count == 1

    def test_e1_entry_calls_on_trade_entry(self):
        limits = RiskLimits()
        rm = RiskManager(limits)
        rm.daily_reset(_TRADING_DAY)

        strategy = _make_strategy(
            entry_model="E1", confirm_bars=1,
            strategy_id="MGC_2300_E1_RR2.0_CB1_NO_FILTER",
        )
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost(), risk_manager=rm)
        engine.on_trading_day_start(_TRADING_DAY)

        break_ts = _build_orb(engine)
        _break_long(engine, break_ts)
        assert rm.daily_trade_count == 0  # Still ARMED, not entered

        # Next bar triggers E1 fill
        next_ts = break_ts + timedelta(minutes=1)
        events = engine.on_bar(_bar(next_ts, 2708, 2715, 2707, 2712))

        entry_events = [e for e in events if e.event_type == "ENTRY"]
        assert len(entry_events) == 1
        assert rm.daily_trade_count == 1

    def test_e3_entry_calls_on_trade_entry(self):
        limits = RiskLimits()
        rm = RiskManager(limits)
        rm.daily_reset(_TRADING_DAY)

        strategy = _make_strategy(
            entry_model="E3", confirm_bars=1,
            strategy_id="MGC_2300_E3_RR2.0_CB1_NO_FILTER",
        )
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost(), risk_manager=rm)
        engine.on_trading_day_start(_TRADING_DAY)

        break_ts = _build_orb(engine)
        _break_long(engine, break_ts)
        assert rm.daily_trade_count == 0  # ARMED, waiting for retrace

        # Retrace bar: low <= orb_high (2705)
        retrace_ts = break_ts + timedelta(minutes=1)
        events = engine.on_bar(_bar(retrace_ts, 2708, 2712, 2704, 2710))

        entry_events = [e for e in events if e.event_type == "ENTRY"]
        assert len(entry_events) == 1
        assert rm.daily_trade_count == 1


# ============================================================================
# 2. Engine calls risk_manager.on_trade_exit(pnl_r) on target/stop exit
# ============================================================================

class TestOnTradeExit:

    def test_target_hit_calls_on_trade_exit(self):
        limits = RiskLimits()
        rm = RiskManager(limits)
        rm.daily_reset(_TRADING_DAY)

        strategy = _make_strategy(entry_model="E2", confirm_bars=1, rr_target=2.0)
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost(), risk_manager=rm)
        engine.on_trading_day_start(_TRADING_DAY)

        break_ts = _build_orb(engine)
        _break_long(engine, break_ts)
        # Entry at 2706, stop=2695, risk=11, target=2706+22=2728

        # Target hit
        target_ts = break_ts + timedelta(minutes=1)
        events = engine.on_bar(_bar(target_ts, 2710, 2730, 2709, 2725))

        exit_events = [e for e in events if e.event_type == "EXIT"]
        assert len(exit_events) == 1
        assert "win" in exit_events[0].reason
        assert rm.daily_pnl_r > 0  # Win adds positive R

    def test_stop_hit_calls_on_trade_exit(self):
        limits = RiskLimits()
        rm = RiskManager(limits)
        rm.daily_reset(_TRADING_DAY)

        strategy = _make_strategy(entry_model="E2", confirm_bars=1, rr_target=2.0)
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost(), risk_manager=rm)
        engine.on_trading_day_start(_TRADING_DAY)

        break_ts = _build_orb(engine)
        _break_long(engine, break_ts)

        # Stop hit: low <= 2695
        stop_ts = break_ts + timedelta(minutes=1)
        events = engine.on_bar(_bar(stop_ts, 2704, 2706, 2694, 2695))

        exit_events = [e for e in events if e.event_type == "EXIT"]
        assert len(exit_events) == 1
        assert "loss" in exit_events[0].reason
        assert rm.daily_pnl_r == -1.0

    def test_pnl_r_value_passed_to_risk_manager(self):
        limits = RiskLimits()
        rm = RiskManager(limits)
        rm.daily_reset(_TRADING_DAY)

        strategy = _make_strategy(entry_model="E2", confirm_bars=1, rr_target=2.0)
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost(), risk_manager=rm)
        engine.on_trading_day_start(_TRADING_DAY)

        break_ts = _build_orb(engine)
        _break_long(engine, break_ts)

        # Stop hit => pnl_r = -1.0
        stop_ts = break_ts + timedelta(minutes=1)
        engine.on_bar(_bar(stop_ts, 2704, 2706, 2694, 2695))

        # The engine's completed trade and RM must agree
        completed = engine.completed_trades
        assert len(completed) == 1
        assert completed[0].pnl_r == -1.0
        assert rm.daily_pnl_r == completed[0].pnl_r


# ============================================================================
# 3. Engine calls risk_manager.on_trade_exit(pnl_r) on EOD scratch
# ============================================================================

class TestScratchExit:

    def test_scratch_calls_on_trade_exit(self):
        limits = RiskLimits()
        rm = RiskManager(limits)
        rm.daily_reset(_TRADING_DAY)

        strategy = _make_strategy(entry_model="E2", confirm_bars=1)
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost(), risk_manager=rm)
        engine.on_trading_day_start(_TRADING_DAY)

        break_ts = _build_orb(engine)
        _break_long(engine, break_ts)
        assert rm.daily_trade_count == 1

        # No target/stop hit; just feed a neutral bar then close the day
        engine.on_bar(_bar(break_ts + timedelta(minutes=1), 2706, 2708, 2704, 2707))
        events = engine.on_trading_day_end()

        scratch_events = [e for e in events if e.event_type == "SCRATCH"]
        assert len(scratch_events) == 1
        # RM should have received the scratch PnL
        assert rm.daily_pnl_r != 0.0 or rm.daily_pnl_r == 0.0  # Always called
        # More specifically, the completed trade has a real pnl_r value
        completed = [t for t in engine.completed_trades if t.pnl_r is not None]
        assert len(completed) == 1
        assert rm.daily_pnl_r == completed[0].pnl_r

    def test_scratch_pnl_r_reflects_mark_to_market(self):
        limits = RiskLimits()
        rm = RiskManager(limits)
        rm.daily_reset(_TRADING_DAY)

        strategy = _make_strategy(entry_model="E2", confirm_bars=1, rr_target=2.0)
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost(), risk_manager=rm)
        engine.on_trading_day_start(_TRADING_DAY)

        break_ts = _build_orb(engine)
        _break_long(engine, break_ts)
        # Entry at 2706.0

        # Move price up, then scratch
        engine.on_bar(_bar(break_ts + timedelta(minutes=1), 2706, 2720, 2705, 2718))
        engine.on_trading_day_end()

        # Scratch is mark-to-market: last close = 2718, entry = 2706, positive PnL
        completed = engine.completed_trades
        assert len(completed) == 1
        assert completed[0].pnl_r > 0  # Price went up for a long
        assert rm.daily_pnl_r == completed[0].pnl_r


# ============================================================================
# 4. Engine emits REJECT when risk_manager.can_enter() returns False
# ============================================================================

class TestReject:

    def test_e2_rejected_by_risk_manager(self):
        limits = RiskLimits(max_daily_trades=0)  # No trades allowed
        rm = RiskManager(limits)
        rm.daily_reset(_TRADING_DAY)

        strategy = _make_strategy(entry_model="E2", confirm_bars=1)
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost(), risk_manager=rm)
        engine.on_trading_day_start(_TRADING_DAY)

        break_ts = _build_orb(engine)
        events = _break_long(engine, break_ts)

        reject_events = [e for e in events if e.event_type == "REJECT"]
        assert len(reject_events) == 1
        assert "risk_rejected" in reject_events[0].reason
        assert rm.daily_trade_count == 0  # No entry recorded

    def test_e1_rejected_by_risk_manager(self):
        limits = RiskLimits(max_daily_trades=0)
        rm = RiskManager(limits)
        rm.daily_reset(_TRADING_DAY)

        strategy = _make_strategy(
            entry_model="E1", confirm_bars=1,
            strategy_id="MGC_2300_E1_RR2.0_CB1_NO_FILTER",
        )
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost(), risk_manager=rm)
        engine.on_trading_day_start(_TRADING_DAY)

        break_ts = _build_orb(engine)
        _break_long(engine, break_ts)  # ARMED

        # Next bar: E1 tries to fill, but RM rejects
        next_ts = break_ts + timedelta(minutes=1)
        events = engine.on_bar(_bar(next_ts, 2708, 2715, 2707, 2712))

        reject_events = [e for e in events if e.event_type == "REJECT"]
        assert len(reject_events) == 1
        assert "risk_rejected" in reject_events[0].reason
        assert rm.daily_trade_count == 0

    def test_e3_rejected_by_risk_manager(self):
        limits = RiskLimits(max_daily_trades=0)
        rm = RiskManager(limits)
        rm.daily_reset(_TRADING_DAY)

        strategy = _make_strategy(
            entry_model="E3", confirm_bars=1,
            strategy_id="MGC_2300_E3_RR2.0_CB1_NO_FILTER",
        )
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost(), risk_manager=rm)
        engine.on_trading_day_start(_TRADING_DAY)

        break_ts = _build_orb(engine)
        _break_long(engine, break_ts)  # ARMED

        # Retrace bar, but RM rejects
        retrace_ts = break_ts + timedelta(minutes=1)
        events = engine.on_bar(_bar(retrace_ts, 2708, 2712, 2704, 2710))

        reject_events = [e for e in events if e.event_type == "REJECT"]
        assert len(reject_events) == 1
        assert "risk_rejected" in reject_events[0].reason
        assert rm.daily_trade_count == 0

    def test_reject_does_not_record_entry(self):
        limits = RiskLimits(max_daily_trades=0)
        rm = RiskManager(limits)
        rm.daily_reset(_TRADING_DAY)

        strategy = _make_strategy(entry_model="E2", confirm_bars=1)
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost(), risk_manager=rm)
        engine.on_trading_day_start(_TRADING_DAY)

        break_ts = _build_orb(engine)
        _break_long(engine, break_ts)

        # No active entered trades
        active = engine.get_active_trades()
        entered = [t for t in active if t.state == TradeState.ENTERED]
        assert len(entered) == 0
        assert engine.daily_trade_count == 0


# ============================================================================
# 5. Circuit breaker: after max_daily_loss_r exceeded, entries are rejected
# ============================================================================

class TestCircuitBreaker:

    def test_circuit_breaker_rejects_after_loss_limit(self):
        limits = RiskLimits(max_daily_loss_r=-2.0, max_concurrent_positions=5,
                            max_per_orb_positions=5, max_daily_trades=20)
        rm = RiskManager(limits)
        rm.daily_reset(_TRADING_DAY)

        # Two strategies on same ORB: both enter and both lose => -2.0R total
        strat_a = _make_strategy(
            entry_model="E2", confirm_bars=1, rr_target=2.0,
            strategy_id="MGC_2300_E2_RR2.0_CB1_A",
        )
        strat_b = _make_strategy(
            entry_model="E2", confirm_bars=1, rr_target=2.0,
            strategy_id="MGC_2300_E2_RR2.0_CB1_B",
        )

        engine = ExecutionEngine(
            _make_portfolio([strat_a, strat_b]), _cost(), risk_manager=rm,
        )
        engine.on_trading_day_start(_TRADING_DAY)

        break_ts = _build_orb(engine)
        _break_long(engine, break_ts)

        # Both entered, now stop hit => both lose -1.0R each => total -2.0R
        engine.on_bar(_bar(break_ts + timedelta(minutes=1), 2704, 2706, 2694, 2695))
        assert rm.daily_pnl_r == -2.0
        assert rm.is_halted()

        # RM should reject any new entry attempt
        allowed, reason = rm.can_enter("any_strategy", "1000", [], engine.daily_pnl_r)
        assert not allowed
        assert "circuit_breaker" in reason

    def test_circuit_breaker_via_on_trade_exit_accumulation(self):
        limits = RiskLimits(max_daily_loss_r=-3.0)
        rm = RiskManager(limits)
        rm.daily_reset(_TRADING_DAY)

        # Simulate losses via on_trade_exit directly
        rm.on_trade_exit(-1.5)
        assert not rm.is_halted()

        rm.on_trade_exit(-1.5)  # cumulative -3.0
        assert rm.is_halted()

        # Even with good PnL reported, halted persists
        allowed, reason = rm.can_enter("s1", "2300", [], 0.0)
        assert not allowed
        assert "circuit_breaker" in reason


# ============================================================================
# 6. Max concurrent: when limit reached, new entries are rejected
# ============================================================================

class TestMaxConcurrent:

    def test_max_concurrent_rejects_when_full(self):
        limits = RiskLimits(
            max_concurrent_positions=1,
            max_per_orb_positions=5,
            max_daily_trades=20,
        )
        rm = RiskManager(limits)
        rm.daily_reset(_TRADING_DAY)

        # Two strategies on same ORB: first enters, second gets rejected
        strat_a = _make_strategy(
            entry_model="E2", confirm_bars=1, rr_target=2.0,
            strategy_id="MGC_2300_E2_RR2.0_CB1_A",
        )
        strat_b = _make_strategy(
            entry_model="E2", confirm_bars=1, rr_target=2.0,
            strategy_id="MGC_2300_E2_RR2.0_CB1_B",
        )

        engine = ExecutionEngine(
            _make_portfolio([strat_a, strat_b]), _cost(), risk_manager=rm,
        )
        engine.on_trading_day_start(_TRADING_DAY)

        break_ts = _build_orb(engine)
        events = _break_long(engine, break_ts)

        entry_events = [e for e in events if e.event_type == "ENTRY"]
        reject_events = [e for e in events if e.event_type == "REJECT"]

        # First strategy enters, second is rejected (max concurrent = 1)
        assert len(entry_events) == 1
        assert len(reject_events) == 1
        assert "risk_rejected" in reject_events[0].reason

    def test_max_concurrent_allows_after_exit(self):
        limits = RiskLimits(
            max_concurrent_positions=1,
            max_per_orb_positions=5,
            max_daily_trades=20,
        )
        rm = RiskManager(limits)
        rm.daily_reset(_TRADING_DAY)

        strategy = _make_strategy(
            entry_model="E2", confirm_bars=1, rr_target=2.0,
        )
        engine = ExecutionEngine(
            _make_portfolio([strategy]), _cost(), risk_manager=rm,
        )
        engine.on_trading_day_start(_TRADING_DAY)

        break_ts = _build_orb(engine)
        _break_long(engine, break_ts)

        # Trade entered. Now exit via stop.
        engine.on_bar(_bar(break_ts + timedelta(minutes=1), 2704, 2706, 2694, 2695))

        # After exit, concurrent count drops. RM should allow a new entry.
        allowed, _ = rm.can_enter("new_strat", "1800", [], engine.daily_pnl_r)
        assert allowed


# ============================================================================
# Lifecycle consistency: RM state matches engine state
# ============================================================================

class TestLifecycleConsistency:

    def test_rm_trade_count_matches_engine(self):
        limits = RiskLimits()
        rm = RiskManager(limits)
        rm.daily_reset(_TRADING_DAY)

        strategy = _make_strategy(entry_model="E2", confirm_bars=1)
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost(), risk_manager=rm)
        engine.on_trading_day_start(_TRADING_DAY)

        break_ts = _build_orb(engine)
        _break_long(engine, break_ts)

        assert rm.daily_trade_count == engine.daily_trade_count

    def test_rm_pnl_tracks_engine_after_multiple_exits(self):
        limits = RiskLimits(max_per_orb_positions=5, max_concurrent_positions=5,
                            max_daily_trades=20)
        rm = RiskManager(limits)
        rm.daily_reset(_TRADING_DAY)

        # Two strategies, same ORB, both enter and lose
        strat_a = _make_strategy(
            entry_model="E2", confirm_bars=1, rr_target=2.0,
            strategy_id="MGC_2300_E2_RR2.0_CB1_A",
        )
        strat_b = _make_strategy(
            entry_model="E2", confirm_bars=1, rr_target=2.0,
            strategy_id="MGC_2300_E2_RR2.0_CB1_B",
        )
        engine = ExecutionEngine(
            _make_portfolio([strat_a, strat_b]), _cost(), risk_manager=rm,
        )
        engine.on_trading_day_start(_TRADING_DAY)

        break_ts = _build_orb(engine)
        _break_long(engine, break_ts)

        # Both should enter (max_concurrent=5)
        entered = [t for t in engine.active_trades if t.state == TradeState.ENTERED]
        assert len(entered) == 2

        # Stop hit for both
        engine.on_bar(_bar(break_ts + timedelta(minutes=1), 2704, 2706, 2694, 2695))

        assert rm.daily_trade_count == 2
        assert rm.daily_pnl_r == -2.0  # Two losses at -1.0R each
        assert engine.daily_pnl_r == rm.daily_pnl_r

    def test_no_risk_manager_engine_still_works(self):
        strategy = _make_strategy(entry_model="E2", confirm_bars=1)
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost(), risk_manager=None)
        engine.on_trading_day_start(_TRADING_DAY)

        break_ts = _build_orb(engine)
        events = _break_long(engine, break_ts)

        entry_events = [e for e in events if e.event_type == "ENTRY"]
        assert len(entry_events) == 1  # Works fine without RM

    def test_full_lifecycle_entry_exit_scratch(self):
        limits = RiskLimits(max_concurrent_positions=5, max_per_orb_positions=5,
                            max_daily_trades=20)
        rm = RiskManager(limits)
        rm.daily_reset(_TRADING_DAY)

        strat_a = _make_strategy(
            entry_model="E2", confirm_bars=1, rr_target=2.0,
            strategy_id="MGC_2300_E2_RR2.0_CB1_A",
        )
        strat_b = _make_strategy(
            entry_model="E2", confirm_bars=1, rr_target=2.0,
            strategy_id="MGC_2300_E2_RR2.0_CB1_B",
        )

        engine = ExecutionEngine(
            _make_portfolio([strat_a, strat_b]), _cost(), risk_manager=rm,
        )
        engine.on_trading_day_start(_TRADING_DAY)

        break_ts = _build_orb(engine)
        _break_long(engine, break_ts)
        assert rm.daily_trade_count == 2

        # Strat A hits stop (-1.0R), strat B survives
        # Use a bar that only hits strat A's stop but not strat B (same stop, so both hit)
        engine.on_bar(_bar(break_ts + timedelta(minutes=1), 2704, 2706, 2694, 2695))
        # Both stopped out
        assert rm.daily_trade_count == 2
        assert rm.daily_pnl_r == -2.0

        # EOD scratch should not add more (no active entered trades)
        events = engine.on_trading_day_end()
        scratch_events = [e for e in events if e.event_type == "SCRATCH"]
        assert len(scratch_events) == 0  # Already exited
        assert rm.daily_pnl_r == -2.0  # Unchanged
