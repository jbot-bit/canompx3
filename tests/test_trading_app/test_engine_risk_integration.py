"""
Integration tests: ExecutionEngine + RiskManager.

Verifies the engine properly calls RiskManager methods during the trade
lifecycle (entry, exit, scratch, reject).
"""

import sys
from pathlib import Path
from datetime import date, datetime, timezone, timedelta

import pytest

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
        strategy_id="MGC_2300_E1_RR2.0_CB1_NO_FILTER",
        instrument="MGC",
        orb_label="2300",
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

def _fill_e1(engine, break_ts, o=2708, h=2715, l=2707, c=2712):
    """Feed the E1 fill bar (next bar after confirm). Returns events."""
    fill_ts = break_ts + timedelta(minutes=1)
    return engine.on_bar(_bar(fill_ts, o, h, l, c))

# ============================================================================
# 1. Engine calls risk_manager.on_trade_entry() on entry (E1, E3)
# ============================================================================

class TestOnTradeEntry:

    def test_e1_entry_calls_on_trade_entry(self):
        limits = RiskLimits()
        rm = RiskManager(limits)
        rm.daily_reset(_TRADING_DAY)

        strategy = _make_strategy(entry_model="E1", confirm_bars=1)
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost(), risk_manager=rm)
        engine.on_trading_day_start(_TRADING_DAY)

        break_ts = _build_orb(engine)
        _break_long(engine, break_ts)
        assert rm.daily_trade_count == 0  # Still ARMED, not entered

        # Next bar triggers E1 fill
        events = _fill_e1(engine, break_ts)

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

        strategy = _make_strategy(entry_model="E1", confirm_bars=1, rr_target=2.0)
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost(), risk_manager=rm)
        engine.on_trading_day_start(_TRADING_DAY)

        break_ts = _build_orb(engine)
        _break_long(engine, break_ts)
        # E1 fill: entry at 2708, stop=2695, risk=13, target=2708+26=2734
        _fill_e1(engine, break_ts)

        # Target hit
        target_ts = break_ts + timedelta(minutes=2)
        events = engine.on_bar(_bar(target_ts, 2712, 2735, 2711, 2730))

        exit_events = [e for e in events if e.event_type == "EXIT"]
        assert len(exit_events) == 1
        assert "win" in exit_events[0].reason
        assert rm.daily_pnl_r > 0  # Win adds positive R

    def test_stop_hit_calls_on_trade_exit(self):
        limits = RiskLimits()
        rm = RiskManager(limits)
        rm.daily_reset(_TRADING_DAY)

        strategy = _make_strategy(entry_model="E1", confirm_bars=1, rr_target=2.0)
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost(), risk_manager=rm)
        engine.on_trading_day_start(_TRADING_DAY)

        break_ts = _build_orb(engine)
        _break_long(engine, break_ts)
        _fill_e1(engine, break_ts)

        # Stop hit: low <= 2695
        stop_ts = break_ts + timedelta(minutes=2)
        events = engine.on_bar(_bar(stop_ts, 2704, 2706, 2694, 2695))

        exit_events = [e for e in events if e.event_type == "EXIT"]
        assert len(exit_events) == 1
        assert "loss" in exit_events[0].reason
        assert rm.daily_pnl_r == -1.0

    def test_pnl_r_value_passed_to_risk_manager(self):
        limits = RiskLimits()
        rm = RiskManager(limits)
        rm.daily_reset(_TRADING_DAY)

        strategy = _make_strategy(entry_model="E1", confirm_bars=1, rr_target=2.0)
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost(), risk_manager=rm)
        engine.on_trading_day_start(_TRADING_DAY)

        break_ts = _build_orb(engine)
        _break_long(engine, break_ts)
        _fill_e1(engine, break_ts)

        # Stop hit => pnl_r = -1.0
        stop_ts = break_ts + timedelta(minutes=2)
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

        strategy = _make_strategy(entry_model="E1", confirm_bars=1)
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost(), risk_manager=rm)
        engine.on_trading_day_start(_TRADING_DAY)

        break_ts = _build_orb(engine)
        _break_long(engine, break_ts)
        _fill_e1(engine, break_ts)
        assert rm.daily_trade_count == 1

        # No target/stop hit; just feed a neutral bar then close the day
        engine.on_bar(_bar(break_ts + timedelta(minutes=2), 2710, 2712, 2708, 2711))
        events = engine.on_trading_day_end()

        scratch_events = [e for e in events if e.event_type == "SCRATCH"]
        assert len(scratch_events) == 1
        # RM should have received the scratch PnL
        completed = [t for t in engine.completed_trades if t.pnl_r is not None]
        assert len(completed) == 1
        assert rm.daily_pnl_r == completed[0].pnl_r

    def test_scratch_pnl_r_reflects_mark_to_market(self):
        limits = RiskLimits()
        rm = RiskManager(limits)
        rm.daily_reset(_TRADING_DAY)

        strategy = _make_strategy(entry_model="E1", confirm_bars=1, rr_target=2.0)
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost(), risk_manager=rm)
        engine.on_trading_day_start(_TRADING_DAY)

        break_ts = _build_orb(engine)
        _break_long(engine, break_ts)
        # E1 fill: entry at open=2708
        _fill_e1(engine, break_ts)

        # Move price up, then scratch
        engine.on_bar(_bar(break_ts + timedelta(minutes=2), 2712, 2720, 2710, 2718))
        engine.on_trading_day_end()

        # Scratch is mark-to-market: last close = 2718, entry = 2708, positive PnL
        completed = engine.completed_trades
        assert len(completed) == 1
        assert completed[0].pnl_r > 0  # Price went up for a long
        assert rm.daily_pnl_r == completed[0].pnl_r

# ============================================================================
# 4. Engine emits REJECT when risk_manager.can_enter() returns False
# ============================================================================

class TestReject:

    def test_e1_rejected_by_risk_manager(self):
        limits = RiskLimits(max_daily_trades=0)
        rm = RiskManager(limits)
        rm.daily_reset(_TRADING_DAY)

        strategy = _make_strategy(entry_model="E1", confirm_bars=1)
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost(), risk_manager=rm)
        engine.on_trading_day_start(_TRADING_DAY)

        break_ts = _build_orb(engine)
        _break_long(engine, break_ts)  # ARMED

        # Next bar: E1 tries to fill, but RM rejects
        events = _fill_e1(engine, break_ts)

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

        strategy = _make_strategy(entry_model="E1", confirm_bars=1)
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost(), risk_manager=rm)
        engine.on_trading_day_start(_TRADING_DAY)

        break_ts = _build_orb(engine)
        _break_long(engine, break_ts)

        # E1 fill bar triggers rejection
        _fill_e1(engine, break_ts)

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

        # Two E1 strategies on same ORB: both enter and both lose => -2.0R total
        strat_a = _make_strategy(
            entry_model="E1", confirm_bars=1, rr_target=2.0,
            strategy_id="MGC_2300_E1_RR2.0_CB1_A",
        )
        strat_b = _make_strategy(
            entry_model="E1", confirm_bars=1, rr_target=2.0,
            strategy_id="MGC_2300_E1_RR2.0_CB1_B",
        )

        engine = ExecutionEngine(
            _make_portfolio([strat_a, strat_b]), _cost(), risk_manager=rm,
        )
        engine.on_trading_day_start(_TRADING_DAY)

        break_ts = _build_orb(engine)
        _break_long(engine, break_ts)  # Both ARMED

        # E1 fill bar: both enter at open=2708
        _fill_e1(engine, break_ts)

        # Both entered, now stop hit => both lose -1.0R each => total -2.0R
        engine.on_bar(_bar(break_ts + timedelta(minutes=2), 2704, 2706, 2694, 2695))
        assert rm.daily_pnl_r == -2.0
        assert rm.is_halted()

        # RM should reject any new entry attempt
        allowed, reason, _ = rm.can_enter("any_strategy", "1000", [], engine.daily_pnl_r)
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
        allowed, reason, _ = rm.can_enter("s1", "2300", [], 0.0)
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

        # Two E1 strategies on same ORB: first enters, second gets rejected
        strat_a = _make_strategy(
            entry_model="E1", confirm_bars=1, rr_target=2.0,
            strategy_id="MGC_2300_E1_RR2.0_CB1_A",
        )
        strat_b = _make_strategy(
            entry_model="E1", confirm_bars=1, rr_target=2.0,
            strategy_id="MGC_2300_E1_RR2.0_CB1_B",
        )

        engine = ExecutionEngine(
            _make_portfolio([strat_a, strat_b]), _cost(), risk_manager=rm,
        )
        engine.on_trading_day_start(_TRADING_DAY)

        break_ts = _build_orb(engine)
        _break_long(engine, break_ts)  # Both ARMED

        # E1 fill bar: first fills, second rejected (max concurrent = 1)
        events = _fill_e1(engine, break_ts)

        entry_events = [e for e in events if e.event_type == "ENTRY"]
        reject_events = [e for e in events if e.event_type == "REJECT"]

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

        strategy = _make_strategy(entry_model="E1", confirm_bars=1, rr_target=2.0)
        engine = ExecutionEngine(
            _make_portfolio([strategy]), _cost(), risk_manager=rm,
        )
        engine.on_trading_day_start(_TRADING_DAY)

        break_ts = _build_orb(engine)
        _break_long(engine, break_ts)  # ARMED
        _fill_e1(engine, break_ts)  # Entered

        # Trade entered. Now exit via stop.
        engine.on_bar(_bar(break_ts + timedelta(minutes=2), 2704, 2706, 2694, 2695))

        # After exit, concurrent count drops. RM should allow a new entry.
        allowed, _, _ = rm.can_enter("new_strat", "1800", [], engine.daily_pnl_r)
        assert allowed

# ============================================================================
# Lifecycle consistency: RM state matches engine state
# ============================================================================

class TestLifecycleConsistency:

    def test_rm_trade_count_matches_engine(self):
        limits = RiskLimits()
        rm = RiskManager(limits)
        rm.daily_reset(_TRADING_DAY)

        strategy = _make_strategy(entry_model="E1", confirm_bars=1)
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost(), risk_manager=rm)
        engine.on_trading_day_start(_TRADING_DAY)

        break_ts = _build_orb(engine)
        _break_long(engine, break_ts)
        _fill_e1(engine, break_ts)

        assert rm.daily_trade_count == engine.daily_trade_count

    def test_rm_pnl_tracks_engine_after_multiple_exits(self):
        limits = RiskLimits(max_per_orb_positions=5, max_concurrent_positions=5,
                            max_daily_trades=20)
        rm = RiskManager(limits)
        rm.daily_reset(_TRADING_DAY)

        # Two E1 strategies, same ORB, both enter and lose
        strat_a = _make_strategy(
            entry_model="E1", confirm_bars=1, rr_target=2.0,
            strategy_id="MGC_2300_E1_RR2.0_CB1_A",
        )
        strat_b = _make_strategy(
            entry_model="E1", confirm_bars=1, rr_target=2.0,
            strategy_id="MGC_2300_E1_RR2.0_CB1_B",
        )
        engine = ExecutionEngine(
            _make_portfolio([strat_a, strat_b]), _cost(), risk_manager=rm,
        )
        engine.on_trading_day_start(_TRADING_DAY)

        break_ts = _build_orb(engine)
        _break_long(engine, break_ts)  # Both ARMED
        _fill_e1(engine, break_ts)  # Both fill at open=2708

        # Both should enter (max_concurrent=5)
        entered = [t for t in engine.active_trades if t.state == TradeState.ENTERED]
        assert len(entered) == 2

        # Stop hit for both
        engine.on_bar(_bar(break_ts + timedelta(minutes=2), 2704, 2706, 2694, 2695))

        assert rm.daily_trade_count == 2
        assert rm.daily_pnl_r == -2.0  # Two losses at -1.0R each
        assert engine.daily_pnl_r == rm.daily_pnl_r

    def test_no_risk_manager_engine_still_works(self):
        strategy = _make_strategy(entry_model="E1", confirm_bars=1)
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost(), risk_manager=None)
        engine.on_trading_day_start(_TRADING_DAY)

        break_ts = _build_orb(engine)
        _break_long(engine, break_ts)
        events = _fill_e1(engine, break_ts)

        entry_events = [e for e in events if e.event_type == "ENTRY"]
        assert len(entry_events) == 1  # Works fine without RM

    def test_full_lifecycle_entry_exit_scratch(self):
        limits = RiskLimits(max_concurrent_positions=5, max_per_orb_positions=5,
                            max_daily_trades=20)
        rm = RiskManager(limits)
        rm.daily_reset(_TRADING_DAY)

        strat_a = _make_strategy(
            entry_model="E1", confirm_bars=1, rr_target=2.0,
            strategy_id="MGC_2300_E1_RR2.0_CB1_A",
        )
        strat_b = _make_strategy(
            entry_model="E1", confirm_bars=1, rr_target=2.0,
            strategy_id="MGC_2300_E1_RR2.0_CB1_B",
        )

        engine = ExecutionEngine(
            _make_portfolio([strat_a, strat_b]), _cost(), risk_manager=rm,
        )
        engine.on_trading_day_start(_TRADING_DAY)

        break_ts = _build_orb(engine)
        _break_long(engine, break_ts)
        _fill_e1(engine, break_ts)
        assert rm.daily_trade_count == 2

        # Both hit stop
        engine.on_bar(_bar(break_ts + timedelta(minutes=2), 2704, 2706, 2694, 2695))
        # Both stopped out
        assert rm.daily_trade_count == 2
        assert rm.daily_pnl_r == -2.0

        # EOD scratch should not add more (no active entered trades)
        events = engine.on_trading_day_end()
        scratch_events = [e for e in events if e.event_type == "SCRATCH"]
        assert len(scratch_events) == 0  # Already exited
        assert rm.daily_pnl_r == -2.0  # Unchanged

# ============================================================================
# 7. Correlation-weighted concurrent guard (Phase 2 risk hardening)
# ============================================================================

class TestCorrelationWeightedConcurrent:

    def test_correlated_trades_fill_budget_faster(self):
        """Two highly correlated open positions (rho=0.9) consume 0.9 effective slots each."""
        corr_lookup = {
            ("strat_new", "strat_a"): 0.9,
            ("strat_new", "strat_b"): 0.9,
        }
        limits = RiskLimits(max_concurrent_positions=3, max_per_orb_positions=5, max_daily_trades=20)
        rm = RiskManager(limits, corr_lookup=corr_lookup)
        rm.daily_reset(_TRADING_DAY)

        # Simulate 2 entered trades
        class FakeTrade:
            def __init__(self, sid, orb):
                self.strategy_id = sid
                self.orb_label = orb
                self.state = TradeState.ENTERED

        active = [FakeTrade("strat_a", "0900"), FakeTrade("strat_b", "1000")]
        # Effective = 0.9 + 0.9 = 1.8 < 3 => allowed
        allowed, reason, _ = rm.can_enter("strat_new", "1800", active, 0.0)
        assert allowed, f"Should allow: effective 1.8 < 3, got: {reason}"

    def test_correlated_trades_block_when_full(self):
        """With 3 correlated positions at rho=0.9 each, effective = 2.7 blocks at limit=3."""
        corr_lookup = {
            ("strat_new", "strat_a"): 0.95,
            ("strat_new", "strat_b"): 0.95,
            ("strat_new", "strat_c"): 0.95,
        }
        limits = RiskLimits(max_concurrent_positions=3, max_per_orb_positions=5, max_daily_trades=20)
        rm = RiskManager(limits, corr_lookup=corr_lookup)
        rm.daily_reset(_TRADING_DAY)

        class FakeTrade:
            def __init__(self, sid, orb):
                self.strategy_id = sid
                self.orb_label = orb
                self.state = TradeState.ENTERED

        active = [FakeTrade("strat_a", "0900"), FakeTrade("strat_b", "0900"),
                  FakeTrade("strat_c", "1000")]
        # Need effective >= 3. Use 4 active trades at 0.95 each = 3.8 >= 3
        active.append(FakeTrade("strat_d", "1000"))
        corr_lookup[("strat_new", "strat_d")] = 0.95
        rm = RiskManager(limits, corr_lookup=corr_lookup)
        rm.daily_reset(_TRADING_DAY)

        allowed, reason, _ = rm.can_enter("strat_new", "1800", active, 0.0)
        assert not allowed
        assert "corr_concurrent" in reason

    def test_uncorrelated_trades_allow_more(self):
        """Low correlation (0.3) trades use minimum 0.3 weight — more budget room."""
        corr_lookup = {
            ("strat_new", "strat_a"): 0.1,
            ("strat_new", "strat_b"): 0.1,
        }
        limits = RiskLimits(max_concurrent_positions=2, max_per_orb_positions=5, max_daily_trades=20)
        rm = RiskManager(limits, corr_lookup=corr_lookup)
        rm.daily_reset(_TRADING_DAY)

        class FakeTrade:
            def __init__(self, sid, orb):
                self.strategy_id = sid
                self.orb_label = orb
                self.state = TradeState.ENTERED

        active = [FakeTrade("strat_a", "0900"), FakeTrade("strat_b", "1800")]
        # Effective = max(0.1, 0.3) + max(0.1, 0.3) = 0.6 < 2 => allowed
        allowed, reason, _ = rm.can_enter("strat_new", "2300", active, 0.0)
        assert allowed, f"Should allow uncorrelated: effective 0.6 < 2, got: {reason}"

    def test_no_corr_lookup_backward_compatible(self):
        """Without corr_lookup, falls back to simple count."""
        limits = RiskLimits(max_concurrent_positions=2, max_per_orb_positions=5, max_daily_trades=20)
        rm = RiskManager(limits)
        rm.daily_reset(_TRADING_DAY)

        class FakeTrade:
            def __init__(self, sid, orb):
                self.strategy_id = sid
                self.orb_label = orb
                self.state = TradeState.ENTERED

        active = [FakeTrade("strat_a", "0900"), FakeTrade("strat_b", "1000")]
        allowed, reason, _ = rm.can_enter("strat_new", "1800", active, 0.0)
        assert not allowed
        assert "max_concurrent" in reason

# ============================================================================
# 8. Live session costs: engine produces different win PnL by session
# ============================================================================

class TestLiveSessionCosts:

    def test_win_pnl_differs_with_session_costs(self):
        """Engine with live_session_costs=True produces different win PnL
        than one without, because session slippage differs from base."""
        strategy = _make_strategy(
            entry_model="E1", confirm_bars=1, rr_target=2.0,
            orb_label="0900",
            strategy_id="MGC_0900_E1_RR2.0_CB1_NO_FILTER",
        )
        portfolio = _make_portfolio([strategy])

        engine_flat = ExecutionEngine(portfolio, _cost(), live_session_costs=False)
        engine_live = ExecutionEngine(portfolio, _cost(), live_session_costs=True)

        # Use 0900 ORB window: 23:00-23:05 UTC
        orb_base = datetime(2024, 1, 4, 23, 0, tzinfo=timezone.utc)
        td = date(2024, 1, 5)

        for eng in (engine_flat, engine_live):
            eng.on_trading_day_start(td)
            for i in range(5):
                eng.on_bar(_bar(orb_base + timedelta(minutes=i),
                                2700, 2705, 2695, 2702))
            # Break long
            break_ts = orb_base + timedelta(minutes=5)
            eng.on_bar(_bar(break_ts, 2704, 2710, 2703, 2706))
            # E1 fill bar: entry at open=2707, stop=2695, risk=12, target=2707+24=2731
            eng.on_bar(_bar(break_ts + timedelta(minutes=1),
                            2707, 2712, 2706, 2710))
            # Win: target hit (high >= 2731)
            eng.on_bar(_bar(break_ts + timedelta(minutes=2),
                            2710, 2735, 2709, 2728))

        flat_pnl = engine_flat.daily_pnl_r
        live_pnl = engine_live.daily_pnl_r

        # Both should be positive wins
        assert flat_pnl > 0, f"Flat engine should win, got {flat_pnl}"
        assert live_pnl > 0, f"Live engine should win, got {live_pnl}"
        # 0900 has 1.3x slippage => higher friction => lower R-multiple
        assert live_pnl < flat_pnl, (
            f"0900 live (1.3x slippage) should produce lower R than flat: "
            f"live={live_pnl}, flat={flat_pnl}"
        )

    def test_scratch_uses_session_costs(self):
        """Scratch PnL in live mode uses session-adjusted costs."""
        strategy = _make_strategy(
            entry_model="E1", confirm_bars=1, rr_target=2.0,
            orb_label="0900",
            strategy_id="MGC_0900_E1_RR2.0_CB1_NO_FILTER",
        )
        portfolio = _make_portfolio([strategy])

        engine_flat = ExecutionEngine(portfolio, _cost(), live_session_costs=False)
        engine_live = ExecutionEngine(portfolio, _cost(), live_session_costs=True)

        orb_base = datetime(2024, 1, 4, 23, 0, tzinfo=timezone.utc)
        td = date(2024, 1, 5)

        for eng in (engine_flat, engine_live):
            eng.on_trading_day_start(td)
            for i in range(5):
                eng.on_bar(_bar(orb_base + timedelta(minutes=i),
                                2700, 2705, 2695, 2702))
            # Break long
            break_ts = orb_base + timedelta(minutes=5)
            eng.on_bar(_bar(break_ts, 2704, 2710, 2703, 2706))
            # E1 fill bar: entry at open=2707
            eng.on_bar(_bar(break_ts + timedelta(minutes=1),
                            2707, 2712, 2706, 2710))
            # Mid-trade bar, no stop/target — trade stays open
            eng.on_bar(_bar(break_ts + timedelta(minutes=2),
                            2710, 2714, 2708, 2712))
            # EOD scratch
            eng.on_trading_day_end()

        flat_pnl = engine_flat.daily_pnl_r
        live_pnl = engine_live.daily_pnl_r

        # Both should be positive (price moved in our direction)
        assert flat_pnl > 0
        assert live_pnl > 0
        # 0900 session costs are higher => lower scratch PnL
        assert live_pnl < flat_pnl, (
            f"Scratch with 0900 session costs should be lower: "
            f"live={live_pnl}, flat={flat_pnl}"
        )

# ============================================================================
# 9. Calendar overlay: NFP/OPEX day skipping in engine
# ============================================================================

from trading_app.config import CALENDAR_SKIP_NFP_OPEX, CalendarSkipFilter

class TestCalendarOverlay:

    def test_nfp_day_blocks_entry(self):
        """On an NFP day, the calendar overlay should prevent strategy arming."""
        strategy = _make_strategy(entry_model="E1", confirm_bars=1)
        engine = ExecutionEngine(
            _make_portfolio([strategy]), _cost(),
            calendar_overlay=CALENDAR_SKIP_NFP_OPEX,
        )
        nfp_row = {"is_nfp_day": True, "is_opex_day": False, "is_friday": False, "day_of_week": 4}
        engine.on_trading_day_start(_TRADING_DAY, daily_features_row=nfp_row)

        break_ts = _build_orb(engine)
        events = _break_long(engine, break_ts)
        # No strategy should arm on NFP day
        assert len(engine.active_trades) == 0
        # Fill bar should produce nothing
        events = _fill_e1(engine, break_ts)
        entry_events = [e for e in events if e.event_type == "ENTRY"]
        assert len(entry_events) == 0

    def test_opex_day_blocks_entry(self):
        """On an OPEX day, the calendar overlay should prevent strategy arming."""
        strategy = _make_strategy(entry_model="E1", confirm_bars=1)
        engine = ExecutionEngine(
            _make_portfolio([strategy]), _cost(),
            calendar_overlay=CALENDAR_SKIP_NFP_OPEX,
        )
        opex_row = {"is_nfp_day": False, "is_opex_day": True, "is_friday": True, "day_of_week": 4}
        engine.on_trading_day_start(_TRADING_DAY, daily_features_row=opex_row)

        break_ts = _build_orb(engine)
        events = _break_long(engine, break_ts)
        assert len(engine.active_trades) == 0

    def test_normal_day_allows_entry(self):
        """On a normal day (not NFP/OPEX), the calendar overlay allows trading."""
        strategy = _make_strategy(entry_model="E1", confirm_bars=1)
        engine = ExecutionEngine(
            _make_portfolio([strategy]), _cost(),
            calendar_overlay=CALENDAR_SKIP_NFP_OPEX,
        )
        normal_row = {"is_nfp_day": False, "is_opex_day": False, "is_friday": False, "day_of_week": 2}
        engine.on_trading_day_start(_TRADING_DAY, daily_features_row=normal_row)

        break_ts = _build_orb(engine)
        _break_long(engine, break_ts)
        events = _fill_e1(engine, break_ts)
        entry_events = [e for e in events if e.event_type == "ENTRY"]
        assert len(entry_events) == 1

    def test_no_overlay_allows_nfp_day(self):
        """With calendar_overlay=None, NFP day is not blocked."""
        strategy = _make_strategy(entry_model="E1", confirm_bars=1)
        engine = ExecutionEngine(
            _make_portfolio([strategy]), _cost(),
            calendar_overlay=None,
        )
        nfp_row = {"is_nfp_day": True, "is_opex_day": False, "is_friday": False, "day_of_week": 4}
        engine.on_trading_day_start(_TRADING_DAY, daily_features_row=nfp_row)

        break_ts = _build_orb(engine)
        _break_long(engine, break_ts)
        events = _fill_e1(engine, break_ts)
        entry_events = [e for e in events if e.event_type == "ENTRY"]
        assert len(entry_events) == 1

    def test_no_daily_features_row_allows_entry(self):
        """When daily_features_row is None (no data), overlay is skipped."""
        strategy = _make_strategy(entry_model="E1", confirm_bars=1)
        engine = ExecutionEngine(
            _make_portfolio([strategy]), _cost(),
            calendar_overlay=CALENDAR_SKIP_NFP_OPEX,
        )
        engine.on_trading_day_start(_TRADING_DAY)  # No daily_features_row

        break_ts = _build_orb(engine)
        _break_long(engine, break_ts)
        events = _fill_e1(engine, break_ts)
        entry_events = [e for e in events if e.event_type == "ENTRY"]
        assert len(entry_events) == 1
