"""
Tests for IB-conditional exit logic and LiveIB tracker.

Covers:
- LiveIB formation and break detection
- 1000 session IB-conditional exits (pending/aligned/opposed)
- 0900 session always uses fixed target (no IB logic)
- 1100 shelved (not in ORB_WINDOWS_UTC for now)
- Hold-7h timeout
"""

import sys
from pathlib import Path
from datetime import date, datetime, timezone, timedelta

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trading_app.execution_engine import (
    ExecutionEngine,
    LiveIB,
    LiveORB,
    ActiveTrade,
    TradeEvent,
    TradeState,
    ORB_WINDOWS_UTC,
)
from trading_app.config import (
    SESSION_EXIT_MODE, IB_DURATION_MINUTES, HOLD_HOURS, ORB_DURATION_MINUTES,
)
from trading_app.portfolio import Portfolio, PortfolioStrategy
from pipeline.cost_model import get_cost_spec


def _cost():
    return get_cost_spec("MGC")


def _make_strategy(**overrides):
    base = dict(
        strategy_id="MGC_1000_E1_RR2.5_CB1_ORB_G5",
        instrument="MGC",
        orb_label="1000",
        entry_model="E1",
        rr_target=2.5,
        confirm_bars=1,
        filter_type="ORB_G5",
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
# LiveIB Unit Tests
# ============================================================================

class TestLiveIB:

    def test_ib_formation(self):
        """IB high/low computed from bars within 120-min window."""
        start = datetime(2024, 1, 4, 23, 0, tzinfo=timezone.utc)
        end = start + timedelta(minutes=120)
        ib = LiveIB(window_start_utc=start, window_end_utc=end)

        ib.update(_bar(start, 2700, 2710, 2695, 2705))
        ib.update(_bar(start + timedelta(minutes=1), 2705, 2715, 2698, 2712))

        assert ib.high == 2715.0
        assert ib.low == 2695.0
        assert not ib.complete

    def test_ib_completes_after_window(self):
        """IB marked complete when bar ts >= window end."""
        start = datetime(2024, 1, 4, 23, 0, tzinfo=timezone.utc)
        end = start + timedelta(minutes=120)
        ib = LiveIB(window_start_utc=start, window_end_utc=end)

        # Feed bars during IB
        for i in range(120):
            ib.update(_bar(start + timedelta(minutes=i), 2700, 2710, 2695, 2705))

        assert not ib.complete

        # Bar at or after end
        ib.update(_bar(end, 2705, 2708, 2702, 2706))
        assert ib.complete

    def test_ib_break_long(self):
        """Close above IB high after completion = long break."""
        start = datetime(2024, 1, 4, 23, 0, tzinfo=timezone.utc)
        end = start + timedelta(minutes=120)
        ib = LiveIB(window_start_utc=start, window_end_utc=end)

        # Set IB range
        ib.update(_bar(start, 2700, 2710, 2690, 2705))
        ib.update(_bar(end, 2705, 2708, 2702, 2706))  # completes IB
        assert ib.complete
        assert ib.high == 2710.0
        assert ib.low == 2690.0

        # Break long
        result = ib.check_break(_bar(end + timedelta(minutes=1), 2708, 2715, 2707, 2711))
        assert result == "long"
        assert ib.break_dir == "long"

    def test_ib_break_short(self):
        """Close below IB low after completion = short break."""
        start = datetime(2024, 1, 4, 23, 0, tzinfo=timezone.utc)
        end = start + timedelta(minutes=120)
        ib = LiveIB(window_start_utc=start, window_end_utc=end)

        ib.update(_bar(start, 2700, 2710, 2690, 2705))
        ib.update(_bar(end, 2705, 2708, 2702, 2706))

        result = ib.check_break(_bar(end + timedelta(minutes=1), 2695, 2698, 2685, 2688))
        assert result == "short"
        assert ib.break_dir == "short"

    def test_ib_no_break_inside_range(self):
        """Close within IB range = no break."""
        start = datetime(2024, 1, 4, 23, 0, tzinfo=timezone.utc)
        end = start + timedelta(minutes=120)
        ib = LiveIB(window_start_utc=start, window_end_utc=end)

        ib.update(_bar(start, 2700, 2710, 2690, 2705))
        ib.update(_bar(end, 2705, 2708, 2702, 2706))

        result = ib.check_break(_bar(end + timedelta(minutes=1), 2700, 2708, 2692, 2705))
        assert result is None
        assert ib.break_dir is None

    def test_ib_break_only_once(self):
        """IB break direction is set once, subsequent bars don't change it."""
        start = datetime(2024, 1, 4, 23, 0, tzinfo=timezone.utc)
        end = start + timedelta(minutes=120)
        ib = LiveIB(window_start_utc=start, window_end_utc=end)

        ib.update(_bar(start, 2700, 2710, 2690, 2705))
        ib.update(_bar(end, 2705, 2708, 2702, 2706))

        ib.check_break(_bar(end + timedelta(minutes=1), 2708, 2715, 2707, 2711))
        assert ib.break_dir == "long"

        # Second break attempt shouldn't change direction
        result = ib.check_break(_bar(end + timedelta(minutes=2), 2695, 2698, 2685, 2688))
        assert result is None
        assert ib.break_dir == "long"

    def test_ib_no_break_before_complete(self):
        """Break check returns None if IB not yet complete."""
        start = datetime(2024, 1, 4, 23, 0, tzinfo=timezone.utc)
        end = start + timedelta(minutes=120)
        ib = LiveIB(window_start_utc=start, window_end_utc=end)

        ib.update(_bar(start, 2700, 2710, 2690, 2705))
        # IB not complete yet
        assert not ib.complete

        result = ib.check_break(_bar(start + timedelta(minutes=60), 2708, 2715, 2707, 2711))
        assert result is None


# ============================================================================
# 1100 Exclusion Tests
# ============================================================================

class Test1100Exclusion:

    def test_1100_not_in_orb_windows(self):
        """1100 session shelved for breakout — not in ORB_WINDOWS_UTC."""
        assert "1100" not in ORB_WINDOWS_UTC

    def test_1100_not_in_session_exit_mode(self):
        """1100 session shelved — not in SESSION_EXIT_MODE."""
        assert "1100" not in SESSION_EXIT_MODE

    def test_1130_in_orb_windows(self):
        """1130 (HK/SG equity open) is registered in ORB_WINDOWS_UTC."""
        assert "1130" in ORB_WINDOWS_UTC
        assert ORB_WINDOWS_UTC["1130"] == (1, 30)

    def test_1130_fixed_target(self):
        """1130 uses fixed_target exit mode."""
        assert SESSION_EXIT_MODE["1130"] == "fixed_target"


# ============================================================================
# Config Tests
# ============================================================================

class TestConfig:

    def test_session_exit_modes_defined(self):
        """All active sessions have exit modes defined."""
        for label in ORB_WINDOWS_UTC:
            assert label in SESSION_EXIT_MODE

    def test_1000_is_ib_conditional(self):
        assert SESSION_EXIT_MODE["1000"] == "ib_conditional"

    def test_0900_is_fixed_target(self):
        assert SESSION_EXIT_MODE["0900"] == "fixed_target"

    def test_ib_duration(self):
        assert IB_DURATION_MINUTES == 120

    def test_hold_hours(self):
        assert HOLD_HOURS == 7


# ============================================================================
# IB-Conditional Exit Integration Tests
# ============================================================================

class TestIBConditionalExits:
    """Test 1000 session IB-conditional exit behavior in the execution engine."""

    def _build_1000_orb_and_enter(self, engine, td=date(2024, 1, 5),
                                   orb_high=2710.0, orb_low=2700.0):
        """Build 1000 ORB, trigger long break, and enter E1 trade.

        1000 ORB window: 00:00 to 00:00+ORB_DURATION UTC on trading day.
        Uses session-specific duration from config (variable aperture).
        Returns the timestamp base for further bar generation.
        """
        orb_dur = ORB_DURATION_MINUTES.get("1000", 5)
        ts_base = datetime(td.year, td.month, td.day, 0, 0, tzinfo=timezone.utc)

        # Feed IB bars before the 1000 window (23:00-00:00 UTC)
        prev_day = td - timedelta(days=1)
        ib_ts = datetime(prev_day.year, prev_day.month, prev_day.day,
                         23, 0, tzinfo=timezone.utc)
        for i in range(60):
            engine.on_bar(_bar(ib_ts + timedelta(minutes=i),
                               2700, 2720, 2680, 2705))

        # Build 1000 ORB (00:00 to 00:00+orb_dur UTC)
        for i in range(orb_dur):
            engine.on_bar(_bar(ts_base + timedelta(minutes=i),
                               orb_low, orb_high, orb_low, orb_low + 5))

        # Break bar at orb_dur: close > ORB high
        engine.on_bar(_bar(ts_base + timedelta(minutes=orb_dur),
                           orb_high - 1, orb_high + 5, orb_high - 2, orb_high + 1))

        # E1 fill bar at orb_dur+1
        fill_bar = _bar(ts_base + timedelta(minutes=orb_dur + 1),
                        orb_high + 2, orb_high + 5, orb_high + 1, orb_high + 3)
        engine.on_bar(fill_bar)

        return ts_base

    def test_1000_trade_starts_ib_pending(self):
        """1000 trade enters in ib_pending mode when IB hasn't broken yet."""
        strategy = _make_strategy()
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))

        ts_base = self._build_1000_orb_and_enter(engine)

        # Trade should be entered in ib_pending mode
        entered = [t for t in engine.active_trades if t.state == TradeState.ENTERED]
        assert len(entered) == 1
        assert entered[0].exit_mode == "ib_pending"
        assert entered[0].ib_alignment is None

    def test_1000_ib_aligned_switches_to_hold_7h(self):
        """IB breaks aligned -> trade switches to hold_7h, target removed."""
        strategy = _make_strategy()
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))

        ts_base = self._build_1000_orb_and_enter(engine)

        # IB completes at 01:00 UTC (23:00 + 120 min)
        # Feed bars to complete IB
        ib_end = datetime(2024, 1, 5, 1, 0, tzinfo=timezone.utc)
        t = ts_base + timedelta(minutes=ORB_DURATION_MINUTES.get("1000", 5) + 2)
        while t < ib_end:
            engine.on_bar(_bar(t, 2710, 2715, 2705, 2712))
            t += timedelta(minutes=1)

        # Bar at IB end that completes it
        engine.on_bar(_bar(ib_end, 2712, 2718, 2710, 2715))

        # Now IB should be complete
        assert engine.ib.complete

        # Bar that breaks IB high (aligned with long trade)
        events = engine.on_bar(_bar(ib_end + timedelta(minutes=1),
                                    2720, 2725, 2718, 2722))

        entered = [t for t in engine.active_trades if t.state == TradeState.ENTERED]
        assert len(entered) == 1
        assert entered[0].exit_mode == "hold_7h"
        assert entered[0].ib_alignment == "aligned"
        assert entered[0].target_price is None  # Target removed

    def test_1000_ib_opposed_exits_at_market(self):
        """IB breaks opposed -> trade exits at bar close."""
        strategy = _make_strategy()
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))

        ts_base = self._build_1000_orb_and_enter(engine)

        # Feed bars to complete IB
        ib_end = datetime(2024, 1, 5, 1, 0, tzinfo=timezone.utc)
        t = ts_base + timedelta(minutes=ORB_DURATION_MINUTES.get("1000", 5) + 2)
        while t < ib_end:
            engine.on_bar(_bar(t, 2710, 2715, 2705, 2712))
            t += timedelta(minutes=1)

        engine.on_bar(_bar(ib_end, 2712, 2718, 2710, 2715))
        assert engine.ib.complete

        # Bar that breaks IB low (opposed to long trade)
        events = engine.on_bar(_bar(ib_end + timedelta(minutes=1),
                                    2685, 2688, 2675, 2678))

        exit_events = [e for e in events if e.event_type == "EXIT"]
        assert len(exit_events) == 1
        assert "ib_opposed" in exit_events[0].reason

        # Trade should be completed
        completed = [t for t in engine.completed_trades
                     if t.ib_alignment == "opposed"]
        assert len(completed) == 1

    def test_1000_ib_pending_keeps_fixed_target(self):
        """While in ib_pending mode, fixed target and stop remain active."""
        strategy = _make_strategy()
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))

        ts_base = self._build_1000_orb_and_enter(engine)

        entered = [t for t in engine.active_trades if t.state == TradeState.ENTERED]
        assert len(entered) == 1
        assert entered[0].exit_mode == "ib_pending"
        assert entered[0].target_price is not None  # Target still active
        assert entered[0].stop_price is not None     # Stop still active

    def test_hold_7h_timeout_exit(self):
        """Trade in hold_7h exits after HOLD_HOURS."""
        strategy = _make_strategy()
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))

        ts_base = self._build_1000_orb_and_enter(engine)

        # Complete IB and break aligned
        ib_end = datetime(2024, 1, 5, 1, 0, tzinfo=timezone.utc)
        t = ts_base + timedelta(minutes=ORB_DURATION_MINUTES.get("1000", 5) + 2)
        while t < ib_end:
            engine.on_bar(_bar(t, 2710, 2715, 2705, 2712))
            t += timedelta(minutes=1)
        engine.on_bar(_bar(ib_end, 2712, 2718, 2710, 2715))
        engine.on_bar(_bar(ib_end + timedelta(minutes=1), 2720, 2725, 2718, 2722))

        entered = [t for t in engine.active_trades if t.state == TradeState.ENTERED]
        assert len(entered) == 1
        assert entered[0].exit_mode == "hold_7h"
        entry_ts = entered[0].entry_ts

        # Feed bars for 7 hours without hitting stop
        t = ib_end + timedelta(minutes=2)
        timeout_ts = entry_ts + timedelta(hours=HOLD_HOURS)
        events_all = []
        while t < timeout_ts:
            events_all.extend(engine.on_bar(_bar(t, 2715, 2720, 2705, 2718)))
            t += timedelta(minutes=1)

        # No exit yet
        assert len([e for e in events_all if e.event_type == "EXIT"]) == 0

        # Bar at timeout
        events = engine.on_bar(_bar(timeout_ts, 2715, 2720, 2705, 2718))
        exit_events = [e for e in events if e.event_type == "EXIT"]
        assert len(exit_events) == 1
        assert "hold_timeout" in exit_events[0].reason

    def test_hold_7h_stop_still_active(self):
        """Stop is still active in hold_7h mode (not just time cutoff)."""
        strategy = _make_strategy()
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))

        ts_base = self._build_1000_orb_and_enter(engine)

        # Complete IB and break aligned
        ib_end = datetime(2024, 1, 5, 1, 0, tzinfo=timezone.utc)
        t = ts_base + timedelta(minutes=ORB_DURATION_MINUTES.get("1000", 5) + 2)
        while t < ib_end:
            engine.on_bar(_bar(t, 2710, 2715, 2705, 2712))
            t += timedelta(minutes=1)
        engine.on_bar(_bar(ib_end, 2712, 2718, 2710, 2715))
        engine.on_bar(_bar(ib_end + timedelta(minutes=1), 2720, 2725, 2718, 2722))

        entered = [t for t in engine.active_trades if t.state == TradeState.ENTERED]
        assert len(entered) == 1
        assert entered[0].exit_mode == "hold_7h"
        stop = entered[0].stop_price

        # Bar that hits stop
        events = engine.on_bar(_bar(ib_end + timedelta(minutes=2),
                                    2702, 2705, stop - 1, 2698))
        exit_events = [e for e in events if e.event_type == "EXIT"]
        assert len(exit_events) == 1
        assert "loss" in exit_events[0].reason


# ============================================================================
# Early Exit Skip in hold_7h Mode
# ============================================================================

class TestEarlyExitSkipInHold7h:

    def test_early_exit_skipped_in_hold_7h(self):
        """Timed early exit must NOT fire when trade is in hold_7h mode."""
        strategy = _make_strategy()
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))

        # Build and enter 1000 trade
        td = date(2024, 1, 5)
        prev_day = td - timedelta(days=1)
        ib_ts = datetime(prev_day.year, prev_day.month, prev_day.day,
                         23, 0, tzinfo=timezone.utc)
        # Feed IB bars
        for i in range(60):
            engine.on_bar(_bar(ib_ts + timedelta(minutes=i), 2700, 2720, 2680, 2705))

        ts_base = datetime(td.year, td.month, td.day, 0, 0, tzinfo=timezone.utc)
        orb_dur = ORB_DURATION_MINUTES.get("1000", 5)
        # Build 1000 ORB (00:00 to 00:00+orb_dur)
        for i in range(orb_dur):
            engine.on_bar(_bar(ts_base + timedelta(minutes=i), 2700, 2710, 2700, 2705))
        # Break long
        engine.on_bar(_bar(ts_base + timedelta(minutes=orb_dur), 2709, 2715, 2708, 2711))
        # E1 fill
        engine.on_bar(_bar(ts_base + timedelta(minutes=orb_dur + 1), 2712, 2715, 2711, 2713))

        entered = [t for t in engine.active_trades if t.state == TradeState.ENTERED]
        assert len(entered) == 1
        assert entered[0].exit_mode == "ib_pending"

        # Complete IB and break aligned (long)
        ib_end = datetime(td.year, td.month, td.day, 1, 0, tzinfo=timezone.utc)
        t = ts_base + timedelta(minutes=ORB_DURATION_MINUTES.get("1000", 5) + 2)
        while t < ib_end:
            engine.on_bar(_bar(t, 2710, 2715, 2705, 2712))
            t += timedelta(minutes=1)
        engine.on_bar(_bar(ib_end, 2712, 2718, 2710, 2715))
        engine.on_bar(_bar(ib_end + timedelta(minutes=1), 2720, 2725, 2718, 2722))

        entered = [t for t in engine.active_trades if t.state == TradeState.ENTERED]
        assert len(entered) == 1
        assert entered[0].exit_mode == "hold_7h"
        entry_ts = entered[0].entry_ts

        # Now feed bars until 30 min after entry, with trade underwater
        t = ib_end + timedelta(minutes=2)
        threshold_ts = entry_ts + timedelta(minutes=30)
        exit_events = []
        while t <= threshold_ts + timedelta(minutes=1):
            # Price dropping — trade is losing
            evts = engine.on_bar(_bar(t, 2708, 2710, 2705, 2706))
            exit_events.extend([e for e in evts if e.event_type == "EXIT"])
            t += timedelta(minutes=1)

        # No exit should have fired — early exit skipped for hold_7h
        assert len(exit_events) == 0, (
            "Early exit must be skipped in hold_7h mode"
        )


# ============================================================================
# IB Already Opposed at Entry Time
# ============================================================================

class TestIBAlreadyOpposed:

    def test_1000_entry_rejected_when_ib_already_opposed_unit(self):
        """Unit test: verify rejection code path when IB is pre-resolved opposed.

        The IB-already-opposed scenario is structurally rare for 1000 E1 CB1
        (IB range contains ORB range, so IB breaks after ORB in same direction).
        But it CAN happen with higher CB or late ORB breaks. Test the code path
        directly by pre-setting IB state.
        """
        strategy = _make_strategy()
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))

        td = date(2024, 1, 5)
        prev_day = td - timedelta(days=1)

        # Build 1000 ORB normally
        ib_ts = datetime(prev_day.year, prev_day.month, prev_day.day,
                         23, 0, tzinfo=timezone.utc)
        for i in range(60):
            engine.on_bar(_bar(ib_ts + timedelta(minutes=i), 2705, 2710, 2702, 2706))

        ts_base = datetime(td.year, td.month, td.day, 0, 0, tzinfo=timezone.utc)
        orb_dur = ORB_DURATION_MINUTES.get("1000", 5)
        for i in range(orb_dur):
            engine.on_bar(_bar(ts_base + timedelta(minutes=i), 2702, 2710, 2700, 2705))

        # Force IB to be complete and broken SHORT (simulating the scenario)
        engine.ib.complete = True
        engine.ib.high = 2715.0
        engine.ib.low = 2695.0
        engine.ib.break_dir = "short"
        engine.ib.break_ts = ts_base + timedelta(minutes=50)

        # Trigger ORB LONG break (close > 2710)
        events = engine.on_bar(_bar(ts_base + timedelta(minutes=orb_dur),
                                    2709, 2715, 2708, 2711))

        # E1 armed on break bar — fills on next bar
        next_events = engine.on_bar(_bar(ts_base + timedelta(minutes=orb_dur + 1),
                                         2712, 2715, 2711, 2713))

        reject_events = [e for e in next_events if e.event_type == "REJECT"]
        entry_events = [e for e in next_events if e.event_type == "ENTRY"]

        assert len(reject_events) == 1, "Trade should be rejected when IB already opposed"
        assert "ib_already_opposed" in reject_events[0].reason
        assert len(entry_events) == 0, "No entry should happen"


# ============================================================================
# 0900 Session (Fixed Target, No IB Logic)
# ============================================================================

class Test0900FixedTarget:

    def test_0900_always_fixed_target(self):
        """0900 session trades always use fixed_target exit mode."""
        strategy = _make_strategy(
            strategy_id="MGC_0900_E1_RR2.5_CB1_ORB_G5",
            orb_label="0900",
        )
        engine = ExecutionEngine(_make_portfolio([strategy]), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))

        # 0900 ORB window: 23:00-23:05 UTC prev day
        prev_day = date(2024, 1, 4)
        ts_base = datetime(prev_day.year, prev_day.month, prev_day.day,
                           23, 0, tzinfo=timezone.utc)

        # Build ORB
        for i in range(5):
            engine.on_bar(_bar(ts_base + timedelta(minutes=i),
                               2700, 2710, 2695, 2705))

        # Break bar
        engine.on_bar(_bar(ts_base + timedelta(minutes=5),
                           2708, 2715, 2707, 2712))

        # E1 fill bar
        engine.on_bar(_bar(ts_base + timedelta(minutes=6),
                           2714, 2718, 2712, 2716))

        entered = [t for t in engine.active_trades if t.state == TradeState.ENTERED]
        assert len(entered) == 1
        assert entered[0].exit_mode == "fixed_target"
        assert entered[0].target_price is not None  # Target active


# ============================================================================
# IB Initialization Tests
# ============================================================================

class TestIBInitialization:

    def test_engine_creates_ib_on_day_start(self):
        """Engine creates LiveIB on trading day start."""
        engine = ExecutionEngine(_make_portfolio(), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))

        assert engine.ib is not None
        assert engine.ib.window_start_utc == datetime(2024, 1, 4, 23, 0, tzinfo=timezone.utc)
        assert engine.ib.window_end_utc == datetime(2024, 1, 5, 1, 0, tzinfo=timezone.utc)

    def test_ib_resets_on_new_day(self):
        """IB resets on new trading day."""
        engine = ExecutionEngine(_make_portfolio(), _cost())
        engine.on_trading_day_start(date(2024, 1, 5))
        engine.ib.high = 2720.0  # Simulate some data
        engine.ib.complete = True

        engine.on_trading_day_start(date(2024, 1, 6))
        assert engine.ib.high is None
        assert not engine.ib.complete
        assert engine.ib.window_start_utc == datetime(2024, 1, 5, 23, 0, tzinfo=timezone.utc)
