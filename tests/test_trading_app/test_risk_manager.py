"""
Tests for trading_app.risk_manager module.
"""

import sys
from pathlib import Path
from datetime import date
from dataclasses import dataclass
from enum import Enum

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trading_app.risk_manager import RiskLimits, RiskManager


# --- Minimal stubs for active trades ---

class _State(Enum):
    ENTERED = "ENTERED"
    ARMED = "ARMED"
    EXITED = "EXITED"


@dataclass
class _FakeTrade:
    orb_label: str
    state: _State = _State.ENTERED


def _entered(orb="2300"):
    return _FakeTrade(orb_label=orb, state=_State.ENTERED)


def _armed(orb="2300"):
    return _FakeTrade(orb_label=orb, state=_State.ARMED)


# ============================================================================
# RiskLimits Tests
# ============================================================================

class TestRiskLimits:

    def test_defaults(self):
        limits = RiskLimits()
        assert limits.max_daily_loss_r == -5.0
        assert limits.max_concurrent_positions == 3
        assert limits.max_per_orb_positions == 1
        assert limits.max_daily_trades == 15
        assert limits.drawdown_warning_r == -3.0

    def test_frozen(self):
        limits = RiskLimits()
        with pytest.raises(Exception):
            limits.max_daily_loss_r = -10.0

    def test_custom_values(self):
        limits = RiskLimits(
            max_daily_loss_r=-3.0,
            max_concurrent_positions=5,
            max_per_orb_positions=2,
            max_daily_trades=20,
            drawdown_warning_r=-2.0,
        )
        assert limits.max_daily_loss_r == -3.0
        assert limits.max_concurrent_positions == 5


# ============================================================================
# Circuit Breaker Tests
# ============================================================================

class TestCircuitBreaker:

    def test_halts_at_limit(self):
        rm = RiskManager(RiskLimits(max_daily_loss_r=-5.0))
        rm.daily_reset(date(2024, 1, 5))

        allowed, reason = rm.can_enter("s1", "2300", [], -5.0)
        assert not allowed
        assert "circuit_breaker" in reason

    def test_halts_below_limit(self):
        rm = RiskManager(RiskLimits(max_daily_loss_r=-5.0))
        rm.daily_reset(date(2024, 1, 5))

        allowed, _ = rm.can_enter("s1", "2300", [], -6.0)
        assert not allowed

    def test_allows_above_limit(self):
        rm = RiskManager(RiskLimits(max_daily_loss_r=-5.0))
        rm.daily_reset(date(2024, 1, 5))

        allowed, reason = rm.can_enter("s1", "2300", [], -4.9)
        assert allowed
        assert reason == ""

    def test_halt_persists_after_trigger(self):
        rm = RiskManager(RiskLimits(max_daily_loss_r=-5.0))
        rm.daily_reset(date(2024, 1, 5))

        # Trigger halt
        rm.can_enter("s1", "2300", [], -5.0)
        assert rm.is_halted()

        # Even with 0 PnL, still halted
        allowed, _ = rm.can_enter("s2", "1800", [], 0.0)
        assert not allowed

    def test_halt_clears_on_daily_reset(self):
        rm = RiskManager(RiskLimits(max_daily_loss_r=-5.0))
        rm.daily_reset(date(2024, 1, 5))
        rm.can_enter("s1", "2300", [], -5.0)
        assert rm.is_halted()

        rm.daily_reset(date(2024, 1, 6))
        assert not rm.is_halted()

    def test_on_trade_exit_triggers_halt(self):
        rm = RiskManager(RiskLimits(max_daily_loss_r=-5.0))
        rm.daily_reset(date(2024, 1, 5))

        rm.on_trade_exit(-3.0)
        assert not rm.is_halted()

        rm.on_trade_exit(-3.0)  # total: -6.0
        assert rm.is_halted()


# ============================================================================
# Max Concurrent Tests
# ============================================================================

class TestMaxConcurrent:

    def test_rejects_at_limit(self):
        rm = RiskManager(RiskLimits(max_concurrent_positions=2))
        rm.daily_reset(date(2024, 1, 5))

        trades = [_entered("2300"), _entered("1800")]
        allowed, reason = rm.can_enter("s3", "0030", trades, 0.0)
        assert not allowed
        assert "max_concurrent" in reason

    def test_allows_below_limit(self):
        rm = RiskManager(RiskLimits(max_concurrent_positions=3))
        rm.daily_reset(date(2024, 1, 5))

        trades = [_entered("2300"), _entered("1800")]
        allowed, _ = rm.can_enter("s3", "0030", trades, 0.0)
        assert allowed

    def test_armed_trades_not_counted(self):
        """ARMED trades are not in a position yet."""
        rm = RiskManager(RiskLimits(max_concurrent_positions=2))
        rm.daily_reset(date(2024, 1, 5))

        trades = [_entered("2300"), _armed("1800")]
        allowed, _ = rm.can_enter("s3", "0030", trades, 0.0)
        assert allowed


# ============================================================================
# Max Per ORB Tests
# ============================================================================

class TestMaxPerOrb:

    def test_rejects_same_orb(self):
        rm = RiskManager(RiskLimits(max_per_orb_positions=1))
        rm.daily_reset(date(2024, 1, 5))

        trades = [_entered("2300")]
        allowed, reason = rm.can_enter("s2", "2300", trades, 0.0)
        assert not allowed
        assert "max_per_orb" in reason

    def test_allows_different_orb(self):
        rm = RiskManager(RiskLimits(max_per_orb_positions=1))
        rm.daily_reset(date(2024, 1, 5))

        trades = [_entered("2300")]
        allowed, _ = rm.can_enter("s2", "1800", trades, 0.0)
        assert allowed

    def test_allows_two_per_orb(self):
        rm = RiskManager(RiskLimits(max_per_orb_positions=2))
        rm.daily_reset(date(2024, 1, 5))

        trades = [_entered("2300")]
        allowed, _ = rm.can_enter("s2", "2300", trades, 0.0)
        assert allowed


# ============================================================================
# Max Daily Trades Tests
# ============================================================================

class TestMaxDailyTrades:

    def test_rejects_at_limit(self):
        rm = RiskManager(RiskLimits(max_daily_trades=3))
        rm.daily_reset(date(2024, 1, 5))

        rm.on_trade_entry()
        rm.on_trade_entry()
        rm.on_trade_entry()

        allowed, reason = rm.can_enter("s4", "0030", [], 0.0)
        assert not allowed
        assert "max_daily_trades" in reason

    def test_allows_below_limit(self):
        rm = RiskManager(RiskLimits(max_daily_trades=3))
        rm.daily_reset(date(2024, 1, 5))

        rm.on_trade_entry()
        rm.on_trade_entry()

        allowed, _ = rm.can_enter("s4", "0030", [], 0.0)
        assert allowed

    def test_resets_on_new_day(self):
        rm = RiskManager(RiskLimits(max_daily_trades=2))
        rm.daily_reset(date(2024, 1, 5))

        rm.on_trade_entry()
        rm.on_trade_entry()
        allowed, _ = rm.can_enter("s1", "2300", [], 0.0)
        assert not allowed

        rm.daily_reset(date(2024, 1, 6))
        allowed, _ = rm.can_enter("s1", "2300", [], 0.0)
        assert allowed


# ============================================================================
# Drawdown Warning Tests
# ============================================================================

class TestDrawdownWarning:

    def test_warning_logged_but_allows(self):
        rm = RiskManager(RiskLimits(drawdown_warning_r=-3.0, max_daily_loss_r=-5.0))
        rm.daily_reset(date(2024, 1, 5))

        allowed, _ = rm.can_enter("s1", "2300", [], -3.5)
        assert allowed
        assert len(rm.warnings) == 1
        assert "drawdown_warning" in rm.warnings[0]

    def test_no_warning_above_threshold(self):
        rm = RiskManager(RiskLimits(drawdown_warning_r=-3.0))
        rm.daily_reset(date(2024, 1, 5))

        rm.can_enter("s1", "2300", [], -2.0)
        assert len(rm.warnings) == 0


# ============================================================================
# Status and Daily Reset Tests
# ============================================================================

class TestStatus:

    def test_get_status(self):
        rm = RiskManager(RiskLimits())
        rm.daily_reset(date(2024, 1, 5))
        rm.on_trade_entry()
        rm.on_trade_exit(-1.0)

        status = rm.get_status()
        assert status["trading_day"] == date(2024, 1, 5)
        assert status["daily_pnl_r"] == -1.0
        assert status["daily_trade_count"] == 1
        assert status["halted"] is False

    def test_daily_reset_clears_everything(self):
        rm = RiskManager(RiskLimits())
        rm.daily_reset(date(2024, 1, 5))
        rm.on_trade_entry()
        rm.on_trade_exit(-2.0)
        rm.can_enter("s1", "2300", [], -3.5)  # triggers warning

        rm.daily_reset(date(2024, 1, 6))
        status = rm.get_status()
        assert status["daily_pnl_r"] == 0.0
        assert status["daily_trade_count"] == 0
        assert status["halted"] is False
        assert status["warnings"] == 0
