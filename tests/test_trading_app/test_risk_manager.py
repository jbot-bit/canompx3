"""
Tests for trading_app.risk_manager module.
"""

import sys
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path

import pytest

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
    orb_minutes: int | None = None  # None for backward-compat tests
    strategy_id: str = ""  # Needed for correlation-weighted checks


def _entered(orb="US_DATA_830", orb_minutes=None):
    return _FakeTrade(orb_label=orb, state=_State.ENTERED, orb_minutes=orb_minutes)


def _armed(orb="US_DATA_830", orb_minutes=None):
    return _FakeTrade(orb_label=orb, state=_State.ARMED, orb_minutes=orb_minutes)


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
        with pytest.raises(Exception):  # noqa: B017
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

        allowed, reason, _ = rm.can_enter("s1", "US_DATA_830", [], -5.0)
        assert not allowed
        assert "circuit_breaker" in reason

    def test_halts_below_limit(self):
        rm = RiskManager(RiskLimits(max_daily_loss_r=-5.0))
        rm.daily_reset(date(2024, 1, 5))

        allowed, _, _ = rm.can_enter("s1", "US_DATA_830", [], -6.0)
        assert not allowed

    def test_allows_above_limit(self):
        rm = RiskManager(RiskLimits(max_daily_loss_r=-5.0))
        rm.daily_reset(date(2024, 1, 5))

        allowed, reason, _ = rm.can_enter("s1", "US_DATA_830", [], -4.9)
        assert allowed
        assert reason == ""

    def test_halt_persists_after_trigger(self):
        rm = RiskManager(RiskLimits(max_daily_loss_r=-5.0))
        rm.daily_reset(date(2024, 1, 5))

        # Trigger halt
        rm.can_enter("s1", "US_DATA_830", [], -5.0)
        assert rm.is_halted()

        # Even with 0 PnL, still halted
        allowed, _, _ = rm.can_enter("s2", "LONDON_METALS", [], 0.0)
        assert not allowed

    def test_halt_clears_on_daily_reset(self):
        rm = RiskManager(RiskLimits(max_daily_loss_r=-5.0))
        rm.daily_reset(date(2024, 1, 5))
        rm.can_enter("s1", "US_DATA_830", [], -5.0)
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

        trades = [_entered("US_DATA_830"), _entered("LONDON_METALS")]
        allowed, reason, _ = rm.can_enter("s3", "NYSE_OPEN", trades, 0.0)
        assert not allowed
        assert "max_concurrent" in reason

    def test_allows_below_limit(self):
        rm = RiskManager(RiskLimits(max_concurrent_positions=3))
        rm.daily_reset(date(2024, 1, 5))

        trades = [_entered("US_DATA_830"), _entered("LONDON_METALS")]
        allowed, _, _ = rm.can_enter("s3", "NYSE_OPEN", trades, 0.0)
        assert allowed

    def test_armed_trades_not_counted(self):
        """ARMED trades are not in a position yet."""
        rm = RiskManager(RiskLimits(max_concurrent_positions=2))
        rm.daily_reset(date(2024, 1, 5))

        trades = [_entered("US_DATA_830"), _armed("LONDON_METALS")]
        allowed, _, _ = rm.can_enter("s3", "NYSE_OPEN", trades, 0.0)
        assert allowed


# ============================================================================
# Max Per ORB Tests
# ============================================================================


class TestMaxPerOrb:
    def test_rejects_same_orb(self):
        rm = RiskManager(RiskLimits(max_per_orb_positions=1))
        rm.daily_reset(date(2024, 1, 5))

        trades = [_entered("US_DATA_830")]
        allowed, reason, _ = rm.can_enter("s2", "US_DATA_830", trades, 0.0)
        assert not allowed
        assert "max_per_orb" in reason

    def test_allows_different_orb(self):
        rm = RiskManager(RiskLimits(max_per_orb_positions=1))
        rm.daily_reset(date(2024, 1, 5))

        trades = [_entered("US_DATA_830")]
        allowed, _, _ = rm.can_enter("s2", "LONDON_METALS", trades, 0.0)
        assert allowed

    def test_allows_two_per_orb(self):
        rm = RiskManager(RiskLimits(max_per_orb_positions=2))
        rm.daily_reset(date(2024, 1, 5))

        trades = [_entered("US_DATA_830")]
        allowed, _, _ = rm.can_enter("s2", "US_DATA_830", trades, 0.0)
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

        allowed, reason, _ = rm.can_enter("s4", "NYSE_OPEN", [], 0.0)
        assert not allowed
        assert "max_daily_trades" in reason

    def test_allows_below_limit(self):
        rm = RiskManager(RiskLimits(max_daily_trades=3))
        rm.daily_reset(date(2024, 1, 5))

        rm.on_trade_entry()
        rm.on_trade_entry()

        allowed, _, _ = rm.can_enter("s4", "NYSE_OPEN", [], 0.0)
        assert allowed

    def test_resets_on_new_day(self):
        rm = RiskManager(RiskLimits(max_daily_trades=2))
        rm.daily_reset(date(2024, 1, 5))

        rm.on_trade_entry()
        rm.on_trade_entry()
        allowed, _, _ = rm.can_enter("s1", "US_DATA_830", [], 0.0)
        assert not allowed

        rm.daily_reset(date(2024, 1, 6))
        allowed, _, _ = rm.can_enter("s1", "US_DATA_830", [], 0.0)
        assert allowed


# ============================================================================
# Drawdown Warning Tests
# ============================================================================


class TestDrawdownWarning:
    def test_warning_logged_but_allows(self):
        rm = RiskManager(RiskLimits(drawdown_warning_r=-3.0, max_daily_loss_r=-5.0))
        rm.daily_reset(date(2024, 1, 5))

        allowed, _, _ = rm.can_enter("s1", "US_DATA_830", [], -3.5)
        assert allowed
        assert len(rm.warnings) == 1
        assert "drawdown_warning" in rm.warnings[0]

    def test_no_warning_above_threshold(self):
        rm = RiskManager(RiskLimits(drawdown_warning_r=-3.0))
        rm.daily_reset(date(2024, 1, 5))

        rm.can_enter("s1", "US_DATA_830", [], -2.0)
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
        rm.can_enter("s1", "US_DATA_830", [], -3.5)  # triggers warning

        rm.daily_reset(date(2024, 1, 6))
        status = rm.get_status()
        assert status["daily_pnl_r"] == 0.0
        assert status["daily_trade_count"] == 0
        assert status["halted"] is False
        assert status["warnings"] == 0


# ============================================================================
# Multi-Day Equity Drawdown Tests
# ============================================================================


class TestEquityDrawdown:
    def test_cumulative_pnl_tracks_across_days(self):
        rm = RiskManager(RiskLimits(max_equity_drawdown_r=-10.0))
        rm.daily_reset(date(2024, 1, 5))
        rm.on_trade_exit(2.0)
        rm.on_trade_exit(1.0)
        rm.daily_reset(date(2024, 1, 6))
        rm.on_trade_exit(-1.5)

        assert rm.cumulative_pnl_r == pytest.approx(1.5)
        assert rm.equity_high_water_r == pytest.approx(3.0)

    def test_drawdown_halts_when_breached(self):
        rm = RiskManager(RiskLimits(max_equity_drawdown_r=-5.0))
        rm.daily_reset(date(2024, 1, 5))
        rm.on_trade_exit(3.0)  # HWM = 3.0
        rm.daily_reset(date(2024, 1, 6))
        rm.on_trade_exit(-4.0)  # cum = -1.0, DD = -4.0 from peak 3.0
        assert not rm._equity_halted

        rm.on_trade_exit(-4.5)  # cum = -5.5, DD = -8.5 from peak 3.0
        assert rm._equity_halted

    def test_equity_halt_blocks_can_enter(self):
        rm = RiskManager(RiskLimits(max_equity_drawdown_r=-3.0))
        rm.daily_reset(date(2024, 1, 5))
        rm.on_trade_exit(1.0)  # HWM = 1.0
        rm.on_trade_exit(-4.5)  # cum = -3.5, DD = -4.5 from peak
        assert rm._equity_halted

        rm.daily_reset(date(2024, 1, 6))
        allowed, reason, _ = rm.can_enter("s1", "US_DATA_830", [], 0.0)
        assert not allowed
        assert "equity_drawdown" in reason

    def test_equity_halt_persists_across_daily_reset(self):
        rm = RiskManager(RiskLimits(max_equity_drawdown_r=-2.0))
        rm.daily_reset(date(2024, 1, 5))
        rm.on_trade_exit(-2.5)  # DD = -2.5 from peak 0.0
        assert rm._equity_halted

        rm.daily_reset(date(2024, 1, 6))
        assert rm._equity_halted
        assert rm.is_halted()

    def test_equity_reset_clears_halt(self):
        rm = RiskManager(RiskLimits(max_equity_drawdown_r=-2.0))
        rm.daily_reset(date(2024, 1, 5))
        rm.on_trade_exit(-3.0)
        assert rm._equity_halted

        rm.equity_reset()
        assert not rm._equity_halted
        assert rm.cumulative_pnl_r == 0.0
        assert rm.equity_high_water_r == 0.0

    def test_disabled_when_none(self):
        rm = RiskManager(RiskLimits(max_equity_drawdown_r=None, max_daily_loss_r=-100.0))
        rm.daily_reset(date(2024, 1, 5))
        rm.on_trade_exit(-50.0)
        assert not rm._equity_halted
        allowed, _, _ = rm.can_enter("s1", "US_DATA_830", [], rm.daily_pnl_r)
        assert allowed

    def test_hwm_updates_on_new_peak(self):
        rm = RiskManager(RiskLimits(max_equity_drawdown_r=-10.0))
        rm.daily_reset(date(2024, 1, 5))
        rm.on_trade_exit(5.0)
        assert rm.equity_high_water_r == pytest.approx(5.0)
        rm.on_trade_exit(-2.0)
        assert rm.equity_high_water_r == pytest.approx(5.0)
        rm.on_trade_exit(3.0)  # cum = 6.0
        assert rm.equity_high_water_r == pytest.approx(6.0)

    def test_get_status_includes_equity(self):
        rm = RiskManager(RiskLimits(max_equity_drawdown_r=-10.0))
        rm.daily_reset(date(2024, 1, 5))
        rm.on_trade_exit(5.0)
        rm.on_trade_exit(-2.0)

        status = rm.get_status()
        assert status["cumulative_pnl_r"] == pytest.approx(3.0)
        assert status["equity_high_water_r"] == pytest.approx(5.0)
        assert status["equity_drawdown_r"] == pytest.approx(-2.0)
        assert status["equity_halted"] is False


# ============================================================================
# Aperture-Aware Multi-RR Risk Tests
# ============================================================================


class TestApertureAware:
    """Tests for multi-aperture position management (O5 + O30 same session)."""

    def test_different_apertures_allowed_concurrently(self):
        """O5 and O30 on same session are different ORBs — both allowed."""
        rm = RiskManager(RiskLimits(max_per_orb_positions=1, max_per_session_positions=2))
        rm.daily_reset(date(2024, 1, 5))

        trades = [_entered("NYSE_OPEN", orb_minutes=5)]
        allowed, reason, _ = rm.can_enter("s2", "NYSE_OPEN", trades, 0.0, orb_minutes=30)
        assert allowed, f"O30 should be allowed alongside O5: {reason}"

    def test_same_aperture_blocked(self):
        """Two O5 on same session blocked by max_per_orb."""
        rm = RiskManager(RiskLimits(max_per_orb_positions=1, max_per_session_positions=2))
        rm.daily_reset(date(2024, 1, 5))

        trades = [_entered("NYSE_OPEN", orb_minutes=5)]
        allowed, reason, _ = rm.can_enter("s2", "NYSE_OPEN", trades, 0.0, orb_minutes=5)
        assert not allowed
        assert "max_per_orb" in reason

    def test_third_aperture_blocked_by_per_session(self):
        """O5 + O30 active, O15 blocked by max_per_session=2."""
        rm = RiskManager(RiskLimits(max_per_orb_positions=1, max_per_session_positions=2))
        rm.daily_reset(date(2024, 1, 5))

        trades = [
            _entered("NYSE_OPEN", orb_minutes=5),
            _entered("NYSE_OPEN", orb_minutes=30),
        ]
        allowed, reason, _ = rm.can_enter("s3", "NYSE_OPEN", trades, 0.0, orb_minutes=15)
        assert not allowed
        assert "max_per_session" in reason

    def test_half_size_when_concurrent_different_aperture(self):
        """contract_factor=0.5 returned when same-session different-aperture active."""
        rm = RiskManager(RiskLimits(max_per_orb_positions=1, max_per_session_positions=2))
        rm.daily_reset(date(2024, 1, 5))

        trades = [_entered("NYSE_OPEN", orb_minutes=5)]
        allowed, reason, factor = rm.can_enter("s2", "NYSE_OPEN", trades, 0.0, orb_minutes=30)
        assert allowed
        assert factor == 0.5

    def test_full_size_when_no_concurrent(self):
        """contract_factor=1.0 when no existing position on that session."""
        rm = RiskManager(RiskLimits(max_per_orb_positions=1, max_per_session_positions=2))
        rm.daily_reset(date(2024, 1, 5))

        trades = [_entered("LONDON_METALS", orb_minutes=5)]
        allowed, reason, factor = rm.can_enter("s2", "NYSE_OPEN", trades, 0.0, orb_minutes=30)
        assert allowed
        assert factor == 1.0

    def test_backward_compat_no_orb_minutes(self):
        """Without orb_minutes param, behaves like old per-ORB check."""
        rm = RiskManager(RiskLimits(max_per_orb_positions=1, max_per_session_positions=2))
        rm.daily_reset(date(2024, 1, 5))

        # _FakeTrade without orb_minutes set (None)
        trades = [_entered("NYSE_OPEN")]
        allowed, reason, _ = rm.can_enter("s2", "NYSE_OPEN", trades, 0.0)
        assert not allowed
        assert "max_per_orb" in reason

    def test_different_sessions_unaffected(self):
        """Aperture logic only applies within same session."""
        rm = RiskManager(RiskLimits(max_per_orb_positions=1, max_per_session_positions=2))
        rm.daily_reset(date(2024, 1, 5))

        trades = [_entered("NYSE_OPEN", orb_minutes=5)]
        allowed, _, factor = rm.can_enter("s2", "US_DATA_1000", trades, 0.0, orb_minutes=30)
        assert allowed
        assert factor == 1.0  # Different session — no half-sizing

    def test_defaults_include_max_per_session(self):
        """RiskLimits default includes max_per_session_positions=2."""
        limits = RiskLimits()
        assert limits.max_per_session_positions == 2

    def test_corr_weighted_plus_aperture_half_sizing(self):
        """Correlation reduction (0.7) further reduced to 0.5 by aperture half-sizing."""
        from trading_app.execution_engine import TradeState

        class _CorrTrade:
            def __init__(self, sid, orb, om):
                self.strategy_id = sid
                self.orb_label = orb
                self.orb_minutes = om
                self.state = TradeState.ENTERED

        corr_lookup = {("s_new", "s_existing"): 0.8}
        limits = RiskLimits(
            max_concurrent_positions=3,
            max_per_orb_positions=1,
            max_per_session_positions=2,
        )
        rm = RiskManager(limits, corr_lookup=corr_lookup)
        rm.daily_reset(date(2024, 1, 5))

        active = [_CorrTrade("s_existing", "NYSE_OPEN", 5)]
        allowed, _, factor = rm.can_enter("s_new", "NYSE_OPEN", active, 0.0, orb_minutes=30)
        assert allowed
        # Correlation gives some factor, aperture caps at 0.5
        assert factor <= 0.5


# ─── F-2: Same-instrument opposite-direction guard (cross-account hedging) ──
# @canonical-source docs/research-input/topstep/topstep_cross_account_hedging.md
# @verbatim "Cross-account hedging occurs when you hold opposite positions across
#            multiple accounts at the same time. This means you're simultaneously
#            long and short the same instrument."
# @verbatim "Yes! You can trade the same instrument across multiple accounts.
#            What's prohibited is holding opposite positions simultaneously."


@dataclass
class _FakeStrategy:
    instrument: str


@dataclass
class _HedgeTrade:
    """Trade fixture with strategy.instrument and direction for F-2 tests."""

    strategy_id: str
    orb_label: str
    direction: str  # "long" or "short"
    strategy: _FakeStrategy
    state: _State = _State.ENTERED


class TestF2HedgingGuard:
    """F-2: refuse entries opposite an existing same-instrument position."""

    def _make_rm(self) -> RiskManager:
        limits = RiskLimits(
            max_concurrent_positions=10,  # generous so we hit the hedge check first
            max_per_orb_positions=10,
            max_per_session_positions=10,
            max_daily_trades=20,
        )
        rm = RiskManager(limits)
        rm.daily_reset(date(2026, 4, 8))
        return rm

    def _trade(self, strategy_id: str, instrument: str, direction: str, session: str = "NYSE_OPEN") -> _HedgeTrade:
        return _HedgeTrade(
            strategy_id=strategy_id,
            orb_label=session,
            direction=direction,
            strategy=_FakeStrategy(instrument=instrument),
        )

    def test_long_then_short_same_instrument_blocked(self):
        rm = self._make_rm()
        active = [self._trade("MNQ_NYSE_OPEN_E2", "MNQ", "long")]
        allowed, reason, _ = rm.can_enter(
            strategy_id="MNQ_TOKYO_OPEN_E2",
            orb_label="TOKYO_OPEN",
            active_trades=active,
            daily_pnl_r=0.0,
            instrument="MNQ",
            direction="short",
        )
        assert allowed is False
        assert "hedging_guard" in reason
        assert "MNQ" in reason

    def test_short_then_long_same_instrument_blocked(self):
        rm = self._make_rm()
        active = [self._trade("MNQ_NYSE_OPEN_E2", "MNQ", "short")]
        allowed, reason, _ = rm.can_enter(
            strategy_id="MNQ_TOKYO_OPEN_E2",
            orb_label="TOKYO_OPEN",
            active_trades=active,
            daily_pnl_r=0.0,
            instrument="MNQ",
            direction="long",
        )
        assert allowed is False
        assert "hedging_guard" in reason

    def test_long_then_long_same_instrument_allowed(self):
        rm = self._make_rm()
        active = [self._trade("MNQ_NYSE_OPEN_E2", "MNQ", "long")]
        allowed, _, _ = rm.can_enter(
            strategy_id="MNQ_TOKYO_OPEN_E2",
            orb_label="TOKYO_OPEN",
            active_trades=active,
            daily_pnl_r=0.0,
            instrument="MNQ",
            direction="long",
        )
        assert allowed is True

    def test_long_mnq_then_short_mgc_allowed(self):
        """Different instruments → no hedge."""
        rm = self._make_rm()
        active = [self._trade("MNQ_NYSE_OPEN_E2", "MNQ", "long")]
        allowed, _, _ = rm.can_enter(
            strategy_id="MGC_CME_REOPEN_E2",
            orb_label="CME_REOPEN",
            active_trades=active,
            daily_pnl_r=0.0,
            instrument="MGC",
            direction="short",
        )
        assert allowed is True

    def test_long_then_short_after_exit_allowed(self):
        """Once the long position EXITED, a new short is fine."""
        rm = self._make_rm()
        exited = self._trade("MNQ_NYSE_OPEN_E2", "MNQ", "long")
        exited.state = _State.EXITED
        active = [exited]
        allowed, _, _ = rm.can_enter(
            strategy_id="MNQ_TOKYO_OPEN_E2",
            orb_label="TOKYO_OPEN",
            active_trades=active,
            daily_pnl_r=0.0,
            instrument="MNQ",
            direction="short",
        )
        assert allowed is True

    def test_no_instrument_disables_check(self):
        """Backward compat: callers that don't pass instrument get the old behavior."""
        rm = self._make_rm()
        active = [self._trade("MNQ_NYSE_OPEN_E2", "MNQ", "long")]
        allowed, _, _ = rm.can_enter(
            strategy_id="MNQ_TOKYO_OPEN_E2",
            orb_label="TOKYO_OPEN",
            active_trades=active,
            daily_pnl_r=0.0,
            # no instrument/direction → check skipped
        )
        assert allowed is True

    def test_armed_position_does_not_count(self):
        """Only ENTERED positions count for hedge detection (ARMED hasn't filled yet)."""
        rm = self._make_rm()
        armed = self._trade("MNQ_NYSE_OPEN_E2", "MNQ", "long")
        armed.state = _State.ARMED
        active = [armed]
        allowed, _, _ = rm.can_enter(
            strategy_id="MNQ_TOKYO_OPEN_E2",
            orb_label="TOKYO_OPEN",
            active_trades=active,
            daily_pnl_r=0.0,
            instrument="MNQ",
            direction="short",
        )
        assert allowed is True

    def test_two_existing_positions_block_on_first_match(self):
        """If multiple longs exist on same instrument, short is blocked."""
        rm = self._make_rm()
        active = [
            self._trade("MNQ_NYSE_OPEN_E2", "MNQ", "long"),
            self._trade("MNQ_TOKYO_OPEN_E2", "MNQ", "long"),
        ]
        allowed, reason, _ = rm.can_enter(
            strategy_id="MNQ_EUROPE_FLOW_E2",
            orb_label="EUROPE_FLOW",
            active_trades=active,
            daily_pnl_r=0.0,
            instrument="MNQ",
            direction="short",
        )
        assert allowed is False
        assert "hedging_guard" in reason


# ─── F-1: TopStep XFA Scaling Plan integration with can_enter ──────────
# @canonical-source docs/research-input/topstep/topstep_scaling_plan_article.md
# @canonical-image docs/research-input/topstep/images/xfa_scaling_chart.png


@dataclass
class _ContractsTrade:
    """Trade fixture for F-1 tests — has contracts + strategy.instrument."""

    strategy_id: str
    orb_label: str
    direction: str
    contracts: int
    strategy: _FakeStrategy
    state: _State = _State.ENTERED


class TestF1ScalingPlanIntegration:
    """F-1 inside RiskManager.can_enter — uses topstep_xfa_account_size."""

    def _make_rm(self, account_size: int = 50_000) -> RiskManager:
        limits = RiskLimits(
            max_concurrent_positions=10,
            max_per_orb_positions=10,
            max_per_session_positions=10,
            max_daily_trades=20,
            topstep_xfa_account_size=account_size,
        )
        rm = RiskManager(limits)
        rm.daily_reset(date(2026, 4, 8))
        return rm

    def _trade(self, sid: str, instrument: str, direction: str, contracts: int = 1) -> _ContractsTrade:
        return _ContractsTrade(
            strategy_id=sid,
            orb_label="NYSE_OPEN",
            direction=direction,
            contracts=contracts,
            strategy=_FakeStrategy(instrument=instrument),
        )

    def test_no_balance_set_fails_closed(self):
        """If orchestrator hasn't set EOD balance, can_enter must refuse."""
        rm = self._make_rm()
        # NOTE: rm._topstep_xfa_eod_balance is intentionally None
        allowed, reason, _ = rm.can_enter(
            strategy_id="MNQ_NYSE_OPEN_E2",
            orb_label="NYSE_OPEN",
            active_trades=[],
            daily_pnl_r=0.0,
            instrument="MNQ",
            direction="long",
        )
        assert allowed is False
        assert "EOD XFA balance unknown" in reason

    def test_day_one_50k_at_cap_allows_first_two(self):
        """Day-1 50K XFA: cap=2, current=1 micro = 1 mini-equiv. New 1-micro entry → 2 total ≤ 2 → OK."""
        rm = self._make_rm(50_000)
        rm.set_topstep_xfa_eod_balance(0.0)
        active = [self._trade("MNQ_NYSE_OPEN_E2", "MNQ", "long", contracts=1)]
        allowed, _, _ = rm.can_enter(
            strategy_id="MNQ_TOKYO_OPEN_E2",
            orb_label="TOKYO_OPEN",
            active_trades=active,
            daily_pnl_r=0.0,
            instrument="MNQ",
            direction="long",  # same direction → not blocked by F-2
        )
        assert allowed is True

    def test_day_one_50k_third_lane_blocked(self):
        """Day-1 50K XFA: cap=2. 2 active + 1 new = 3 > 2 → blocked by F-1."""
        rm = self._make_rm(50_000)
        rm.set_topstep_xfa_eod_balance(0.0)
        active = [
            self._trade("MNQ_NYSE_OPEN_E2", "MNQ", "long", contracts=1),
            self._trade("MNQ_TOKYO_OPEN_E2", "MNQ", "long", contracts=1),
        ]
        allowed, reason, _ = rm.can_enter(
            strategy_id="MNQ_EUROPE_FLOW_E2",
            orb_label="EUROPE_FLOW",
            active_trades=active,
            daily_pnl_r=0.0,
            instrument="MNQ",
            direction="long",
        )
        assert allowed is False
        assert "topstep_scaling_plan" in reason
        assert "day_max 2" in reason

    def test_after_2k_profit_5_lanes_allowed(self):
        """After $2K profit (top tier), 50K XFA cap=5. 4 active + 1 new = 5 ≤ 5 → OK."""
        rm = self._make_rm(50_000)
        rm.set_topstep_xfa_eod_balance(2_000.0)
        active = [
            self._trade(f"MNQ_S{i}", "MNQ", "long", contracts=1) for i in range(4)
        ]
        allowed, _, _ = rm.can_enter(
            strategy_id="MNQ_S5",
            orb_label="EUROPE_FLOW",
            active_trades=active,
            daily_pnl_r=0.0,
            instrument="MNQ",
            direction="long",
        )
        assert allowed is True

    def test_after_2k_profit_6th_lane_blocked(self):
        """At top tier (cap=5), 5 active + 1 new = 6 > 5 → blocked."""
        rm = self._make_rm(50_000)
        rm.set_topstep_xfa_eod_balance(2_000.0)
        active = [
            self._trade(f"MNQ_S{i}", "MNQ", "long", contracts=1) for i in range(5)
        ]
        allowed, _, _ = rm.can_enter(
            strategy_id="MNQ_S6",
            orb_label="EUROPE_FLOW",
            active_trades=active,
            daily_pnl_r=0.0,
            instrument="MNQ",
            direction="long",
        )
        assert allowed is False

    def test_disabled_when_account_size_none(self):
        """Non-TopStep deployments don't use topstep_xfa_account_size → check skipped."""
        limits = RiskLimits(
            max_concurrent_positions=10,
            max_per_orb_positions=10,
            max_per_session_positions=10,
            max_daily_trades=20,
            # topstep_xfa_account_size left at default None
        )
        rm = RiskManager(limits)
        rm.daily_reset(date(2026, 4, 8))
        # No set_topstep_xfa_eod_balance call. Should still allow entries.
        allowed, _, _ = rm.can_enter(
            strategy_id="MNQ_S1",
            orb_label="NYSE_OPEN",
            active_trades=[],
            daily_pnl_r=0.0,
            instrument="MNQ",
            direction="long",
        )
        assert allowed is True

    def test_100k_day1_allows_3_blocks_4(self):
        """100K XFA Day-1: cap=3. 2 active + 1 new = 3 ≤ 3 OK; 3 active + 1 new = 4 > 3 blocked."""
        rm = self._make_rm(100_000)
        rm.set_topstep_xfa_eod_balance(0.0)

        # 3 lots OK
        active2 = [self._trade(f"S{i}", "MNQ", "long", contracts=1) for i in range(2)]
        allowed, _, _ = rm.can_enter("S3", "NYSE_OPEN", active2, 0.0, instrument="MNQ", direction="long")
        assert allowed is True

        # 4 lots blocked
        active3 = [self._trade(f"S{i}", "MNQ", "long", contracts=1) for i in range(3)]
        allowed, reason, _ = rm.can_enter("S4", "NYSE_OPEN", active3, 0.0, instrument="MNQ", direction="long")
        assert allowed is False
        assert "100K XFA" in reason
