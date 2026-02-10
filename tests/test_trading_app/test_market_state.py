"""Tests for MarketState, scoring, and cascade_table (all DB-free)."""

import pytest
from datetime import date, datetime, timezone
from dataclasses import dataclass

from trading_app.market_state import (
    MarketState, OrbSnapshot, SessionSignals, RegimeContext, ORB_LABELS,
)
from trading_app.scoring import score_strategy, ScoringWeights, MIN_SCORE_THRESHOLD
from trading_app.cascade_table import lookup_cascade


# =========================================================================
# Fixtures
# =========================================================================

@dataclass(frozen=True)
class FakeStrategy:
    """Minimal strategy stub for scoring tests."""
    strategy_id: str
    orb_label: str
    expectancy_r: float
    win_rate: float
    entry_model: str = "E1"
    rr_target: float = 2.5
    confirm_bars: int = 2
    filter_type: str = "G4"
    sample_size: int = 100
    sharpe_ratio: float = 0.2
    max_drawdown_r: float = 5.0
    median_risk_points: float = 3.0
    source: str = "baseline"


def _make_state(**orb_kwargs) -> MarketState:
    """Build a MarketState with specified ORBs (no DB)."""
    state = MarketState(trading_day=date(2025, 6, 15))
    for label in ORB_LABELS:
        state.orbs[label] = OrbSnapshot(label=label)
    # Apply overrides
    for label, data in orb_kwargs.items():
        orb = state.orbs[label]
        for k, v in data.items():
            setattr(orb, k, v)
    return state


# =========================================================================
# Fix 1: visible_sessions masks future outcomes
# =========================================================================

class TestVisibleSessions:
    def test_masks_future_outcomes(self):
        """Outcomes for non-visible sessions are None."""
        state = _make_state(
            **{
                "0900": {"outcome": "win", "high": 100.0, "low": 95.0, "size": 5.0, "complete": True},
                "1800": {"outcome": "loss", "high": 200.0, "low": 195.0, "size": 5.0, "complete": True},
            }
        )
        # Simulate visible_sessions masking (what from_trading_day does)
        visible = {"0900"}
        for label, orb in state.orbs.items():
            if label not in visible:
                orb.outcome = None

        assert state.orbs["0900"].outcome == "win"
        assert state.orbs["1800"].outcome is None
        # Structural fields remain
        assert state.orbs["1800"].high == 200.0
        assert state.orbs["1800"].size == 5.0

    def test_empty_visible_sessions_masks_all(self):
        """Empty set masks all outcomes."""
        state = _make_state(
            **{
                "0900": {"outcome": "win", "high": 100.0, "low": 95.0, "size": 5.0},
                "1000": {"outcome": "loss", "high": 100.0, "low": 95.0, "size": 5.0},
            }
        )
        visible = set()
        for label, orb in state.orbs.items():
            if label not in visible:
                orb.outcome = None

        assert all(orb.outcome is None for orb in state.orbs.values())

    def test_none_visible_sessions_preserves_all(self):
        """None (default) preserves all outcomes."""
        state = _make_state(
            **{
                "0900": {"outcome": "win"},
                "1800": {"outcome": "loss"},
            }
        )
        # visible_sessions=None -> no masking
        assert state.orbs["0900"].outcome == "win"
        assert state.orbs["1800"].outcome == "loss"

    def test_progressive_reveal(self):
        """Outcomes revealed one at a time as trades resolve."""
        state = _make_state(
            **{
                "0900": {"outcome": "win", "high": 100.0, "low": 95.0, "size": 5.0,
                          "break_dir": "long", "complete": True},
                "1000": {"outcome": "loss", "high": 100.0, "low": 95.0, "size": 5.0,
                          "break_dir": "short", "complete": True},
            }
        )
        # Start: nothing visible
        for orb in state.orbs.values():
            orb.outcome = None

        # Reveal 0900
        state.orbs["0900"].outcome = "win"
        state.update_signals()
        assert state.signals.prior_outcomes == {"0900": "win"}

        # Reveal 1000
        state.orbs["1000"].outcome = "loss"
        state.update_signals()
        assert "1000" in state.signals.prior_outcomes


# =========================================================================
# Signals: reversal, chop, continuation
# =========================================================================

class TestSignals:
    def test_reversal_detected(self):
        """0900 loss + 1000 opposite direction = reversal."""
        state = _make_state(
            **{
                "0900": {"outcome": "loss", "break_dir": "long", "complete": True,
                          "high": 100.0, "low": 95.0, "size": 5.0},
                "1000": {"outcome": "win", "break_dir": "short", "complete": True,
                          "high": 100.0, "low": 95.0, "size": 5.0},
            }
        )
        state.update_signals()
        assert state.signals.reversal_active is True
        assert state.signals.chop_detected is False

    def test_chop_detected(self):
        """0900 loss + 1000 loss = chop."""
        state = _make_state(
            **{
                "0900": {"outcome": "loss", "break_dir": "long", "complete": True,
                          "high": 100.0, "low": 95.0, "size": 5.0},
                "1000": {"outcome": "loss", "break_dir": "long", "complete": True,
                          "high": 100.0, "low": 95.0, "size": 5.0},
            }
        )
        state.update_signals()
        assert state.signals.chop_detected is True

    def test_continuation_detected(self):
        """0900 win + 1000 same direction = continuation."""
        state = _make_state(
            **{
                "0900": {"outcome": "win", "break_dir": "long", "complete": True,
                          "high": 100.0, "low": 95.0, "size": 5.0},
                "1000": {"outcome": "win", "break_dir": "long", "complete": True,
                          "high": 100.0, "low": 95.0, "size": 5.0},
            }
        )
        state.update_signals()
        assert state.signals.continuation is True

    def test_no_signals_when_outcomes_masked(self):
        """No signals when outcomes are not yet visible."""
        state = _make_state(
            **{
                "0900": {"break_dir": "long", "complete": True,
                          "high": 100.0, "low": 95.0, "size": 5.0},
                "1000": {"break_dir": "short", "complete": True,
                          "high": 100.0, "low": 95.0, "size": 5.0},
            }
        )
        # outcomes are None (default)
        state.update_signals()
        assert state.signals.reversal_active is False
        assert state.signals.chop_detected is False
        assert state.signals.continuation is False


# =========================================================================
# Scoring
# =========================================================================

class TestScoring:
    def test_orb_size_below_4_dampens(self):
        """ORB size < 4 applies 0.5x multiplier."""
        state = _make_state(**{"0900": {"size": 2.0}})
        strategy = FakeStrategy("S1", "0900", expectancy_r=0.4, win_rate=0.45)
        score = score_strategy(strategy, state)
        assert score == pytest.approx(0.4 * 0.5, abs=0.01)

    def test_orb_size_above_8_amplifies(self):
        """ORB size >= 8 applies 1.2x multiplier."""
        state = _make_state(**{"0900": {"size": 10.0}})
        strategy = FakeStrategy("S1", "0900", expectancy_r=0.4, win_rate=0.45)
        score = score_strategy(strategy, state)
        assert score == pytest.approx(0.4 * 1.2, abs=0.01)

    def test_orb_size_4_to_8_no_adjustment(self):
        """ORB size in [4, 8) = no adjustment."""
        state = _make_state(**{"0900": {"size": 5.0}})
        strategy = FakeStrategy("S1", "0900", expectancy_r=0.4, win_rate=0.45)
        score = score_strategy(strategy, state)
        assert score == pytest.approx(0.4, abs=0.01)

    def test_chop_penalty_on_1100(self):
        """Chop signal penalizes 1100 strategies."""
        state = _make_state(**{"1100": {"size": 5.0}})
        state.signals.chop_detected = True
        strategy = FakeStrategy("S1", "1100", expectancy_r=0.4, win_rate=0.45)
        weights = ScoringWeights()
        score = score_strategy(strategy, state, weights)
        assert score < 0.4  # penalized

    def test_reversal_bonus_on_1000(self):
        """Reversal signal gives bonus to 1000 strategies."""
        state = _make_state(**{"1000": {"size": 5.0}})
        state.signals.reversal_active = True
        strategy = FakeStrategy("S1", "1000", expectancy_r=0.4, win_rate=0.45)
        weights = ScoringWeights()
        score = score_strategy(strategy, state, weights)
        assert score > 0.4  # boosted

    def test_regime_delta_adjustment(self):
        """Regime delta adjusts score."""
        state = _make_state(**{"0900": {"size": 5.0}})
        state.regime = RegimeContext(label="2025", deltas={"S1": 0.2})
        strategy = FakeStrategy("S1", "0900", expectancy_r=0.4, win_rate=0.45)
        weights = ScoringWeights()
        score = score_strategy(strategy, state, weights)
        expected = 0.4 + 0.2 * weights.regime_delta
        assert score == pytest.approx(expected, abs=0.01)


# =========================================================================
# Cascade table lookup
# =========================================================================

class TestCascadeLookup:
    def test_lookup_found(self):
        table = {("0900", "loss", "opposite"): {"1000_wr": 0.52, "n": 148}}
        result = lookup_cascade(table, "0900", "loss", "opposite")
        assert result is not None
        assert result["1000_wr"] == 0.52

    def test_lookup_missing_returns_none(self):
        table = {("0900", "loss", "opposite"): {"1000_wr": 0.52, "n": 148}}
        result = lookup_cascade(table, "0900", "win", "same")
        assert result is None

    def test_lookup_empty_table(self):
        result = lookup_cascade({}, "0900", "loss", "opposite")
        assert result is None


# =========================================================================
# Risk manager chop warning
# =========================================================================

class TestRiskManagerChopWarning:
    def test_chop_warning_recorded(self):
        from trading_app.risk_manager import RiskManager, RiskLimits
        rm = RiskManager(RiskLimits())
        rm.daily_reset(date(2025, 6, 15))

        state = MarketState()
        state.signals.chop_detected = True

        allowed, reason = rm.can_enter(
            "S1", "1100", [], 0.0, market_state=state
        )
        assert allowed is True
        assert any("chop_warning" in w for w in rm.warnings)

    def test_no_warning_without_chop(self):
        from trading_app.risk_manager import RiskManager, RiskLimits
        rm = RiskManager(RiskLimits())
        rm.daily_reset(date(2025, 6, 15))

        state = MarketState()
        state.signals.chop_detected = False

        allowed, _ = rm.can_enter(
            "S1", "1100", [], 0.0, market_state=state
        )
        assert allowed is True
        assert not any("chop_warning" in w for w in rm.warnings)
