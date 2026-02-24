"""Tests for trading_app.scoring — strategy context scoring engine."""

import pytest
from types import SimpleNamespace

from trading_app.scoring import score_strategy, ScoringWeights, MIN_SCORE_THRESHOLD


def _make_strategy(expectancy_r=0.3, orb_label="TOKYO_OPEN", strategy_id="test_strat",
                   win_rate=0.50):
    return SimpleNamespace(
        expectancy_r=expectancy_r,
        orb_label=orb_label,
        strategy_id=strategy_id,
        win_rate=win_rate,
    )


def _make_state(regime=None, chop_detected=False, reversal_active=False,
                continuation=False, cascade_wr=None, orbs=None):
    signals = SimpleNamespace(
        chop_detected=chop_detected,
        reversal_active=reversal_active,
        continuation=continuation,
        cascade_wr=cascade_wr,
    )
    return SimpleNamespace(
        regime=regime,
        signals=signals,
        orbs=orbs or {},
    )


class TestScoreStrategy:
    def test_base_score_only(self):
        """Base expectancy with no context signals."""
        strat = _make_strategy(expectancy_r=0.3)
        state = _make_state()
        score = score_strategy(strat, state)
        assert score == pytest.approx(0.3, abs=1e-4)

    def test_regime_delta_applied(self):
        """Regime delta adjusts score."""
        strat = _make_strategy(expectancy_r=0.3, strategy_id="s1")
        regime = SimpleNamespace(deltas={"s1": 0.1})
        state = _make_state(regime=regime)
        score = score_strategy(strat, state)
        # 0.3 + 0.1 * 0.3 (regime_delta weight) = 0.33
        assert score == pytest.approx(0.33, abs=1e-4)

    def test_chop_penalty_on_singapore_open(self):
        """Chop detection penalizes SINGAPORE_OPEN session."""
        strat = _make_strategy(orb_label="SINGAPORE_OPEN")
        state = _make_state(chop_detected=True)
        score = score_strategy(strat, state)
        # 0.3 + (-0.5) = -0.2
        assert score == pytest.approx(-0.2, abs=1e-4)

    def test_chop_no_effect_on_cme_reopen(self):
        """Chop detection does NOT penalize CME_REOPEN session."""
        strat = _make_strategy(orb_label="CME_REOPEN")
        state = _make_state(chop_detected=True)
        score = score_strategy(strat, state)
        assert score == pytest.approx(0.3, abs=1e-4)

    def test_reversal_bonus_on_tokyo_open(self):
        """Reversal signal boosts TOKYO_OPEN."""
        strat = _make_strategy(orb_label="TOKYO_OPEN")
        state = _make_state(reversal_active=True)
        score = score_strategy(strat, state)
        # 0.3 + 0.3 = 0.6
        assert score == pytest.approx(0.6, abs=1e-4)

    def test_orb_size_strong(self):
        """ORB size >= 8.0 gives 1.2x multiplier."""
        strat = _make_strategy(orb_label="TOKYO_OPEN")
        orb = SimpleNamespace(size=10.0)
        state = _make_state(orbs={"TOKYO_OPEN": orb})
        score = score_strategy(strat, state)
        # 0.3 * 1.2 = 0.36
        assert score == pytest.approx(0.36, abs=1e-4)

    def test_orb_size_weak(self):
        """ORB size < 4.0 gives 0.5x multiplier."""
        strat = _make_strategy(orb_label="TOKYO_OPEN")
        orb = SimpleNamespace(size=2.0)
        state = _make_state(orbs={"TOKYO_OPEN": orb})
        score = score_strategy(strat, state)
        # 0.3 * 0.5 = 0.15
        assert score == pytest.approx(0.15, abs=1e-4)

    def test_cascade_wr_adjustment(self):
        """Cascade win rate delta adjusts score."""
        strat = _make_strategy(win_rate=0.50)
        state = _make_state(cascade_wr=0.60)
        score = score_strategy(strat, state)
        # 0.3 + (0.60 - 0.50) * 0.2 = 0.3 + 0.02 = 0.32
        assert score == pytest.approx(0.32, abs=1e-4)

    def test_custom_weights(self):
        """Custom weights override defaults."""
        strat = _make_strategy(expectancy_r=0.5)
        state = _make_state()
        w = ScoringWeights(base_expectancy=2.0)
        score = score_strategy(strat, state, weights=w)
        assert score == pytest.approx(1.0, abs=1e-4)

    def test_no_orb_no_crash(self):
        """Missing ORB for strategy's session doesn't crash."""
        strat = _make_strategy(orb_label="CME_REOPEN")
        state = _make_state(orbs={})  # no CME_REOPEN ORB
        score = score_strategy(strat, state)
        assert isinstance(score, float)


class TestScoringWeights:
    def test_default_values(self):
        w = ScoringWeights()
        assert w.base_expectancy == 1.0
        assert w.chop_penalty == -0.5

    def test_min_score_threshold_is_zero(self):
        assert MIN_SCORE_THRESHOLD == 0.0
