"""
Hypothesis-driven strategy scoring engine.

Combines multiple context signals into a single confidence score per strategy.
Used by MarketState.score_strategies() to rank strategies for the current day.

Read-only -- never writes to DB.

Usage:
    from trading_app.scoring import score_strategy, ScoringWeights
    score = score_strategy(strategy, market_state)
"""

from dataclasses import dataclass


@dataclass
class ScoringWeights:
    """Tunable weights for context signals."""
    base_expectancy: float = 1.0
    regime_delta: float = 0.3
    cross_session: float = 0.2
    volume_signal: float = 0.15
    orb_size_signal: float = 0.15
    chop_penalty: float = -0.5
    reversal_bonus: float = 0.3
    continuation_bonus: float = 0.2


# Minimum score threshold -- strategies below this are not worth arming
MIN_SCORE_THRESHOLD = 0.0


def score_strategy(strategy, state, weights: ScoringWeights | None = None) -> float:
    """Compute context-adjusted score for a strategy given current market state.

    Args:
        strategy: PortfolioStrategy with strategy_id, orb_label, expectancy_r
        state: MarketState with orbs, signals, regime
        weights: Tunable scoring weights (defaults to ScoringWeights())

    Returns:
        Float score. Higher = more confidence in this strategy today.
    """
    if weights is None:
        weights = ScoringWeights()

    score = strategy.expectancy_r * weights.base_expectancy

    # Regime adjustment: if current regime shows different ExpR, adjust
    if state.regime and strategy.strategy_id in state.regime.deltas:
        delta = state.regime.deltas[strategy.strategy_id]
        score += delta * weights.regime_delta

    # Cross-session signals
    if state.signals.chop_detected and strategy.orb_label in ("SINGAPORE_OPEN",):
        score += weights.chop_penalty

    if state.signals.reversal_active and strategy.orb_label == "TOKYO_OPEN":
        score += weights.reversal_bonus

    if state.signals.continuation and strategy.orb_label in ("TOKYO_OPEN", "SINGAPORE_OPEN"):
        score += weights.continuation_bonus

    # Cascade win rate adjustment
    if state.signals.cascade_wr is not None:
        # Adjust based on how cascade_wr differs from base win rate
        cascade_delta = state.signals.cascade_wr - strategy.win_rate
        score += cascade_delta * weights.cross_session

    # ORB size signal for this strategy's session
    orb = state.orbs.get(strategy.orb_label)
    if orb and orb.size is not None:
        if orb.size >= 8.0:
            score *= 1.2  # Strong volatility regime
        elif orb.size >= 4.0:
            pass  # Normal -- no adjustment
        else:
            score *= 0.5  # Weak ORB -- house wins territory

    return round(score, 4)
