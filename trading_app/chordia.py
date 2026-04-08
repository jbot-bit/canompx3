"""
Chordia t-statistic gate — Chordia, Goyal, Saretto (2018).

Computes the t-statistic of a strategy's mean per-trade return against the null
hypothesis of zero mean. Used as Criterion 4 of the locked institutional
criteria in `docs/institutional/pre_registered_criteria.md`.

The mathematical identity used here:
    sharpe_per_trade = mean[R] / std[R]
    t_statistic     = mean[R] / (std[R] / sqrt(N))
                    = (mean[R] / std[R]) * sqrt(N)
                    = sharpe_per_trade * sqrt(N)

So when the validator stores `sharpe_ratio` as the per-trade Sharpe (which it
does, see `trading_app/strategy_validator.py` and `validated_setups` schema),
the t-statistic is simply `sharpe_ratio * sqrt(sample_size)`. No separate std
column is required.

@research-source: docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md
@canonical-source: trading_app/chordia.py
@revalidated-for: Phase 4 Stage 4.0 (2026-04-08, criterion 4 enforcement)

Thresholds (locked in `pre_registered_criteria.md` Criterion 4):
- t >= 3.00 for strategies with strong pre-registered economic theory support
  (Harvey-Liu-Zhu 2015 hurdle)
- t >= 3.79 for strategies without such theoretical support
  (Chordia et al 2018 hurdle, K=2 million strategies tested)

These are HARD GATES — no post-hoc relaxation allowed per the locked criteria
file. If a strategy is borderline, the institutional response is to declare it
research-provisional, not to lower the threshold.
"""

from __future__ import annotations

import math

# Locked thresholds — sourced from docs/institutional/literature/.
# Modifying these requires an amendment to pre_registered_criteria.md
# Criterion 4 with explicit literature citation.
CHORDIA_T_WITH_THEORY: float = 3.00
CHORDIA_T_WITHOUT_THEORY: float = 3.79


def chordia_threshold(has_theory: bool) -> float:
    """Return the locked Chordia t-statistic threshold for the given theory state.

    Parameters
    ----------
    has_theory
        True if the strategy has a pre-registered economic theory citation in
        its hypothesis file. False if not (the discovery stands purely on
        empirical fit).

    Returns
    -------
    float
        3.00 if has_theory else 3.79.
    """
    return CHORDIA_T_WITH_THEORY if has_theory else CHORDIA_T_WITHOUT_THEORY


def compute_chordia_t(sharpe_ratio: float, sample_size: int) -> float:
    """Compute the Chordia t-statistic from per-trade Sharpe and sample size.

    Uses the identity ``t = sharpe_per_trade * sqrt(N)`` which follows directly
    from ``sharpe_per_trade = mean[R] / std[R]`` and the standard t-statistic
    definition ``t = mean[R] / (std[R] / sqrt(N))``.

    Parameters
    ----------
    sharpe_ratio
        Per-trade Sharpe ratio (mean of per-trade R-multiples divided by their
        standard deviation). This is the canonical form stored in
        ``validated_setups.sharpe_ratio`` and ``experimental_strategies.sharpe_ratio``.
    sample_size
        Number of trades in the sample. Must be at least 2 to compute the
        t-statistic; the standard error of the mean is undefined for N < 2.

    Returns
    -------
    float
        The t-statistic. Larger absolute values indicate stronger evidence
        against the null (mean per-trade return = 0).

    Raises
    ------
    ValueError
        If ``sample_size < 2``. The Chordia gate cannot be applied to a sample
        with fewer than 2 trades.
    """
    if sample_size < 2:
        raise ValueError(
            f"Chordia t-statistic requires sample_size >= 2, got {sample_size}. "
            f"The standard error of the mean is undefined for N < 2."
        )
    return sharpe_ratio * math.sqrt(sample_size)


def chordia_gate(
    sharpe_ratio: float,
    sample_size: int,
    has_theory: bool,
) -> tuple[bool, float, float]:
    """Apply the Chordia t-statistic gate to a strategy.

    Parameters
    ----------
    sharpe_ratio
        Per-trade Sharpe ratio.
    sample_size
        Number of trades.
    has_theory
        True if a pre-registered theory citation exists for the hypothesis
        family this strategy belongs to.

    Returns
    -------
    tuple[bool, float, float]
        ``(passed, t_statistic, threshold)`` where ``passed`` is True iff
        ``t_statistic >= threshold``. The threshold is selected by
        ``chordia_threshold(has_theory)``.

    Raises
    ------
    ValueError
        If ``sample_size < 2`` (propagated from ``compute_chordia_t``).
    """
    threshold = chordia_threshold(has_theory)
    t_stat = compute_chordia_t(sharpe_ratio, sample_size)
    return (t_stat >= threshold, t_stat, threshold)


__all__ = [
    "CHORDIA_T_WITH_THEORY",
    "CHORDIA_T_WITHOUT_THEORY",
    "chordia_threshold",
    "compute_chordia_t",
    "chordia_gate",
]
