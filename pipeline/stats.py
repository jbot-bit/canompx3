"""Shared statistical utilities.

Provides per-trade Sharpe ratio and Jobson-Korkie test (1981) for Sharpe
comparison. Used by trading_app/ml/evaluate.py, evaluate_validated.py, and
scripts/tools/select_family_rr.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def per_trade_sharpe(pnl: pd.Series) -> float:
    """Per-trade Sharpe ratio (no annualization).

    Our data is per-trade pnl_r, NOT daily returns. Trade frequency varies
    by filter (G0 ~500/yr, G8 ~50/yr), so sqrt(252) annualization is wrong.
    Per-trade Sharpe = mean/std — comparable across all strategies.
    """
    if pnl.std() == 0 or len(pnl) < 2:
        return 0.0
    return float(pnl.mean() / pnl.std())


def jobson_korkie_p(
    sharpe_a: float,
    sharpe_b: float,
    n_a: int,
    n_b: int,
    rho: float = 0.0,
) -> float:
    """Two-sided Jobson-Korkie (1981) test for Sharpe equality.

    Returns p-value. Large p means Sharpes are statistically indistinguishable.
    Small p (< 0.05) means the difference is significant.

    @research-source Jobson & Korkie (1981) "Performance Hypothesis Testing
    with the Sharpe and Treynor Measures"
    @research-source Memmel (2003) corrected SE for correlated streams
    @research-source Lo (2002) "The Statistics of Sharpe Ratios" — Sharpe SE
    is O(1/sqrt(N)), so visual differences are often noise with N < 500

    Args:
        sharpe_a: Per-trade Sharpe of stream A (e.g., baseline).
        sharpe_b: Per-trade Sharpe of stream B (e.g., filtered).
        n_a: Number of trades in stream A.
        n_b: Number of trades in stream B.
        rho: Assumed correlation between the two return streams.
             Use 0.7 for same-trade-subset comparisons (meta-label filter).
    """
    n_eff = min(n_a, n_b)
    if n_eff < 5:
        return 1.0  # insufficient data, treat as equal

    se_sq = (2.0 / n_eff) * (1 - rho) + (1.0 / (2 * n_eff)) * (
        sharpe_a**2 + sharpe_b**2 - 2 * sharpe_a * sharpe_b * rho**2
    )
    if se_sq <= 0:
        se_sq = 2.0 / n_eff

    diff = abs(sharpe_a - sharpe_b)
    z = diff / np.sqrt(se_sq)
    # Lazy import — scipy.stats is ~3-4s to import. Only this function uses it.
    # PEP 8 explicitly endorses delayed imports for performance.
    from scipy import stats as scipy_stats

    return float(2 * (1 - scipy_stats.norm.cdf(z)))
