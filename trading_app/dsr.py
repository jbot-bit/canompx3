"""
Deflated Sharpe Ratio (DSR) — Bailey & Lopez de Prado (2014).

Computes the probability that a strategy's Sharpe ratio exceeds what
pure noise would produce given N independent trials.

The key insight: when you test N strategies, the BEST one's observed Sharpe
is inflated by selection bias. DSR deflates it back to honest terms.

Usage:
    from trading_app.dsr import compute_dsr, compute_sr0

    sr0 = compute_sr0(n_eff=253, var_sr=0.047)
    dsr = compute_dsr(sr_hat=0.30, sr0=sr0, t_obs=150, skewness=0.2, kurtosis_excess=-1.5)
    # dsr > 0.95 → 95% confidence the strategy isn't noise

References:
    Bailey, D.H. and Lopez de Prado, M. (2014) "The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
    Journal of Portfolio Management, 40(5), 94-107.

    Bailey, D.H., Borwein, J.M., Lopez de Prado, M. and Zhu, Q.J. (2014)
    "Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest
    Overfitting on Out-of-Sample Performance." AMS Notices, 61(5), 458-471.

IMPORTANT — N_eff sensitivity:
    DSR results are HIGHLY sensitive to N_eff (effective independent trials).
    With V[SR]=0.047 (per-trade):
      N_eff=5   → SR0=0.26 → 4 strategies pass
      N_eff=10  → SR0=0.34 → 0 pass
      N_eff=253 → SR0=0.62 → 0 pass
    The true N_eff is unknown. Use edge family count as upper bound,
    instrument×session combinations as lower bound. ONC algorithm
    (Lopez de Prado) gives proper estimation but is not yet implemented.
    Until N_eff is properly estimated, DSR is INFORMATIONAL, not a hard gate.
"""

from __future__ import annotations

import math

# Euler-Mascheroni constant
_GAMMA = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    """Standard normal CDF (no scipy dependency for production code)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Standard normal inverse CDF (rational approximation, Abramowitz & Stegun 26.2.23).

    Accurate to ~4.5e-4 for 0.0027 < p < 0.9973. Good enough for DSR.
    """
    if p <= 0:
        return -10.0
    if p >= 1:
        return 10.0
    if p < 0.5:
        return -_norm_ppf(1 - p)

    # Rational approximation for 0.5 <= p < 1
    t = math.sqrt(-2 * math.log(1 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)


def compute_sr0(n_eff: int | float, var_sr: float) -> float:
    """Expected maximum Sharpe ratio from noise (False Strategy Theorem).

    Args:
        n_eff: Effective number of independent strategy trials.
            Conservative: count of distinct edge families.
            Aggressive: count of instrument × session combos.
            Proper: ONC algorithm (not yet implemented).
        var_sr: Cross-sectional variance of per-trade Sharpe ratios
            across all experimental strategies with sample_size >= 30.

    Returns:
        SR0: the per-trade Sharpe ratio you'd expect from the BEST of
        N_eff pure-noise strategies. Any observed SR below this is
        indistinguishable from selection bias.
    """
    if n_eff < 2:
        return 0.0
    if var_sr <= 0:
        return 0.0

    std_sr = math.sqrt(var_sr)
    z1 = _norm_ppf(1 - 1.0 / n_eff)
    z2 = _norm_ppf(1 - 1.0 / (n_eff * math.e))
    return std_sr * ((1 - _GAMMA) * z1 + _GAMMA * z2)


def compute_dsr(
    sr_hat: float,
    sr0: float,
    t_obs: int,
    skewness: float = 0.0,
    kurtosis_excess: float = 0.0,
) -> float:
    """Deflated Sharpe Ratio — probability strategy is real after correction.

    Args:
        sr_hat: Observed per-trade Sharpe ratio of the strategy.
        sr0: Expected max Sharpe from noise (from compute_sr0).
        t_obs: Number of trade observations (sample_size).
        skewness: Skewness of per-trade returns.
        kurtosis_excess: Excess kurtosis of per-trade returns (kurtosis - 3).

    Returns:
        DSR: probability in [0, 1]. DSR > 0.95 means 95% confidence
        the strategy's Sharpe exceeds what noise would produce.
    """
    if t_obs < 2:
        return 0.0

    numerator = (sr_hat - sr0) * math.sqrt(t_obs - 1)

    # Denominator: Var[SR] per Lo (2002) Prop.2, Mertens (2002).
    # Normal baseline contributes 0.5*SR^2; excess kurtosis adds (kurt_excess/4)*SR^2.
    # Combined: (kurtosis_excess + 2) / 4 * SR^2.
    denom_sq = 1.0 - skewness * sr_hat + ((kurtosis_excess + 2) / 4.0) * sr_hat**2
    if denom_sq <= 0:
        return 0.0

    z = numerator / math.sqrt(denom_sq)
    return _norm_cdf(z)


def estimate_var_sr_from_db(db_path, min_sample: int = 30) -> float:
    """Estimate V[SR] from experimental_strategies (per-trade Sharpe).

    Uses canonical strategies with sample_size >= min_sample.
    Returns cross-sectional variance of sharpe_ratio column.
    """
    import duckdb

    with duckdb.connect(str(db_path), read_only=True) as con:
        row = con.execute(
            """SELECT VAR_SAMP(sharpe_ratio)
               FROM experimental_strategies
               WHERE sample_size >= ?
               AND sharpe_ratio IS NOT NULL
               AND is_canonical = TRUE""",
            [min_sample],
        ).fetchone()
        return row[0] if row and row[0] is not None else 0.0


def estimate_n_eff_from_db(db_path) -> dict:
    """Estimate N_eff bounds from the database.

    Returns dict with:
        n_raw: raw trial count (upper bound, inflated by correlation)
        n_families_all: all edge families (moderate estimate)
        n_families_active: non-purged families (moderate estimate)
        n_instrument_session: instrument × session combos (lower bound)
    """
    import duckdb

    with duckdb.connect(str(db_path), read_only=True) as con:
        n_raw = con.execute(
            "SELECT COUNT(*) FROM experimental_strategies WHERE is_canonical = TRUE"
        ).fetchone()[0]

        n_fam_all = con.execute(
            "SELECT COUNT(DISTINCT family_hash) FROM edge_families"
        ).fetchone()[0]

        n_fam_active = con.execute(
            "SELECT COUNT(DISTINCT family_hash) FROM edge_families WHERE robustness_status NOT IN ('PURGED')"
        ).fetchone()[0]

        n_inst_sess = con.execute(
            """SELECT COUNT(DISTINCT instrument || '_' || orb_label)
               FROM experimental_strategies WHERE is_canonical = TRUE"""
        ).fetchone()[0]

        return {
            "n_raw": n_raw,
            "n_families_all": n_fam_all,
            "n_families_active": n_fam_active,
            "n_instrument_session": n_inst_sess,
        }
