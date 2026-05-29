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
    The true N_eff is unknown without estimation. Two principled estimators
    are provided (both grounded, both clamped to [1, M]):
      - `bailey_neff_correlation(rho_hat, m)` — closed-form Bailey-LdP 2014
        Eq. 9 / Exhibit 4: N̂ = ρ̂ + (1-ρ̂)·M (linear ρ̂-interpolation).
      - `estimate_n_eff_onc(returns_matrix)` — López de Prado Optimal Number
        of Clusters: correlation → distance → cluster → silhouette-optimal k.
    Whether DSR is a hard gate is a doctrine decision (see Criterion 5 /
    Amendment 3.5 in docs/institutional/pre_registered_criteria.md), not a
    property of this module. This module supplies the math only.
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
            Proper: ONC clustering via `estimate_n_eff_onc`, or the
            closed-form correlation correction via `bailey_neff_correlation`.
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
        n_raw = con.execute("SELECT COUNT(*) FROM experimental_strategies WHERE is_canonical = TRUE").fetchone()[0]

        n_fam_all = con.execute("SELECT COUNT(DISTINCT family_hash) FROM edge_families").fetchone()[0]

        # Count all families — PURGED label is member-count heuristic, not fitness.
        # Allocator trailing window handles actual fitness (2026-04-03).
        n_fam_active = con.execute("SELECT COUNT(DISTINCT family_hash) FROM edge_families").fetchone()[0]

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


def bailey_neff_correlation(rho_hat: float, m: int) -> float:
    """Effective trial count via the closed-form Bailey-LdP 2014 Eq. 9.

    N̂ = ρ̂ + (1 - ρ̂)·M  — a linear interpolation between full independence
    (ρ̂ → 0 ⇒ N̂ = M) and full dependence (ρ̂ → 1 ⇒ N̂ = 1).

    Args:
        rho_hat: Average pairwise correlation between the M trial return
            series (the off-diagonal mean of the correlation matrix).
            Clamped to [0, 1]: a negative ρ̂ would push N̂ above M, which is
            meaningless for the False Strategy Theorem (you cannot have more
            independent trials than series), so it is floored at 0.
        m: Number of raw trials (columns).

    Returns:
        N̂ in [1, m]. Conservative posture: a larger N̂ raises SR_0 and makes
        the DSR gate stricter, never looser.

    Reference:
        Bailey, D.H. & López de Prado, M. (2014), Appendix A.3 + Exhibit 4.
        docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md
    """
    if m < 1:
        return 1.0
    if m == 1:
        return 1.0
    rho = min(1.0, max(0.0, rho_hat))
    n_eff = rho + (1.0 - rho) * m
    # Clamp defensively — the formula is bounded to [1, m] for rho in [0, 1],
    # but float drift on the endpoints should never escape the contract.
    return min(float(m), max(1.0, n_eff))


def estimate_n_eff_onc(
    returns_matrix,
    *,
    min_overlap: int = 30,
    max_clusters: int | None = None,
    random_state: int = 0,
) -> dict:
    """Effective trial count via López de Prado's Optimal Number of Clusters.

    Clusters the strategy return-correlation matrix and uses the optimal
    cluster count as N̂. Near-clone strategies (same session-day, different
    aperture/RR) collapse into one cluster, so N̂ << raw column count when the
    universe is correlated — which is the whole point: counting raw cells
    overstates the number of independent trials and inflates SR_0.

    Algorithm (ML for Asset Managers Ch. 4 / AFML 2018 Ch. 4):
      1. Pearson correlation matrix C over the columns (pairwise-complete,
         requiring `min_overlap` shared observations).
      2. Correlation distance D = sqrt(0.5·(1 - C)) (a proper metric: 0 when
         ρ=1, 1 when ρ=-1).
      3. For k in [2, K_max], KMeans on the distance rows; score each k by the
         mean silhouette (on the precomputed distance).
      4. N̂ = argmax-silhouette k. Ties / degenerate silhouette → fall back to
         the conservative raw column count.

    Args:
        returns_matrix: An (observations × M) array-like of per-period
            returns, one column per trial/strategy. NaN marks "no trade this
            period" and is handled pairwise — the matrix need not be dense.
        min_overlap: Minimum shared non-NaN observations for a pair's
            correlation to count; sparser pairs are treated as NaN→0 distance
            contribution avoided (see notes). Mirrors the 30-day floor used in
            research/mnq_pr51_dsr_effective_n_v2.py.
        max_clusters: Upper bound on k searched. Defaults to min(M-1, 50).
        random_state: KMeans seed (determinism — required; argless RNG is
            banned in this repo's reproducible paths).

    Returns:
        dict with:
          n_eff: int in [1, M] — the optimal cluster count (N̂).
          m: int — raw column count.
          best_silhouette: float | None — silhouette at n_eff (None if
            clustering was degenerate and the raw-count fallback was used).
          method: "onc_silhouette" | "fallback_raw_count" — which path ran.
          rho_hat: float — off-diagonal mean correlation (for cross-check /
            feeding bailey_neff_correlation; NaN-safe mean, 0.0 if undefined).

    Conservative contract: any degenerate input (M < 2, all-NaN, no finite
    pairwise correlations, silhouette undefined) returns
    `n_eff = max(1, M)` via the fallback path. Clustering can never report
    MORE independent trials than columns — n_eff is clamped to [1, M].

    Reference:
        López de Prado, M. (2018) Advances in Financial ML, Ch. 4 (clustering).
        López de Prado, M. (2020) ML for Asset Managers, Ch. 4 (ONC).
        docs/institutional/literature/lopez_de_prado_2018_afml_ch_3_7_8.md
    """
    import numpy as np

    arr = np.asarray(returns_matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"returns_matrix must be 2-D (observations × trials); got ndim={arr.ndim}")

    m = arr.shape[1]

    def _fallback(rho: float) -> dict:
        return {
            "n_eff": max(1, m),
            "m": m,
            "best_silhouette": None,
            "method": "fallback_raw_count",
            "rho_hat": rho,
        }

    if m < 2 or arr.shape[0] < 2:
        return _fallback(0.0)

    # 1) Pairwise-complete Pearson correlation. pandas handles the NaN masking
    #    with an overlap floor the way the prior research script did.
    import pandas as pd

    corr = pd.DataFrame(arr).corr(method="pearson", min_periods=min_overlap).to_numpy()

    # Off-diagonal mean correlation (NaN-safe) — reported for cross-check and
    # to feed the closed-form estimator.
    upper = np.triu(np.ones((m, m), dtype=bool), k=1)
    off_diag = corr[upper]
    finite = off_diag[np.isfinite(off_diag)]
    rho_hat = float(finite.mean()) if finite.size else 0.0

    # If correlations are too sparse to define a distance matrix, fall back.
    if not np.isfinite(corr).all():
        # Replace NaN correlations with 0 (max-distance-neutral) so the metric
        # is well-defined; if EVERY off-diagonal was NaN we cannot cluster.
        if finite.size == 0:
            return _fallback(rho_hat)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)

    # 2) Correlation distance — proper metric, 0 at ρ=1.
    corr = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt(0.5 * (1.0 - corr))
    np.fill_diagonal(dist, 0.0)

    # 3) Silhouette-optimal k over KMeans on the distance rows.
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    k_cap = max_clusters if max_clusters is not None else min(m - 1, 50)
    k_cap = max(2, min(k_cap, m - 1))

    best_k: int | None = None
    best_score = -1.0
    for k in range(2, k_cap + 1):
        labels = KMeans(n_clusters=k, n_init="auto", random_state=random_state).fit_predict(dist)
        if len(set(labels)) < 2:
            continue
        try:
            score = float(silhouette_score(dist, labels, metric="precomputed"))
        except ValueError:
            # silhouette undefined (e.g. a singleton-only partition) — skip k.
            continue
        if score > best_score:
            best_score = score
            best_k = k

    if best_k is None:
        return _fallback(rho_hat)

    return {
        "n_eff": int(min(m, max(1, best_k))),
        "m": m,
        "best_silhouette": best_score,
        "method": "onc_silhouette",
        "rho_hat": rho_hat,
    }
