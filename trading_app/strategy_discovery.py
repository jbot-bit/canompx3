"""
Grid search over strategy variants and save results to experimental_strategies.

For each combination of (orb_label, rr_target, confirm_bars, filter),
queries pre-computed orb_outcomes, computes performance metrics, and
writes results to experimental_strategies.

Usage:
    python trading_app/strategy_discovery.py --instrument MGC --start 2021-01-01 --end 2025-12-31
    python trading_app/strategy_discovery.py --instrument MGC --start 2021-01-01 --end 2025-12-31 --dry-run
"""

import json
import sys
from collections import defaultdict
from datetime import UTC, date
from pathlib import Path

from pipeline.log import get_logger

logger = get_logger(__name__)

# pandas + duckdb lazy-loaded inside the functions that use them
# (each in exactly one site). PEP 8 endorses delayed imports for
# performance — cost is modest on warm OS-cache (~0.3s pandas alone)
# but noticeable on cold boot where the DLL is not in filesystem cache.
from pipeline.asset_configs import get_enabled_sessions
from pipeline.cost_model import get_cost_spec
from pipeline.dst import (
    DOW_MISALIGNED_SESSIONS,
    DST_AFFECTED_SESSIONS,
    classify_dst_verdict,
    is_winter_for_session,
)
from pipeline.paths import GOLD_DB_PATH
from trading_app.config import (
    ALL_FILTERS,
    ENTRY_MODELS,
    SKIP_ENTRY_MODELS,
    STOP_MULTIPLIERS,
    CompositeFilter,
    CrossAssetATRFilter,
    DayOfWeekSkipFilter,
    VolumeFilter,
    apply_tight_stop,
    get_filters_for_grid,
    is_e2_lookahead_filter,
)
from trading_app.db_manager import compute_trade_day_hash, init_trading_app_schema
from trading_app.hypothesis_loader import (
    HypothesisLoaderError,
    ScopePredicate,
    check_mode_a_consistency,
    enforce_minbtl_bound,
    extract_scope_predicate,
    load_hypothesis_metadata,
)
from trading_app.outcome_builder import CONFIRM_BARS_OPTIONS, RR_TARGETS
from trading_app.phase_4_discovery_gates import check_git_cleanliness, check_single_use

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

# Filter specificity ranking: higher = more specific = preferred as canonical
_FILTER_SPECIFICITY_BASE = {
    "ORB_G8": 5,
    "ORB_G6": 4,
    "ORB_G5_L12": 4,
    "ORB_G5": 3,
    "ORB_G4_L12": 3,
    "ORB_G4": 2,
    "DIR_LONG": 2,
    "DIR_SHORT": 2,
    "VOL_RV12_N20": 1,
    "NO_FILTER": 0,
}
# Sorted longest-first so ORB_G5_L12 matches before ORB_G5
_SPECIFICITY_PREFIXES = sorted(_FILTER_SPECIFICITY_BASE.items(), key=lambda x: -len(x[0]))


def _get_filter_specificity(filter_key: str) -> int:
    """Get specificity score. Composites (e.g. ORB_G4_NOFRI) inherit base + 1."""
    if filter_key in _FILTER_SPECIFICITY_BASE:
        return _FILTER_SPECIFICITY_BASE[filter_key]
    for base, spec in _SPECIFICITY_PREFIXES:
        if filter_key.startswith(base + "_"):
            return spec + 1
    return 0  # unknown defaults to same as NO_FILTER


_BATCH_COLUMNS = [
    "strategy_id",
    "instrument",
    "orb_label",
    "orb_minutes",
    "rr_target",
    "confirm_bars",
    "entry_model",
    "filter_type",
    "filter_params",
    "stop_multiplier",
    "sample_size",
    "win_rate",
    "avg_win_r",
    "avg_loss_r",
    "expectancy_r",
    "sharpe_ratio",
    "max_drawdown_r",
    "median_risk_points",
    "avg_risk_points",
    "median_risk_dollars",
    "avg_risk_dollars",
    "avg_win_dollars",
    "avg_loss_dollars",
    "trades_per_year",
    "sharpe_ann",
    "yearly_results",
    "entry_signals",
    "scratch_count",
    "early_exit_count",
    "trade_day_hash",
    "is_canonical",
    "canonical_strategy_id",
    "dst_winter_n",
    "dst_winter_avg_r",
    "dst_summer_n",
    "dst_summer_avg_r",
    "dst_verdict",
    "validation_status",
    "validation_notes",
    "created_at",
    # Audit metrics (F-04, F-11)
    "p_value",
    "sharpe_ann_adj",
    "autocorr_lag1",
    # Haircut Sharpe (Bailey & Lopez de Prado, 2014)
    "sharpe_haircut",
    "skewness",
    "kurtosis_excess",
    # Multiple testing audit (Chordia et al 2018, Bailey & Lopez de Prado 2018)
    "n_trials_at_discovery",
    "fst_hurdle",
    # BH FDR at discovery (Mar 2026 — Bloomey statistical hardening)
    "fdr_significant_discovery",
    "fdr_adjusted_p_discovery",
    # Phase 4 Stage 4.1 — pre-registered hypothesis file SHA stamp.
    # Populated when run_discovery is called with hypothesis_file=<Path>;
    # NULL on legacy-mode runs. See docs/audit/hypotheses/README.md.
    "hypothesis_file_sha",
]


def _flush_batch_df(con, insert_batch: list[list]) -> None:
    # Flush a batch of strategy rows via DataFrame replacement scan.
    #
    # Phase 4 Stage 4.1 defensive guard (D-7): every row in ``insert_batch``
    # MUST have exactly ``len(_BATCH_COLUMNS)`` elements. Without this
    # check, a mismatch between ``_BATCH_COLUMNS``, the SQL column list
    # below, and the batch-assembly append loop silently shifts values
    # into the neighbouring slots (Pandas/DuckDB positional alignment).
    # The Stage 4.1 SHA-stamping edit is spread across three coordinated
    # locations, so a mismatch is a real possibility — better to raise
    # loudly at write time than to silently corrupt experimental_strategies.
    #
    # (Comment form instead of docstring to avoid the trading-app schema
    # drift check treating this as embedded SQL — the word "INSERT" plus
    # "INTO" patterns in a docstring trigger false positives in
    # pipeline.check_drift.check_schema_query_consistency_trading_app.)
    import pandas as pd

    expected_width = len(_BATCH_COLUMNS)
    for i, row in enumerate(insert_batch):
        if len(row) != expected_width:
            raise ValueError(
                f"_flush_batch_df column alignment error: row {i} has "
                f"{len(row)} values but _BATCH_COLUMNS has {expected_width}. "
                f"Check _BATCH_COLUMNS + INSERT column list + batch assembly "
                f"loop for a three-way mismatch (Phase 4 Stage 4.1 SHA "
                f"stamping guard)."
            )
    batch_df = pd.DataFrame(insert_batch, columns=_BATCH_COLUMNS)  # noqa: F841
    con.execute("""
        INSERT OR REPLACE INTO experimental_strategies
        (strategy_id, instrument, orb_label, orb_minutes,
         rr_target, confirm_bars, entry_model,
         filter_type, filter_params, stop_multiplier,
         sample_size, win_rate, avg_win_r, avg_loss_r,
         expectancy_r, sharpe_ratio, max_drawdown_r,
         median_risk_points, avg_risk_points,
         median_risk_dollars, avg_risk_dollars, avg_win_dollars, avg_loss_dollars,
         trades_per_year, sharpe_ann,
         yearly_results,
         entry_signals, scratch_count, early_exit_count,
         trade_day_hash, is_canonical, canonical_strategy_id,
         dst_winter_n, dst_winter_avg_r, dst_summer_n, dst_summer_avg_r, dst_verdict,
         validation_status, validation_notes,
         created_at,
         p_value, sharpe_ann_adj, autocorr_lag1,
         sharpe_haircut, skewness, kurtosis_excess,
         n_trials_at_discovery, fst_hurdle,
         fdr_significant_discovery, fdr_adjusted_p_discovery,
         hypothesis_file_sha)
        SELECT strategy_id, instrument, orb_label, orb_minutes,
               rr_target, confirm_bars, entry_model,
               filter_type, filter_params, stop_multiplier,
               sample_size, win_rate, avg_win_r, avg_loss_r,
               expectancy_r, sharpe_ratio, max_drawdown_r,
               median_risk_points, avg_risk_points,
               median_risk_dollars, avg_risk_dollars, avg_win_dollars, avg_loss_dollars,
               trades_per_year, sharpe_ann,
               yearly_results,
               entry_signals, scratch_count, early_exit_count,
               trade_day_hash, is_canonical, canonical_strategy_id,
               dst_winter_n, dst_winter_avg_r, dst_summer_n, dst_summer_avg_r, dst_verdict,
               validation_status, validation_notes,
               COALESCE(created_at, CURRENT_TIMESTAMP),
               p_value, sharpe_ann_adj, autocorr_lag1,
               sharpe_haircut, skewness, kurtosis_excess,
               n_trials_at_discovery, fst_hurdle,
               fdr_significant_discovery, fdr_adjusted_p_discovery,
               hypothesis_file_sha
        FROM batch_df
    """)


def _mark_canonical(strategies: list[dict]) -> None:
    """Mark canonical vs alias within each dedup group.

    Groups by (instrument, orb_label, entry_model, rr_target, confirm_bars, trade_day_hash).
    Within each group, the strategy with highest filter specificity is canonical;
    ties broken by filter_key alphabetically.
    """
    groups = defaultdict(list)
    for s in strategies:
        key = (
            s["instrument"],
            s["orb_label"],
            s["entry_model"],
            s["rr_target"],
            s["confirm_bars"],
            s["trade_day_hash"],
            s.get("stop_multiplier", 1.0),
        )
        groups[key].append(s)

    for group in groups.values():
        # Sort by specificity descending, then filter_key for determinism
        group.sort(
            key=lambda s: (
                -_get_filter_specificity(s["filter_key"]),
                s["filter_key"],
            )
        )
        head = group[0]
        head["is_canonical"] = True
        head["canonical_strategy_id"] = head["strategy_id"]
        for alias in group[1:]:
            alias["is_canonical"] = False
            alias["canonical_strategy_id"] = head["strategy_id"]


def _t_test_pvalue(t_stat: float, df: int) -> float:
    """Two-tailed p-value from Student's t-distribution (no scipy needed).

    Uses the regularized incomplete beta function relationship:
      p = I_x(a, b)  where x = df/(df + t^2), a = df/2, b = 0.5

    For large df (>100), uses normal approximation.
    Returns two-tailed p-value.
    """
    import math

    if df <= 0:
        return 1.0

    t_abs = abs(t_stat)

    # Normal approximation for large df
    if df > 100:
        # Two-tailed p-value from normal distribution
        p = math.erfc(t_abs / math.sqrt(2))
        return max(p, 1e-16)

    # Regularized incomplete beta function via continued fraction
    x = df / (df + t_abs * t_abs)
    a = df / 2.0
    b = 0.5

    # Regularized incomplete beta function via continued fraction
    # (Numerical Recipes betacf pattern — includes critical d₁ init term)
    def _betainc(x_val, a_val, b_val):
        """Regularized incomplete beta function I_x(a, b)."""
        if x_val <= 0:
            return 0.0
        if x_val >= 1:
            return 1.0

        # Symmetry relation for better convergence when x is large
        if x_val > (a_val + 1.0) / (a_val + b_val + 2.0):
            return 1.0 - _betainc(1.0 - x_val, b_val, a_val)

        # Log-beta prefactor
        log_beta = math.lgamma(a_val) + math.lgamma(b_val) - math.lgamma(a_val + b_val)
        front = math.exp(a_val * math.log(x_val) + b_val * math.log(1.0 - x_val) - log_beta) / a_val

        # Continued fraction (Numerical Recipes betacf)
        TINY = 1e-30
        qab = a_val + b_val
        qap = a_val + 1.0
        qam = a_val - 1.0
        c = 1.0
        d = 1.0 - qab * x_val / qap  # Critical d₁ initialization
        if abs(d) < TINY:
            d = TINY
        d = 1.0 / d
        h = d
        for m in range(1, 200):
            m2 = 2 * m
            # Even step
            aa = m * (b_val - m) * x_val / ((qam + m2) * (a_val + m2))
            d = 1.0 + aa * d
            if abs(d) < TINY:
                d = TINY
            c = 1.0 + aa / c
            if abs(c) < TINY:
                c = TINY
            d = 1.0 / d
            h *= d * c

            # Odd step
            aa = -(a_val + m) * (qab + m) * x_val / ((a_val + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < TINY:
                d = TINY
            c = 1.0 + aa / c
            if abs(c) < TINY:
                c = TINY
            d = 1.0 / d
            delta = d * c
            h *= delta

            if abs(delta - 1.0) < 1e-10:
                break

        return min(1.0, front * h)

    # I_x(df/2, 1/2) gives the CDF
    p_one_tail = _betainc(x, a, b) / 2.0
    p_two_tail = 2.0 * p_one_tail
    return max(min(p_two_tail, 1.0), 1e-16)


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF (rational approximation, Abramowitz & Stegun).

    Used by Deflated Sharpe Ratio and False Strategy Theorem calculations.
    Accuracy: ~4.5e-4 absolute error, sufficient for SR computations.
    """
    import math

    if p <= 0:
        return -6.0
    if p >= 1:
        return 6.0
    if p == 0.5:
        return 0.0
    if p > 0.5:
        return -_norm_ppf(1.0 - p)
    t = math.sqrt(-2.0 * math.log(p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return -(t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t))


def _compute_haircut_sharpe(
    sharpe_per_trade: float,
    n_obs: int,
    skewness: float,
    kurtosis_excess: float,
    n_trials: int,
    trades_per_year: float,
) -> float | None:
    """Deflated Sharpe Ratio per Bailey & Lopez de Prado (2014).

    Adjusts observed Sharpe for both multiple testing bias AND non-normality.
    Returns the DSR Z-score: positive values mean the strategy's per-trade
    Sharpe genuinely exceeds the noise floor from testing K combos.

    Steps:
      1. Compute V(SR_obs) using Mertens (2002) non-normality correction:
         V[SR] = (1/T) * (1 - gamma3*SR + (gamma4/4)*SR^2)
      2. Compute E[max(SR|null)] for K trials with zero skill.
         Under null (SR=0), V_null = 1/T (Mertens terms vanish).
      3. DSR = (SR_obs - E[max(SR|null)]) / sqrt(V[SR_obs])
         This Z-score accounts for BOTH selection bias AND non-normality.

    @research-source: Bailey & Lopez de Prado (2014) deflated-sharpe.pdf
    @research-source: Mertens (2002) variance formula — non-normality correction
    @revalidated-for: E1, E2 entry models (bimodal ORB payoffs)

    Returns DSR Z-score (positive = exceeds noise floor) or None if insufficient data.
    """
    import math

    if n_obs < 10 or n_trials < 2 or sharpe_per_trade is None:
        return None
    if trades_per_year <= 0:
        return None

    sr = sharpe_per_trade
    gamma3 = skewness if skewness is not None else 0.0
    gamma4 = kurtosis_excess if kurtosis_excess is not None else 0.0

    # Mertens (2002): variance of the per-trade Sharpe estimator
    # V[SR] = (1/T) * (1 - gamma3*SR + (gamma4/4)*SR^2)
    # For normal data: gamma3=0, gamma4=0 -> V = 1/T (simple formula)
    # For bimodal ORB payoffs: gamma4 >> 0 -> V > 1/T (wider uncertainty)
    v_sr_obs = (1.0 / n_obs) * (1.0 - gamma3 * sr + (gamma4 / 4.0) * sr * sr)
    if v_sr_obs <= 0:
        v_sr_obs = 1.0 / n_obs  # Degenerate case: fallback to simple

    # Under null (SR=0), all Mertens terms vanish: V_null = 1/T
    std_sr_null = math.sqrt(1.0 / n_obs)

    # Expected max per-trade Sharpe under null (BLP 2014, Euler-Mascheroni approx)
    gamma_em = 0.5772156649
    p1 = 1.0 - 1.0 / n_trials
    p2 = 1.0 - 1.0 / (n_trials * math.e)

    e_max_null = std_sr_null * ((1.0 - gamma_em) * _norm_ppf(p1) + gamma_em * _norm_ppf(p2))

    # Deflated Sharpe Ratio: Z-score normalized by Mertens std error
    # DSR > 0 means SR genuinely exceeds multiple-testing noise floor
    dsr = (sr - e_max_null) / math.sqrt(v_sr_obs)

    return round(dsr, 4)


def compute_fst_hurdle(n_trials: int) -> float | None:
    """False Strategy Theorem hurdle: expected max Sharpe under zero skill.

    For K independent trials with true SR=0, the expected maximum observed
    per-trade Sharpe follows the extreme value distribution:
      E[max{SR}] = (1-gamma)*ppf(1 - 1/K) + gamma*ppf(1 - 1/(K*e))
    where gamma = 0.5772 (Euler-Mascheroni constant).

    Any strategy with per-trade Sharpe below this hurdle is indistinguishable
    from noise under multiple testing.

    @research-source: Bailey & Lopez de Prado (2018) false-strategy-lopez.pdf

    Args:
        n_trials: K, total number of strategy combos tested.

    Returns:
        Per-trade FST hurdle (float), or None if n_trials < 2.
    """
    import math

    if n_trials < 2:
        return None

    gamma_em = 0.5772156649
    p1 = 1.0 - 1.0 / n_trials
    p2 = 1.0 - 1.0 / (n_trials * math.e)

    hurdle = (1.0 - gamma_em) * _norm_ppf(p1) + gamma_em * _norm_ppf(p2)
    return round(hurdle, 4)


def compute_metrics(outcomes: list[dict], cost_spec=None, n_trials: int = 0) -> dict:
    """
    Compute performance metrics from a list of outcome rows.

    sample_size = wins + losses ONLY (scratches/early_exits excluded).
    entry_signals = wins + losses + scratches + early_exits (total entries).
    win_rate denominator = wins + losses.

    Args:
        outcomes: List of outcome dicts from orb_outcomes.
        cost_spec: Optional CostSpec for dollar calculations. If provided,
            computes median_risk_dollars, avg_risk_dollars, avg_win_dollars,
            avg_loss_dollars alongside R-multiple metrics.

    Returns dict with: sample_size, win_rate, avg_win_r, avg_loss_r,
    expectancy_r, sharpe_ratio, max_drawdown_r, median_risk_points,
    avg_risk_points, median_risk_dollars, avg_risk_dollars,
    avg_win_dollars, avg_loss_dollars, yearly_results, entry_signals,
    scratch_count, early_exit_count.
    """
    _empty = {
        "sample_size": 0,
        "win_rate": None,
        "avg_win_r": None,
        "avg_loss_r": None,
        "expectancy_r": None,
        "sharpe_ratio": None,
        "max_drawdown_r": None,
        "median_risk_points": None,
        "avg_risk_points": None,
        "median_risk_dollars": None,
        "avg_risk_dollars": None,
        "avg_win_dollars": None,
        "avg_loss_dollars": None,
        "trades_per_year": 0,
        "sharpe_ann": None,
        "sharpe_ann_adj": None,
        "autocorr_lag1": None,
        "p_value": None,
        "yearly_results": "{}",
        "entry_signals": 0,
        "scratch_count": 0,
        "early_exit_count": 0,
    }
    if not outcomes:
        return dict(_empty)

    # Split wins/losses (scratches/early_exits excluded from W/L stats)
    wins = [o for o in outcomes if o["outcome"] == "win"]
    losses = [o for o in outcomes if o["outcome"] == "loss"]
    scratches = [o for o in outcomes if o["outcome"] == "scratch"]
    early_exits = [o for o in outcomes if o["outcome"] == "early_exit"]
    traded = [o for o in outcomes if o["outcome"] in ("win", "loss")]

    n_traded = len(traded)
    entry_signals = len(wins) + len(losses) + len(scratches) + len(early_exits)
    if n_traded == 0:
        result = dict(_empty)
        result["entry_signals"] = entry_signals
        result["scratch_count"] = len(scratches)
        result["early_exit_count"] = len(early_exits)
        return result

    win_rate = len(wins) / n_traded
    loss_rate = 1.0 - win_rate

    avg_win_r = sum(o["pnl_r"] for o in wins) / len(wins) if wins else 0.0
    avg_loss_r = abs(sum(o["pnl_r"] for o in losses) / len(losses)) if losses else 0.0

    # E = (WR * AvgWin_R) - (LR * AvgLoss_R)  [CANONICAL_LOGIC.txt section 4]
    expectancy_r = (win_rate * avg_win_r) - (loss_rate * avg_loss_r)

    # Sharpe ratio: mean(R) / std(R)
    r_values = [o["pnl_r"] for o in traded]
    mean_r = sum(r_values) / len(r_values)
    if len(r_values) > 1:
        variance = sum((r - mean_r) ** 2 for r in r_values) / (len(r_values) - 1)
        std_r = variance**0.5
        sharpe_ratio = mean_r / std_r if std_r > 0 else None
    else:
        sharpe_ratio = None

    # Max drawdown in R (cumulative R-equity curve)
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for o in traded:
        cumulative += o["pnl_r"]
        peak = max(peak, cumulative)
        dd = peak - cumulative
        max_dd = max(max_dd, dd)

    # Yearly breakdown
    yearly = {}
    for o in traded:
        year = str(o["trading_day"].year) if hasattr(o["trading_day"], "year") else str(o["trading_day"])[:4]
        if year not in yearly:
            yearly[year] = {"trades": 0, "wins": 0, "total_r": 0.0}
        yearly[year]["trades"] += 1
        if o["outcome"] == "win":
            yearly[year]["wins"] += 1
        yearly[year]["total_r"] += o["pnl_r"]

    # Compute per-year metrics
    for year_data in yearly.values():
        year_data["win_rate"] = round(year_data["wins"] / year_data["trades"], 4) if year_data["trades"] > 0 else 0.0
        year_data["avg_r"] = round(year_data["total_r"] / year_data["trades"], 4) if year_data["trades"] > 0 else 0.0
        year_data["total_r"] = round(year_data["total_r"], 4)

    # Annualized Sharpe = per_trade_sharpe * sqrt(trades_per_year)
    # Use actual date span (not count of distinct calendar years)
    if traded:
        trading_days = [o["trading_day"] for o in traded]
        min_day = min(trading_days)
        max_day = max(trading_days)
        if hasattr(min_day, "toordinal"):
            span_days = (max_day - min_day).days + 1
        else:
            from datetime import date as _date

            min_day = _date.fromisoformat(str(min_day)[:10])
            max_day = _date.fromisoformat(str(max_day)[:10])
            span_days = (max_day - min_day).days + 1
        years_span = max(span_days / 365.25, 0.25)  # floor at 3 months
    else:
        years_span = 0
    trades_per_year = (n_traded / years_span) if years_span > 0 else 0
    sharpe_ann = sharpe_ratio * (trades_per_year**0.5) if sharpe_ratio is not None and trades_per_year > 0 else None

    # FIX (F-11): Lo (2002) autocorrelation-adjusted annualized Sharpe.
    # Raw sharpe_ann assumes iid returns. If returns are positively
    # autocorrelated, this inflates the annualized figure. Compute lag-1
    # autocorrelation and apply the adjustment factor q = 1 + 2*rho_1.
    # sharpe_ann_adj = sharpe_ann / sqrt(q)
    # Store both: sharpe_ann (backward-compatible) + sharpe_ann_adj (honest).
    sharpe_ann_adj = None
    autocorr_lag1 = None
    if sharpe_ann is not None and len(r_values) >= 10:
        # Lag-1 autocorrelation of R-multiples
        n_r = len(r_values)
        r_demeaned = [r - mean_r for r in r_values]
        numerator = sum(r_demeaned[i] * r_demeaned[i + 1] for i in range(n_r - 1))
        denominator = sum(d * d for d in r_demeaned)
        if denominator > 0:
            autocorr_lag1 = numerator / denominator
            # Lo (2002): q = 1 + 2 * rho_1 (first-order approximation)
            # Clamp q to [0.1, 10] to prevent extreme adjustments from noisy estimates
            q = max(0.1, min(10.0, 1.0 + 2.0 * autocorr_lag1))
            sharpe_ann_adj = sharpe_ann / (q**0.5)

    # FIX (F-04): One-sample t-test p-value for H0: mean(pnl_r) = 0.
    # This tests whether the strategy's edge is statistically distinguishable
    # from random. Uses Welch's t-statistic: t = mean / (std / sqrt(n)).
    # P-value computed from the Student's t-distribution via regularized
    # incomplete beta function (no scipy dependency).
    p_value = None
    if len(r_values) >= 5 and sharpe_ratio is not None:
        t_stat = mean_r / (std_r / (len(r_values) ** 0.5)) if std_r > 0 else None
        if t_stat is not None:
            p_value = _t_test_pvalue(t_stat, len(r_values) - 1)

    # Haircut Sharpe: Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio.
    # Adjusts per-trade Sharpe for selection bias from testing n_trials combos,
    # then annualizes the result. Requires skewness and excess kurtosis.
    sharpe_haircut = None
    skewness = None
    kurtosis_excess = None
    if sharpe_ratio is not None and len(r_values) >= 10 and n_trials > 1 and trades_per_year > 0:
        n_r = len(r_values)
        # Sample skewness: m3 / s^3
        m3 = sum((r - mean_r) ** 3 for r in r_values) / n_r
        skewness = m3 / (std_r**3) if std_r > 0 else 0.0
        # Sample excess kurtosis: m4 / s^4 - 3
        m4 = sum((r - mean_r) ** 4 for r in r_values) / n_r
        kurtosis_excess = (m4 / (std_r**4) - 3.0) if std_r > 0 else 0.0

        sharpe_haircut = _compute_haircut_sharpe(
            sharpe_per_trade=sharpe_ratio,
            n_obs=n_r,
            skewness=skewness,
            kurtosis_excess=kurtosis_excess,
            n_trials=n_trials,
            trades_per_year=trades_per_year,
        )

    # Risk stats (from entry_price and stop_price)
    risk_points_list = [
        abs(o["entry_price"] - o["stop_price"])
        for o in traded
        if o.get("entry_price") is not None and o.get("stop_price") is not None
    ]
    if risk_points_list:
        sorted_risks = sorted(risk_points_list)
        mid = len(sorted_risks) // 2
        if len(sorted_risks) % 2 == 0:
            median_risk = (sorted_risks[mid - 1] + sorted_risks[mid]) / 2
        else:
            median_risk = sorted_risks[mid]
        avg_risk = sum(risk_points_list) / len(risk_points_list)
    else:
        median_risk = None
        avg_risk = None

    # Dollar aggregates (per-contract, approximate)
    # Uses average risk_points across all trades, not per-trade sums.
    # risk_dollars = risk_points * point_value (stop distance only, no friction)
    median_risk_dollars = None
    avg_risk_dollars = None
    avg_win_dollars = None
    avg_loss_dollars = None
    if cost_spec is not None and avg_risk is not None:
        avg_risk_dollars = round(avg_risk * cost_spec.point_value, 2)
        avg_win_dollars = round(avg_win_r * avg_risk_dollars, 2)
        avg_loss_dollars = round(avg_loss_r * avg_risk_dollars, 2)
    if cost_spec is not None and median_risk is not None:
        median_risk_dollars = round(median_risk * cost_spec.point_value, 2)

    return {
        "sample_size": n_traded,
        "win_rate": round(win_rate, 4),
        "avg_win_r": round(avg_win_r, 4),
        "avg_loss_r": round(avg_loss_r, 4),
        "expectancy_r": round(expectancy_r, 4),
        "sharpe_ratio": round(sharpe_ratio, 4) if sharpe_ratio is not None else None,
        "max_drawdown_r": round(max_dd, 4),
        "median_risk_points": round(median_risk, 4) if median_risk is not None else None,
        "avg_risk_points": round(avg_risk, 4) if avg_risk is not None else None,
        "median_risk_dollars": median_risk_dollars,
        "avg_risk_dollars": avg_risk_dollars,
        "avg_win_dollars": avg_win_dollars,
        "avg_loss_dollars": avg_loss_dollars,
        "trades_per_year": round(trades_per_year, 1),
        "sharpe_ann": round(sharpe_ann, 4) if sharpe_ann is not None else None,
        "sharpe_ann_adj": round(sharpe_ann_adj, 4) if sharpe_ann_adj is not None else None,
        "autocorr_lag1": round(autocorr_lag1, 4) if autocorr_lag1 is not None else None,
        "p_value": round(p_value, 6) if p_value is not None else None,
        "sharpe_haircut": sharpe_haircut,
        "skewness": round(skewness, 4) if skewness is not None else None,
        "kurtosis_excess": round(kurtosis_excess, 4) if kurtosis_excess is not None else None,
        "n_trials_at_discovery": n_trials if n_trials > 0 else None,
        "fst_hurdle": compute_fst_hurdle(n_trials) if n_trials > 0 else None,
        "yearly_results": json.dumps(yearly),
        "entry_signals": entry_signals,
        "scratch_count": len(scratches),
        "early_exit_count": len(early_exits),
    }


def compute_dst_split_from_outcomes(outcomes: list[dict], orb_label: str) -> dict:
    """Compute winter/summer split metrics from in-memory outcome list.

    Returns dict with: winter_n, winter_avg_r, summer_n, summer_avg_r, verdict.
    Returns verdict='CLEAN' for non-affected sessions.
    """
    if orb_label not in DST_AFFECTED_SESSIONS:
        return {
            "winter_n": None,
            "winter_avg_r": None,
            "summer_n": None,
            "summer_avg_r": None,
            "verdict": "CLEAN",
        }

    winter_rs = []
    summer_rs = []

    for o in outcomes:
        if o["outcome"] not in ("win", "loss"):
            continue
        td = o["trading_day"]
        if hasattr(td, "date"):
            td = td.date()
        elif not isinstance(td, date):
            td = date.fromisoformat(str(td)[:10])

        is_w = is_winter_for_session(td, orb_label)
        if is_w is None:
            continue
        if is_w:
            winter_rs.append(o["pnl_r"])
        else:
            summer_rs.append(o["pnl_r"])

    winter_n = len(winter_rs)
    summer_n = len(summer_rs)
    winter_avg_r = sum(winter_rs) / winter_n if winter_n > 0 else None
    summer_avg_r = sum(summer_rs) / summer_n if summer_n > 0 else None

    verdict = classify_dst_verdict(winter_avg_r, summer_avg_r, winter_n, summer_n)

    return {
        "winter_n": winter_n,
        "winter_avg_r": round(winter_avg_r, 4) if winter_avg_r is not None else None,
        "summer_n": summer_n,
        "summer_avg_r": round(summer_avg_r, 4) if summer_avg_r is not None else None,
        "verdict": verdict,
    }


def make_strategy_id(
    instrument: str,
    orb_label: str,
    entry_model: str,
    rr_target: float,
    confirm_bars: int,
    filter_type: str,
    dst_regime: str | None = None,
    orb_minutes: int = 5,
    stop_multiplier: float = 1.0,
) -> str:
    """Generate deterministic strategy ID.

    Format: {instrument}_{orb_label}_{entry_model}_RR{rr}_CB{cb}_{filter_type}[_O{min}][_S075][_W|_S]
    Example: MGC_CME_REOPEN_E1_RR2.5_CB2_ORB_G4          (5m default — no suffix)
             MGC_CME_REOPEN_E1_RR2.5_CB2_ORB_G4_O15      (15m ORB)
             MGC_CME_REOPEN_E1_RR2.5_CB2_ORB_G4_O30_W    (30m ORB, winter-only)
             MGC_CME_REOPEN_E1_RR2.5_CB2_ORB_G4_S075     (5m, tight stop 0.75x)
             MGC_CME_REOPEN_E1_RR2.5_CB2_ORB_G4_O15_S075_W  (15m, tight stop, winter)

    Components:
      instrument      - Trading instrument (MGC = Micro Gold Futures)
      orb_label       - ORB session name (CME_REOPEN, TOKYO_OPEN, LONDON_METALS, etc.)
      entry_model     - E1 (next bar open), E2 (stop-market at ORB + slippage), E3 (limit retrace)
      RR              - Risk/Reward target (1.0 to 4.0)
      CB              - Confirm bars required (1 to 5)
      filter_type     - ORB size filter (NO_FILTER, ORB_G4, ORB_L3, etc.)
      _O{min}         - ORB duration suffix; omitted for default 5m
      _S075           - Tight stop suffix (0.75x ORB range); omitted for default 1.0x
      _W/_S           - DST regime suffix (winter/summer); omitted for blended/clean sessions
    """
    base = f"{instrument}_{orb_label}_{entry_model}_RR{rr_target}_CB{confirm_bars}_{filter_type}"
    if orb_minutes != 5:
        base = f"{base}_O{orb_minutes}"
    if stop_multiplier != 1.0:
        # Encode as S075 for 0.75x — integer percentage of 100
        sm_pct = int(round(stop_multiplier * 100))
        base = f"{base}_S{sm_pct:03d}"
    if dst_regime == "winter":
        return f"{base}_W"
    if dst_regime == "summer":
        return f"{base}_S"
    return base


def parse_dst_regime(strategy_id: str) -> str | None:
    """Extract DST regime from strategy_id suffix (_W or _S), or None if blended/clean."""
    if strategy_id.endswith("_W"):
        return "winter"
    if strategy_id.endswith("_S"):
        return "summer"
    return None


def parse_stop_multiplier(strategy_id: str) -> float:
    """Extract stop multiplier from strategy_id _S{pct} suffix, or 1.0 if absent.

    Example: 'MGC_TOKYO_OPEN_E2_RR2.5_CB1_ORB_G4_S075' -> 0.75
             'MGC_TOKYO_OPEN_E2_RR2.5_CB1_ORB_G4_S075_W' -> 0.75
             'MGC_TOKYO_OPEN_E2_RR2.5_CB1_ORB_G4' -> 1.0
    """
    import re

    # Strip DST suffix first
    sid = strategy_id
    if sid.endswith("_W") or sid.endswith("_S"):
        sid = sid[:-2]
    m = re.search(r"_S(\d{3})$", sid)
    if m:
        return int(m.group(1)) / 100.0
    return 1.0


def _load_daily_features(con, instrument, orb_minutes, start_date, end_date):
    """Load all daily_features rows once into a list of dicts."""
    params = [instrument, orb_minutes]
    where = ["symbol = ?", "orb_minutes = ?"]
    if start_date:
        where.append("trading_day >= ?")
        params.append(start_date)
    if end_date:
        where.append("trading_day <= ?")
        params.append(end_date)

    rows = con.execute(
        f"SELECT * FROM daily_features WHERE {' AND '.join(where)} ORDER BY trading_day",
        params,
    ).fetchall()
    cols = [desc[0] for desc in con.description]
    return [dict(zip(cols, r, strict=False)) for r in rows]


def _build_filter_day_sets(features, orb_labels, all_filters):
    """Pre-compute matching day sets for every (filter, orb) combo.

    NOTE (Feb 2026): Double-break exclusion REMOVED. Double-break days are
    real losses in live trading (you can't predict them in advance — the
    opposite break stops you out after entry). Including them gives honest
    discovery metrics. Walk-forward validation already loads unfiltered
    outcomes, so validated strategies were always tested honestly.
    """
    result = {}
    for filter_key, strategy_filter in all_filters.items():
        for orb_label in orb_labels:
            days = set()
            for row in features:
                if row.get(f"orb_{orb_label}_break_dir") is None:
                    continue
                if strategy_filter.matches_row(row, orb_label):
                    days.add(row["trading_day"])
            result[(filter_key, orb_label)] = days
    return result


def _ts_minute_key(ts):
    """Normalize a timestamp to UTC (year, month, day, hour, minute) tuple.

    DuckDB returns TIMESTAMPTZ as local timezone (e.g., Brisbane AEST+10),
    while Python datetime may use timezone.utc. Normalize to UTC for
    consistent comparison.
    """
    utc_ts = ts.astimezone(UTC) if ts.tzinfo is not None else ts
    return (utc_ts.year, utc_ts.month, utc_ts.day, utc_ts.hour, utc_ts.minute)


def _compute_relative_volumes(con, features, instrument, orb_labels, all_filters):
    """
    Pre-compute relative volume at break bar for each (trading_day, orb_label).

    Enriches each feature row dict with rel_vol_{orb_label} key.
    Only runs if at least one VolumeFilter is in all_filters.
    Fail-closed: missing data -> rel_vol stays absent -> filter rejects.
    """
    import statistics

    # Determine max lookback needed across all volume filters
    vol_filters = [f for f in all_filters.values() if isinstance(f, VolumeFilter)]
    if not vol_filters:
        return
    max_lookback = max(f.lookback_days for f in vol_filters)

    # Step 1: Collect all break timestamps and unique UTC minutes-of-day
    break_ts_list = []
    unique_minutes = set()
    for row in features:
        for orb_label in orb_labels:
            break_ts = row.get(f"orb_{orb_label}_break_ts")
            if break_ts is not None and hasattr(break_ts, "hour"):
                break_ts_list.append(break_ts)
                utc_ts = break_ts.astimezone(UTC) if break_ts.tzinfo is not None else break_ts
                unique_minutes.add(utc_ts.hour * 60 + utc_ts.minute)

    if not unique_minutes:
        return

    # Step 2: Load historical volumes for each unique minute-of-day
    # Keyed by minute-of-day, each entry is [(minute_key, volume), ...] sorted chronologically
    minute_history = {}
    for mod in sorted(unique_minutes):
        h, m = divmod(mod, 60)
        rows = con.execute(
            """SELECT ts_utc, volume FROM bars_1m
               WHERE symbol = ?
               AND EXTRACT(HOUR FROM (ts_utc AT TIME ZONE 'UTC')) = ?
               AND EXTRACT(MINUTE FROM (ts_utc AT TIME ZONE 'UTC')) = ?
               ORDER BY ts_utc""",
            [instrument, h, m],
        ).fetchall()
        minute_history[mod] = [(_ts_minute_key(ts), vol) for ts, vol in rows]

    # Step 3: Compute relative volume for each (day, orb_label) break
    for row in features:
        for orb_label in orb_labels:
            break_ts = row.get(f"orb_{orb_label}_break_ts")
            if break_ts is None:
                continue

            break_key = _ts_minute_key(break_ts)
            utc_ts = break_ts.astimezone(UTC) if break_ts.tzinfo is not None else break_ts
            mod = utc_ts.hour * 60 + utc_ts.minute
            history = minute_history.get(mod, [])
            if not history:
                continue  # fail-closed

            # Find this break bar in the chronological history
            idx = None
            for j, (k, _) in enumerate(history):
                if k == break_key:
                    idx = j
                    break
            if idx is None:
                continue  # fail-closed: break bar not in bars_1m

            break_vol = history[idx][1]
            if break_vol is None or break_vol == 0:
                continue  # fail-closed

            # Take prior N entries (up to max_lookback)
            start = max(0, idx - max_lookback)
            prior_vols = [v for _, v in history[start:idx] if v > 0]

            if not prior_vols:
                continue  # fail-closed: no baseline

            baseline = statistics.median(prior_vols)
            if baseline <= 0:
                continue  # fail-closed

            row[f"rel_vol_{orb_label}"] = break_vol / baseline


def _inject_cross_asset_atrs(con, features, instrument, all_filters):
    """Inject cross-asset ATR percentile into feature row dicts.

    For each CrossAssetATRFilter in all_filters, loads the source instrument's
    atr_20_pct from daily_features and injects cross_atr_{source}_pct into
    each target feature row, matched by trading_day.

    Only runs if at least one CrossAssetATRFilter is in all_filters.
    Fail-closed: missing source data → key stays absent → filter rejects.
    """
    cross_filters = [f for f in all_filters.values() if isinstance(f, CrossAssetATRFilter)]
    if not cross_filters:
        return

    # Collect unique source instruments
    sources = {f.source_instrument for f in cross_filters}

    for source in sources:
        if source == instrument:
            continue  # skip self-referencing

        # Bulk-load source instrument's ATR percentile
        rows = con.execute(
            """SELECT trading_day, atr_20_pct FROM daily_features
               WHERE symbol = ? AND orb_minutes = 5 AND atr_20_pct IS NOT NULL
               ORDER BY trading_day""",
            [source],
        ).fetchall()

        if not rows:
            logger.warning(f"No atr_20_pct data for source instrument {source}")
            continue

        # Build lookup: trading_day → atr_20_pct
        source_atr = {}
        for td, pct in rows:
            key = td.date() if hasattr(td, "date") else td
            source_atr[key] = pct

        # Inject into target feature rows
        col_name = f"cross_atr_{source}_pct"
        injected = 0
        for row in features:
            td = row.get("trading_day")
            if td is None:
                continue
            td_key = td.date() if hasattr(td, "date") else td
            pct = source_atr.get(td_key)
            if pct is not None:
                row[col_name] = pct
                injected += 1

        logger.info(f"  Injected {col_name} for {injected}/{len(features)} rows")


def _load_outcomes_bulk(con, instrument, orb_minutes, orb_labels, entry_models, holdout_date=None, start_date=None):
    """
    Load all non-NULL outcomes in one query per (orb, entry_model).

    Args:
        holdout_date: If set, only load outcomes with trading_day < holdout_date.
            This implements true temporal holdout (F-02 audit fix) — discovery
            only sees pre-holdout data, leaving post-holdout for OOS validation.
        start_date: If set, only load outcomes with trading_day >= start_date.
            Used to exclude structurally non-representative early data (e.g.,
            micro-contract launch period).  Defaults to WF_START_OVERRIDE for
            the instrument when called from run_discovery without explicit
            --start CLI override.
            @research-source: 2026-04-09 structural data audit (ATR, volume,
              G-filter pass rates confirm pre-2020 MNQ/MES data is non-
              representative of mature-contract microstructure)

    Returns dict keyed by (orb_label, entry_model, rr_target, confirm_bars)
    with value = list of outcome dicts.
    """
    grouped = {}
    for orb_label in orb_labels:
        for em in entry_models:
            sql = """SELECT trading_day, rr_target, confirm_bars,
                          outcome, pnl_r, mae_r, mfe_r,
                          entry_price, stop_price
                   FROM orb_outcomes
                   WHERE symbol = ? AND orb_minutes = ?
                     AND orb_label = ? AND entry_model = ?
                     AND outcome IS NOT NULL"""
            params = [instrument, orb_minutes, orb_label, em]

            if start_date is not None:
                sql += "\n                     AND trading_day >= ?"
                params.append(start_date)

            if holdout_date is not None:
                sql += "\n                     AND trading_day < ?"
                params.append(holdout_date)

            sql += "\n                   ORDER BY trading_day"
            rows = con.execute(sql, params).fetchall()

            for r in rows:
                key = (orb_label, em, r[1], r[2])  # (orb, em, rr, cb)
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(
                    {
                        "trading_day": r[0],
                        "outcome": r[3],
                        "pnl_r": r[4],
                        "mae_r": r[5],
                        "mfe_r": r[6],
                        "entry_price": r[7],
                        "stop_price": r[8],
                    }
                )

    return grouped


def _inject_hypothesis_filters(
    *,
    scope_predicate: ScopePredicate,
    sessions: list[str],
    all_grid_filters: dict,
    hypothesis_extra_by_session: dict[str, dict],
) -> None:
    """Phase 4 Stage 4.1b: inject hypothesis-declared filter types into the grid.

    When a hypothesis file's ``ScopePredicate`` declares filter_types that are
    NOT in the legacy discovery grid for the current instrument, add them to
    ``all_grid_filters`` (so bulk pre-computation covers them) and to
    ``hypothesis_extra_by_session`` (so the per-session loop merges them in).
    Both dicts are mutated in place.

    Safety rules:
    - Filter must exist in ``ALL_FILTERS``. Unknown strings are silently
      skipped here; ``scope_predicate.accepts()`` will reject the combo at
      combo-enumeration time, producing a clean zero-combos result.
    - DOW composites (``CompositeFilter`` with a ``DayOfWeekSkipFilter``
      overlay) are skipped for sessions in ``DOW_MISALIGNED_SESSIONS``
      because Brisbane DOW != exchange DOW at NYSE_OPEN. A Friday-skip
      would fire on the wrong exchange-local day there.
    - Filters already present in ``all_grid_filters`` are no-ops — the
      legacy grid wins, no duplication into the per-session map.
    - E2 look-ahead filters are still blocked downstream by
      ``is_e2_lookahead_filter()`` during combo enumeration.
    """
    declared_filter_types = scope_predicate.allowed_filter_types()
    for ft in declared_filter_types:
        if ft in all_grid_filters:
            continue  # already in legacy grid
        if ft not in ALL_FILTERS:
            continue  # unknown — scope predicate will reject anyway
        filter_obj = ALL_FILTERS[ft]
        is_dow_composite = isinstance(filter_obj, CompositeFilter) and isinstance(
            filter_obj.overlay, DayOfWeekSkipFilter
        )
        for s in sessions:
            if is_dow_composite and s in DOW_MISALIGNED_SESSIONS:
                continue  # DOW misalignment guard
            hypothesis_extra_by_session[s][ft] = filter_obj
            all_grid_filters[ft] = filter_obj


def run_discovery(
    db_path: Path | None = None,
    instrument: str = "MGC",
    start_date: date | None = None,
    end_date: date | None = None,
    orb_minutes: int = 5,
    dry_run: bool = False,
    dst_regime: str | None = None,
    holdout_date: date | None = None,
    unlock_holdout: str | None = None,
    hypothesis_file: Path | None = None,
) -> int:
    """
    Grid search over all strategy variants.

    Bulk-loads data upfront (1 features query + 18 outcome queries),
    then iterates the grid in Python with no further DB reads.

    Args:
        dst_regime: If 'winter' or 'summer', restrict DST-affected sessions
            (CME_REOPEN/LONDON_METALS/NYSE_OPEN/US_DATA_830) to that regime only. Produces strategy IDs
            with _W or _S suffix. Clean sessions (TOKYO_OPEN/SINGAPORE_OPEN etc.) are unaffected.
            If None (default), produces blended strategies (existing behaviour).
        holdout_date: If set, discovery only uses outcomes with trading_day <
            holdout_date. This creates a true temporal holdout (F-02 audit fix)
            for OOS validation. Use with strategy_validator.py --oos-start to
            test discovered strategies on post-holdout data.
        unlock_holdout: Optional override token to allow holdout_date past the
            sacred boundary (HOLDOUT_SACRED_FROM = 2026-01-01 per Mode A /
            Amendment 2.7). Required for any 2026+ data access. Without this
            token, post-sacred holdout dates raise ValueError. With the
            correct token, a LOUD WARNING is logged and the override is
            allowed — but the resulting strategies are research-provisional
            and CANNOT be promoted to deployment without separate validation
            against a fresh untouched holdout window.
        hypothesis_file: Phase 4 Stage 4.1 — path to a committed pre-registered
            hypothesis YAML file in ``docs/audit/hypotheses/``. When set, the
            discovery routine:
            (a) verifies the file is tracked + clean in git (check_git_cleanliness),
            (b) enforces Criterion 2 MinBTL on the declared trial count
                (enforce_minbtl_bound),
            (c) enforces Amendment 2.7 Mode A consistency on the hypothesis
                holdout_date (check_mode_a_consistency),
            (d) verifies the file SHA has not been used in a prior run
                (check_single_use, runs after connection open),
            (e) builds a ScopePredicate that limits enumeration to the
                hypothesis file's declared (filter_type, sessions, rr_targets,
                entry_models, confirm_bars, stop_multipliers) bundles — this
                is the pre-facto MinBTL constraint per the Stage 4.1 design,
                not a post-facto count check,
            (f) stamps every experimental_strategies row with the file's
                content SHA so downstream drift check #94 can verify
                integrity.
            When None (default), discovery runs in legacy mode with no
            enforcement and no SHA stamping. Legacy mode exists to preserve
            pre-Stage-4.1 test fixtures, null-seed scripts, and
            pipeline/run_full_pipeline.py which is updated in a follow-up
            stage. The validator's Phase 4 gates treat NULL SHA as
            legacy bypass per ``_is_phase_4_grandfathered``.

    Returns count of strategies written.
    """
    # Mode A holdout enforcement — function-level gate (Amendment 2.7).
    # This is the chokepoint: every caller (CLI, tests, research, internal,
    # nested/regime discovery wrappers) goes through enforce_holdout_date here.
    # The CLI main() also calls it (defense in depth) but THIS call is the
    # authoritative one — no Python caller can bypass.
    # Override mechanism: pass unlock_holdout="3656" to allow post-sacred dates.
    # The override is logged loudly and destroys OOS validity.
    from trading_app.holdout_policy import enforce_holdout_date

    holdout_date = enforce_holdout_date(holdout_date, override_token=unlock_holdout)

    if dst_regime not in (None, "winter", "summer"):
        raise ValueError(f"dst_regime must be 'winter', 'summer', or None; got {dst_regime!r}")
    if db_path is None:
        db_path = GOLD_DB_PATH

    # ---- Phase 4 Stage 4.1 enforcement (pre-connection) ----
    # When ``hypothesis_file`` is supplied, run the pure-YAML discipline
    # gates that do not need a DB connection. The single-use check runs
    # below after the connection opens. When ``hypothesis_file`` is None
    # (legacy mode), ALL of this is skipped — the run produces rows with
    # NULL hypothesis_file_sha which the validator treats as grandfathered
    # via _is_phase_4_grandfathered.
    #
    # Failure semantics: any HypothesisLoaderError from these gates
    # propagates out of run_discovery to the caller. The CLI main()
    # translates it into parser.error for a clean exit 2. Function-path
    # callers (tests, research) see the raised exception and can catch
    # it if desired.
    scope_predicate: ScopePredicate | None = None
    hypothesis_sha: str | None = None
    if hypothesis_file is not None:
        logger.info(f"Phase 4 enforcement active: hypothesis file {hypothesis_file}")
        # Gate 1: file must be tracked + clean in git (lock point integrity).
        check_git_cleanliness(hypothesis_file)
        # Gate 2: load + schema-validate YAML.
        h_meta = load_hypothesis_metadata(hypothesis_file)
        # Gate 3: Amendment 2.7 Mode A consistency on holdout_date field.
        check_mode_a_consistency(h_meta)
        # Gate 4: Criterion 2 MinBTL bound. The proxy-mode flag is read
        # from metadata.data_source_mode; default is clean (300 cap).
        source_meta = h_meta.get("metadata", {})
        on_proxy_data = isinstance(source_meta, dict) and source_meta.get("data_source_mode") == "proxy"
        verdict, reason = enforce_minbtl_bound(h_meta, on_proxy_data=on_proxy_data)
        if verdict is not None:
            raise HypothesisLoaderError(reason or "criterion_2: unknown rejection")
        # Gate 5: build the scope predicate for this instrument. Fails
        # closed if no hypothesis in the file declares this instrument.
        scope_predicate = extract_scope_predicate(h_meta, instrument=instrument)
        hypothesis_sha = str(h_meta["sha"])
        logger.info(
            f"  SHA={hypothesis_sha[:12]} "
            f"declared_trials={scope_predicate.total_declared_trials} "
            f"hypotheses={len(scope_predicate.hypotheses)} "
            f"mode={'proxy' if on_proxy_data else 'clean'}"
        )

    import duckdb

    if not dry_run:
        init_trading_app_schema(db_path=db_path)

    with duckdb.connect(str(db_path)) as con:
        from pipeline.db_config import configure_connection

        configure_connection(con, writing=True)

        # Phase 4 Stage 4.1: single-use check runs here (needs DB).
        # Must fire BEFORE the DELETE+INSERT idempotent wipe below, otherwise
        # the prior SHA evidence is gone and re-runs silently succeed.
        # Scout Risk 3 and Phase A reviewer rationale captured in
        # docs/runtime/stages/phase-4-1-discovery-hypothesis-file.md D-4.
        if hypothesis_sha is not None:
            check_single_use(hypothesis_sha, con, orb_minutes=orb_minutes)

        # Determine which sessions to search
        sessions = get_enabled_sessions(instrument)
        if not sessions:
            raise ValueError(
                f"get_enabled_sessions returned empty for {instrument} — "
                f"check pipeline/asset_configs.py enabled_sessions configuration"
            )
        logger.info(f"Sessions: {len(sessions)} enabled for {instrument}")

        # ---- Bulk load phase (all DB reads happen here) ----
        # When holdout_date is set, cap end_date to holdout_date to prevent
        # feature leakage (e.g., relative volume computed with future data)
        effective_end = end_date
        if holdout_date is not None:
            if effective_end is None or holdout_date < effective_end:
                effective_end = holdout_date

        # Structural start boundary: when no explicit --start is provided,
        # default to the instrument's WF_START_OVERRIDE. This ensures the
        # IS sample used for discovery does NOT include structurally non-
        # representative contract-launch data (e.g., MNQ/MES 2019 with
        # ATR 0.42x, volume 0.16x, G8 pass rate 39%).
        # Applied to BOTH daily_features AND outcomes so stats, filter-day
        # sets, and yearly_results are all consistent.
        # @research-source: 2026-04-09 structural data audit, Amendment 3.1
        effective_start = start_date
        if effective_start is None:
            from trading_app.config import WF_START_OVERRIDE

            wf_override = WF_START_OVERRIDE.get(instrument)
            if wf_override is not None:
                effective_start = wf_override
                logger.info(
                    f"  STRUCTURAL START: using WF_START_OVERRIDE={wf_override} "
                    f"for {instrument} (no explicit --start provided)"
                )

        logger.info("Loading daily features...")
        features = _load_daily_features(con, instrument, orb_minutes, effective_start, effective_end)
        logger.info(f"  {len(features)} daily_features rows loaded")

        # Build union of all session-specific filters for bulk pre-computation
        all_grid_filters: dict = {}
        for s in sessions:
            all_grid_filters.update(get_filters_for_grid(instrument, s))

        # Phase 4 Stage 4.1b: Hypothesis-mode filter injection.
        # When a scope predicate is set and declares filter_types that are NOT
        # in the legacy grid for this instrument, inject them from ALL_FILTERS.
        # This allows pre-registered hypothesis files to test filter/session
        # pairs that the legacy grid doesn't enumerate (e.g., GAP_R015 outside
        # MGC/GC CME_REOPEN, or DOW composites outside LONDON_METALS).
        #
        # Extracted into _inject_hypothesis_filters() for isolated unit testing
        # — see TestHypothesisFilterInjection in test_strategy_discovery.py.
        # Legacy mode (scope_predicate is None) is unaffected.
        hypothesis_extra_by_session: dict[str, dict] = {s: {} for s in sessions}
        if scope_predicate is not None:
            _inject_hypothesis_filters(
                scope_predicate=scope_predicate,
                sessions=sessions,
                all_grid_filters=all_grid_filters,
                hypothesis_extra_by_session=hypothesis_extra_by_session,
            )
            injected_count = sum(len(v) for v in hypothesis_extra_by_session.values())
            if injected_count > 0:
                injected_types = sorted(
                    {ft for session_map in hypothesis_extra_by_session.values() for ft in session_map}
                )
                logger.info(
                    f"Phase 4: injected {len(injected_types)} hypothesis filter type(s) "
                    f"across {injected_count} filter/session combinations: {injected_types}"
                )

        logger.info("Computing relative volumes for volume filters...")
        _compute_relative_volumes(con, features, instrument, sessions, all_grid_filters)

        logger.info("Injecting cross-asset ATR percentiles...")
        _inject_cross_asset_atrs(con, features, instrument, all_grid_filters)

        logger.info("Building filter/ORB day sets...")
        filter_days = _build_filter_day_sets(features, sessions, all_grid_filters)

        logger.info("Loading outcomes (bulk)...")
        if holdout_date is not None:
            logger.info(f"  HOLDOUT MODE: only using outcomes before {holdout_date}")
        outcomes_by_key = _load_outcomes_bulk(
            con,
            instrument,
            orb_minutes,
            sessions,
            ENTRY_MODELS,
            holdout_date=holdout_date,
            start_date=effective_start,
        )
        logger.info(f"  {sum(len(v) for v in outcomes_by_key.values())} outcome rows loaded")

        # ---- Grid iteration (pure Python, no DB reads) ----
        # Cost spec for dollar calculations
        cost_spec = get_cost_spec(instrument)

        # Collect all strategies in memory first, then dedup before writing
        all_strategies = []  # list of (strategy_id, filter_key, trade_days, row_data)
        total_combos = 0
        for s in sessions:
            # Include hypothesis-injected filters for per-session count
            nf = len(get_filters_for_grid(instrument, s)) + len(hypothesis_extra_by_session.get(s, {}))
            total_combos += nf * len(RR_TARGETS) * len(CONFIRM_BARS_OPTIONS)  # E1 (all CBs)
            total_combos += nf * len(RR_TARGETS) * 2  # E2+E3 CB1 — E3 skipped at runtime (SKIP_ENTRY_MODELS)
            # but counted here for conservative n_trials_at_discovery (higher FST hurdle)
        # Each base combo is tested at every stop multiplier — count them as
        # separate hypotheses per Harvey & Liu (2015). Different stop distances
        # produce distinct P&L series and distinct p-values.
        total_combos *= len(STOP_MULTIPLIERS)
        combo_idx = 0

        # Phase 4 Stage 4.1: per-hypothesis scope predicate early-exit sets.
        # When scope_predicate is set (Phase 4 enforcement mode), each outer
        # loop level pre-checks the allowed-set to skip iterations early.
        # The UNION-based early exits are a performance optimization — they
        # may over-accept combos that cross-pollinate across hypotheses, but
        # the final per-hypothesis bundle check inside the innermost loop is
        # the authoritative gate. Correctness is preserved; only wasted
        # iterations are eliminated.
        p4_allowed_sessions: frozenset[str] | None = None
        p4_allowed_filter_types: frozenset[str] | None = None
        p4_allowed_entry_models: frozenset[str] | None = None
        p4_allowed_rr_targets: frozenset[float] | None = None
        p4_allowed_confirm_bars: frozenset[int] | None = None
        p4_allowed_stop_mults: frozenset[float] | None = None
        if scope_predicate is not None:
            p4_allowed_sessions = scope_predicate.allowed_sessions()
            p4_allowed_filter_types = scope_predicate.allowed_filter_types()
            p4_allowed_entry_models = scope_predicate.allowed_entry_models()
            p4_allowed_rr_targets = scope_predicate.allowed_rr_targets()
            p4_allowed_confirm_bars = scope_predicate.allowed_confirm_bars()
            p4_allowed_stop_mults = scope_predicate.allowed_stop_multipliers()
        # Raw count of tuples the predicate accepted — the MinBTL-relevant
        # trial count for the safety net assertion after the loop.
        phase_4_accepted_count = 0

        for orb_label in sessions:
            # Phase 4 early-exit: skip sessions not referenced by any hypothesis
            if p4_allowed_sessions is not None and orb_label not in p4_allowed_sessions:
                continue
            session_filters = dict(get_filters_for_grid(instrument, orb_label))
            # Merge hypothesis-injected filters for this session
            session_filters.update(hypothesis_extra_by_session.get(orb_label, {}))
            for filter_key, strategy_filter in session_filters.items():
                # Phase 4 early-exit: skip filter_types not referenced by any
                # hypothesis. strategy_filter.filter_type is the canonical
                # source (Phase A review M-2: never hardcode filter key format).
                if (
                    p4_allowed_filter_types is not None
                    and getattr(strategy_filter, "filter_type", None) not in p4_allowed_filter_types
                ):
                    continue
                matching_day_set = filter_days[(filter_key, orb_label)]

                for em in ENTRY_MODELS:
                    if em in SKIP_ENTRY_MODELS:
                        continue
                    # E2 look-ahead exclusion: skip filters that reference
                    # break-bar data (volume, continuation, speed). E2 enters
                    # on the first touch after ORB end, before the break bar
                    # closes — these filter values are unknowable at entry time.
                    # E1 is unaffected (enters after break bar closes).
                    if em == "E2" and is_e2_lookahead_filter(filter_key):
                        continue
                    # Phase 4 early-exit: skip entry_models not referenced
                    if p4_allowed_entry_models is not None and em not in p4_allowed_entry_models:
                        continue
                    for rr_target in RR_TARGETS:
                        # Phase 4 early-exit: skip rr_targets not referenced
                        if p4_allowed_rr_targets is not None and rr_target not in p4_allowed_rr_targets:
                            continue
                        for cb in CONFIRM_BARS_OPTIONS:
                            if em in ("E2", "E3") and cb > 1:
                                continue
                            # Phase 4 early-exit: skip confirm_bars not referenced
                            if p4_allowed_confirm_bars is not None and cb not in p4_allowed_confirm_bars:
                                continue
                            combo_idx += 1

                            if not matching_day_set:
                                continue

                            # Filter pre-loaded outcomes by matching days
                            all_outcomes = outcomes_by_key.get((orb_label, em, rr_target, cb), [])
                            outcomes = [o for o in all_outcomes if o["trading_day"] in matching_day_set]

                            # Apply DST regime filter for affected sessions
                            session_is_dst_affected = orb_label in DST_AFFECTED_SESSIONS
                            if dst_regime is not None and session_is_dst_affected:
                                want_winter = dst_regime == "winter"
                                outcomes = [
                                    o
                                    for o in outcomes
                                    if is_winter_for_session(o["trading_day"], orb_label) == want_winter
                                ]

                            if not outcomes:
                                continue

                            for stop_mult in STOP_MULTIPLIERS:
                                # Phase 4 early-exit: skip stop_mults not referenced
                                if p4_allowed_stop_mults is not None and stop_mult not in p4_allowed_stop_mults:
                                    continue

                                # Phase 4 FULL per-hypothesis bundle check.
                                # The early-exit sets above filter out most
                                # non-matching combos via cheap set membership
                                # lookups, but they use the UNION across
                                # hypotheses which is permissive. This
                                # innermost check is the authoritative gate:
                                # it requires the (session, filter_type, em,
                                # rr, cb, stop) tuple to match ALL six
                                # dimensions of a SINGLE hypothesis bundle,
                                # preventing cross-pollination across
                                # hypotheses with disjoint scopes.
                                if scope_predicate is not None:
                                    filter_type_str = getattr(strategy_filter, "filter_type", "")
                                    if not scope_predicate.accepts(
                                        orb_label=orb_label,
                                        filter_type=filter_type_str,
                                        entry_model=em,
                                        rr_target=float(rr_target),
                                        confirm_bars=int(cb),
                                        stop_multiplier=float(stop_mult),
                                    ):
                                        continue
                                    # Only count AFTER the predicate accepts.
                                    # phase_4_accepted_count is the RAW trial
                                    # count for the safety-net assertion below;
                                    # it counts every combo the predicate
                                    # accepts AND has non-empty outcomes
                                    # (empty-outcome combos are skipped at the
                                    # `if not outcomes: continue` guard above),
                                    # BEFORE dedup and metric compute. Phase D
                                    # review CONSIDER #3 — doc drift tightening.
                                    phase_4_accepted_count += 1

                                # Apply tight stop simulation (no-op for 1.0x).
                                # Stop variants ARE counted in total_combos (K includes
                                # len(STOP_MULTIPLIERS) factor). Each stop multiplier
                                # produces a distinct P&L series and distinct p-value —
                                # they are separate hypotheses per Harvey & Liu (2015).
                                sim_outcomes = apply_tight_stop(outcomes, stop_mult, cost_spec)

                                metrics = compute_metrics(sim_outcomes, cost_spec=cost_spec, n_trials=total_combos)
                                if metrics["sample_size"] == 0:
                                    continue

                                # Determine effective regime for strategy_id suffix:
                                # use dst_regime if this session is DST-affected, else no suffix
                                effective_regime = dst_regime if session_is_dst_affected else None
                                strategy_id = make_strategy_id(
                                    instrument,
                                    orb_label,
                                    em,
                                    rr_target,
                                    cb,
                                    filter_key,
                                    dst_regime=effective_regime,
                                    orb_minutes=orb_minutes,
                                    stop_multiplier=stop_mult,
                                )
                                trade_days = sorted({o["trading_day"] for o in sim_outcomes})
                                trade_day_hash = compute_trade_day_hash(trade_days)

                                # Compute DST split metadata:
                                # For regime-specific strategies, split is degenerate (one regime only).
                                # For blended strategies, compute the full winter/summer breakdown.
                                if dst_regime is not None and session_is_dst_affected:
                                    # All outcomes are already one regime; set split fields explicitly
                                    n = len(
                                        [o for o in sim_outcomes if o.get("outcome") not in ("scratch", "early_exit")]
                                    )
                                    avg_r = metrics["expectancy_r"]
                                    if dst_regime == "winter":
                                        dst_split = {
                                            "winter_n": n,
                                            "winter_avg_r": avg_r,
                                            "summer_n": 0,
                                            "summer_avg_r": None,
                                            "verdict": "WINTER-ONLY",
                                        }
                                    else:
                                        dst_split = {
                                            "winter_n": 0,
                                            "winter_avg_r": None,
                                            "summer_n": n,
                                            "summer_avg_r": avg_r,
                                            "verdict": "SUMMER-ONLY",
                                        }
                                else:
                                    dst_split = compute_dst_split_from_outcomes(sim_outcomes, orb_label)

                                all_strategies.append(
                                    {
                                        "strategy_id": strategy_id,
                                        "instrument": instrument,
                                        "orb_label": orb_label,
                                        "orb_minutes": orb_minutes,
                                        "rr_target": rr_target,
                                        "confirm_bars": cb,
                                        "entry_model": em,
                                        "filter_key": filter_key,
                                        "filter_params": strategy_filter.to_json(),
                                        "stop_multiplier": stop_mult,
                                        "metrics": metrics,
                                        "trade_day_hash": trade_day_hash,
                                        "dst_split": dst_split,
                                    }
                                )

                if combo_idx % 500 == 0:
                    logger.info(f"  Progress: {combo_idx}/{total_combos} combos, {len(all_strategies)} strategies")

        # Phase 4 Stage 4.1 safety net: raw accepted count must not exceed
        # the hypothesis file's declared total_expected_trials. This is the
        # post-enumeration MinBTL assertion — catches the case where the
        # scope predicate allows more combinations than the author declared
        # (predicate-too-permissive bug). Fails closed BEFORE the DB write
        # phase so no rows land in experimental_strategies from an
        # overshoot run.
        #
        # Phase D review CONSIDER #4: raised as HypothesisLoaderError (not
        # ValueError) so the CLI's main() try/except catches it uniformly
        # with the other Phase 4 discipline failures and exits with a clean
        # parser.error code 2. A predicate-vs-declared mismatch is morally
        # a discipline violation (the hypothesis file's own scope block and
        # declared count disagree), even though its root cause is internal
        # inconsistency rather than operator action.
        if scope_predicate is not None:
            declared = scope_predicate.total_declared_trials
            if phase_4_accepted_count > declared:
                raise HypothesisLoaderError(
                    f"Phase 4 Stage 4.1 safety net: scope predicate accepted "
                    f"{phase_4_accepted_count} raw trials but the hypothesis "
                    f"file declared total_expected_trials={declared}. This "
                    f"is a predicate-too-permissive bug — either the scope "
                    f"blocks need tightening OR the declared count needs "
                    f"increasing in a NEW dated hypothesis file (existing "
                    f"files are immutable per registry README). Aborting "
                    f"before DB write."
                )
            logger.info(
                f"Phase 4 safety net: {phase_4_accepted_count}/{declared} raw trials accepted by scope predicate"
            )
            if phase_4_accepted_count == 0:
                logger.warning(
                    "Phase 4 WARNING: scope predicate accepted ZERO combos. "
                    "Verify that the hypothesis file's filter.type and scope "
                    "fields are consistent with ALL_FILTERS in "
                    "trading_app/config.py. No rows will be written."
                )

        # ---- Dedup: mark canonical vs alias within each group ----
        logger.info(f"Dedup: {len(all_strategies)} strategies, computing canonical...")
        _mark_canonical(all_strategies)
        n_canonical = sum(1 for s in all_strategies if s["is_canonical"])
        n_alias = len(all_strategies) - n_canonical
        logger.info(f"  {n_canonical} canonical, {n_alias} aliases")

        # ---- BH FDR at discovery (Bloomey statistical hardening, Mar 2026) ----
        # Annotate each strategy with BH FDR significance across all K trials.
        # Informational — DSR/FST gates in validation are the hard filters.
        from trading_app.strategy_validator import benjamini_hochberg

        p_pairs = [
            (s["strategy_id"], s["metrics"].get("p_value"))
            for s in all_strategies
            if s["metrics"].get("p_value") is not None
        ]
        fdr_results = benjamini_hochberg(p_pairs, alpha=0.05) if p_pairs else {}
        n_fdr_sig = sum(1 for v in fdr_results.values() if v.get("fdr_significant"))
        logger.info(f"BH FDR at discovery: {n_fdr_sig}/{len(p_pairs)} significant at q=0.05")

        # ---- Batch write (idempotent: DELETE then INSERT OR REPLACE) ----
        if not dry_run:
            if hypothesis_sha is None:
                # LEGACY MODE (no hypothesis file): DELETE all for instrument+
                # aperture to prevent zombie strategies from grid changes
                # (e.g., filter removed from config, entry model disabled).
                # FK-safe: skip rows referenced by validated_setups.promoted_from
                # (grandfathered research-provisional per Amendment 2.4).
                con.execute(
                    """DELETE FROM experimental_strategies
                    WHERE instrument = ? AND orb_minutes = ?
                      AND strategy_id NOT IN (
                          SELECT promoted_from FROM validated_setups
                          WHERE promoted_from IS NOT NULL
                      )""",
                    [instrument, orb_minutes],
                )
            # HYPOTHESIS MODE: skip DELETE. Each hypothesis file defines its
            # own scope — strategies from OTHER files are intentional research
            # artifacts, not zombies. INSERT OR REPLACE (line ~177) handles
            # same-ID overwrites. Single-use SHA enforcement prevents accidental
            # re-runs. This allows multiple hypothesis files to coexist for the
            # same instrument. Fix: 2026-04-10 (GC proxy + MNQ multi-RR wipe).
            insert_batch = []
            for s in all_strategies:
                m = s["metrics"]
                dst = s["dst_split"]
                fdr = fdr_results.get(s["strategy_id"], {})
                insert_batch.append(
                    [
                        s["strategy_id"],
                        s["instrument"],
                        s["orb_label"],
                        s["orb_minutes"],
                        s["rr_target"],
                        s["confirm_bars"],
                        s["entry_model"],
                        s["filter_key"],
                        s["filter_params"],
                        s["stop_multiplier"],
                        m["sample_size"],
                        m["win_rate"],
                        m["avg_win_r"],
                        m["avg_loss_r"],
                        m["expectancy_r"],
                        m["sharpe_ratio"],
                        m["max_drawdown_r"],
                        m["median_risk_points"],
                        m["avg_risk_points"],
                        m["median_risk_dollars"],
                        m["avg_risk_dollars"],
                        m["avg_win_dollars"],
                        m["avg_loss_dollars"],
                        m["trades_per_year"],
                        m["sharpe_ann"],
                        m["yearly_results"],
                        m["entry_signals"],
                        m["scratch_count"],
                        m["early_exit_count"],
                        s["trade_day_hash"],
                        s["is_canonical"],
                        s["canonical_strategy_id"],
                        dst["winter_n"],
                        dst["winter_avg_r"],
                        dst["summer_n"],
                        dst["summer_avg_r"],
                        dst["verdict"],
                        None,
                        None,  # Reset validation_status/notes
                        None,  # Fresh timestamp — COALESCE(NULL, CURRENT_TIMESTAMP)
                        # Audit metrics (F-04, F-11)
                        m.get("p_value"),
                        m.get("sharpe_ann_adj"),
                        m.get("autocorr_lag1"),
                        # Haircut Sharpe (Bailey & Lopez de Prado, 2014)
                        m.get("sharpe_haircut"),
                        m.get("skewness"),
                        m.get("kurtosis_excess"),
                        # Multiple testing audit (Chordia et al 2018, Bailey 2018)
                        m.get("n_trials_at_discovery"),
                        m.get("fst_hurdle"),
                        # BH FDR at discovery (Bloomey hardening)
                        fdr.get("fdr_significant"),
                        fdr.get("adjusted_p"),
                        # Phase 4 Stage 4.1: pre-registered hypothesis file
                        # SHA. Single-valued for the whole batch — either the
                        # CLI was invoked with --hypothesis-file (non-None)
                        # or it was not (None = legacy-mode, all rows NULL).
                        hypothesis_sha,
                    ]
                )

                if len(insert_batch) >= 500:
                    _flush_batch_df(con, insert_batch)
                    insert_batch = []

            if insert_batch:
                _flush_batch_df(con, insert_batch)
            con.commit()

        total_strategies = len(all_strategies)
        logger.info(
            f"Discovered {total_strategies} strategies "
            f"({n_canonical} canonical, {n_alias} aliases) "
            f"from {total_combos} combos"
        )
        if dry_run:
            logger.info("  (DRY RUN -- no data written)")

        return total_strategies


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Grid search over strategy variants")
    parser.add_argument("--instrument", default="MGC", help="Instrument symbol")
    parser.add_argument("--start", type=date.fromisoformat, help="Start date")
    parser.add_argument("--end", type=date.fromisoformat, help="End date")
    parser.add_argument("--orb-minutes", type=int, default=5, help="ORB duration")
    parser.add_argument("--dry-run", action="store_true", help="No DB writes")
    parser.add_argument("--db", type=str, default=None, help="Database path (default: gold.db)")
    parser.add_argument(
        "--dst-regime",
        choices=["winter", "summer"],
        default=None,
        help="Restrict DST-affected sessions (CME_REOPEN/LONDON_METALS/NYSE_OPEN/US_DATA_830) to one regime. "
        "Produces _W or _S strategy IDs. Clean sessions unaffected. "
        "Run twice (--dst-regime winter AND --dst-regime summer) to replace all blended strategies.",
    )
    parser.add_argument(
        "--holdout-date",
        type=date.fromisoformat,
        default=None,
        help="Temporal holdout cutoff (YYYY-MM-DD). Discovery only uses outcomes "
        "BEFORE this date. Use with validator --oos-start for true OOS testing. "
        "Example: --holdout-date 2025-01-01 discovers on pre-2025 data. "
        "DEFAULT: Amendment 2.7 sacred-from date "
        "(trading_app.holdout_policy.HOLDOUT_SACRED_FROM). Values later than "
        "the sacred-from date are rejected per Mode A discipline UNLESS "
        "--unlock-holdout TOKEN is also passed.",
    )
    parser.add_argument(
        "--unlock-holdout",
        type=str,
        default=None,
        help="Override token to allow --holdout-date past the sacred boundary. "
        "Required for any 2026+ data access. Without this token, post-sacred "
        "holdout dates raise ValueError. With the correct token, a LOUD WARNING "
        "is logged and the override is allowed — but the resulting strategies "
        "are RESEARCH-PROVISIONAL and CANNOT be promoted to deployment without "
        "separate validation against a fresh untouched holdout window. "
        "Token value: see trading_app.holdout_policy.HOLDOUT_OVERRIDE_TOKEN. "
        "If you are seeing this help text and don't already know the token, "
        "you almost certainly should NOT be using this flag.",
    )
    parser.add_argument(
        "--hypothesis-file",
        type=Path,
        default=None,
        help="Phase 4 Stage 4.1 — path to a committed pre-registered hypothesis "
        "YAML file in docs/audit/hypotheses/. When set, discovery runs in "
        "Phase 4 enforcement mode: (1) git cleanliness verified, (2) MinBTL "
        "bound enforced on declared trial count, (3) Mode A holdout consistency "
        "verified, (4) single-use enforced (file can only be run once), "
        "(5) enumeration is scope-limited to the hypothesis file's declared "
        "(filter_type, sessions, rr_targets, entry_models, confirm_bars, "
        "stop_multipliers) bundles — NOT a post-facto count check, (6) every "
        "experimental_strategies row is stamped with the file's content SHA. "
        "When unset (default), discovery runs in legacy mode with no "
        "enforcement and no SHA stamping. See docs/audit/hypotheses/README.md "
        "and docs/institutional/pre_registered_criteria.md.",
    )
    args = parser.parse_args()

    # Mode A enforcement (Amendment 2.7 / RESEARCH_RULES.md § "2026 holdout is
    # sacred"). enforce_holdout_date() defaults None to HOLDOUT_SACRED_FROM
    # and raises ValueError on post-sacred values with a clear error citing
    # the canonical source. See trading_app/holdout_policy.py and
    # docs/institutional/pre_registered_criteria.md Amendment 2.7.
    # NOTE: this CLI-level call is defense in depth — run_discovery() also
    # calls enforce_holdout_date as the function-level chokepoint.
    from trading_app.holdout_policy import enforce_holdout_date

    try:
        effective_holdout = enforce_holdout_date(args.holdout_date, override_token=args.unlock_holdout)
    except ValueError as e:
        parser.error(str(e))  # exits with code 2 and prints the message

    db_path = Path(args.db) if args.db else None

    # Phase 4 Stage 4.1: wrap run_discovery to translate HypothesisLoaderError
    # (raised by the loader + gates chain when --hypothesis-file is provided
    # and a discipline violation is caught) into a clean parser.error exit.
    # Without this wrap, the exception would propagate as an unhandled
    # traceback, which is ugly for operators and inconsistent with the
    # enforce_holdout_date error path above.
    try:
        run_discovery(
            db_path=db_path,
            instrument=args.instrument,
            start_date=args.start,
            end_date=args.end,
            orb_minutes=args.orb_minutes,
            dry_run=args.dry_run,
            dst_regime=args.dst_regime,
            holdout_date=effective_holdout,
            unlock_holdout=args.unlock_holdout,
            hypothesis_file=args.hypothesis_file,
        )
    except HypothesisLoaderError as e:
        parser.error(f"Phase 4 hypothesis discipline: {e}")


if __name__ == "__main__":
    main()
