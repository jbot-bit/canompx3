"""
DISCOVERY SAFETY: UNSAFE — writes to validated_setups (derived layer).
Do not use validated_setups output as discovery truth. See CLAUDE.md Project Truth Protocol.

7-phase strategy validation per CANONICAL_LOGIC.txt section 9.

Validates strategies from experimental_strategies and promotes passing
ones to validated_setups. Rejected strategies get validation_notes.

Phases:
  1. Sample size (reject < 30, warn < 100)
  2. Post-cost expectancy > 0
  3. Yearly robustness (positive in ALL years; DORMANT regime waivers available)
  4. Stress test (ExpR > 0 at +50% costs)
  4b. Walk-forward OOS validation (anchored expanding, 6m test windows)
  FDR. BH FDR hard gate (global K across all instruments, α=0.05)
  5. Sharpe ratio (optional quality filter)
  6. Max drawdown (optional risk filter)

Usage:
    python trading_app/strategy_validator.py --instrument MGC
    python trading_app/strategy_validator.py --instrument MGC --no-walkforward
    python trading_app/strategy_validator.py --instrument MGC --min-sample 100 --dry-run
"""

import json
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path

# duckdb lazy-loaded inside the 4 functions that use it (PEP 8 — delayed
# imports for performance; duckdb's own import is modest but deferring keeps
# this module's cold-import path free of the binary DLL load).
from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.audit_log import get_git_sha
from pipeline.cost_model import get_cost_spec, stress_test_costs
from pipeline.dst import (
    DST_AFFECTED_SESSIONS,
    classify_dst_verdict,
    is_winter_for_session,
)
from pipeline.log import get_logger
from pipeline.paths import GOLD_DB_PATH
from trading_app.config import (
    CORE_MIN_SAMPLES,
    MIN_WFE,
    NOISE_FLOOR_BY_INSTRUMENT,
    REGIME_MIN_SAMPLES,
    REGIME_WF_MIN_TRADES_PER_WINDOW,
    REGIME_WF_MIN_TRAIN_TRADES,
    REGIME_WF_MIN_WINDOWS,
    REGIME_WF_TRADE_COUNT,
    WF_MIN_TRAIN_TRADES,
    WF_START_OVERRIDE,
    WF_TRADE_COUNT_OVERRIDE,
)
from trading_app.db_manager import init_trading_app_schema
from trading_app.edge_families import build_edge_families_for_instrument

# Phase 4 Stage 4.0 (2026-04-08) — institutional criteria gates.
# Stage 4.0 enforces criteria 1 (hypothesis file presence), 2 (MinBTL bound),
# 8 (2026 OOS positive, N/A-safe), and 9 (era stability) as pre-flight gates.
#
# Criteria 4 (Chordia) and 5 (DSR) are EXPLICITLY DEFERRED to Stage 4.0b
# for institutional compliance with the locked amendments:
# - Amendment 2.1: DSR is CROSS-CHECK ONLY until N_eff is formally solved per
#   Bailey-LdP 2014 Eq. 9. The existing informational DSR block at the
#   bottom of run_validation already implements this correctly; Stage 4.0
#   does NOT touch it.
# - Amendment 2.2: Chordia is BANDED with a 4-tier ladder that requires
#   composition with BH FDR + WFE + 2026 OOS results. That composition must
#   fire AFTER the legacy validator's FDR and walkforward phases, so it
#   cannot be a pre-flight gate. Stage 4.0b will implement the banded rule
#   as a post-validation check.
from trading_app.holdout_policy import HOLDOUT_GRANDFATHER_CUTOFF, HOLDOUT_SACRED_FROM
from trading_app.hypothesis_loader import (
    HypothesisLoaderError,
    enforce_minbtl_bound,
    load_hypothesis_by_sha,
)
from trading_app.strategy_discovery import parse_dst_regime
from trading_app.validated_shelf import validated_shelf_lifecycle
from trading_app.validation_provenance import StrategyTradeWindowResolver
from trading_app.walkforward import append_walkforward_result

logger = get_logger(__name__)

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]


# =========================================================================
# FDR correction (F-01: Multiple comparison adjustment)
# =========================================================================


def benjamini_hochberg(
    p_values: list[tuple[str, float]],
    alpha: float = 0.05,
    total_tests: int | None = None,
) -> dict[str, dict]:
    """Apply Benjamini-Hochberg FDR correction to a set of p-values.

    Addresses the Bailey Rule: when testing thousands of strategies, some will
    appear profitable by chance. BH controls the False Discovery Rate — the
    expected proportion of false positives among all rejections.

    Args:
        p_values: List of (strategy_id, raw_p_value) tuples. Strategies with
            None p-values are excluded from correction.
        alpha: FDR significance level (default 0.05 = 5% expected false discoveries).
        total_tests: Fixed number of total tests (K) for BH denominator.
            When running validation across instruments in separate batches,
            pass the pre-computed global K to ensure consistent correction.
            If None, uses len(valid_p_values) as the denominator.

    Returns:
        Dict keyed by strategy_id with:
          - raw_p: original p-value
          - adjusted_p: BH-adjusted p-value
          - fdr_significant: True if adjusted_p < alpha
          - fdr_rank: rank position (1 = smallest p-value)

    Reference: Benjamini & Hochberg (1995), "Controlling the False Discovery
    Rate: a Practical and Powerful Approach to Multiple Testing."
    """
    # Filter out None p-values
    valid = [(sid, p) for sid, p in p_values if p is not None]
    if not valid:
        return {}

    # Sort by p-value ascending
    valid.sort(key=lambda x: x[1])
    m = total_tests if total_tests is not None else len(valid)

    # Fail-closed: BH requires m >= n to maintain FDR control.
    # m < n would make the correction anti-conservative (fewer tests than p-values).
    if m < len(valid):
        raise ValueError(
            f"total_tests ({m}) < valid p-values ({len(valid)}): "
            f"BH requires m >= n to maintain FDR control (Benjamini & Hochberg 1995)"
        )

    n = len(valid)
    results = {}
    # BH procedure: adjusted_p[i] = min(p[i] * m / rank, 1.0)
    # m = total tests (may exceed n when some p-values are None / from other batches)
    # Loop iterates over the n valid p-values; m is only used in the adjustment formula.
    # Enforce monotonicity: adjusted_p[i] = min(adjusted_p[i], adjusted_p[i+1])
    prev_adj = 1.0
    for rank_idx in range(n - 1, -1, -1):
        sid, raw_p = valid[rank_idx]
        rank = rank_idx + 1  # 1-indexed rank
        adj_p = min(raw_p * m / rank, 1.0)
        adj_p = min(adj_p, prev_adj)  # monotonicity
        prev_adj = adj_p
        results[sid] = {
            "raw_p": raw_p,
            "adjusted_p": round(adj_p, 6),
            "fdr_significant": adj_p < alpha,
            "fdr_rank": rank,
        }

    return results


def _parse_orb_size_bounds(filter_type: str | None, filter_params: str | None) -> tuple[float | None, float | None]:
    """Extract min_size/max_size from filter_type or filter_params JSON.

    Returns (min_size, max_size). Either may be None.
    """
    # Try filter_params JSON first (most reliable)
    if filter_params:
        try:
            params = json.loads(filter_params) if isinstance(filter_params, str) else filter_params
            min_s = params.get("min_size")
            max_s = params.get("max_size")
            if min_s is not None or max_s is not None:
                return (float(min_s) if min_s is not None else None, float(max_s) if max_s is not None else None)
            # CompositeFilter: size bounds are in params["base"]
            base = params.get("base")
            if isinstance(base, dict):
                min_s = base.get("min_size")
                max_s = base.get("max_size")
                if min_s is not None or max_s is not None:
                    return (float(min_s) if min_s is not None else None, float(max_s) if max_s is not None else None)
        except (json.JSONDecodeError, TypeError, ValueError):
            logger.warning("Corrupt filter_params in _parse_orb_size_bounds: %s", filter_params)

    # Fallback: parse from filter_type string (ORB_G5 -> min=5, ORB_G4_L12 -> min=4/max=12)
    if filter_type and filter_type.startswith("ORB_G"):
        rest = filter_type[5:]  # after "ORB_G"
        # Strip DOW skip suffixes for composite filter_types (e.g. ORB_G4_NOFRI)
        for dow_suffix in ("_NOFRI", "_NOMON", "_NOTUE"):
            if rest.endswith(dow_suffix):
                rest = rest[: -len(dow_suffix)]
                break
        if "_L" in rest:
            parts = rest.split("_L")
            try:
                return (float(parts[0]), float(parts[1]))
            except (ValueError, IndexError):
                pass
        else:
            try:
                return (float(rest), None)
            except ValueError:
                pass

    return (None, None)


def _parse_cost_ratio_cap_pct(filter_type: str | None, filter_params: str | None) -> float | None:
    """Extract max_cost_ratio_pct from filter_type or filter_params JSON."""
    if filter_params:
        try:
            params = json.loads(filter_params) if isinstance(filter_params, str) else filter_params
            max_pct = params.get("max_cost_ratio_pct")
            if max_pct is not None:
                return float(max_pct)
            base = params.get("base")
            if isinstance(base, dict):
                max_pct = base.get("max_cost_ratio_pct")
                if max_pct is not None:
                    return float(max_pct)
        except (json.JSONDecodeError, TypeError, ValueError):
            logger.warning("Corrupt filter_params in _parse_cost_ratio_cap_pct: %s", filter_params)

    if filter_type and filter_type.startswith("COST_LT"):
        try:
            return float(int(filter_type[7:]))
        except ValueError:
            return None

    return None


def _parse_skip_days(filter_params: str | None) -> list[int] | None:
    """Extract skip_days from filter_params JSON (DayOfWeekSkipFilter or CompositeFilter overlay).

    Returns list of weekday ints to exclude, or None if no DOW filter.
    """
    if not filter_params:
        return None
    try:
        params = json.loads(filter_params) if isinstance(filter_params, str) else filter_params
        # Direct DayOfWeekSkipFilter
        sd = params.get("skip_days")
        if sd is not None and len(sd) > 0:
            return [int(d) for d in sd]
        # CompositeFilter: check overlay
        overlay = params.get("overlay")
        if isinstance(overlay, dict):
            sd = overlay.get("skip_days")
            if sd is not None and len(sd) > 0:
                return [int(d) for d in sd]
    except (json.JSONDecodeError, TypeError, ValueError):
        logger.warning("Corrupt filter_params in _parse_skip_days: %s", filter_params)
    return None


def compute_dst_split(
    con,
    strategy_id: str,
    instrument: str,
    orb_label: str,
    entry_model: str,
    rr_target: float,
    confirm_bars: int,
    filter_type: str,
    filter_params: str | None = None,
    orb_minutes: int = 5,
) -> dict:
    """Compute winter/summer split metrics for a strategy at a DST-affected session.

    Queries orb_outcomes joined with daily_features to apply the ORB size filter,
    then tags each trading day as winter or summer and computes separate metrics.

    CRITICAL: The daily_features JOIN must include orb_minutes to avoid duplicate
    rows (daily_features has rows for orb_minutes 5, 15, 30).  Without this filter,
    each outcome joins to 3 daily_features rows, inflating counts ~3-5x.

    Returns dict with keys: winter_n, winter_avg_r, summer_n, summer_avg_r, verdict.
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

    # Build ORB size filter clause from strategy's filter
    min_size, max_size = _parse_orb_size_bounds(filter_type, filter_params)
    max_cost_ratio_pct = _parse_cost_ratio_cap_pct(filter_type, filter_params)
    size_col = f"orb_{orb_label}_size"

    # NOTE: Double-break exclusion removed (Feb 2026). Double-break days
    # are real losses in live trading — including them gives honest metrics.
    size_clauses = []
    size_params = []
    if min_size is not None:
        size_clauses.append(f"df.{size_col} >= ?")
        size_params.append(min_size)
    if max_size is not None:
        size_clauses.append(f"df.{size_col} < ?")
        size_params.append(max_size)
    if max_cost_ratio_pct is not None:
        cost_spec = get_cost_spec(instrument)
        size_clauses.append(f"(100.0 * ? / NULLIF((df.{size_col} * ?) + ?, 0)) < ?")
        size_params.extend(
            [
                cost_spec.total_friction,
                cost_spec.point_value,
                cost_spec.total_friction,
                max_cost_ratio_pct,
            ]
        )

    # DOW skip filter (CompositeFilter with DayOfWeekSkipFilter overlay)
    skip_days = _parse_skip_days(filter_params)
    if skip_days:
        placeholders = ", ".join("?" * len(skip_days))
        size_clauses.append(f"df.day_of_week NOT IN ({placeholders})")
        size_params.extend(skip_days)

    size_where = (" AND " + " AND ".join(size_clauses)) if size_clauses else ""

    rows = con.execute(
        f"""
        SELECT o.trading_day, o.pnl_r
        FROM orb_outcomes o
        JOIN daily_features df
          ON o.trading_day = df.trading_day
          AND df.symbol = ?
          AND df.orb_minutes = ?
        WHERE o.symbol = ?
          AND o.orb_label = ?
          AND o.orb_minutes = df.orb_minutes
          AND o.entry_model = ?
          AND o.rr_target = ?
          AND o.confirm_bars = ?
          AND o.outcome IN ('win', 'loss')
          {size_where}
        ORDER BY o.trading_day
    """,
        [instrument, orb_minutes, instrument, orb_label, entry_model, rr_target, confirm_bars] + size_params,
    ).fetchall()

    if not rows:
        return {
            "winter_n": 0,
            "winter_avg_r": None,
            "summer_n": 0,
            "summer_avg_r": None,
            "verdict": "LOW-N",
        }

    # Split by DST regime
    winter_rs = []
    summer_rs = []

    for trading_day, pnl_r in rows:
        # trading_day may come back as date or datetime
        if hasattr(trading_day, "date"):
            td = trading_day.date()
        elif isinstance(trading_day, date):
            td = trading_day
        else:
            td = date.fromisoformat(str(trading_day))

        is_w = is_winter_for_session(td, orb_label)
        if is_w is None:
            continue  # Should not happen for affected sessions
        if is_w:
            winter_rs.append(pnl_r)
        else:
            summer_rs.append(pnl_r)

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


def classify_regime(atr_20: float) -> str:
    """Classify market regime from mean ATR(20)."""
    # @research-source: MGC regime analysis (Mar 2026), see memory/mgc_regime_analysis.md
    # ATR < 20 = dormant (pre-2022 MGC regime), ATR 20-30 = marginal transition zone
    if atr_20 < 20.0:
        return "DORMANT"
    elif atr_20 < 30.0:
        return "MARGINAL"
    return "ACTIVE"


def validate_strategy(
    row: dict,
    cost_spec,
    # @research-source: engineering safety margin (50% cost buffer). No academic
    # derivation. The concept of cost stress testing is standard (Chan, Carver) but
    # the 1.5x multiplier is a project heuristic.
    # @sensitivity-tested: 2026-03-23. All 9 live strategies have ExpR 0.11-0.40,
    # wide margin above breakeven. Robust at 1.2x-1.8x. Not currently load-bearing.
    stress_multiplier: float = 1.5,
    min_sample: int = REGIME_MIN_SAMPLES,
    min_sharpe: float | None = None,
    max_drawdown: float | None = None,
    exclude_years: set[int] | None = None,
    min_years_positive_pct: float = 0.75,  # @research-source Fitschen "Building Reliable Trading Systems" — 85% of top CTAs have ≥1 losing year
    min_trades_per_year: int = 1,
    atr_by_year: dict[int, float] | None = None,
    enable_regime_waivers: bool = True,
) -> tuple[str, str, list[int]]:
    """
    Run 6-phase validation on a single strategy row.

    Args:
        row: Dict from experimental_strategies query
        cost_spec: CostSpec for the instrument
        stress_multiplier: Cost increase multiplier for stress test
        min_sample: Minimum sample size (reject below this)
        min_sharpe: Optional minimum Sharpe ratio (Phase 5)
        max_drawdown: Optional max drawdown in R (Phase 6)
        exclude_years: Years to exclude from Phase 3 yearly check
        min_years_positive_pct: Fraction of included years that must be
            positive (0.0-1.0). Default 0.75 per Fitschen (85% of top CTAs have a bad year).
        min_trades_per_year: Minimum trades for a year to count in Phase 3.
            Years below this are excluded. Default 1 = include all years.
            REGIME strategies (N<100) use 5 via run_validation() dispatch.
        atr_by_year: Mean ATR(20) per year for regime classification.
            Pre-fetched in run_validation(), None disables waivers.
        enable_regime_waivers: If True (default), grant DORMANT waivers
            for negative years with mean ATR < 20 and <= 5 trades.

    Returns:
        (status, notes, regime_waivers): "PASSED" or "REJECTED", with
        explanation, plus list of waived years (empty if none).
    """
    notes = []
    if exclude_years is None:
        exclude_years = set()

    # Phase 1: Sample size
    sample = row.get("sample_size") or 0
    if sample < min_sample:
        return "REJECTED", f"Phase 1: Sample size {sample} < {min_sample}", []
    if sample < CORE_MIN_SAMPLES:
        notes.append(f"Phase 1 WARN: sample={sample} (< {CORE_MIN_SAMPLES})")

    # Phase 2: Post-cost expectancy
    exp_r = row.get("expectancy_r")
    if exp_r is None or exp_r <= 0:
        return "REJECTED", f"Phase 2: ExpR={exp_r} <= 0", []

    # Phase 2b: REMOVED (2026-03-21).
    # Noise floor is now a post-validation flag (noise_risk), not a pre-WF hard gate.
    # See: methodology audit — floor kills strategies before WF/FDR can test them,
    # destroying evidence of OOS consistency. Moved downstream per canon lock.

    # Phase 3: Yearly robustness
    yearly_json = row.get("yearly_results", "{}")
    try:
        yearly = json.loads(yearly_json) if isinstance(yearly_json, str) else yearly_json
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("Corrupt yearly_results JSON for %s: %s", row.get("strategy_id", "?"), exc)
        yearly = {}

    if not yearly:
        return "REJECTED", "Phase 3: No yearly data", []

    included_years = {
        y: d for y, d in yearly.items() if int(y) not in exclude_years and d.get("trades", 0) >= min_trades_per_year
    }
    if not included_years:
        return "REJECTED", "Phase 3: No yearly data after exclusions", []

    neg_years = {y: d for y, d in included_years.items() if d.get("avg_r", 0) <= 0}
    pos_count = len(included_years) - len(neg_years)
    regime_waivers = []

    if neg_years:
        # Regime waiver logic: DORMANT years (ATR < 20, <= 5 trades) can be waived.
        # Waived years count as passing for the % threshold calculation.
        # FIX (2026-03-26): Previously, ANY unwaived negative year caused immediate
        # rejection, bypassing min_years_positive_pct. This made the threshold
        # effectively 100% for instruments without DORMANT years (MNQ, MES).
        # Now: waived years are excluded from the failure count, then the standard
        # min_years_positive_pct threshold applies to the remaining set.
        if enable_regime_waivers and atr_by_year:
            waived = []
            for y, d in neg_years.items():
                yr_int = int(y)
                mean_atr = atr_by_year.get(yr_int)
                trades = d.get("trades", 0)
                # @research-source: MGC regime analysis (Mar 2026) — DORMANT years
                # typically have 0-5 trades; waiving years with >5 trades risks
                # masking real negative signal
                if mean_atr is not None and classify_regime(mean_atr) == "DORMANT" and trades <= 5:
                    waived.append(yr_int)

            unwaived_neg = [y for y in neg_years if int(y) not in waived]

            if pos_count == 0:
                return (
                    "REJECTED",
                    "Phase 3: All years require DORMANT waiver, need at least 1 clean positive year",
                    [],
                )

            # Count waived years as passing, then apply pct threshold
            effective_pos = pos_count + len(waived)
            effective_total = len(included_years)
            pct_positive = effective_pos / effective_total if effective_total > 0 else 0

            if pct_positive < min_years_positive_pct:
                neg_list = sorted(unwaived_neg)
                return (
                    "REJECTED",
                    (
                        f"Phase 3: {effective_pos}/{effective_total} years positive "
                        f"({pct_positive:.0%} < {min_years_positive_pct:.0%}), "
                        f"unwaived negative: {', '.join(neg_list)}" + (f", waived: {sorted(waived)}" if waived else "")
                    ),
                    [],
                )

            regime_waivers = sorted(waived)
            for yr in regime_waivers:
                y_str = str(yr)
                d = neg_years[y_str]
                notes.append(
                    f"Year {yr} waived: DORMANT regime (mean_atr={atr_by_year[yr]:.1f}, trades={d.get('trades', 0)})"
                )
            if unwaived_neg:
                notes.append(
                    f"Phase 3: {len(unwaived_neg)} non-waivable negative year(s) "
                    f"({', '.join(sorted(unwaived_neg))}), within {min_years_positive_pct:.0%} threshold"
                )
        else:
            # Strict mode (no waiver data available)
            pct_positive = pos_count / len(included_years)
            if pct_positive < min_years_positive_pct:
                neg_list = sorted(neg_years.keys())
                return (
                    "REJECTED",
                    (
                        f"Phase 3: {pos_count}/{len(included_years)} years positive "
                        f"({pct_positive:.0%} < {min_years_positive_pct:.0%}), "
                        f"negative: {', '.join(neg_list)}"
                    ),
                    [],
                )

    # Phase 4: Stress test — "ExpR > 0 at +50% costs"
    # Compute extra friction per trade in R-multiples.
    # delta_r = extra_cost_dollars / risk_dollars
    # Risk source (preferred → fallback):
    #   1. median_risk_points from row (outcome-based, if available)
    #   2. avg_risk_points from row (outcome-based, if available)
    #   3. tick-based floor: min_ticks_floor * tick_size (from CostSpec)
    stressed = stress_test_costs(cost_spec, stress_multiplier)
    stress_friction_delta = stressed.total_friction - cost_spec.total_friction

    # Determine risk denominator
    median_risk = row.get("median_risk_points")
    avg_risk = row.get("avg_risk_points")
    if median_risk is not None and median_risk > 0:
        strategy_risk_points = median_risk
    elif avg_risk is not None and avg_risk > 0:
        strategy_risk_points = avg_risk
    else:
        strategy_risk_points = cost_spec.min_risk_floor_points
        logger.warning(
            "Risk fallback to min_risk_floor_points=%.4f for %s — "
            "median_risk=%s, avg_risk=%s. Stress test denominator may be compressed.",
            cost_spec.min_risk_floor_points,
            row.get("strategy_id", "?"),
            median_risk,
            avg_risk,
        )

    strategy_risk_dollars = strategy_risk_points * cost_spec.point_value
    # Floor: never below tick-based minimum
    denom = max(strategy_risk_dollars, cost_spec.min_risk_floor_dollars)
    extra_cost_per_trade_r = stress_friction_delta / denom
    stress_exp = exp_r - extra_cost_per_trade_r

    if stress_exp <= 0:
        return (
            "REJECTED",
            f"Phase 4: Stress ExpR={stress_exp:.4f} <= 0 (base={exp_r}, delta_r={extra_cost_per_trade_r:.4f}, risk_pts={strategy_risk_points:.2f})",
            [],
        )

    # Phase 4c/4d (DSR, FST) — REMOVED as fake gates (2026-03-18 adversarial review).
    # Both used raw n_trials (K≈120K) without N_eff correction for correlated tests
    # (BLP 2014 Appendix A.3). FST passed 13/116,900 strategies — the hurdle was
    # broken, not selective. DSR had the same N_eff inflation.
    # Multiple testing is now handled by BH FDR hard gate (global K) in Phase C.
    # Data columns (sharpe_haircut, fst_hurdle) remain in experimental_strategies
    # for future use if N_eff estimation is implemented.

    # Phase 5: Sharpe ratio (optional)
    if min_sharpe is not None:
        sharpe = row.get("sharpe_ratio")
        if sharpe is None or sharpe < min_sharpe:
            return "REJECTED", f"Phase 5: Sharpe={sharpe} < {min_sharpe}", []

    # Phase 6: Max drawdown (optional)
    if max_drawdown is not None:
        dd = row.get("max_drawdown_r")
        if dd is not None and dd > max_drawdown:
            return "REJECTED", f"Phase 6: MaxDD={dd}R > {max_drawdown}R", []

    status = "PASSED"
    if notes:
        return status, "; ".join(notes), regime_waivers
    return status, "All phases passed", regime_waivers


def _walkforward_worker(
    strategy_id: str,
    instrument: str,
    orb_label: str,
    entry_model: str,
    rr_target: float,
    confirm_bars: int,
    filter_type: str,
    filter_params: str | None,
    orb_minutes: int,
    db_path_str: str,
    wf_params: dict,
    dst_regime: str | None,
    dst_verdict_from_discovery: str | None,
    dst_cols_from_discovery: dict | None,
    wf_start_date: date | None = None,
    stop_multiplier: float = 1.0,
    wf_test_trades: int | None = None,
    wf_min_train_trades: int | None = None,
) -> dict:
    """Worker function for parallel walkforward. Runs in a subprocess.

    Opens its own read-only DuckDB connection. Returns a plain dict
    (must be serialization-safe — no connection objects, no DataFrames).
    """
    t0 = time.monotonic()

    import duckdb

    from pipeline.db_config import configure_connection
    from trading_app.walkforward import run_walkforward

    result = {
        "strategy_id": strategy_id,
        "wf_result": None,
        "dst_split": None,
        "wf_duration_s": 0.0,
        "error": None,
    }

    try:
        with duckdb.connect(db_path_str, read_only=True) as con:
            configure_connection(con, writing=False)

            # Phase 4b: Walk-forward
            # Load cost spec for tight stop re-simulation in WF windows
            from pipeline.cost_model import get_cost_spec

            wf_cost_spec = get_cost_spec(instrument) if stop_multiplier != 1.0 else None

            wf_result = run_walkforward(
                con=con,
                strategy_id=strategy_id,
                instrument=instrument,
                orb_label=orb_label,
                entry_model=entry_model,
                rr_target=rr_target,
                confirm_bars=confirm_bars,
                filter_type=filter_type,
                orb_minutes=orb_minutes,
                test_window_months=wf_params["test_window_months"],
                min_train_months=wf_params["min_train_months"],
                min_trades_per_window=wf_params["min_trades"],
                min_valid_windows=wf_params["min_windows"],
                min_pct_positive=wf_params["min_pct_positive"],
                dst_regime=dst_regime,
                wf_start_date=wf_start_date,
                stop_multiplier=stop_multiplier,
                cost_spec=wf_cost_spec,
                test_window_trades=wf_test_trades,
                min_train_trades=wf_min_train_trades,
            )
            result["wf_result"] = {
                "passed": wf_result.passed,
                "rejection_reason": wf_result.rejection_reason,
                "as_dict": {k: v for k, v in wf_result.__dict__.items()},
            }

            # DST split (recompute only for blended strategies missing data)
            if dst_verdict_from_discovery is not None:
                result["dst_split"] = dst_cols_from_discovery
            elif dst_regime is None:
                dst_split = compute_dst_split(
                    con,
                    strategy_id,
                    instrument,
                    orb_label=orb_label,
                    entry_model=entry_model,
                    rr_target=rr_target,
                    confirm_bars=confirm_bars,
                    filter_type=filter_type,
                    filter_params=filter_params,
                    orb_minutes=orb_minutes,
                )
                result["dst_split"] = dst_split
            else:
                result["dst_split"] = {
                    "winter_n": None,
                    "winter_avg_r": None,
                    "summer_n": None,
                    "summer_avg_r": None,
                    "verdict": None,
                }

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        logger.warning("Walk-forward failed for %s: %s", result.get("strategy_id", "?"), e)

    result["wf_duration_s"] = time.monotonic() - t0
    return result


# =========================================================================
# Phase 4 Stage 4.0 — institutional criteria pre-flight gates
# =========================================================================
#
# These gates implement criteria 1, 2, 8, 9 of the locked institutional
# criteria in docs/institutional/pre_registered_criteria.md. They run BEFORE
# the existing validate_strategy() per-row pipeline as a pre-flight pass.
#
# Criteria 4 (Chordia) and 5 (DSR) are DEFERRED to Stage 4.0b per
# Amendments 2.1 and 2.2 of the locked criteria file. See the imports-
# section comment earlier in this file for the full deferral rationale.
# Short version: Amendment 2.1 makes DSR cross-check only until N_eff is
# formally solved; Amendment 2.2 makes Chordia a 4-band ladder requiring
# BH FDR + WFE + 2026 OOS composition, which cannot fire as a pre-flight
# gate. Stage 4.0b will implement both as post-validation checks.
#
# Grandfather skip: rows with created_at <= HOLDOUT_GRANDFATHER_CUTOFF
# (2026-04-08 00:00:00 UTC) are exempt from these new gates and continue to
# use the legacy validator path. This protects the 124 existing
# validated_setups rows from retroactive rejection per Amendment 2.4.
# Rows with NULL hypothesis_file_sha also bypass — the Stage 4.1 drift
# check catches post-cutoff bypass at validation-startup time.
#
# Gate order: grandfather → C1 → C2 → C9 → C8. C9 fires before C8 because
# era stability is a pure JSON parse while C8 issues a DuckDB query per
# row; short-circuit the expensive IO gate whenever a cheaper local gate
# can reject first.
#
# See docs/plans/2026-04-08-phase-4-clean-rediscovery-design.md § Stage 4.0.


def _is_phase_4_grandfathered(row_dict: dict) -> bool:
    """Return True if the experimental row should bypass Phase 4 gates.

    Two conditions trigger bypass; either alone is sufficient:

    1. ``created_at <= HOLDOUT_GRANDFATHER_CUTOFF`` — the row pre-dates the
       Amendment 2.7 commit (2026-04-08 00:00:00 UTC), so it is one of the
       124 historical experimental_strategies rows that Amendment 2.4
       grandfathered as research-provisional. These rows must not be
       retroactively rejected.

    2. ``hypothesis_file_sha IS NULL`` — the row was created without
       Phase 4 awareness (legacy code path, synthetic test fixture, or
       a discovery run that bypassed Stage 4.1's --hypothesis-file
       requirement). The validator treats this as "not opt-in to Phase 4"
       and applies the legacy gates only. The Stage 4.1 drift check is
       the bridge that catches the "post-cutoff bypass" case — it
       asserts that every post-cutoff experimental row carries a non-null
       SHA, and fires at validation-startup time, not per-row.

    The distinction matters because the validator must remain compatible
    with synthetic test fixtures and legacy callers (nested/regime
    discovery, null seed runs). A row that genuinely opts in to Phase 4
    by carrying a SHA gets the strict gates; everything else passes
    through to the legacy path.

    A NULL ``created_at`` is treated as grandfathered (defensive: legacy
    rows with missing timestamps must not be retroactively rejected).
    """
    if row_dict.get("hypothesis_file_sha") is None:
        return True
    created_at = row_dict.get("created_at")
    if created_at is None:
        return True
    # DuckDB returns datetime with tzinfo for TIMESTAMPTZ columns. The
    # cutoff is also tz-aware. Direct comparison is well-defined.
    if isinstance(created_at, datetime):
        return created_at <= HOLDOUT_GRANDFATHER_CUTOFF
    # Defensive: any other type → grandfather (do not retroactively reject)
    return True


def _check_criterion_1_hypothesis_file(row_dict: dict) -> tuple[str | None, str | None]:
    """Criterion 1: pre-registered hypothesis file present and discoverable.

    Locked text (`docs/institutional/pre_registered_criteria.md`):
    "Before any discovery run, a pre-registered hypothesis file must exist at
    `docs/audit/hypotheses/YYYY-MM-DD-<slug>.yaml`."

    Implementation:
    - The experimental row must carry a non-null ``hypothesis_file_sha``
    - The SHA must resolve to a real file in the registry directory
    - The file must parse as a valid hypothesis schema

    Returns
    -------
    tuple[str | None, str | None]
        ``(None, None)`` if the criterion passes.
        ``("REJECTED", "criterion_1: ...")`` if the criterion fails.
    """
    sha = row_dict.get("hypothesis_file_sha")
    if not sha:
        return (
            "REJECTED",
            "criterion_1: experimental row has no hypothesis_file_sha "
            "(post-cutoff row requires pre-registered hypothesis file)",
        )
    try:
        meta = load_hypothesis_by_sha(sha)
    except HypothesisLoaderError as exc:
        return ("REJECTED", f"criterion_1: hypothesis file load error: {exc}")
    if meta is None:
        return (
            "REJECTED",
            f"criterion_1: hypothesis_file_sha={sha[:12]}... not found in "
            "docs/audit/hypotheses/ (was the file committed?)",
        )
    return (None, None)


def _check_criterion_2_minbtl(meta: dict, on_proxy_data: bool = False) -> tuple[str | None, str | None]:
    """Criterion 2: hypothesis-file declared trial count must satisfy MinBTL.

    **Phase 4 Stage 4.1b delegation.** This function is now a thin wrapper
    that delegates to the canonical implementation in
    ``trading_app.hypothesis_loader.enforce_minbtl_bound``. The 300/2000
    bounds, the proxy-mode disclosure opt-in, and all Criterion 2 semantics
    live exclusively in the loader module so that the validator and the
    discovery-side CLI cannot drift. See
    ``.claude/rules/institutional-rigor.md`` rule 4 ("delegate to canonical
    sources — never re-encode").

    The function name, signature, and return shape are PRESERVED so that:
    - Drift check #93 (``check_phase_4_validator_gates_present``) continues
      to assert this function exists in the validator module.
    - The call site in ``_check_phase_4_pre_flight_gates`` needs zero
      changes across the Stage 4.0 → 4.1b transition.

    The ``on_proxy_data`` flag passthrough is also preserved. Stage 4.0
    reserved it for Stage 4.1 to activate based on whether a hypothesis
    file's scope includes pre-2024-02-05 trading days; that activation is
    a future sub-stage concern, not Stage 4.1b's.

    Parameters
    ----------
    meta
        Hypothesis file metadata dict from ``load_hypothesis_metadata``.
        Must contain ``total_expected_trials``; proxy mode additionally
        requires ``metadata.data_source_mode == 'proxy'`` and non-empty
        ``metadata.data_source_disclosure``.
    on_proxy_data
        When True, the proxy-extended bound (2000) is consulted and the
        disclosure opt-in is enforced. Default False = clean-data bound (300).

    Returns
    -------
    tuple[str | None, str | None]
        ``(None, None)`` if the criterion passes.
        ``("REJECTED", "criterion_2: ...")`` otherwise. The reason text
        comes from the canonical loader implementation.
    """
    # Translate loader-level HypothesisLoaderError (malformed metadata) into
    # a Criterion 2 rejection so the validator's run_validation loop treats
    # it as a per-row soft failure rather than a stop-the-world exception.
    # Stage 4.0's inline version returned this shape for the missing-field
    # case; the delegation preserves that behavior at the boundary.
    try:
        return enforce_minbtl_bound(meta, on_proxy_data=on_proxy_data)
    except HypothesisLoaderError as exc:
        return (
            "REJECTED",
            f"criterion_2: {exc}",
        )


# Criteria 4 (Chordia) and 5 (DSR) are deferred to Stage 4.0b. See the
# module-level comment near the imports for the institutional reasoning.
# Amendment 2.1 (locked) makes Criterion 5 cross-check only; the existing
# informational DSR block at the bottom of run_validation already writes
# dsr_score to validated_setups for audit. Amendment 2.2 (locked) makes
# Criterion 4 a 4-band ladder that requires BH FDR + WFE + 2026 OOS
# composition and therefore cannot fire as a pre-flight gate.


def _check_criterion_8_oos(
    row_dict: dict, db_path: Path | None, *, strict_oos_n: bool = False
) -> tuple[str | None, str | None]:
    verdict = _evaluate_criterion_8_oos(row_dict, db_path, strict_oos_n=strict_oos_n)
    return verdict["status"], verdict["reason"]


def _evaluate_criterion_8_oos(row_dict: dict, db_path: Path | None, *, strict_oos_n: bool = False) -> dict[str, object]:
    """Criterion 8: 2026 out-of-sample positive (with N/A safety).

    Queries ``orb_outcomes`` joined with ``daily_features`` for trading days
    on or after ``HOLDOUT_SACRED_FROM`` (2026-01-01), applies the candidate's
    canonical filter via ``ALL_FILTERS[filter_type].matches_row()``, and
    computes the OOS expectancy-R from matching trades.

    The locked threshold is ``OOS_ExpR >= 0`` AND ``OOS_ExpR >= 0.40 * IS_ExpR``.

    Parameters
    ----------
    strict_oos_n
        When True (Pathway B / individual testing mode), reject hard if
        ``N_oos`` is below the minimum threshold instead of silently passing
        through.  Amendment 3.0 condition 4 forbids "insufficient OOS data
        exemptions" for Pathway B.  Default False preserves Pathway A
        legacy permissive behaviour.

    N/A safety: if zero OOS trades exist (e.g., synthetic test DB with no
    2026 data), the gate returns N/A pass-through. This prevents pre-2026
    integration tests and null seed runs from regressing.

    Triple-join trap: the join MUST include ``orb_minutes`` to prevent the
    3x row inflation. Pattern follows ``compute_dst_split`` at validator
    line 314.

    Reading the sacred 2026 window for VALIDATION is allowed under Mode A;
    only DISCOVERY writes are forbidden. See ``trading_app.holdout_policy``.
    """
    # Lazy import to avoid a circular import at module load time:
    # trading_app.config is heavy and imports many siblings.
    from trading_app.config import ALL_FILTERS

    filter_type = row_dict.get("filter_type")
    if not filter_type:
        return {
            "status": "REJECTED",
            "reason": "criterion_8: experimental row missing filter_type (cannot apply OOS filter)",
            "c8_oos_status": None,
            "n_oos": None,
            "oos_expectancy_r": None,
            "oos_is_ratio": None,
        }
    if filter_type not in ALL_FILTERS:
        return {
            "status": "REJECTED",
            "reason": f"criterion_8: filter_type='{filter_type}' not in ALL_FILTERS registry",
            "c8_oos_status": None,
            "n_oos": None,
            "oos_expectancy_r": None,
            "oos_is_ratio": None,
        }
    filter_obj = ALL_FILTERS[filter_type]
    instrument = row_dict.get("instrument")
    orb_label = row_dict.get("orb_label")
    orb_minutes = row_dict.get("orb_minutes")
    entry_model = row_dict.get("entry_model")
    confirm_bars = row_dict.get("confirm_bars")
    rr_target = row_dict.get("rr_target")
    is_expr = row_dict.get("expectancy_r")
    if any(v is None for v in (instrument, orb_label, orb_minutes, entry_model, confirm_bars, rr_target, is_expr)):
        return {
            "status": "REJECTED",
            "reason": "criterion_8: experimental row missing required dimensions for OOS query",
            "c8_oos_status": None,
            "n_oos": None,
            "oos_expectancy_r": None,
            "oos_is_ratio": None,
        }
    # Re-bind to local typed names so the static checker sees the narrowing
    # the runtime check above already enforced.
    orb_label_str: str = str(orb_label)
    is_expr_f: float = float(is_expr)  # type: ignore[arg-type]

    import duckdb

    effective_db = db_path if db_path is not None else GOLD_DB_PATH
    oos_pnl_r: list[float] = []
    with duckdb.connect(str(effective_db), read_only=True) as oos_con:
        # Triple-join with orb_minutes prevents the 3x inflation trap.
        oos_rows = oos_con.execute(
            """
            SELECT o.*, d.*
            FROM orb_outcomes o
            JOIN daily_features d
              ON o.symbol = d.symbol
             AND o.trading_day = d.trading_day
             AND o.orb_minutes = d.orb_minutes
            WHERE o.symbol = ?
              AND o.orb_label = ?
              AND o.orb_minutes = ?
              AND o.entry_model = ?
              AND o.confirm_bars = ?
              AND o.rr_target = ?
              AND o.trading_day >= ?
            """,
            [instrument, orb_label, orb_minutes, entry_model, confirm_bars, rr_target, HOLDOUT_SACRED_FROM],
        ).fetchall()
        col_names = [desc[0] for desc in oos_con.description]

    # Apply the canonical filter to each joined row. matches_row takes a
    # daily_features-shaped dict; the joined SELECT * produces a superset
    # which is fine — extra keys are ignored.
    for raw_row in oos_rows:
        joined = dict(zip(col_names, raw_row, strict=False))
        try:
            if filter_obj.matches_row(joined, orb_label_str):
                pnl = joined.get("pnl_r")
                if pnl is not None:
                    oos_pnl_r.append(float(pnl))
        except Exception as exc:
            # Filter implementation defect: log loud, do not crash the
            # validator. The row is dropped from the OOS sample.
            logger.warning(
                "criterion_8 filter %s.matches_row raised on (%s, %s, %s): %s",
                filter_type,
                instrument,
                orb_label_str,
                joined.get("trading_day"),
                exc,
            )
            continue

    n_oos = len(oos_pnl_r)
    if n_oos == 0:
        # N/A pass-through: absence of OOS data is a measurement
        # unavailability, not a criterion violation. This protects
        # synthetic test DBs and null seed runs from false rejection.
        return {
            "status": None,
            "reason": None,
            "c8_oos_status": "NO_OOS_DATA",
            "n_oos": 0,
            "oos_expectancy_r": None,
            "oos_is_ratio": None,
        }

    # Minimum OOS sample gate.  N=30 is a CLT heuristic — NOT a Bailey
    # or Harvey-Liu prescription.  The real institutional fix is a power-
    # grounded threshold derived from each strategy's IS effect size
    # (deferred; see docs/plans/2026-04-09-bloomey-pathway-b-fixes-design.md
    # § "Out-of-scope" item 1).
    #
    # Two modes (added per Bloomey review finding A-2):
    #   strict_oos_n=False (Pathway A / default):  silent pass-through,
    #     logged at WARNING so the event is auditable.  Preserves legacy
    #     behaviour for the 124 grandfathered strategies.
    #   strict_oos_n=True  (Pathway B / individual):  hard REJECT.
    #     Amendment 3.0 condition 4 forbids "insufficient OOS data
    #     exemptions" for individual-mode hypotheses.
    _OOS_MIN_TRADES_CLT_HEURISTIC = 30
    if n_oos < _OOS_MIN_TRADES_CLT_HEURISTIC:
        if strict_oos_n:
            return {
                "status": "REJECTED",
                "reason": (
                    f"criterion_8: N_oos={n_oos} < {_OOS_MIN_TRADES_CLT_HEURISTIC} "
                    f"(Amendment 3.0 condition 4: no insufficient-OOS-data exemptions "
                    f"for Pathway B individual testing mode)"
                ),
                "c8_oos_status": "INSUFFICIENT_N_PATHWAY_B_REJECT",
                "n_oos": n_oos,
                "oos_expectancy_r": None,
                "oos_is_ratio": None,
            }
        logger.warning(
            "  Criterion 8: N_oos=%d < %d — insufficient for judgment, pass-through (Pathway A permissive mode)",
            n_oos,
            _OOS_MIN_TRADES_CLT_HEURISTIC,
        )
        return {
            "status": None,
            "reason": None,
            "c8_oos_status": "INSUFFICIENT_N_PATHWAY_A_PASS_THROUGH",
            "n_oos": n_oos,
            "oos_expectancy_r": None,
            "oos_is_ratio": None,
        }

    oos_expr = sum(oos_pnl_r) / n_oos
    if oos_expr < 0:
        return {
            "status": "REJECTED",
            "reason": f"criterion_8: OOS ExpR={oos_expr:+.4f} < 0 (N_oos={n_oos})",
            "c8_oos_status": "NEGATIVE_OOS_EXPR",
            "n_oos": n_oos,
            "oos_expectancy_r": oos_expr,
            "oos_is_ratio": None,
        }
    if is_expr_f > 0:
        ratio = oos_expr / is_expr_f
        if ratio < 0.40:
            return {
                "status": "REJECTED",
                "reason": (
                    f"criterion_8: OOS/IS ratio={ratio:.3f} < 0.40 "
                    f"(OOS={oos_expr:+.4f} IS={is_expr_f:+.4f} N_oos={n_oos})"
                ),
                "c8_oos_status": "FAILED_RATIO",
                "n_oos": n_oos,
                "oos_expectancy_r": oos_expr,
                "oos_is_ratio": ratio,
            }
    else:
        ratio = None
    return {
        "status": None,
        "reason": None,
        "c8_oos_status": "PASSED",
        "n_oos": n_oos,
        "oos_expectancy_r": oos_expr,
        "oos_is_ratio": ratio,
    }


def _check_criterion_9_era_stability(
    row_dict: dict, *, wf_start_year: int | None = None
) -> tuple[str | None, str | None]:
    """Criterion 9: era stability — no era with ExpR < -0.05 (N >= 50).

    Lifted from informational ``era_dependent`` flag to enforced gate per
    Stage 4.0. Reads the strategy's ``yearly_results`` JSON, era-bins by
    (2015-2019, 2020-2022, 2023, 2024-2025, 2026), and rejects if any era
    has ExpR < -0.05 with at least 50 trades.

    Eras with < 50 trades are exempt (insufficient data to judge).

    Parameters
    ----------
    wf_start_year
        When set, years BEFORE this year are excluded from era bin
        aggregation.  Derived from ``WF_START_OVERRIDE`` for the
        instrument.  Per Amendment 3.1, the same structural data audit
        that makes pre-override data unreliable for walk-forward also
        makes it unreliable for era stability assessment.
    """
    yearly_raw = row_dict.get("yearly_results")
    if not yearly_raw:
        return (None, None)  # no yearly data → cannot evaluate, do not reject
    try:
        yearly = json.loads(yearly_raw) if isinstance(yearly_raw, str) else yearly_raw
    except (json.JSONDecodeError, TypeError):
        return (None, None)  # corrupt JSON → do not reject from this gate
    if not isinstance(yearly, dict):
        return (None, None)

    # Era bins per Criterion 9
    era_bins = {
        "2015-2019": range(2015, 2020),
        "2020-2022": range(2020, 2023),
        "2023": range(2023, 2024),
        "2024-2025": range(2024, 2026),
        "2026": range(2026, 2027),
    }
    for era_label, year_range in era_bins.items():
        n_total = 0
        r_total = 0.0
        for y_str, data in yearly.items():
            try:
                y = int(y_str)
            except (TypeError, ValueError):
                continue
            if y not in year_range:
                continue
            # Amendment 3.1: skip years before the structural override.
            # Same data audit that excludes pre-override from WF also
            # excludes it from era stability (contract-launch artifact,
            # not a real regime signal).
            if wf_start_year is not None and y < wf_start_year:
                continue
            if not isinstance(data, dict):
                continue
            trades = data.get("trades", 0) or 0
            avg_r = data.get("avg_r", 0) or 0
            n_total += trades
            r_total += trades * avg_r
        if n_total >= 50:
            era_expr = r_total / n_total
            if era_expr < -0.05:
                return (
                    "REJECTED",
                    f"criterion_9: era {era_label} ExpR={era_expr:+.4f} < -0.05 (N={n_total})",
                )
    return (None, None)


def _check_phase_4_pre_flight_gates(
    row_dict: dict,
    db_path: Path | None,
    hypothesis_meta_cache: dict[str, dict],
    *,
    testing_mode: str = "family",
) -> tuple[str | None, str | None]:
    verdict = _evaluate_phase_4_pre_flight_gates(
        row_dict,
        db_path,
        hypothesis_meta_cache,
        testing_mode=testing_mode,
    )
    return verdict["status"], verdict["reason"]


def _evaluate_phase_4_pre_flight_gates(
    row_dict: dict,
    db_path: Path | None,
    hypothesis_meta_cache: dict[str, dict],
    *,
    testing_mode: str = "family",
) -> dict[str, object]:
    """Apply all Phase 4 Stage 4.0 pre-flight gates to a single experimental row.

    Stage 4.0 enforces Criteria 1 (hypothesis file presence), 2 (MinBTL),
    9 (era stability), and 8 (2026 OOS positive) in that order. Criteria 4
    (Chordia) and 5 (DSR) are deferred to Stage 4.0b per Amendments 2.1 and
    2.2 of the locked criteria file; they do not appear here.

    Parameters
    ----------
    testing_mode
        "family" (Pathway A) or "individual" (Pathway B).  When
        "individual", the C8 gate uses strict mode (hard reject at
        N_oos < 30 instead of silent pass-through).

    Returns
    -------
    tuple[str | None, str | None]
        ``(None, None)`` if grandfathered or all Stage 4.0 gates pass.
        ``("REJECTED", "criterion_N: ...")`` on first gate failure.
    """
    if _is_phase_4_grandfathered(row_dict):
        return {"status": None, "reason": None, "c8_oos_status": None}

    # Criterion 1: hypothesis file presence (C2 depends on the loaded meta)
    rejection = _check_criterion_1_hypothesis_file(row_dict)
    if rejection != (None, None):
        return {"status": rejection[0], "reason": rejection[1], "c8_oos_status": None}

    # Load + cache the hypothesis metadata for downstream gates.
    sha = row_dict["hypothesis_file_sha"]
    if sha not in hypothesis_meta_cache:
        meta_loaded = load_hypothesis_by_sha(sha)
        if meta_loaded is None:
            # Defensive: criterion_1 already verified this; reaching here
            # implies a race or test fixture inconsistency.
            return {
                "status": "REJECTED",
                "reason": f"criterion_1: hypothesis sha={sha[:12]} disappeared between checks",
                "c8_oos_status": None,
            }
        hypothesis_meta_cache[sha] = meta_loaded
    meta = hypothesis_meta_cache[sha]

    # Criterion 2: MinBTL trial count bound (per file, not per row)
    rejection = _check_criterion_2_minbtl(meta)
    if rejection != (None, None):
        return {"status": rejection[0], "reason": rejection[1], "c8_oos_status": None}

    # Criterion 9: era stability (cheap, pure JSON parse, no DB access).
    # Fires before C8 to short-circuit expensive DB work (reviewer HIGH #5 —
    # gate ordering: cheap local gates before expensive IO gates).
    # Amendment 3.1: pass WF_START_OVERRIDE year so pre-override data
    # (contract-launch artifacts) is excluded from era bins.
    _instrument = row_dict.get("instrument", "")
    _wf_override = WF_START_OVERRIDE.get(_instrument)
    _wf_start_year = _wf_override.year if _wf_override is not None else None
    rejection = _check_criterion_9_era_stability(row_dict, wf_start_year=_wf_start_year)
    if rejection != (None, None):
        return {"status": rejection[0], "reason": rejection[1], "c8_oos_status": None}

    # Criterion 8: 2026 OOS positive (N/A safe when no OOS data).
    # Last in the sequence because it issues a DuckDB read-only query per row.
    # Pathway B (individual mode) uses strict_oos_n=True: hard reject at N<30
    # per Amendment 3.0 condition 4 ("no insufficient OOS data exemptions").
    return _evaluate_criterion_8_oos(
        row_dict,
        db_path,
        strict_oos_n=(testing_mode == "individual"),
    )


def _validation_pathway_for_row(row_dict: dict, testing_mode: str) -> str | None:
    """Structured pathway label for new validation results."""
    if _is_phase_4_grandfathered(row_dict):
        return None
    if testing_mode == "individual":
        return "individual"
    if testing_mode == "family":
        return "family"
    return None


def run_validation(
    db_path: Path | None = None,
    instrument: str = "MGC",
    min_sample: int = REGIME_MIN_SAMPLES,
    stress_multiplier: float = 1.5,
    min_sharpe: float | None = None,
    max_drawdown: float | None = None,
    exclude_years: set[int] | None = None,
    min_years_positive_pct: float = 0.75,  # @research-source Fitschen — see validate_strategy()
    min_trades_per_year: int = 1,
    dry_run: bool = False,
    enable_walkforward: bool = True,
    wf_test_months: int = 6,
    wf_min_train_months: int = 12,
    wf_min_trades: int = 15,  # @research-source Lopez de Prado AFML Ch.11 — min trades/window for stable WF estimate; @revalidated-for E1/E2 (2026-03-10)
    wf_min_windows: int = 3,  # @research-source Bailey et al. 2014 — min 3 OOS windows for meaningful WF; @revalidated-for E1/E2 (2026-03-10)
    wf_min_pct_positive: float = 0.60,  # @research-source Fitschen "Building Reliable Trading Systems" — 60% positive windows; @revalidated-for E1/E2 (2026-03-10)
    wf_output_path: str = "data/walkforward_results.jsonl",
    enable_regime_waivers: bool = True,
    workers: int | None = None,
    fdr_k_overrides: dict[str, int] | None = None,
    testing_mode: str = "family",
) -> tuple[int, int]:
    """
    Validate all experimental_strategies and promote passing ones.

    Args:
        fdr_k_overrides: Per-session K overrides for BH FDR. When provided,
            uses these K values instead of auto-counting from the DB. Required
            for null seed testing where the DB only contains one instrument
            but the real pipeline has multiple instruments contributing to K.
            Format: {"CME_PRECLOSE": 3254, "NYSE_OPEN": 5184, ...}
        testing_mode: "family" (Pathway A, BH FDR) or "individual" (Pathway B,
            raw p < 0.05 + positive sharpe_ann). Per Amendment 3.0. When
            "individual", the BH FDR gate is replaced with a raw significance
            + direction gate. Criteria 6 (WFE), 8 (OOS), 9 (era stability)
            are non-waivable under individual mode.

    Returns (passed_count, rejected_count).
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if db_path is None:
        db_path = GOLD_DB_PATH

    cost_spec = get_cost_spec(instrument)
    wf_start_date = WF_START_OVERRIDE.get(instrument)
    wf_test_trades = WF_TRADE_COUNT_OVERRIDE.get(instrument)
    wf_min_train = WF_MIN_TRAIN_TRADES.get(instrument)

    if workers is None:
        workers = min(8, max(1, (os.cpu_count() or 2) - 1))
    use_parallel = workers > 1 and enable_walkforward

    import duckdb

    # ── Phase A: Load strategies + serial cull (phases 1-5) ──────────
    with duckdb.connect(str(db_path)) as con:
        from pipeline.db_config import configure_connection

        configure_connection(con, writing=True)

        if not dry_run:
            init_trading_app_schema(db_path=db_path)

        rows = con.execute(
            """SELECT * FROM experimental_strategies
               WHERE instrument = ?
               AND (validation_status IS NULL OR validation_status = '')
               ORDER BY strategy_id""",
            [instrument],
        ).fetchall()
        col_names = [desc[0] for desc in con.description]

        atr_by_year = {}
        if enable_regime_waivers:
            # ATR is a daily stat — identical across all orb_minutes rows for the
            # same (trading_day, symbol).  Filter to orb_minutes=5 to get exactly
            # one row per day (avoids 3x inflation from 5/15/30m rows).
            # Using 5 is safe: 5m daily_features always exist for all instruments.
            atr_rows = con.execute(
                """
                SELECT EXTRACT(YEAR FROM trading_day) as yr, AVG(atr_20) as mean_atr
                FROM daily_features
                WHERE symbol = ? AND orb_minutes = 5 AND atr_20 IS NOT NULL
                GROUP BY yr
            """,
                [instrument],
            ).fetchall()
            atr_by_year = {int(r[0]): r[1] for r in atr_rows}

    # Connection closed — DuckDB requires no write connection open
    # while worker processes open read-only connections.

    passed = 0
    rejected = 0
    skipped_aliases = 0
    passed_strategy_ids = []

    serial_results = []  # Each: {row_dict, status, notes, regime_waivers, dst_split, rejection_reason}
    wf_candidates = []  # Survivors needing walkforward

    # Phase 4 Stage 4.0: hypothesis-file metadata cache, populated lazily
    # as the loop encounters new SHAs. Avoids re-parsing the same YAML
    # for every row that points at the same hypothesis file.
    phase_4_hypothesis_cache: dict[str, dict] = {}
    phase_4_rejected_count = 0

    for row in rows:
        row_dict = dict(zip(col_names, row, strict=False))
        strategy_id = row_dict["strategy_id"]

        if row_dict.get("is_canonical") is False:
            skipped_aliases += 1
            serial_results.append(
                {
                    "row_dict": row_dict,
                    "status": "SKIPPED",
                    "notes": "Alias (non-canonical)",
                    "regime_waivers": [],
                    "rejection_reason": None,
                    "validation_pathway": None,
                    "c8_oos_status": None,
                    "dst_split": {
                        "winter_n": None,
                        "winter_avg_r": None,
                        "summer_n": None,
                        "summer_avg_r": None,
                        "verdict": None,
                    },
                }
            )
            continue

        # ── Phase 4 Stage 4.0 pre-flight gates ────────────────────────
        # Apply institutional criteria 1, 2, 8, 9 before the legacy
        # validate_strategy pipeline. Criteria 4 and 5 are deferred to
        # Stage 4.0b per Amendments 2.1 (DSR cross-check only) and 2.2
        # (Chordia banded post-validation). Grandfathered rows (created_at
        # <= HOLDOUT_GRANDFATHER_CUTOFF OR hypothesis_file_sha IS NULL)
        # skip these gates and fall through to the legacy validator path.
        # See _check_phase_4_pre_flight_gates docstring for details.
        phase_4_result = _evaluate_phase_4_pre_flight_gates(
            row_dict,
            db_path,
            phase_4_hypothesis_cache,
            testing_mode=testing_mode,
        )
        phase_4_status = phase_4_result["status"]
        phase_4_reason = phase_4_result["reason"]
        validation_pathway = _validation_pathway_for_row(row_dict, testing_mode)
        c8_oos_status = phase_4_result["c8_oos_status"]
        if phase_4_status is not None:
            serial_results.append(
                {
                    "row_dict": row_dict,
                    "status": phase_4_status,
                    "notes": phase_4_reason,
                    "regime_waivers": [],
                    "rejection_reason": phase_4_reason,
                    "validation_pathway": validation_pathway,
                    "c8_oos_status": c8_oos_status,
                    "dst_split": {
                        "winter_n": None,
                        "winter_avg_r": None,
                        "summer_n": None,
                        "summer_avg_r": None,
                        "verdict": None,
                    },
                }
            )
            rejected += 1
            phase_4_rejected_count += 1
            continue

        # REGIME strategies (N<100) use min_trades_per_year=5 — sparse years
        # have insufficient power for yearly characterization. CORE keeps
        # the caller-provided value (default 1).
        sample_n = row_dict.get("sample_size") or 0
        effective_min_tpy = 5 if sample_n < CORE_MIN_SAMPLES else min_trades_per_year

        status, notes, regime_waivers = validate_strategy(
            row_dict,
            cost_spec,
            stress_multiplier=stress_multiplier,
            min_sample=min_sample,
            min_sharpe=min_sharpe,
            max_drawdown=max_drawdown,
            exclude_years=exclude_years,
            min_years_positive_pct=min_years_positive_pct,
            min_trades_per_year=effective_min_tpy,
            atr_by_year=atr_by_year if enable_regime_waivers else None,
            enable_regime_waivers=enable_regime_waivers,
        )

        strat_dst_regime = parse_dst_regime(strategy_id)

        if status == "PASSED" and enable_walkforward:
            wf_candidates.append(
                {
                    "row_dict": row_dict,
                    "status": status,
                    "notes": notes,
                    "regime_waivers": regime_waivers,
                    "validation_pathway": validation_pathway,
                    "c8_oos_status": c8_oos_status,
                    "strat_dst_regime": strat_dst_regime,
                }
            )
        else:
            dst_split = {
                "winter_n": None,
                "winter_avg_r": None,
                "summer_n": None,
                "summer_avg_r": None,
                "verdict": None,
            }
            serial_results.append(
                {
                    "row_dict": row_dict,
                    "status": status,
                    "notes": notes,
                    "regime_waivers": regime_waivers,
                    "rejection_reason": notes if status == "REJECTED" else None,
                    "validation_pathway": validation_pathway,
                    "c8_oos_status": c8_oos_status,
                    "dst_split": dst_split,
                }
            )
            if status == "PASSED":
                passed += 1
                passed_strategy_ids.append(strategy_id)
            else:
                rejected += 1

    logger.info(
        f"Phase A complete: {len(wf_candidates)} survivors for walkforward, "
        f"{rejected} rejected, {skipped_aliases} aliases skipped "
        f"(of {len(rows)} strategies)"
    )

    # ── Phase B: Parallel walkforward for survivors ──────────────────
    wf_results_map = {}  # strategy_id -> worker result dict

    if wf_candidates:
        wf_params = {
            "test_window_months": wf_test_months,
            "min_train_months": wf_min_train_months,
            "min_trades": wf_min_trades,
            "min_windows": wf_min_windows,
            "min_pct_positive": wf_min_pct_positive,
        }

        wall_start = time.monotonic()
        total_wf_duration = 0.0

        def _build_worker_kwargs(cand):
            rd = cand["row_dict"]
            sample_n = rd.get("sample_size") or 0
            is_regime = sample_n < CORE_MIN_SAMPLES

            # REGIME strategies get smaller WF windows (both must be positive)
            if is_regime:
                cand_wf_params = {
                    **wf_params,
                    "min_windows": REGIME_WF_MIN_WINDOWS,
                    "min_trades": REGIME_WF_MIN_TRADES_PER_WINDOW,
                }
                cand_wf_test_trades = REGIME_WF_TRADE_COUNT.get(instrument) or wf_test_trades
                cand_wf_min_train = REGIME_WF_MIN_TRAIN_TRADES.get(instrument) or wf_min_train
            else:
                cand_wf_params = wf_params
                cand_wf_test_trades = wf_test_trades
                cand_wf_min_train = wf_min_train

            return dict(
                strategy_id=rd["strategy_id"],
                instrument=instrument,
                orb_label=rd["orb_label"],
                entry_model=rd.get("entry_model") or "E1",  # NOT NULL in schema — fallback unreachable
                rr_target=rd["rr_target"],
                confirm_bars=rd["confirm_bars"],
                filter_type=rd.get("filter_type") or "NO_FILTER",  # NOT NULL in schema — fallback unreachable
                filter_params=rd.get("filter_params"),
                orb_minutes=rd.get("orb_minutes", 5),
                db_path_str=str(db_path),
                wf_params=cand_wf_params,
                dst_regime=cand["strat_dst_regime"],
                dst_verdict_from_discovery=rd.get("dst_verdict"),
                wf_start_date=wf_start_date,
                stop_multiplier=rd.get("stop_multiplier", 1.0),
                wf_test_trades=cand_wf_test_trades,
                wf_min_train_trades=cand_wf_min_train,
                dst_cols_from_discovery={
                    "winter_n": rd.get("dst_winter_n"),
                    "winter_avg_r": rd.get("dst_winter_avg_r"),
                    "summer_n": rd.get("dst_summer_n"),
                    "summer_avg_r": rd.get("dst_summer_avg_r"),
                    "verdict": rd.get("dst_verdict"),
                }
                if rd.get("dst_verdict") is not None
                else None,
            )

        if use_parallel:
            logger.info(f"Starting parallel walkforward with {workers} workers for {len(wf_candidates)} strategies...")

            with ProcessPoolExecutor(max_workers=workers) as executor:
                future_to_sid = {}
                for cand in wf_candidates:
                    kwargs = _build_worker_kwargs(cand)
                    future = executor.submit(_walkforward_worker, **kwargs)  # type: ignore[arg-type]
                    future_to_sid[future] = kwargs["strategy_id"]

                for future in as_completed(future_to_sid):
                    sid = future_to_sid[future]
                    try:
                        result = future.result()
                        wf_results_map[sid] = result
                        total_wf_duration += result.get("wf_duration_s", 0)
                    except Exception as e:
                        logger.error(f"Worker exception for {sid}: {e}")
                        wf_results_map[sid] = {
                            "strategy_id": sid,
                            "wf_result": None,
                            "dst_split": None,
                            "error": str(e),
                            "wf_duration_s": 0,
                        }
        else:
            # Serial fallback (--workers 1 or walkforward disabled already handled)
            logger.info(f"Running walkforward serially for {len(wf_candidates)} strategies...")
            for cand in wf_candidates:
                kwargs = _build_worker_kwargs(cand)
                result = _walkforward_worker(**kwargs)  # type: ignore[arg-type]
                wf_results_map[kwargs["strategy_id"]] = result
                total_wf_duration += result.get("wf_duration_s", 0)

        wall_elapsed = time.monotonic() - wall_start
        speedup = total_wf_duration / wall_elapsed if wall_elapsed > 0 else 1.0
        logger.info(
            f"Walkforward complete: wall={wall_elapsed:.1f}s, "
            f"sum(worker)={total_wf_duration:.1f}s, "
            f"speedup={speedup:.1f}x ({workers} workers)"
        )

        # Merge walkforward results into serial_results
        for cand in wf_candidates:
            rd = cand["row_dict"]
            sid = rd["strategy_id"]
            wr = wf_results_map.get(sid, {})

            status = cand["status"]
            notes = cand["notes"]

            if wr.get("error"):
                status = "REJECTED"
                notes = f"Phase 4b: Worker error: {wr['error']}"
            elif wr.get("wf_result") and not wr["wf_result"]["passed"]:
                status = "REJECTED"
                notes = f"Phase 4b: {wr['wf_result']['rejection_reason']}"
            elif not wr.get("wf_result") and not wr.get("error"):
                status = "REJECTED"
                notes = "Phase 4b: No walkforward result received"

            dst_split = wr.get("dst_split") or {
                "winter_n": None,
                "winter_avg_r": None,
                "summer_n": None,
                "summer_avg_r": None,
                "verdict": None,
            }

            serial_results.append(
                {
                    "row_dict": rd,
                    "status": status,
                    "notes": notes,
                    "regime_waivers": cand["regime_waivers"],
                    "rejection_reason": notes if status == "REJECTED" else None,
                    "validation_pathway": cand["validation_pathway"],
                    "c8_oos_status": cand["c8_oos_status"],
                    "dst_split": dst_split,
                    "wf_result_dict": wr.get("wf_result"),
                }
            )

            if status == "PASSED":
                passed += 1
                passed_strategy_ids.append(sid)
            else:
                rejected += 1

    # Log WF error summary
    wf_errors = sum(1 for sr in serial_results if sr["notes"] and "Worker error" in sr["notes"])
    if wf_errors > 0:
        logger.warning(f"Walkforward: {wf_errors} strategies rejected due to worker errors")

    # ── Phase C: Batch write all results ─────────────────────────────
    n_fdr_rejected = 0  # set by FDR hard gate below; used in Phase D logging
    processed_sids = [sr["row_dict"]["strategy_id"] for sr in serial_results]
    validation_run_id = None
    if not dry_run and processed_sids:
        import uuid

        validation_run_id = f"{instrument}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        with duckdb.connect(str(db_path)) as con:
            from pipeline.db_config import configure_connection

            configure_connection(con, writing=True)
            promotion_git_sha = get_git_sha()
            if promotion_git_sha is None:
                raise RuntimeError(
                    "Cannot resolve git SHA for promotion provenance; "
                    "refusing to write native validated_setups rows fail-open."
                )
            provenance_resolver = StrategyTradeWindowResolver(con)
            promotion_written_at = datetime.now(UTC)

            # Purge validated_setups + edge_families ONLY for strategy_ids
            # being processed in this run.  Prior code used instrument +
            # orb_minutes which nuked unrelated strategies (e.g. RR1.0
            # wiped when validating RR1.5/2.0).  Fix: 2026-04-11.
            sid_placeholders = ", ".join("?" * len(processed_sids))
            con.execute(
                f"""DELETE FROM edge_families
                    WHERE head_strategy_id IN ({sid_placeholders})""",
                processed_sids,
            )
            con.execute(
                f"""DELETE FROM validated_setups
                    WHERE strategy_id IN ({sid_placeholders})""",
                processed_sids,
            )

            for sr in serial_results:
                rd = sr["row_dict"]
                sid = rd["strategy_id"]
                status = sr["status"]
                notes = sr["notes"]
                dst_split = sr["dst_split"]
                validation_pathway = sr.get("validation_pathway")
                c8_oos_status = sr.get("c8_oos_status")
                # Phase 4 Stage 4.0: capture the structured criterion-tagged
                # rejection_reason for Phase 4 gate rejections. Legacy
                # rejections leave this NULL — Stage 4.4 audit distinguishes
                # the two via "rejection_reason IS NOT NULL = Phase 4 gate
                # rejection, criterion N is in the value."
                rejection_reason = sr.get("rejection_reason")

                if status == "SKIPPED":
                    con.execute(
                        """UPDATE experimental_strategies
                           SET validation_status = 'SKIPPED',
                               validation_notes = 'Alias (non-canonical)',
                               validation_pathway = NULL,
                               c8_oos_status = NULL
                           WHERE strategy_id = ?""",
                        [sid],
                    )
                    continue

                # Update experimental_strategies
                con.execute(
                    """UPDATE experimental_strategies
                       SET validation_status = ?, validation_notes = ?,
                           dst_winter_n = ?, dst_winter_avg_r = ?,
                           dst_summer_n = ?, dst_summer_avg_r = ?,
                           dst_verdict = ?,
                           rejection_reason = ?,
                           validation_pathway = ?,
                           c8_oos_status = ?
                       WHERE strategy_id = ?""",
                    [
                        status,
                        notes,
                        dst_split.get("winter_n"),
                        dst_split.get("winter_avg_r"),
                        dst_split.get("summer_n"),
                        dst_split.get("summer_avg_r"),
                        dst_split.get("verdict"),
                        rejection_reason,
                        validation_pathway,
                        c8_oos_status,
                        sid,
                    ],
                )

                if status == "PASSED":
                    yearly = rd.get("yearly_results", "{}")
                    try:
                        yearly_data = json.loads(yearly) if isinstance(yearly, str) else yearly
                    except (json.JSONDecodeError, TypeError) as exc:
                        logger.warning("Corrupt yearly_results JSON for %s: %s", sid, exc)
                        yearly_data = {}

                    included = {
                        y: d
                        for y, d in yearly_data.items()
                        if int(y) not in (exclude_years or set()) and d.get("trades", 0) >= min_trades_per_year
                    }
                    years_tested = len(included)
                    all_positive = all(d.get("avg_r", 0) > 0 for d in included.values())
                    regime_waivers = sr["regime_waivers"]

                    # Concentration check: max single year's % of total R
                    year_totals = {
                        y: d.get("total_r", d.get("avg_r", 0) * d.get("trades", 0)) for y, d in included.items()
                    }
                    total_r_sum = sum(year_totals.values())
                    if total_r_sum > 0:
                        max_year_pct_val = max(yr / total_r_sum for yr in year_totals.values())
                    else:
                        max_year_pct_val = None
                    era_dependent_val = max_year_pct_val is not None and max_year_pct_val > 0.50

                    wf_result_dict = sr.get("wf_result_dict")
                    wf_tested = wf_result_dict is not None
                    wf_passed = (wf_result_dict or {}).get("passed", False) if wf_tested else None
                    wf_windows_val = (
                        (wf_result_dict or {}).get("as_dict", {}).get("n_valid_windows") if wf_tested else None
                    )
                    wfe_val = (wf_result_dict or {}).get("as_dict", {}).get("wfe") if wf_tested else None
                    oos_exp_r = (wf_result_dict or {}).get("as_dict", {}).get("agg_oos_exp_r") if wf_tested else None

                    # noise_risk: OOS ExpR at or below per-instrument p95 null floor
                    entry_model_key = rd.get("entry_model") or "E2"
                    inst_floors = NOISE_FLOOR_BY_INSTRUMENT.get(instrument, {})
                    noise_floor = inst_floors.get(entry_model_key, inst_floors.get("E2"))
                    if oos_exp_r is not None and noise_floor is not None:
                        noise_risk_val = oos_exp_r <= noise_floor
                    else:
                        noise_risk_val = None  # not computed (live_config treats as fail-closed)

                    trade_window = provenance_resolver.resolve(
                        instrument=rd["instrument"],
                        orb_label=rd["orb_label"],
                        orb_minutes=rd["orb_minutes"],
                        entry_model=rd.get("entry_model") or "E1",
                        rr_target=rd["rr_target"],
                        confirm_bars=rd["confirm_bars"],
                        filter_type=rd.get("filter_type", ""),
                    )
                    if (
                        trade_window.trade_day_count <= 0
                        or trade_window.first_trade_day is None
                        or trade_window.last_trade_day is None
                    ):
                        raise RuntimeError(f"{sid}: cannot recompute canonical trade window for promotion provenance")

                    shelf_lifecycle = validated_shelf_lifecycle(rd["instrument"])

                    con.execute(
                        """INSERT OR REPLACE INTO validated_setups
                           (strategy_id, promoted_from, instrument, orb_label,
                            orb_minutes, rr_target, confirm_bars, entry_model,
                            filter_type, filter_params, stop_multiplier,
                            sample_size, win_rate, expectancy_r,
                            years_tested, all_years_positive, stress_test_passed,
                            sharpe_ratio, max_drawdown_r,
                            trades_per_year, sharpe_ann,
                            yearly_results, status,
                            median_risk_dollars, avg_risk_dollars,
                            avg_win_dollars, avg_loss_dollars,
                            first_trade_day, last_trade_day, trade_day_count,
                            validation_run_id, promotion_git_sha, promotion_provenance,
                            deployment_scope,
                            regime_waivers, regime_waiver_count,
                            dst_winter_n, dst_winter_avg_r,
                            dst_summer_n, dst_summer_avg_r,
                            dst_verdict,
                            wf_tested, wf_passed, wf_windows, wfe,
                            sharpe_haircut, skewness, kurtosis_excess,
                            oos_exp_r, noise_risk,
                            era_dependent, max_year_pct,
                            p_value, n_trials_at_discovery, fst_hurdle)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        [
                            sid,
                            sid,
                            rd["instrument"],
                            rd["orb_label"],
                            rd["orb_minutes"],
                            rd["rr_target"],
                            rd["confirm_bars"],
                            rd.get("entry_model") or "E1",  # NOT NULL in schema — fallback unreachable
                            rd.get("filter_type", ""),
                            rd.get("filter_params", ""),
                            rd.get("stop_multiplier", 1.0),
                            rd.get("sample_size", 0),
                            rd.get("win_rate", 0),
                            rd.get("expectancy_r", 0),
                            years_tested,
                            all_positive,
                            True,
                            rd.get("sharpe_ratio"),
                            rd.get("max_drawdown_r"),
                            rd.get("trades_per_year"),
                            rd.get("sharpe_ann"),
                            yearly,
                            "active" if rd["instrument"] in ACTIVE_ORB_INSTRUMENTS else "retired",
                            rd.get("median_risk_dollars"),
                            rd.get("avg_risk_dollars"),
                            rd.get("avg_win_dollars"),
                            rd.get("avg_loss_dollars"),
                            trade_window.first_trade_day,
                            trade_window.last_trade_day,
                            trade_window.trade_day_count,
                            validation_run_id,
                            promotion_git_sha,
                            "VALIDATOR_NATIVE",
                            shelf_lifecycle.deployment_scope,
                            json.dumps(regime_waivers) if regime_waivers else None,
                            len(regime_waivers),
                            dst_split.get("winter_n"),
                            dst_split.get("winter_avg_r"),
                            dst_split.get("summer_n"),
                            dst_split.get("summer_avg_r"),
                            dst_split.get("verdict"),
                            wf_tested,
                            wf_passed,
                            wf_windows_val,
                            wfe_val,
                            rd.get("sharpe_haircut"),
                            rd.get("skewness"),
                            rd.get("kurtosis_excess"),
                            oos_exp_r,
                            noise_risk_val,
                            era_dependent_val,
                            round(max_year_pct_val, 4) if max_year_pct_val is not None else None,
                            rd.get("p_value"),
                            rd.get("n_trials_at_discovery"),
                            rd.get("fst_hurdle"),
                        ],
                    )
                    con.execute(
                        """UPDATE validated_setups
                           SET validation_pathway = COALESCE(validation_pathway, ?),
                               c8_oos_status = ?
                           WHERE strategy_id = ?""",
                        [validation_pathway, c8_oos_status, sid],
                    )
                    if shelf_lifecycle.retirement_reason is not None:
                        con.execute(
                            """
                            UPDATE validated_setups
                            SET retired_at = ?,
                                retirement_reason = ?
                            WHERE strategy_id = ?
                            """,
                            [
                                promotion_written_at,
                                shelf_lifecycle.retirement_reason,
                                sid,
                            ],
                        )

            # Write walkforward JSONL (batch)
            from trading_app.walkforward import WalkForwardResult

            for sr in serial_results:
                wfr = sr.get("wf_result_dict")
                if wfr and wfr.get("as_dict"):
                    wd = wfr["as_dict"]
                    wf_obj = WalkForwardResult(
                        **{k: v for k, v in wd.items() if k in WalkForwardResult.__dataclass_fields__}
                    )
                    append_walkforward_result(wf_obj, wf_output_path)

            # FDR hard gate — stratified K by session (orb_label)
            #
            # BH FDR applied per session: each orb_label is a structurally
            # distinct hypothesis family (different market microstructure,
            # participant pool, liquidity regime). K = canonical strategies
            # within that session across ALL active instruments and apertures.
            #
            # Grounding:
            #   BH (1995): m = "family" of hypotheses under simultaneous consideration
            #   Efron Separate-Class model: stratified FDR valid and more powerful
            #     when group structure is pre-specified on logical grounds
            #   Harvey et al (2016): BH-FDR robust to K specification
            #   RESEARCH_RULES.md: "instrument/family K for promotion decisions"
            #
            # Sessions are pre-specified strata defined by exchange events
            # (CME settlement, NYSE open, etc.), not by data outcomes.
            # Dead instruments excluded (ACTIVE_ORB_INSTRUMENTS only).
            #
            # History:
            #   2026-03-18: FDR gate made real (was cosmetic before)
            #   2026-03-24: scoped to active instruments (killed dead M2K/SIL/etc.)
            #   2026-03-24: stratified by session (global K=78K killed CME_PRECLOSE
            #     and TOKYO_OPEN with 0 survivors; stratified K restores valid strata)
            if passed_strategy_ids and testing_mode == "individual":
                # ── Pathway B: Amendment 3.0 individual hypothesis testing ──
                #
                # Gate sequence (ALL three must pass, per Amendment 3.0):
                #   Criterion 3 (Pathway B): raw bootstrap p < 0.05 (two-tailed)
                #   Criterion 3 (direction): sharpe_ann > 0 (rules out significant-
                #     but-negative strategies; Amendment 3.0 condition 2b)
                #   Criterion 6: wfe >= MIN_WFE (Pardo walk-forward efficiency
                #     floor; Amendment 3.0 condition 4 "non-waivable")
                #
                # Criteria 8 (2026 OOS) and 9 (era stability) already enforced
                # in the pre-flight gate stack earlier in the main loop.
                #
                # @research-source: docs/institutional/pre_registered_criteria.md
                #   Amendment 3.0 (2026-04-09) conditions 2b and 4.
                logger.info(
                    "Pathway B (individual): raw p < 0.05 + sharpe > 0 + wfe >= %.2f gate...",
                    MIN_WFE,
                )
                n_pathway_b_pass = 0
                n_pathway_b_rejected = 0
                pathway_b_rejected_ids = []
                for sid in passed_strategy_ids:
                    exp_row = con.execute(
                        "SELECT p_value, sharpe_ann FROM experimental_strategies WHERE strategy_id = ?",
                        [sid],
                    ).fetchone()
                    raw_p = exp_row[0] if exp_row and exp_row[0] is not None else 1.0
                    sharpe = exp_row[1] if exp_row and exp_row[1] is not None else 0.0

                    # WFE gate — mirror the Pathway A fail-closed pattern at
                    # line ~1905: null WFE → treat as 0.0 → reject.
                    # @research-source: Pardo "Evaluation and Optimization" Ch.7
                    #   WFE < 0.50 = lost >50% of edge OOS → likely overfit.
                    wfe_row = con.execute("SELECT wfe FROM validated_setups WHERE strategy_id = ?", [sid]).fetchone()
                    wfe_val = wfe_row[0] if wfe_row and wfe_row[0] is not None else 0.0

                    pass_raw_p = raw_p < 0.05
                    pass_direction = sharpe > 0
                    pass_wfe = wfe_val >= MIN_WFE

                    if pass_raw_p and pass_direction and pass_wfe:
                        n_pathway_b_pass += 1
                        con.execute(
                            """UPDATE validated_setups
                               SET fdr_significant = TRUE,
                                   fdr_adjusted_p = ?,
                                   p_value = ?,
                                   validation_pathway = 'individual',
                                   discovery_k = CASE
                                       WHEN discovery_k IS NULL THEN 1
                                       ELSE discovery_k
                                   END,
                                   discovery_date = CASE
                                       WHEN discovery_date IS NULL THEN ?
                                       ELSE discovery_date
                                   END
                               WHERE strategy_id = ?""",
                            [raw_p, raw_p, date.today(), sid],
                        )
                    else:
                        pathway_b_rejected_ids.append(sid)
                        n_pathway_b_rejected += 1
                        # Tag each failing gate separately so the audit trail
                        # distinguishes Criterion 3 vs Criterion 6 rejections.
                        failures: list[str] = []
                        if not pass_raw_p:
                            failures.append(f"criterion_3_pathway_b: raw p={raw_p:.4f}>=0.05")
                        if not pass_direction:
                            failures.append(f"criterion_3_pathway_b: sharpe_ann={sharpe:.4f}<=0")
                        if not pass_wfe:
                            failures.append(
                                f"criterion_6_pathway_b: wfe={wfe_val:.4f}<{MIN_WFE} (Amendment 3.0 non-waivable)"
                            )
                        reason = "; ".join(failures)
                        con.execute("DELETE FROM validated_setups WHERE strategy_id = ?", [sid])
                        con.execute(
                            """UPDATE experimental_strategies
                               SET validation_status = 'REJECTED',
                                   validation_notes = ?,
                                   rejection_reason = ?
                               WHERE strategy_id = ?""",
                            [reason, reason, sid],
                        )
                if pathway_b_rejected_ids:
                    passed -= n_pathway_b_rejected
                    rejected += n_pathway_b_rejected
                    passed_strategy_ids = [s for s in passed_strategy_ids if s not in set(pathway_b_rejected_ids)]
                # Reuse n_fdr_rejected so Phase D counters (line ~2066) log
                # Pathway B rejections. Name is historical, not methodological.
                n_fdr_rejected = n_pathway_b_rejected
                logger.info(
                    f"  Pathway B gate: {n_pathway_b_pass} survived, "
                    f"{n_pathway_b_rejected} REJECTED "
                    f"(of {n_pathway_b_pass + n_pathway_b_rejected} passed prior phases)"
                )

            elif passed_strategy_ids:
                logger.info("Computing FDR correction (Benjamini-Hochberg, stratified K by session)...")
                # Include the current run's instrument in the FDR pool even if
                # it's not in ACTIVE_ORB_INSTRUMENTS (e.g., GC research runs).
                # Without this, Pathway A strategies for research instruments
                # would silently skip FDR entirely — fail-closed gap.
                # Fix: 2026-04-11.
                active_instruments = list(ACTIVE_ORB_INSTRUMENTS)
                if instrument not in active_instruments:
                    active_instruments.append(instrument)
                placeholders = ", ".join(["?"] * len(active_instruments))

                # Build per-session p-value pools
                session_p_pools: dict[str, list[tuple[str, float]]] = {}
                rows = con.execute(
                    f"""SELECT strategy_id, p_value, orb_label
                        FROM experimental_strategies
                        WHERE is_canonical = TRUE
                        AND p_value IS NOT NULL
                        AND instrument IN ({placeholders})""",
                    active_instruments,
                ).fetchall()
                for sid_row, pval_row, orb_row in rows:
                    session_p_pools.setdefault(orb_row, []).append((sid_row, pval_row))

                total_k = sum(len(v) for v in session_p_pools.values())
                logger.info(f"  Total active K={total_k}, split across {len(session_p_pools)} sessions")

                # Run BH per session, tracking effective K per session for auditability
                fdr_results: dict[str, dict] = {}
                effective_k_by_session: dict[str, int] = {}
                for session_name, pool in sorted(session_p_pools.items()):
                    if fdr_k_overrides and session_name in fdr_k_overrides:
                        k_session = fdr_k_overrides[session_name]
                        k_source = "override"
                    else:
                        k_session = len(pool)
                        k_source = "auto"
                    effective_k_by_session[session_name] = k_session
                    session_fdr = benjamini_hochberg(pool, alpha=0.05, total_tests=k_session)
                    fdr_results.update(session_fdr)
                    n_sig = sum(1 for v in session_fdr.values() if v["fdr_significant"])
                    logger.info(f"  {session_name}: K={k_session} ({k_source}), {n_sig} significant")

                n_fdr_sig = 0
                n_fdr_rejected = 0
                fdr_rejected_ids = []
                _today = date.today()
                # Build p-value lookup to distinguish NULL-p (legitimately
                # unFDR-able) from pool-build errors.
                p_value_by_sid: dict[str, float | None] = {}
                for sid_r in passed_strategy_ids:
                    pv_row = con.execute(
                        "SELECT p_value FROM experimental_strategies WHERE strategy_id = ?",
                        [sid_r],
                    ).fetchone()
                    p_value_by_sid[sid_r] = pv_row[0] if pv_row else None

                for sid in passed_strategy_ids:
                    fdr = fdr_results.get(sid)
                    if fdr is None:
                        if p_value_by_sid.get(sid) is not None:
                            # Fail-closed: strategy has a p_value but wasn't
                            # in the FDR pool. Indicates pool build error
                            # (e.g., instrument missing). Reject rather than
                            # silently promote. Fix: 2026-04-11.
                            logger.warning(
                                f"  FDR MISSING: {sid} has p_value but no FDR result — "
                                "pool build error, failing closed."
                            )
                            fdr_rejected_ids.append(sid)
                            n_fdr_rejected += 1
                        # If p_value is NULL, strategy legitimately can't be
                        # FDR-evaluated. Pass through to promotion.
                        continue
                    # fdr is not None below — look up the effective K that
                    # was ACTUALLY used for BH on this strategy's session
                    _sid_session = con.execute(
                        "SELECT orb_label FROM experimental_strategies WHERE strategy_id = ?",
                        [sid],
                    ).fetchone()
                    _sess_k = effective_k_by_session.get(_sid_session[0], total_k) if _sid_session else total_k
                    # Freeze discovery_k: only set on first write.
                    # Subsequent rebuilds update fdr_significant/adjusted_p
                    # (which may change as the canonical pool grows) but
                    # preserve the K under which the strategy was originally
                    # promoted. This maintains audit trail integrity.
                    # WFE gate: WFE < MIN_WFE → overfit, demote regardless of FDR
                    # Pardo: WFE < 0.50 = lost >50% of edge OOS → likely overfit
                    _wfe_row = con.execute("SELECT wfe FROM validated_setups WHERE strategy_id = ?", [sid]).fetchone()
                    _wfe = _wfe_row[0] if _wfe_row and _wfe_row[0] is not None else 0.0  # fail-closed
                    _fdr_sig = fdr["fdr_significant"] and _wfe >= MIN_WFE
                    if fdr["fdr_significant"] and not _fdr_sig:
                        logger.info(f"  WFE gate: {sid} demoted (WFE={_wfe:.2f} < {MIN_WFE})")

                    con.execute(
                        """UPDATE validated_setups
                           SET fdr_significant = ?,
                               fdr_adjusted_p = ?,
                               p_value = ?,
                               validation_pathway = 'family',
                               n_trials_at_discovery = CASE
                                   WHEN n_trials_at_discovery IS NULL THEN ?
                                   ELSE n_trials_at_discovery
                               END,
                               discovery_k = CASE
                                   WHEN discovery_k IS NULL THEN ?
                                   ELSE discovery_k
                               END,
                               discovery_date = CASE
                                   WHEN discovery_date IS NULL THEN ?
                                   ELSE discovery_date
                               END
                           WHERE strategy_id = ?""",
                        [_fdr_sig, fdr["adjusted_p"], fdr["raw_p"], _sess_k, _sess_k, _today, sid],
                    )
                    if _fdr_sig:
                        n_fdr_sig += 1
                    else:
                        fdr_rejected_ids.append(sid)
                        n_fdr_rejected += 1

                # Hard gate: remove FDR-failing strategies from validated_setups
                if fdr_rejected_ids:
                    for sid in fdr_rejected_ids:
                        # Look up session for rejection note
                        sid_session = con.execute(
                            "SELECT orb_label FROM experimental_strategies WHERE strategy_id = ?",
                            [sid],
                        ).fetchone()
                        sess_name = sid_session[0] if sid_session else "?"
                        sess_k = len(session_p_pools.get(sess_name, []))
                        con.execute(
                            "DELETE FROM validated_setups WHERE strategy_id = ?",
                            [sid],
                        )
                        con.execute(
                            """UPDATE experimental_strategies
                               SET validation_status = 'REJECTED',
                                   validation_notes = 'Phase FDR: BH adjusted p >= 0.05 (session='
                                       || ? || ', K=' || ? || ')',
                                   rejection_reason = 'criterion_3: BH FDR adjusted p >= 0.05 (session='
                                       || ? || ', K=' || ? || ')'
                               WHERE strategy_id = ?""",
                            [sess_name, str(sess_k), sess_name, str(sess_k), sid],
                        )
                    passed -= n_fdr_rejected
                    rejected += n_fdr_rejected
                    # Remove from passed list so downstream counts are correct
                    passed_strategy_ids = [s for s in passed_strategy_ids if s not in set(fdr_rejected_ids)]

                logger.info(
                    f"  FDR hard gate (stratified, total K={total_k}): "
                    f"{n_fdr_sig} survived, {n_fdr_rejected} REJECTED "
                    f"(of {n_fdr_sig + n_fdr_rejected} passed prior phases)"
                )

            if instrument in ACTIVE_ORB_INSTRUMENTS:
                n_families = build_edge_families_for_instrument(con, instrument)
                logger.info("  Edge families rebuilt for %s: %s families", instrument, n_families)
            else:
                con.execute("DELETE FROM edge_families WHERE instrument = ?", [instrument])
                logger.info(
                    "  Non-tradeable instrument %s kept off active shelf; edge families cleared",
                    instrument,
                )

            # ── DSR: Deflated Sharpe Ratio (informational, not a gate) ──
            # Computed per Bailey & Lopez de Prado (2014). Stored for analysis.
            # NOT a hard gate because N_eff is uncertain (see dsr.py docstring).
            if passed_strategy_ids:
                from trading_app.dsr import compute_dsr, compute_sr0

                # V[SR] partitioned by entry model (cross-model review finding:
                # mixing E1+E2 inflates V[SR] due to structural cost gap).
                var_sr_by_em = {}
                for em_query in ["E1", "E2"]:
                    vr = con.execute(
                        """SELECT VAR_SAMP(sharpe_ratio)
                           FROM experimental_strategies
                           WHERE entry_model = ?
                           AND sample_size >= 30
                           AND sharpe_ratio IS NOT NULL
                           AND is_canonical = TRUE""",
                        [em_query],
                    ).fetchone()
                    var_sr_by_em[em_query] = vr[0] if vr and vr[0] else 0.047

                # N_eff: use edge family count as conservative estimate.
                # True N_eff requires ONC algorithm (action queue #9).
                n_eff_row = con.execute("SELECT COUNT(DISTINCT family_hash) FROM edge_families").fetchone()
                n_eff = max(n_eff_row[0] if n_eff_row and n_eff_row[0] else 253, 2)

                n_dsr_pass = 0

                for sid in passed_strategy_ids:
                    row_data = con.execute(
                        """SELECT sharpe_ratio, sample_size, skewness, kurtosis_excess, entry_model
                           FROM validated_setups WHERE strategy_id = ?""",
                        [sid],
                    ).fetchone()
                    if row_data:
                        sr_hat = row_data[0] or 0
                        t_obs = row_data[1] or 30
                        skew = row_data[2] or 0
                        kurt = row_data[3] or 0
                        em_val = row_data[4] or "E2"
                        var_sr = var_sr_by_em.get(em_val, 0.047)
                        sr0 = compute_sr0(n_eff, var_sr)
                        dsr_val = compute_dsr(sr_hat, sr0, t_obs, skew, kurt)
                        con.execute(
                            "UPDATE validated_setups SET dsr_score = ?, sr0_at_discovery = ? WHERE strategy_id = ?",
                            [dsr_val, sr0, sid],
                        )
                        if dsr_val > 0.95:
                            n_dsr_pass += 1

                logger.info(
                    f"  DSR (informational, N_eff={n_eff}, "
                    f"V[SR] E1={var_sr_by_em.get('E1', 0):.4f} E2={var_sr_by_em.get('E2', 0):.4f}): "
                    f"{n_dsr_pass}/{len(passed_strategy_ids)} pass DSR>0.95"
                )

            con.commit()

    logger.info(
        f"Validation complete: {passed} PASSED, {rejected} REJECTED, "
        f"{skipped_aliases} aliases skipped "
        f"(of {len(rows)} strategies)"
    )
    if dry_run:
        logger.info("  (DRY RUN — no data written)")

    # ── Phase D: Log rejection rate per phase ─────────────────────────
    if not dry_run:
        # Count rejections per phase from notes
        # phase4c/phase4d: REMOVED as fake gates (2026-03-18). Counters kept at 0 for schema compat.
        phase_counts = {
            "phase1": 0,
            "phase2": 0,
            "phase2b": 0,  # REMOVED as hard gate (2026-03-21). Counter kept at 0 for schema compat.
            "phase3": 0,
            "phase4": 0,
            "phase4c": 0,  # removed — was never a gate (N_eff broken)
            "phase4d": 0,  # removed — was never a gate (N_eff + unit mismatch)
            "phase4b": 0,
            "phase_fdr": 0,  # BH FDR hard gate (global K) — added 2026-03-18
        }
        for sr in serial_results:
            notes_str = sr.get("notes", "")
            if sr["status"] == "REJECTED":
                if notes_str.startswith("Phase 1:"):
                    phase_counts["phase1"] += 1
                elif notes_str.startswith("Phase 2:"):
                    phase_counts["phase2"] += 1
                elif notes_str.startswith("Phase 3:"):
                    phase_counts["phase3"] += 1
                elif notes_str.startswith("Phase 4b:"):
                    phase_counts["phase4b"] += 1
                elif notes_str.startswith("Phase 4:"):
                    phase_counts["phase4"] += 1
        # FDR rejections happen post-write (Phase C), not in serial_results.
        phase_counts["phase_fdr"] = n_fdr_rejected

        candidates = len(rows) - skipped_aliases
        rejection_rate = 1.0 - (passed / candidates) if candidates > 0 else 1.0

        if validation_run_id is None:
            import uuid

            validation_run_id = f"{instrument}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        with duckdb.connect(str(db_path)) as con:
            from pipeline.db_config import configure_connection

            configure_connection(con, writing=True)
            # Migration: add phase_fdr_rejected column if missing
            try:
                con.execute("ALTER TABLE validation_run_log ADD COLUMN phase_fdr_rejected INTEGER DEFAULT 0")
            except duckdb.CatalogException:
                pass  # column already exists

            con.execute(
                """INSERT INTO validation_run_log
                   (run_id, instrument, candidates,
                    phase1_rejected, phase2_rejected, phase3_rejected,
                    phase4_rejected, phase4c_rejected, phase4d_rejected,
                    phase4b_rejected, phase_fdr_rejected,
                   final_passed, rejection_rate)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    validation_run_id,
                    instrument,
                    candidates,
                    phase_counts["phase1"],
                    phase_counts["phase2"],
                    phase_counts["phase3"],
                    phase_counts["phase4"],
                    phase_counts["phase4c"],
                    phase_counts["phase4d"],
                    phase_counts["phase4b"],
                    phase_counts["phase_fdr"],
                    passed,
                    round(rejection_rate, 4),
                ],
            )
            con.commit()
        logger.info(
            f"  Run logged: {validation_run_id} — rejection_rate={rejection_rate:.1%}, "
            f"P1={phase_counts['phase1']}, P2={phase_counts['phase2']}, "
            f"P2b={phase_counts['phase2b']}, P3={phase_counts['phase3']}, "
            f"P4={phase_counts['phase4']}, P4b={phase_counts['phase4b']}, "
            f"FDR={phase_counts['phase_fdr']}"
        )

    return passed, rejected


def _check_mode_a_holdout_integrity(db_path: Path | None, instrument: str) -> None:
    """Pre-flight gate: refuse to validate when NEW Mode A contamination exists.

    Amendment 2.7 (pre_registered_criteria.md, 2026-04-08) mandates that any
    ``experimental_strategies`` row created AFTER the grandfather cutoff
    containing sacred-window (currently ``HOLDOUT_SACRED_FROM.year``) yearly
    results must have been discovered with ``--holdout-date
    HOLDOUT_SACRED_FROM`` or earlier. The discovery CLI enforces this at
    entry (Stage 3 of the Amendment 2.7 enforcement refactor), but if a
    row lands in ``experimental_strategies`` via a bypass path (direct SQL,
    a legacy script, a test fixture), the validator must refuse to promote
    it rather than silently bless contaminated work.

    This function is a belt-and-suspenders check: ``check_drift.py``
    ``check_holdout_contamination()`` surfaces the same violation at drift
    check time, and the discovery CLI prevents it at entry, so in normal
    operation this pre-flight is a no-op (zero new violations). It fails
    loud only when the two upstream gates have been bypassed.

    Parameters
    ----------
    db_path
        Path to gold.db (``None`` uses the canonical default).
    instrument
        Instrument symbol to check (validator runs per-instrument).

    Raises
    ------
    ValueError
        If any experimental_strategies row for this instrument has
        ``created_at > HOLDOUT_GRANDFATHER_CUTOFF`` AND contains
        ``yearly_results['<sacred_year>']``. The error message cites
        Amendment 2.7 and the canonical source module.
    """
    from trading_app.holdout_policy import (
        HOLDOUT_GRANDFATHER_CUTOFF,
        HOLDOUT_SACRED_FROM,
    )

    effective_path = db_path if db_path else GOLD_DB_PATH
    if not Path(effective_path).exists():
        # Fail-open: if the DB isn't there, discovery hasn't run anyway.
        # The discovery CLI gate would have caught contamination at entry.
        return

    import duckdb

    sacred_year = str(HOLDOUT_SACRED_FROM.year)
    with duckdb.connect(str(effective_path), read_only=True) as con:
        row = con.execute(
            """SELECT COUNT(*) FROM experimental_strategies
               WHERE instrument = ?
               AND created_at > ?
               AND yearly_results IS NOT NULL
               AND json_extract_string(yearly_results, '$."' || ? || '"') IS NOT NULL""",
            [instrument, HOLDOUT_GRANDFATHER_CUTOFF, sacred_year],
        ).fetchone()
        new_contam = row[0] if row is not None else 0

    if new_contam > 0:
        raise ValueError(
            f"Validator refuses to promote {new_contam} {instrument} "
            f"experimental_strategies with {sacred_year} data created after "
            f"{HOLDOUT_GRANDFATHER_CUTOFF.date().isoformat()}. "
            f"This violates Mode A (pre_registered_criteria.md Amendment 2.7). "
            f"Either re-run discovery with --holdout-date "
            f"{HOLDOUT_SACRED_FROM.isoformat()} or promote from a "
            f"pre-grandfather DB snapshot. "
            f"Canonical source: trading_app.holdout_policy"
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate strategies and promote to validated_setups")
    parser.add_argument("--instrument", default="MGC", help="Instrument symbol")
    parser.add_argument("--min-sample", type=int, default=REGIME_MIN_SAMPLES, help="Min sample size")
    parser.add_argument("--stress-multiplier", type=float, default=1.5, help="Cost stress multiplier")
    parser.add_argument("--min-sharpe", type=float, default=None, help="Min Sharpe ratio (optional)")
    parser.add_argument("--max-drawdown", type=float, default=None, help="Max drawdown in R (optional)")
    parser.add_argument(
        "--exclude-years",
        type=int,
        nargs="*",
        default=None,
        help="Years to exclude from Phase 3 (e.g. --exclude-years 2021)",
    )
    parser.add_argument(
        "--min-years-positive-pct",
        type=float,
        default=0.75,
        help="Fraction of included years that must be positive (0.0-1.0, default 0.75)",
    )
    parser.add_argument(
        "--min-trades-per-year",
        type=int,
        default=1,
        help="Min trades for a year to count in Phase 3 robustness check (default 1)",
    )
    parser.add_argument("--dry-run", action="store_true", help="No DB writes")
    parser.add_argument("--db", type=str, default=None, help="Database path (default: gold.db)")
    # Walk-forward (Phase 4b)
    parser.add_argument("--no-walkforward", action="store_true", help="Disable walk-forward validation (Phase 4b)")
    parser.add_argument("--wf-test-months", type=int, default=6, help="Walk-forward test window months (default: 6)")
    parser.add_argument(
        "--wf-min-train-months", type=int, default=12, help="Walk-forward min training months (default: 12)"
    )
    parser.add_argument(
        "--wf-min-trades", type=int, default=15, help="Walk-forward min trades per window (default: 15)"
    )
    parser.add_argument("--wf-min-windows", type=int, default=3, help="Walk-forward min valid windows (default: 3)")
    parser.add_argument(
        "--wf-min-pct-positive", type=float, default=0.60, help="Walk-forward min pct positive windows (default: 0.60)"
    )
    parser.add_argument(
        "--no-regime-waivers", action="store_true", help="Disable DORMANT regime waivers (strict all-years-positive)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel workers for walkforward (default: min(8, cpu_count-1), 1=serial)",
    )
    parser.add_argument(
        "--fdr-k-file",
        type=str,
        default=None,
        help="JSON file with per-session K overrides for BH FDR. "
        'Format: {"CME_PRECLOSE": 3254, ...}. Use for null seed testing '
        "where the DB has fewer strategies than the real pipeline.",
    )
    parser.add_argument(
        "--testing-mode",
        type=str,
        choices=["family", "individual"],
        default="family",
        help="Amendment 3.0: 'family' = Pathway A (BH FDR), "
        "'individual' = Pathway B (raw p < 0.05 + positive sharpe_ann). "
        "Default: family.",
    )
    args = parser.parse_args()

    exclude = set(args.exclude_years) if args.exclude_years else None
    db_path = Path(args.db) if args.db else None

    # Mode A holdout integrity gate (Amendment 2.7). Belt-and-suspenders check
    # that refuses to promote any experimental_strategies rows created after
    # the grandfather cutoff with sacred-window data. Discovery CLI enforces
    # this at entry (Stage 3 refactor), but this validator gate catches any
    # bypass path. In normal operation this is a no-op.
    try:
        _check_mode_a_holdout_integrity(db_path, args.instrument)
    except ValueError as e:
        parser.error(str(e))  # exits code 2 with standard argparse error format

    # Load per-session K overrides if provided
    fdr_k_overrides = None
    if args.fdr_k_file:
        import json as _json

        with open(args.fdr_k_file) as _f:
            fdr_k_overrides = _json.load(_f)
        logger.info(f"Loaded FDR K overrides from {args.fdr_k_file}: {len(fdr_k_overrides)} sessions")

    run_validation(
        db_path=db_path,
        instrument=args.instrument,
        min_sample=args.min_sample,
        stress_multiplier=args.stress_multiplier,
        min_sharpe=args.min_sharpe,
        max_drawdown=args.max_drawdown,
        exclude_years=exclude,
        min_years_positive_pct=args.min_years_positive_pct,
        min_trades_per_year=args.min_trades_per_year,
        dry_run=args.dry_run,
        enable_walkforward=not args.no_walkforward,
        wf_test_months=args.wf_test_months,
        wf_min_train_months=args.wf_min_train_months,
        wf_min_trades=args.wf_min_trades,
        wf_min_windows=args.wf_min_windows,
        wf_min_pct_positive=args.wf_min_pct_positive,
        enable_regime_waivers=not args.no_regime_waivers,
        workers=args.workers,
        fdr_k_overrides=fdr_k_overrides,
        testing_mode=args.testing_mode,
    )


if __name__ == "__main__":
    main()
