"""
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

from pipeline.log import get_logger

logger = get_logger(__name__)

import duckdb

from pipeline.cost_model import get_cost_spec, stress_test_costs
from pipeline.dst import (
    DST_AFFECTED_SESSIONS,
    classify_dst_verdict,
    is_winter_for_session,
)
from pipeline.paths import GOLD_DB_PATH
from trading_app.config import (
    CORE_MIN_SAMPLES,
    NOISE_EXPR_FLOOR,
    REGIME_MIN_SAMPLES,
    WF_MIN_TRAIN_TRADES,
    WF_START_OVERRIDE,
    WF_TRADE_COUNT_OVERRIDE,
)
from trading_app.db_manager import init_trading_app_schema
from trading_app.strategy_discovery import parse_dst_regime
from trading_app.walkforward import append_walkforward_result

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]


# =========================================================================
# FDR correction (F-01: Multiple comparison adjustment)
# =========================================================================


def benjamini_hochberg(p_values: list[tuple[str, float]], alpha: float = 0.05) -> dict[str, dict]:
    """Apply Benjamini-Hochberg FDR correction to a set of p-values.

    Addresses the Bailey Rule: when testing thousands of strategies, some will
    appear profitable by chance. BH controls the False Discovery Rate — the
    expected proportion of false positives among all rejections.

    Args:
        p_values: List of (strategy_id, raw_p_value) tuples. Strategies with
            None p-values are excluded from correction.
        alpha: FDR significance level (default 0.05 = 5% expected false discoveries).

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
    m = len(valid)

    results = {}
    # BH procedure: adjusted_p[i] = min(p[i] * m / rank, 1.0)
    # Enforce monotonicity: adjusted_p[i] = min(adjusted_p[i], adjusted_p[i+1])
    prev_adj = 1.0
    for rank_idx in range(m - 1, -1, -1):
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

    # DOW skip filter (CompositeFilter with DayOfWeekSkipFilter overlay)
    skip_days = _parse_skip_days(filter_params)
    if skip_days:
        placeholders = ", ".join("?" * len(skip_days))
        size_clauses.append(f"df.day_of_week NOT IN ({placeholders})")
        size_params.extend(skip_days)

    size_where = (" AND " + " AND ".join(size_clauses)) if size_clauses else ""

    rows = con.execute(
        f"""
        SELECT o.trading_day, COALESCE(o.ts_pnl_r, o.pnl_r) AS pnl_r
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
          AND o.outcome IN ('win', 'loss', 'time_stop')
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

    # Phase 2b: Noise floor — reject strategies indistinguishable from noise
    # @research-source null_test_10_seeds (White's Reality Check 2026-03-18)
    entry_model = row.get("entry_model") or "E1"
    noise_floor = NOISE_EXPR_FLOOR.get(entry_model, NOISE_EXPR_FLOOR.get("E2", 0.32))
    if exp_r <= noise_floor:
        return (
            "REJECTED",
            f"Phase 2b: ExpR={exp_r:.4f} <= noise floor {noise_floor} for {entry_model}",
            [],
        )

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
        # NOTE: When regime waivers are active, they supersede min_years_positive_pct.
        # Waiver path requires ALL negative years to be DORMANT-waivable (or reject).
        # The pct threshold only applies in strict mode (waivers disabled/no ATR data).
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

            if unwaived_neg:
                return (
                    "REJECTED",
                    (
                        f"Phase 3: {len(unwaived_neg)} year(s) negative and not waivable: "
                        f"{', '.join(sorted(unwaived_neg))}"
                    ),
                    [],
                )

            if pos_count == 0:
                return (
                    "REJECTED",
                    ("Phase 3: All years require DORMANT waiver, need at least 1 clean positive year"),
                    [],
                )

            regime_waivers = sorted(waived)
            for yr in regime_waivers:
                y_str = str(yr)
                d = neg_years[y_str]
                notes.append(
                    f"Year {yr} waived: DORMANT regime (mean_atr={atr_by_year[yr]:.1f}, trades={d.get('trades', 0)})"
                )
        else:
            # Original strict logic
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

    result["wf_duration_s"] = time.monotonic() - t0
    return result


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
) -> tuple[int, int]:
    """
    Validate all experimental_strategies and promote passing ones.

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

    serial_results = []  # Each: {row_dict, status, notes, regime_waivers, dst_split}
    wf_candidates = []  # Survivors needing walkforward

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

        status, notes, regime_waivers = validate_strategy(
            row_dict,
            cost_spec,
            stress_multiplier=stress_multiplier,
            min_sample=min_sample,
            min_sharpe=min_sharpe,
            max_drawdown=max_drawdown,
            exclude_years=exclude_years,
            min_years_positive_pct=min_years_positive_pct,
            min_trades_per_year=min_trades_per_year,
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
                wf_params=wf_params,
                dst_regime=cand["strat_dst_regime"],
                dst_verdict_from_discovery=rd.get("dst_verdict"),
                wf_start_date=wf_start_date,
                stop_multiplier=rd.get("stop_multiplier", 1.0),
                wf_test_trades=wf_test_trades,
                wf_min_train_trades=wf_min_train,
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
    processed_orb_minutes = sorted({sr["row_dict"].get("orb_minutes", 5) for sr in serial_results})
    if not dry_run and processed_orb_minutes:
        with duckdb.connect(str(db_path)) as con:
            from pipeline.db_config import configure_connection

            configure_connection(con, writing=True)

            # Purge stale validated_setups for this instrument + processed
            # apertures only.  edge_families has no orb_minutes column, so
            # scope via head_strategy_id → validated_setups FK.
            placeholders = ", ".join("?" * len(processed_orb_minutes))
            con.execute(
                f"""DELETE FROM edge_families
                    WHERE instrument = ?
                    AND head_strategy_id IN (
                        SELECT strategy_id FROM validated_setups
                        WHERE instrument = ?
                        AND orb_minutes IN ({placeholders})
                    )""",
                [instrument, instrument] + processed_orb_minutes,
            )
            con.execute(
                f"""DELETE FROM validated_setups
                    WHERE instrument = ?
                    AND orb_minutes IN ({placeholders})""",
                [instrument] + processed_orb_minutes,
            )

            for sr in serial_results:
                rd = sr["row_dict"]
                sid = rd["strategy_id"]
                status = sr["status"]
                notes = sr["notes"]
                dst_split = sr["dst_split"]

                if status == "SKIPPED":
                    con.execute(
                        """UPDATE experimental_strategies
                           SET validation_status = 'SKIPPED',
                               validation_notes = 'Alias (non-canonical)'
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
                           dst_verdict = ?
                       WHERE strategy_id = ?""",
                    [
                        status,
                        notes,
                        dst_split.get("winter_n"),
                        dst_split.get("winter_avg_r"),
                        dst_split.get("summer_n"),
                        dst_split.get("summer_avg_r"),
                        dst_split.get("verdict"),
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

                    wf_result_dict = sr.get("wf_result_dict")
                    wf_tested = wf_result_dict is not None
                    wf_passed = (wf_result_dict or {}).get("passed", False) if wf_tested else None
                    wf_windows_val = (
                        (wf_result_dict or {}).get("as_dict", {}).get("n_valid_windows") if wf_tested else None
                    )
                    wfe_val = (wf_result_dict or {}).get("as_dict", {}).get("wfe") if wf_tested else None

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
                            regime_waivers, regime_waiver_count,
                            dst_winter_n, dst_winter_avg_r,
                            dst_summer_n, dst_summer_avg_r,
                            dst_verdict,
                            wf_tested, wf_passed, wf_windows, wfe,
                            sharpe_haircut, skewness, kurtosis_excess)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                            "active",
                            rd.get("median_risk_dollars"),
                            rd.get("avg_risk_dollars"),
                            rd.get("avg_win_dollars"),
                            rd.get("avg_loss_dollars"),
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

            # FDR hard gate — global K across ALL instruments
            # BH FDR valid under PRDS (Benjamini & Yekutieli 2001). Cross-instrument
            # equity correlations (MNQ-MES rho=0.44-0.77) satisfy PRDS. Global K is
            # the honest test count: you searched all instruments, not just this one.
            # Adversarial review 2026-03-18: FDR was cosmetic — now it rejects.
            if passed_strategy_ids:
                logger.info("Computing FDR correction (Benjamini-Hochberg, global K)...")
                all_p_values = con.execute(
                    """SELECT strategy_id, p_value FROM experimental_strategies
                       WHERE is_canonical = TRUE
                       AND p_value IS NOT NULL""",
                ).fetchall()
                p_value_list = [(r[0], r[1]) for r in all_p_values]
                fdr_results = benjamini_hochberg(p_value_list, alpha=0.05)
                global_k = len(p_value_list)

                n_fdr_sig = 0
                n_fdr_rejected = 0
                fdr_rejected_ids = []
                for sid in passed_strategy_ids:
                    fdr = fdr_results.get(sid)
                    if fdr is not None:
                        con.execute(
                            """UPDATE validated_setups
                               SET fdr_significant = ?,
                                   fdr_adjusted_p = ?
                               WHERE strategy_id = ?""",
                            [fdr["fdr_significant"], fdr["adjusted_p"], sid],
                        )
                        if fdr["fdr_significant"]:
                            n_fdr_sig += 1
                        else:
                            fdr_rejected_ids.append(sid)
                            n_fdr_rejected += 1

                # Hard gate: remove FDR-failing strategies from validated_setups
                if fdr_rejected_ids:
                    for sid in fdr_rejected_ids:
                        con.execute(
                            "DELETE FROM validated_setups WHERE strategy_id = ?",
                            [sid],
                        )
                        con.execute(
                            """UPDATE experimental_strategies
                               SET validation_status = 'REJECTED',
                                   validation_notes = 'Phase FDR: BH adjusted p >= 0.05 (global K='
                                       || ? || ')'
                               WHERE strategy_id = ?""",
                            [str(global_k), sid],
                        )
                    passed -= n_fdr_rejected
                    rejected += n_fdr_rejected
                    # Remove from passed list so downstream counts are correct
                    passed_strategy_ids = [s for s in passed_strategy_ids if s not in set(fdr_rejected_ids)]

                logger.info(
                    f"  FDR hard gate (global K={global_k}): "
                    f"{n_fdr_sig} survived, {n_fdr_rejected} REJECTED "
                    f"(of {n_fdr_sig + n_fdr_rejected} passed prior phases)"
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
            "phase2b": 0,  # noise floor gate — added 2026-03-18
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
                elif notes_str.startswith("Phase 2b:"):
                    phase_counts["phase2b"] += 1
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

        import uuid

        run_id = f"{instrument}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

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
                    run_id,
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
            f"  Run logged: {run_id} — rejection_rate={rejection_rate:.1%}, "
            f"P1={phase_counts['phase1']}, P2={phase_counts['phase2']}, "
            f"P2b={phase_counts['phase2b']}, P3={phase_counts['phase3']}, "
            f"P4={phase_counts['phase4']}, P4b={phase_counts['phase4b']}, "
            f"FDR={phase_counts['phase_fdr']}"
        )

    return passed, rejected


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
    args = parser.parse_args()

    exclude = set(args.exclude_years) if args.exclude_years else None
    db_path = Path(args.db) if args.db else None

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
    )


if __name__ == "__main__":
    main()
