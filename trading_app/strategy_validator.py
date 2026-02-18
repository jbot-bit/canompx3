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
  5. Sharpe ratio (optional quality filter)
  6. Max drawdown (optional risk filter)

Usage:
    python trading_app/strategy_validator.py --instrument MGC
    python trading_app/strategy_validator.py --instrument MGC --no-walkforward
    python trading_app/strategy_validator.py --instrument MGC --min-sample 100 --dry-run
"""

import sys
import json
from datetime import date
from pathlib import Path

from pipeline.log import get_logger
logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import duckdb

from pipeline.paths import GOLD_DB_PATH
from pipeline.cost_model import get_cost_spec, stress_test_costs
from pipeline.dst import (
    DST_AFFECTED_SESSIONS,
    is_winter_for_session, classify_dst_verdict,
)
from trading_app.db_manager import init_trading_app_schema
from trading_app.walkforward import run_walkforward, append_walkforward_result

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

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
                return (float(min_s) if min_s is not None else None,
                        float(max_s) if max_s is not None else None)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # Fallback: parse from filter_type string (ORB_G5 -> min=5, ORB_G4_L12 -> min=4/max=12)
    if filter_type and filter_type.startswith("ORB_G"):
        rest = filter_type[5:]  # after "ORB_G"
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


def compute_dst_split(con, strategy_id: str, instrument: str, orb_label: str,
                      entry_model: str, rr_target: float, confirm_bars: int,
                      filter_type: str, filter_params: str | None = None,
                      orb_minutes: int = 5) -> dict:
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
            "winter_n": None, "winter_avg_r": None,
            "summer_n": None, "summer_avg_r": None,
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

    size_where = (" AND " + " AND ".join(size_clauses)) if size_clauses else ""

    rows = con.execute(f"""
        SELECT o.trading_day, o.pnl_r
        FROM orb_outcomes o
        JOIN daily_features df
          ON o.trading_day = df.trading_day
          AND df.symbol = ?
          AND df.orb_minutes = ?
        WHERE o.symbol = ?
          AND o.orb_label = ?
          AND o.entry_model = ?
          AND o.rr_target = ?
          AND o.confirm_bars = ?
          AND o.outcome IN ('win', 'loss')
          {size_where}
        ORDER BY o.trading_day
    """, [instrument, orb_minutes, instrument, orb_label,
          entry_model, rr_target, confirm_bars] + size_params).fetchall()

    if not rows:
        return {
            "winter_n": 0, "winter_avg_r": None,
            "summer_n": 0, "summer_avg_r": None,
            "verdict": "LOW-N",
        }

    # Split by DST regime
    winter_rs = []
    summer_rs = []

    for trading_day, pnl_r in rows:
        # trading_day may come back as date or datetime
        if hasattr(trading_day, 'date'):
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
    if atr_20 < 20.0:
        return "DORMANT"
    elif atr_20 < 30.0:
        return "MARGINAL"
    return "ACTIVE"

def validate_strategy(row: dict, cost_spec, stress_multiplier: float = 1.5,
                      min_sample: int = 30, min_sharpe: float | None = None,
                      max_drawdown: float | None = None,
                      exclude_years: set[int] | None = None,
                      min_years_positive_pct: float = 1.0,
                      atr_by_year: dict[int, float] | None = None,
                      enable_regime_waivers: bool = True) -> tuple[str, str, list[int]]:
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
            positive (0.0-1.0). Default 1.0 = ALL years must be positive.
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
    if sample < 100:
        notes.append(f"Phase 1 WARN: sample={sample} (< 100)")

    # Phase 2: Post-cost expectancy
    exp_r = row.get("expectancy_r")
    if exp_r is None or exp_r <= 0:
        return "REJECTED", f"Phase 2: ExpR={exp_r} <= 0", []

    # Phase 3: Yearly robustness
    yearly_json = row.get("yearly_results", "{}")
    try:
        yearly = json.loads(yearly_json) if isinstance(yearly_json, str) else yearly_json
    except (json.JSONDecodeError, TypeError):
        yearly = {}

    if not yearly:
        return "REJECTED", "Phase 3: No yearly data", []

    included_years = {y: d for y, d in yearly.items() if int(y) not in exclude_years}
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
                if mean_atr is not None and classify_regime(mean_atr) == "DORMANT" and trades <= 5:
                    waived.append(yr_int)

            unwaived_neg = [y for y in neg_years if int(y) not in waived]

            if unwaived_neg:
                return "REJECTED", (
                    f"Phase 3: {len(unwaived_neg)} year(s) negative and not waivable: "
                    f"{', '.join(sorted(unwaived_neg))}"
                ), []

            if pos_count == 0:
                return "REJECTED", (
                    "Phase 3: All years require DORMANT waiver, "
                    "need at least 1 clean positive year"
                ), []

            regime_waivers = sorted(waived)
            for yr in regime_waivers:
                y_str = str(yr)
                d = neg_years[y_str]
                notes.append(
                    f"Year {yr} waived: DORMANT regime "
                    f"(mean_atr={atr_by_year[yr]:.1f}, trades={d.get('trades', 0)})"
                )
        else:
            # Original strict logic
            pct_positive = pos_count / len(included_years)
            if pct_positive < min_years_positive_pct:
                neg_list = sorted(neg_years.keys())
                return "REJECTED", (
                    f"Phase 3: {pos_count}/{len(included_years)} years positive "
                    f"({pct_positive:.0%} < {min_years_positive_pct:.0%}), "
                    f"negative: {', '.join(neg_list)}"
                ), []

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

    strategy_risk_dollars = strategy_risk_points * cost_spec.point_value
    # Floor: never below tick-based minimum
    denom = max(strategy_risk_dollars, cost_spec.min_risk_floor_dollars)
    extra_cost_per_trade_r = stress_friction_delta / denom
    stress_exp = exp_r - extra_cost_per_trade_r

    if stress_exp <= 0:
        return "REJECTED", f"Phase 4: Stress ExpR={stress_exp:.4f} <= 0 (base={exp_r}, delta_r={extra_cost_per_trade_r:.4f}, risk_pts={strategy_risk_points:.2f})", []

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

def run_validation(
    db_path: Path | None = None,
    instrument: str = "MGC",
    min_sample: int = 30,
    stress_multiplier: float = 1.5,
    min_sharpe: float | None = None,
    max_drawdown: float | None = None,
    exclude_years: set[int] | None = None,
    min_years_positive_pct: float = 1.0,
    dry_run: bool = False,
    enable_walkforward: bool = True,
    wf_test_months: int = 6,
    wf_min_train_months: int = 12,
    wf_min_trades: int = 15,
    wf_min_windows: int = 3,
    wf_min_pct_positive: float = 0.60,
    wf_output_path: str = "data/walkforward_results.jsonl",
    enable_regime_waivers: bool = True,
) -> tuple[int, int]:
    """
    Validate all experimental_strategies and promote passing ones.

    Returns (passed_count, rejected_count).
    """
    if db_path is None:
        db_path = GOLD_DB_PATH

    cost_spec = get_cost_spec(instrument)

    with duckdb.connect(str(db_path)) as con:
        from pipeline.db_config import configure_connection
        configure_connection(con, writing=True)

        if not dry_run:
            init_trading_app_schema(db_path=db_path)

        # Fetch unvalidated strategies
        rows = con.execute(
            """SELECT * FROM experimental_strategies
               WHERE instrument = ?
               AND (validation_status IS NULL OR validation_status = '')
               ORDER BY strategy_id""",
            [instrument],
        ).fetchall()
        col_names = [desc[0] for desc in con.description]

        # Pre-fetch ATR by year for regime waivers (one query total)
        atr_by_year = {}
        if enable_regime_waivers:
            atr_rows = con.execute("""
                SELECT EXTRACT(YEAR FROM trading_day) as yr, AVG(atr_20) as mean_atr
                FROM daily_features
                WHERE symbol = ? AND orb_minutes = 5 AND atr_20 IS NOT NULL
                GROUP BY yr
            """, [instrument]).fetchall()
            atr_by_year = {int(r[0]): r[1] for r in atr_rows}

        passed = 0
        rejected = 0
        skipped_aliases = 0

        for row in rows:
            row_dict = dict(zip(col_names, row))
            strategy_id = row_dict["strategy_id"]

            # Skip aliases (non-canonical strategies)
            if row_dict.get("is_canonical") is False:
                skipped_aliases += 1
                if not dry_run:
                    con.execute(
                        """UPDATE experimental_strategies
                           SET validation_status = 'SKIPPED',
                               validation_notes = 'Alias (non-canonical)'
                           WHERE strategy_id = ?""",
                        [strategy_id],
                    )
                continue

            status, notes, regime_waivers = validate_strategy(
                row_dict, cost_spec,
                stress_multiplier=stress_multiplier,
                min_sample=min_sample,
                min_sharpe=min_sharpe,
                max_drawdown=max_drawdown,
                exclude_years=exclude_years,
                min_years_positive_pct=min_years_positive_pct,
                atr_by_year=atr_by_year if enable_regime_waivers else None,
                enable_regime_waivers=enable_regime_waivers,
            )

            # Phase 4b: Walk-forward gate
            if status == "PASSED" and enable_walkforward:
                wf_result = run_walkforward(
                    con=con,
                    strategy_id=strategy_id,
                    instrument=instrument,
                    orb_label=row_dict["orb_label"],
                    entry_model=row_dict.get("entry_model", "E1"),
                    rr_target=row_dict["rr_target"],
                    confirm_bars=row_dict["confirm_bars"],
                    filter_type=row_dict.get("filter_type", "NO_FILTER"),
                    orb_minutes=row_dict.get("orb_minutes", 5),
                    test_window_months=wf_test_months,
                    min_train_months=wf_min_train_months,
                    min_trades_per_window=wf_min_trades,
                    min_valid_windows=wf_min_windows,
                    min_pct_positive=wf_min_pct_positive,
                )
                if not dry_run:
                    append_walkforward_result(wf_result, wf_output_path)
                if not wf_result.passed:
                    status = "REJECTED"
                    notes = f"Phase 4b: {wf_result.rejection_reason}"

            # DST regime split (INFO phase — does not reject, adds visibility)
            # Prefer discovery-computed values (correct for ALL filter types
            # including VolumeFilter which compute_dst_split cannot replicate).
            # Only recompute if discovery left them NULL.
            if row_dict.get("dst_verdict") is not None:
                dst_split = {
                    "winter_n": row_dict.get("dst_winter_n"),
                    "winter_avg_r": row_dict.get("dst_winter_avg_r"),
                    "summer_n": row_dict.get("dst_summer_n"),
                    "summer_avg_r": row_dict.get("dst_summer_avg_r"),
                    "verdict": row_dict.get("dst_verdict"),
                }
            else:
                dst_split = compute_dst_split(
                    con, strategy_id, instrument,
                    orb_label=row_dict["orb_label"],
                    entry_model=row_dict.get("entry_model", "E1"),
                    rr_target=row_dict["rr_target"],
                    confirm_bars=row_dict["confirm_bars"],
                    filter_type=row_dict.get("filter_type", "NO_FILTER"),
                    filter_params=row_dict.get("filter_params"),
                    orb_minutes=row_dict.get("orb_minutes", 5),
                )

            # Log DST info for affected sessions
            if dst_split["verdict"] not in ("CLEAN", None):
                w_r = dst_split["winter_avg_r"]
                s_r = dst_split["summer_avg_r"]
                w_str = f"{w_r:+.3f}" if w_r is not None else "N/A"
                s_str = f"{s_r:+.3f}" if s_r is not None else "N/A"
                logger.info(
                    f"  DST split [{strategy_id}]: "
                    f"W={w_str}({dst_split['winter_n']}) "
                    f"S={s_str}({dst_split['summer_n']}) "
                    f"-> {dst_split['verdict']}"
                )

            if not dry_run:
                # Update experimental_strategies (include DST columns)
                con.execute(
                    """UPDATE experimental_strategies
                       SET validation_status = ?, validation_notes = ?,
                           dst_winter_n = ?, dst_winter_avg_r = ?,
                           dst_summer_n = ?, dst_summer_avg_r = ?,
                           dst_verdict = ?
                       WHERE strategy_id = ?""",
                    [status, notes,
                     dst_split["winter_n"], dst_split["winter_avg_r"],
                     dst_split["summer_n"], dst_split["summer_avg_r"],
                     dst_split["verdict"],
                     strategy_id],
                )

                if status == "PASSED":
                    # Promote to validated_setups
                    yearly = row_dict.get("yearly_results", "{}")
                    try:
                        yearly_data = json.loads(yearly) if isinstance(yearly, str) else yearly
                    except (json.JSONDecodeError, TypeError):
                        yearly_data = {}

                    included = {y: d for y, d in yearly_data.items()
                                if int(y) not in (exclude_years or set())}
                    years_tested = len(included)
                    all_positive = all(
                        d.get("avg_r", 0) > 0 for d in included.values()
                    )

                    con.execute(
                        """INSERT OR REPLACE INTO validated_setups
                           (strategy_id, promoted_from, instrument, orb_label,
                            orb_minutes, rr_target, confirm_bars, entry_model,
                            filter_type, filter_params,
                            sample_size, win_rate, expectancy_r,
                            years_tested, all_years_positive, stress_test_passed,
                            sharpe_ratio, max_drawdown_r,
                            trades_per_year, sharpe_ann,
                            yearly_results, status,
                            regime_waivers, regime_waiver_count,
                            dst_winter_n, dst_winter_avg_r,
                            dst_summer_n, dst_summer_avg_r,
                            dst_verdict)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        [
                            strategy_id, strategy_id,
                            row_dict["instrument"], row_dict["orb_label"],
                            row_dict["orb_minutes"], row_dict["rr_target"],
                            row_dict["confirm_bars"], row_dict.get("entry_model", "E1"),
                            row_dict.get("filter_type", ""),
                            row_dict.get("filter_params", ""),
                            row_dict.get("sample_size", 0),
                            row_dict.get("win_rate", 0),
                            row_dict.get("expectancy_r", 0),
                            years_tested, all_positive, True,
                            row_dict.get("sharpe_ratio"),
                            row_dict.get("max_drawdown_r"),
                            row_dict.get("trades_per_year"),
                            row_dict.get("sharpe_ann"),
                            yearly, "active",
                            json.dumps(regime_waivers) if regime_waivers else None,
                            len(regime_waivers),
                            dst_split["winter_n"], dst_split["winter_avg_r"],
                            dst_split["summer_n"], dst_split["summer_avg_r"],
                            dst_split["verdict"],
                        ],
                    )

            if status == "PASSED":
                passed += 1
            else:
                rejected += 1

        if not dry_run:
            con.commit()

        logger.info(f"Validation complete: {passed} PASSED, {rejected} REJECTED, "
                    f"{skipped_aliases} aliases skipped "
                    f"(of {len(rows)} strategies)")
        if dry_run:
            logger.info("  (DRY RUN — no data written)")

        return passed, rejected

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate strategies and promote to validated_setups"
    )
    parser.add_argument("--instrument", default="MGC", help="Instrument symbol")
    parser.add_argument("--min-sample", type=int, default=30, help="Min sample size")
    parser.add_argument("--stress-multiplier", type=float, default=1.5, help="Cost stress multiplier")
    parser.add_argument("--min-sharpe", type=float, default=None, help="Min Sharpe ratio (optional)")
    parser.add_argument("--max-drawdown", type=float, default=None, help="Max drawdown in R (optional)")
    parser.add_argument("--exclude-years", type=int, nargs="*", default=None,
                        help="Years to exclude from Phase 3 (e.g. --exclude-years 2021)")
    parser.add_argument("--min-years-positive-pct", type=float, default=1.0,
                        help="Fraction of included years that must be positive (0.0-1.0, default 1.0)")
    parser.add_argument("--dry-run", action="store_true", help="No DB writes")
    parser.add_argument("--db", type=str, default=None,
                        help="Database path (default: gold.db)")
    # Walk-forward (Phase 4b)
    parser.add_argument("--no-walkforward", action="store_true",
                        help="Disable walk-forward validation (Phase 4b)")
    parser.add_argument("--wf-test-months", type=int, default=6,
                        help="Walk-forward test window months (default: 6)")
    parser.add_argument("--wf-min-train-months", type=int, default=12,
                        help="Walk-forward min training months (default: 12)")
    parser.add_argument("--wf-min-trades", type=int, default=15,
                        help="Walk-forward min trades per window (default: 15)")
    parser.add_argument("--wf-min-windows", type=int, default=3,
                        help="Walk-forward min valid windows (default: 3)")
    parser.add_argument("--wf-min-pct-positive", type=float, default=0.60,
                        help="Walk-forward min pct positive windows (default: 0.60)")
    parser.add_argument("--no-regime-waivers", action="store_true",
                        help="Disable DORMANT regime waivers (strict all-years-positive)")
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
        dry_run=args.dry_run,
        enable_walkforward=not args.no_walkforward,
        wf_test_months=args.wf_test_months,
        wf_min_train_months=args.wf_min_train_months,
        wf_min_trades=args.wf_min_trades,
        wf_min_windows=args.wf_min_windows,
        wf_min_pct_positive=args.wf_min_pct_positive,
        enable_regime_waivers=not args.no_regime_waivers,
    )

if __name__ == "__main__":
    main()
