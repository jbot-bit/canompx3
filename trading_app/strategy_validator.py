"""
6-phase strategy validation per CANONICAL_LOGIC.txt section 9.

Validates strategies from experimental_strategies and promotes passing
ones to validated_setups. Rejected strategies get validation_notes.

Phases:
  1. Sample size (reject < 30, warn < 100)
  2. Post-cost expectancy > 0
  3. Yearly robustness (positive in ALL years)
  4. Stress test (ExpR > 0 at +50% costs)
  5. Sharpe ratio (optional quality filter)
  6. Max drawdown (optional risk filter)

Usage:
    python trading_app/strategy_validator.py --instrument MGC
    python trading_app/strategy_validator.py --instrument MGC --min-sample 100 --dry-run
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb

from pipeline.paths import GOLD_DB_PATH
from pipeline.cost_model import get_cost_spec, stress_test_costs
from trading_app.db_manager import init_trading_app_schema

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)


def validate_strategy(row: dict, cost_spec, stress_multiplier: float = 1.5,
                      min_sample: int = 30, min_sharpe: float | None = None,
                      max_drawdown: float | None = None,
                      exclude_years: set[int] | None = None,
                      min_years_positive_pct: float = 1.0) -> tuple[str, str]:
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

    Returns:
        (status, notes): "PASSED" or "REJECTED", with explanation.
    """
    notes = []
    if exclude_years is None:
        exclude_years = set()

    # Phase 1: Sample size
    sample = row.get("sample_size") or 0
    if sample < min_sample:
        return "REJECTED", f"Phase 1: Sample size {sample} < {min_sample}"
    if sample < 100:
        notes.append(f"Phase 1 WARN: sample={sample} (< 100)")

    # Phase 2: Post-cost expectancy
    exp_r = row.get("expectancy_r")
    if exp_r is None or exp_r <= 0:
        return "REJECTED", f"Phase 2: ExpR={exp_r} <= 0"

    # Phase 3: Yearly robustness
    yearly_json = row.get("yearly_results", "{}")
    try:
        yearly = json.loads(yearly_json) if isinstance(yearly_json, str) else yearly_json
    except (json.JSONDecodeError, TypeError):
        yearly = {}

    if not yearly:
        return "REJECTED", "Phase 3: No yearly data"

    included_years = {y: d for y, d in yearly.items() if int(y) not in exclude_years}
    if not included_years:
        return "REJECTED", "Phase 3: No yearly data after exclusions"

    positive_count = sum(1 for d in included_years.values() if d.get("avg_r", 0) > 0)
    total_included = len(included_years)
    pct_positive = positive_count / total_included

    if pct_positive < min_years_positive_pct:
        neg_years = [y for y, d in included_years.items() if d.get("avg_r", 0) <= 0]
        return "REJECTED", (
            f"Phase 3: {positive_count}/{total_included} years positive "
            f"({pct_positive:.0%} < {min_years_positive_pct:.0%}), "
            f"negative: {', '.join(sorted(neg_years))}"
        )

    # Phase 4: Stress test
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
        return "REJECTED", f"Phase 4: Stress ExpR={stress_exp:.4f} <= 0 (base={exp_r}, delta_r={extra_cost_per_trade_r:.4f}, risk_pts={strategy_risk_points:.2f})"

    # Phase 5: Sharpe ratio (optional)
    if min_sharpe is not None:
        sharpe = row.get("sharpe_ratio")
        if sharpe is None or sharpe < min_sharpe:
            return "REJECTED", f"Phase 5: Sharpe={sharpe} < {min_sharpe}"

    # Phase 6: Max drawdown (optional)
    if max_drawdown is not None:
        dd = row.get("max_drawdown_r")
        if dd is not None and dd > max_drawdown:
            return "REJECTED", f"Phase 6: MaxDD={dd}R > {max_drawdown}R"

    status = "PASSED"
    if notes:
        return status, "; ".join(notes)
    return status, "All phases passed"


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
) -> tuple[int, int]:
    """
    Validate all experimental_strategies and promote passing ones.

    Returns (passed_count, rejected_count).
    """
    if db_path is None:
        db_path = GOLD_DB_PATH

    cost_spec = get_cost_spec(instrument)

    con = duckdb.connect(str(db_path))
    try:
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

        passed = 0
        rejected = 0

        for row in rows:
            row_dict = dict(zip(col_names, row))
            strategy_id = row_dict["strategy_id"]

            status, notes = validate_strategy(
                row_dict, cost_spec,
                stress_multiplier=stress_multiplier,
                min_sample=min_sample,
                min_sharpe=min_sharpe,
                max_drawdown=max_drawdown,
                exclude_years=exclude_years,
                min_years_positive_pct=min_years_positive_pct,
            )

            if not dry_run:
                # Update experimental_strategies
                con.execute(
                    """UPDATE experimental_strategies
                       SET validation_status = ?, validation_notes = ?
                       WHERE strategy_id = ?""",
                    [status, notes, strategy_id],
                )

                if status == "PASSED":
                    # Promote to validated_setups
                    yearly = row_dict.get("yearly_results", "{}")
                    try:
                        yearly_data = json.loads(yearly) if isinstance(yearly, str) else yearly
                    except (json.JSONDecodeError, TypeError):
                        yearly_data = {}

                    years_tested = len(yearly_data)
                    all_positive = all(
                        d.get("avg_r", 0) > 0 for d in yearly_data.values()
                    )

                    con.execute(
                        """INSERT OR REPLACE INTO validated_setups
                           (strategy_id, promoted_from, instrument, orb_label,
                            orb_minutes, rr_target, confirm_bars, entry_model,
                            filter_type, filter_params,
                            sample_size, win_rate, expectancy_r,
                            years_tested, all_years_positive, stress_test_passed,
                            sharpe_ratio, max_drawdown_r, yearly_results, status)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                            yearly, "active",
                        ],
                    )

            if status == "PASSED":
                passed += 1
            else:
                rejected += 1

        if not dry_run:
            con.commit()

        print(f"Validation complete: {passed} PASSED, {rejected} REJECTED "
              f"(of {len(rows)} strategies)")
        if dry_run:
            print("  (DRY RUN — no data written)")

        return passed, rejected

    finally:
        con.close()


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
    args = parser.parse_args()

    exclude = set(args.exclude_years) if args.exclude_years else None

    run_validation(
        instrument=args.instrument,
        min_sample=args.min_sample,
        stress_multiplier=args.stress_multiplier,
        min_sharpe=args.min_sharpe,
        max_drawdown=args.max_drawdown,
        exclude_years=exclude,
        min_years_positive_pct=args.min_years_positive_pct,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
