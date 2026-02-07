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
from datetime import date, datetime, timezone

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
                      max_drawdown: float | None = None) -> tuple[str, str]:
    """
    Run 6-phase validation on a single strategy row.

    Args:
        row: Dict from experimental_strategies query
        cost_spec: CostSpec for the instrument
        stress_multiplier: Cost increase multiplier for stress test
        min_sample: Minimum sample size (reject below this)
        min_sharpe: Optional minimum Sharpe ratio (Phase 5)
        max_drawdown: Optional max drawdown in R (Phase 6)

    Returns:
        (status, notes): "PASSED" or "REJECTED", with explanation.
    """
    notes = []

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

    for year, data in yearly.items():
        avg_r = data.get("avg_r", 0)
        if avg_r <= 0:
            return "REJECTED", f"Phase 3: Year {year} avg_r={avg_r} <= 0"

    # Phase 4: Stress test
    # Recompute expectancy with inflated costs
    # stress_test increases friction by multiplier. The effect on R is:
    # friction_increase_in_points = (stress_friction - base_friction) / point_value
    # This reduces each win's R and increases each loss's R (makes it more negative)
    stressed = stress_test_costs(cost_spec, stress_multiplier)
    friction_delta_points = stressed.friction_in_points - cost_spec.friction_in_points

    win_rate = row.get("win_rate", 0)
    avg_win_r = row.get("avg_win_r", 0)
    avg_loss_r = row.get("avg_loss_r", 0)

    # Under stress: wins shrink, losses grow (in absolute terms)
    # Approximate: each trade's R shifts by -friction_delta_points/risk_points
    # But we don't have risk_points per trade. Conservative: assume risk=ORB size.
    # Simpler approach: scale the friction impact proportionally.
    # friction_delta_r ≈ friction_delta_points * point_value / (avg_risk_dollars)
    # Since we work in R-multiples already, the additional friction per trade in R is:
    # delta_r = total_friction_increase / risk_in_dollars
    # Without knowing exact risk, use the friction_in_points as a fraction of 1R.
    # This is approximate but conservative.
    # Better: just check if expectancy can absorb the extra cost.
    # extra_cost_per_trade_R = friction_delta_points / (avg risk in points)
    # Since 1R = risk in points, and friction_delta is in points:
    # extra_cost_R ≈ friction_delta_points (when risk = 1 point... not realistic)
    #
    # Most practical: recompute E with adjusted win/loss R values.
    # Each trade pays friction_delta more. In R terms, if we assume the average
    # risk for MGC ORBs is ~3-5 points:
    # extra_cost_r = friction_delta_points / avg_risk_points
    # We don't track avg_risk_points, so use a conservative estimate.
    # For now: reduce ExpR by (friction_delta * point_value) / (point_value * 1)
    # = friction_delta (in points) ... which would be too aggressive.
    #
    # Simplest correct approach: just check if ExpR > stress margin.
    # stress_margin = friction_delta_points * (avg trades per unit risk)
    # ACTUALLY: the cleanest way is to ask "does the strategy survive if
    # every trade loses an extra fraction of R?"
    # extra_friction_per_trade = friction_delta (dollars) / risk_dollars
    # We know friction_delta dollars = (stress - base) total friction.
    # Without risk_dollars, approximate using ExpR > some threshold.
    # The CANONICAL approach: "ExpR > 0 at +50% costs"
    # Conservative approximation: reduce ExpR by the per-trade friction increase
    # expressed in R. For MGC: base friction = $8.40, stress = $12.60.
    # Extra = $4.20. Average ORB risk ~ 3-5 points = $30-50. So extra ~0.08-0.14R.
    # Use: stress_exp = exp_r - (stress_friction - base_friction) / point_value
    # This assumes risk = 1 point (maximally conservative). In practice risk > 1 pt.
    stress_friction_delta = stressed.total_friction - cost_spec.total_friction
    # Conservative: assume minimum realistic risk of 2 points for MGC
    min_risk_dollars = 2.0 * cost_spec.point_value
    extra_cost_per_trade_r = stress_friction_delta / min_risk_dollars
    stress_exp = exp_r - extra_cost_per_trade_r

    if stress_exp <= 0:
        return "REJECTED", f"Phase 4: Stress ExpR={stress_exp:.4f} <= 0 (base={exp_r}, delta_r={extra_cost_per_trade_r:.4f})"

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
                            orb_minutes, rr_target, confirm_bars, filter_type,
                            filter_params, sample_size, win_rate, expectancy_r,
                            years_tested, all_years_positive, stress_test_passed,
                            sharpe_ratio, max_drawdown_r, yearly_results, status)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        [
                            strategy_id, strategy_id,
                            row_dict["instrument"], row_dict["orb_label"],
                            row_dict["orb_minutes"], row_dict["rr_target"],
                            row_dict["confirm_bars"], row_dict.get("filter_type", ""),
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
    parser.add_argument("--dry-run", action="store_true", help="No DB writes")
    args = parser.parse_args()

    run_validation(
        instrument=args.instrument,
        min_sample=args.min_sample,
        stress_multiplier=args.stress_multiplier,
        min_sharpe=args.min_sharpe,
        max_drawdown=args.max_drawdown,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
