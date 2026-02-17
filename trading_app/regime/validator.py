"""
Validate regime strategies and promote to regime_validated.

Same validation logic as strategy_validator.py but reads/writes regime tables.

Usage:
    python -m trading_app.regime.validator --instrument MGC --run-label 2025_only
    python -m trading_app.regime.validator --instrument MGC --run-label 2025_only --min-sample 20 --dry-run
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

import duckdb

from pipeline.paths import GOLD_DB_PATH
from pipeline.cost_model import get_cost_spec
from trading_app.strategy_validator import validate_strategy
from trading_app.regime.schema import init_regime_schema

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

def run_regime_validation(
    db_path: Path | None = None,
    instrument: str = "MGC",
    run_label: str = "default",
    min_sample: int = 30,
    stress_multiplier: float = 1.5,
    min_sharpe: float | None = None,
    max_drawdown: float | None = None,
    exclude_years: set[int] | None = None,
    min_years_positive_pct: float = 1.0,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Validate all regime_strategies for a run_label and promote passing ones.

    Returns (passed_count, rejected_count).
    """
    if db_path is None:
        db_path = GOLD_DB_PATH

    cost_spec = get_cost_spec(instrument)

    con = duckdb.connect(str(db_path))
    try:
        if not dry_run:
            init_regime_schema(con=con)

        # Fetch unvalidated strategies for this run
        rows = con.execute(
            """SELECT * FROM regime_strategies
               WHERE run_label = ? AND instrument = ?
               AND (validation_status IS NULL OR validation_status = '')
               ORDER BY strategy_id""",
            [run_label, instrument],
        ).fetchall()
        col_names = [desc[0] for desc in con.description]

        passed = 0
        rejected = 0

        for row in rows:
            row_dict = dict(zip(col_names, row))
            strategy_id = row_dict["strategy_id"]

            status, notes, _ = validate_strategy(
                row_dict, cost_spec,
                stress_multiplier=stress_multiplier,
                min_sample=min_sample,
                min_sharpe=min_sharpe,
                max_drawdown=max_drawdown,
                exclude_years=exclude_years,
                min_years_positive_pct=min_years_positive_pct,
            )

            if not dry_run:
                # Update regime_strategies
                con.execute(
                    """UPDATE regime_strategies
                       SET validation_status = ?, validation_notes = ?
                       WHERE run_label = ? AND strategy_id = ?""",
                    [status, notes, run_label, strategy_id],
                )

                if status == "PASSED":
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
                        """INSERT OR REPLACE INTO regime_validated
                           (run_label, strategy_id,
                            start_date, end_date,
                            instrument, orb_label, orb_minutes,
                            rr_target, confirm_bars, entry_model,
                            filter_type, filter_params,
                            sample_size, win_rate, expectancy_r,
                            years_tested, all_years_positive, stress_test_passed,
                            sharpe_ratio, max_drawdown_r, yearly_results, status)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        [
                            run_label, strategy_id,
                            row_dict.get("start_date"), row_dict.get("end_date"),
                            row_dict["instrument"], row_dict["orb_label"],
                            row_dict["orb_minutes"],
                            row_dict["rr_target"], row_dict["confirm_bars"],
                            row_dict.get("entry_model", "E1"),
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

        print(f"Regime validation complete (run_label={run_label}): "
              f"{passed} PASSED, {rejected} REJECTED (of {len(rows)} strategies)")
        if dry_run:
            print("  (DRY RUN -- no data written)")

        return passed, rejected

    finally:
        con.close()

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate regime strategies and promote to regime_validated"
    )
    parser.add_argument("--instrument", default="MGC", help="Instrument symbol")
    parser.add_argument("--run-label", required=True, help="Regime run label")
    parser.add_argument("--min-sample", type=int, default=30, help="Min sample size")
    parser.add_argument("--stress-multiplier", type=float, default=1.5)
    parser.add_argument("--min-sharpe", type=float, default=None)
    parser.add_argument("--max-drawdown", type=float, default=None)
    parser.add_argument("--exclude-years", type=int, nargs="*", default=None)
    parser.add_argument("--min-years-positive-pct", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    exclude = set(args.exclude_years) if args.exclude_years else None

    run_regime_validation(
        instrument=args.instrument,
        run_label=args.run_label,
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
