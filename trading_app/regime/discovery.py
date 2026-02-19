"""
Grid search over strategy variants for a date-bounded regime.

Same logic as strategy_discovery.py but reads from orb_outcomes with date
filter and writes to regime_strategies with a run_label tag.

Usage:
    python -m trading_app.regime.discovery --instrument MGC --start 2025-01-01 --end 2025-12-31 --run-label 2025_only
    python -m trading_app.regime.discovery --instrument MGC --start 2025-01-01 --end 2025-12-31 --run-label 2025_only --dry-run
"""

import sys
from pathlib import Path
from datetime import date

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

import duckdb

from pipeline.paths import GOLD_DB_PATH
from pipeline.init_db import ORB_LABELS
from trading_app.config import get_filters_for_grid, ENTRY_MODELS
from trading_app.outcome_builder import RR_TARGETS, CONFIRM_BARS_OPTIONS
from trading_app.strategy_discovery import (
    compute_metrics,
    make_strategy_id,
    _load_daily_features,
    _build_filter_day_sets,
    _compute_relative_volumes,
    _load_outcomes_bulk,
)
from trading_app.regime.schema import init_regime_schema

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

def run_regime_discovery(
    db_path: Path | None = None,
    instrument: str = "MGC",
    start_date: date | None = None,
    end_date: date | None = None,
    run_label: str = "default",
    orb_minutes: int = 5,
    dry_run: bool = False,
) -> int:
    """Grid search over strategy variants for a date-bounded regime.

    Uses same strategy_id format as production (enables direct JOIN for comparison).
    Results tagged with run_label for coexistence of multiple regime runs.

    Returns count of strategies written.
    """
    if db_path is None:
        db_path = GOLD_DB_PATH

    con = duckdb.connect(str(db_path))
    try:
        if not dry_run:
            init_regime_schema(con=con)
            # Idempotent: clear previous run with same label
            con.execute(
                "DELETE FROM regime_validated WHERE run_label = ?", [run_label]
            )
            con.execute(
                "DELETE FROM regime_strategies WHERE run_label = ?", [run_label]
            )
            con.commit()

        # ---- Bulk load phase ----
        print(f"Regime discovery: run_label={run_label}, "
              f"dates={start_date} to {end_date}, orb_minutes={orb_minutes}")

        print("Loading daily features...")
        features = _load_daily_features(con, instrument, orb_minutes, start_date, end_date)
        print(f"  {len(features)} daily_features rows loaded")

        if not features:
            print("  No daily_features found. Exiting.")
            return 0

        # Build session-aware filter sets (session 1100/2300/0030 get base only)
        session_filters = {orb: get_filters_for_grid(instrument, orb) for orb in ORB_LABELS}
        union_filters: dict = {}
        for sf in session_filters.values():
            union_filters.update(sf)

        print("Computing relative volumes for volume filters...")
        _compute_relative_volumes(con, features, instrument, ORB_LABELS, union_filters)

        print("Building filter/ORB day sets...")
        filter_days = _build_filter_day_sets(features, ORB_LABELS, union_filters)

        # Build the set of days covered by daily_features (for outcome filtering)
        feature_day_set = {row["trading_day"] for row in features}

        print("Loading outcomes (bulk)...")
        outcomes_by_key = _load_outcomes_bulk(
            con, instrument, orb_minutes, ORB_LABELS, ENTRY_MODELS,
        )
        print(f"  {sum(len(v) for v in outcomes_by_key.values())} outcome rows loaded")

        # ---- Grid iteration (session-aware: only session-appropriate filters per ORB) ----
        total_strategies = 0
        total_combos = sum(
            len(sf) * len(RR_TARGETS) * (len(CONFIRM_BARS_OPTIONS) * 2 + 1)
            for sf in session_filters.values()
        )
        combo_idx = 0
        insert_batch = []

        for orb_label in ORB_LABELS:
            for filter_key, strategy_filter in session_filters[orb_label].items():
                matching_day_set = filter_days[(filter_key, orb_label)]

                for em in ENTRY_MODELS:
                    for rr_target in RR_TARGETS:
                        for cb in CONFIRM_BARS_OPTIONS:
                            if em == "E3" and cb > 1:
                                continue
                            combo_idx += 1

                            if not matching_day_set:
                                continue

                            # Filter outcomes by matching days AND date range
                            all_outcomes = outcomes_by_key.get(
                                (orb_label, em, rr_target, cb), [],
                            )
                            outcomes = [
                                o for o in all_outcomes
                                if o["trading_day"] in matching_day_set
                                and o["trading_day"] in feature_day_set
                            ]

                            if not outcomes:
                                continue

                            metrics = compute_metrics(outcomes)
                            strategy_id = make_strategy_id(
                                instrument, orb_label, em, rr_target, cb, filter_key,
                            )

                            if not dry_run:
                                insert_batch.append([
                                    run_label, strategy_id,
                                    start_date, end_date,
                                    instrument, orb_label, orb_minutes,
                                    rr_target, cb, em, filter_key,
                                    strategy_filter.to_json(),
                                    metrics["sample_size"], metrics["win_rate"],
                                    metrics["avg_win_r"], metrics["avg_loss_r"],
                                    metrics["expectancy_r"], metrics["sharpe_ratio"],
                                    metrics["max_drawdown_r"],
                                    metrics["median_risk_points"],
                                    metrics["avg_risk_points"],
                                    metrics["yearly_results"],
                                ])

                                if len(insert_batch) >= 500:
                                    con.executemany(
                                        """INSERT INTO regime_strategies
                                           (run_label, strategy_id,
                                            start_date, end_date,
                                            instrument, orb_label, orb_minutes,
                                            rr_target, confirm_bars, entry_model,
                                            filter_type, filter_params,
                                            sample_size, win_rate, avg_win_r, avg_loss_r,
                                            expectancy_r, sharpe_ratio, max_drawdown_r,
                                            median_risk_points, avg_risk_points,
                                            yearly_results)
                                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                                   ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                        insert_batch,
                                    )
                                    insert_batch = []

                            total_strategies += 1

                if combo_idx % 500 == 0:
                    print(f"  Progress: {combo_idx}/{total_combos} combos, "
                          f"{total_strategies} strategies")

        # Flush remaining batch
        if insert_batch and not dry_run:
            con.executemany(
                """INSERT INTO regime_strategies
                   (run_label, strategy_id,
                    start_date, end_date,
                    instrument, orb_label, orb_minutes,
                    rr_target, confirm_bars, entry_model,
                    filter_type, filter_params,
                    sample_size, win_rate, avg_win_r, avg_loss_r,
                    expectancy_r, sharpe_ratio, max_drawdown_r,
                    median_risk_points, avg_risk_points,
                    yearly_results)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                insert_batch,
            )

        if not dry_run:
            con.commit()

        print(f"\nDone: {total_strategies} regime strategies "
              f"(run_label={run_label}) from {total_combos} combos")
        if dry_run:
            print("  (DRY RUN -- no data written)")

        return total_strategies

    finally:
        con.close()

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Grid search over strategy variants for a date-bounded regime"
    )
    parser.add_argument("--instrument", default="MGC", help="Instrument symbol")
    parser.add_argument("--start", type=date.fromisoformat, help="Start date",
                        required=True)
    parser.add_argument("--end", type=date.fromisoformat, help="End date",
                        required=True)
    parser.add_argument("--run-label", required=True,
                        help="Label for this regime run (e.g. 2025_only)")
    parser.add_argument("--orb-minutes", type=int, default=5, help="ORB duration")
    parser.add_argument("--dry-run", action="store_true", help="No DB writes")
    args = parser.parse_args()

    run_regime_discovery(
        instrument=args.instrument,
        start_date=args.start,
        end_date=args.end,
        run_label=args.run_label,
        orb_minutes=args.orb_minutes,
        dry_run=args.dry_run,
    )

if __name__ == "__main__":
    main()
