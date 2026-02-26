"""
Grid search over nested ORB strategy variants.

Same logic as strategy_discovery.py but reads from nested_outcomes
and writes to nested_strategies. Strategy IDs have NESTED_ prefix.

Usage:
    python -m trading_app.nested.discovery --instrument MGC
    python -m trading_app.nested.discovery --instrument MGC --dry-run
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
    _load_daily_features,
    _build_filter_day_sets,
    _compute_relative_volumes,
)
from trading_app.nested.schema import init_nested_schema

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

# Nested entry resolution (always 5m)
ENTRY_RESOLUTION = 5

def make_nested_strategy_id(
    instrument: str,
    orb_label: str,
    orb_minutes: int,
    entry_model: str,
    rr_target: float,
    confirm_bars: int,
    filter_type: str,
) -> str:
    """Generate deterministic nested strategy ID.

    Format: NESTED_{instrument}_{orb}_{minutes}m_{em}_RR{rr}_CB{cb}_{filter}
    Example: NESTED_MGC_CME_REOPEN_15m_E1_RR2.5_CB2_ORB_G4
    """
    return (
        f"NESTED_{instrument}_{orb_label}_{orb_minutes}m_"
        f"{entry_model}_RR{rr_target}_CB{confirm_bars}_{filter_type}"
    )

def _load_nested_outcomes_bulk(con, instrument, orb_minutes, entry_resolution,
                               orb_labels, entry_models):
    """Load all non-NULL nested outcomes in bulk.

    Returns dict keyed by (orb_label, entry_model, rr_target, confirm_bars)
    with value = list of outcome dicts.
    """
    grouped = {}
    for orb_label in orb_labels:
        for em in entry_models:
            rows = con.execute(
                """SELECT trading_day, rr_target, confirm_bars,
                          outcome, pnl_r, mae_r, mfe_r,
                          entry_price, stop_price
                   FROM nested_outcomes
                   WHERE symbol = ? AND orb_minutes = ?
                     AND entry_resolution = ?
                     AND orb_label = ? AND entry_model = ?
                     AND outcome IS NOT NULL
                   ORDER BY trading_day""",
                [instrument, orb_minutes, entry_resolution, orb_label, em],
            ).fetchall()

            for r in rows:
                key = (orb_label, em, r[1], r[2])
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append({
                    "trading_day": r[0],
                    "outcome": r[3],
                    "pnl_r": r[4],
                    "mae_r": r[5],
                    "mfe_r": r[6],
                    "entry_price": r[7],
                    "stop_price": r[8],
                })

    return grouped

def run_nested_discovery(
    db_path: Path | None = None,
    instrument: str = "MGC",
    start_date: date | None = None,
    end_date: date | None = None,
    orb_minutes_list: list[int] | None = None,
    dry_run: bool = False,
) -> int:
    """Grid search over nested strategy variants.

    Bulk-loads data upfront, then iterates the full grid in Python.
    Returns count of strategies written.
    """
    if db_path is None:
        db_path = GOLD_DB_PATH
    if orb_minutes_list is None:
        orb_minutes_list = [15, 30]

    con = duckdb.connect(str(db_path))
    try:
        from pipeline.db_config import configure_connection
        configure_connection(con, writing=True)

        if not dry_run:
            init_nested_schema(con=con)

        total_strategies = 0

        for orb_minutes in orb_minutes_list:
            print(f"\n--- Discovery for orb_minutes={orb_minutes} ---")

            # Bulk load phase
            print("Loading daily features...")
            features = _load_daily_features(con, instrument, orb_minutes, start_date, end_date)
            print(f"  {len(features)} daily_features rows loaded")

            if not features:
                print(f"  No daily_features for orb_minutes={orb_minutes}. Skipping.")
                continue

            # Build union of all session-specific filters for bulk pre-computation
            all_grid_filters: dict = {}
            for s in ORB_LABELS:
                all_grid_filters.update(get_filters_for_grid(instrument, s))

            print("Computing relative volumes for volume filters...")
            _compute_relative_volumes(con, features, instrument, ORB_LABELS, all_grid_filters)

            print("Building filter/ORB day sets...")
            filter_days = _build_filter_day_sets(features, ORB_LABELS, all_grid_filters)

            print("Loading nested outcomes (bulk)...")
            outcomes_by_key = _load_nested_outcomes_bulk(
                con, instrument, orb_minutes, ENTRY_RESOLUTION, ORB_LABELS, ENTRY_MODELS,
            )
            print(f"  {sum(len(v) for v in outcomes_by_key.values())} outcome rows loaded")

            # Grid iteration (session-aware: each session gets only its justified filters)
            total_combos = 0
            for s in ORB_LABELS:
                nf = len(get_filters_for_grid(instrument, s))
                total_combos += nf * len(RR_TARGETS) * len(CONFIRM_BARS_OPTIONS)  # E1 (all CBs)
                total_combos += nf * len(RR_TARGETS) * 2                        # E2+E3 (CB1 only)
            combo_idx = 0
            insert_batch = []

            for orb_label in ORB_LABELS:
                session_filters = get_filters_for_grid(instrument, orb_label)
                for filter_key, strategy_filter in session_filters.items():
                    matching_day_set = filter_days[(filter_key, orb_label)]

                    for em in ENTRY_MODELS:
                        for rr_target in RR_TARGETS:
                            for cb in CONFIRM_BARS_OPTIONS:
                                if em in ("E2", "E3") and cb > 1:
                                    continue
                                combo_idx += 1

                                if not matching_day_set:
                                    continue

                                all_outcomes = outcomes_by_key.get(
                                    (orb_label, em, rr_target, cb), [],
                                )
                                outcomes = [
                                    o for o in all_outcomes
                                    if o["trading_day"] in matching_day_set
                                ]

                                if not outcomes:
                                    continue

                                metrics = compute_metrics(outcomes)
                                strategy_id = make_nested_strategy_id(
                                    instrument, orb_label, orb_minutes,
                                    em, rr_target, cb, filter_key,
                                )

                                if not dry_run:
                                    insert_batch.append([
                                        strategy_id, instrument, orb_label,
                                        orb_minutes, ENTRY_RESOLUTION,
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
                                            """INSERT OR REPLACE INTO nested_strategies
                                               (strategy_id, instrument, orb_label,
                                                orb_minutes, entry_resolution,
                                                rr_target, confirm_bars, entry_model,
                                                filter_type, filter_params,
                                                sample_size, win_rate, avg_win_r, avg_loss_r,
                                                expectancy_r, sharpe_ratio, max_drawdown_r,
                                                median_risk_points, avg_risk_points,
                                                yearly_results)
                                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                                       ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                    """INSERT OR REPLACE INTO nested_strategies
                       (strategy_id, instrument, orb_label,
                        orb_minutes, entry_resolution,
                        rr_target, confirm_bars, entry_model,
                        filter_type, filter_params,
                        sample_size, win_rate, avg_win_r, avg_loss_r,
                        expectancy_r, sharpe_ratio, max_drawdown_r,
                        median_risk_points, avg_risk_points,
                        yearly_results)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                               ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    insert_batch,
                )

            if not dry_run:
                con.commit()

        print(f"\nDone: {total_strategies} nested strategies")
        if dry_run:
            print("  (DRY RUN -- no data written)")

        return total_strategies

    finally:
        con.close()

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Grid search over nested ORB strategy variants"
    )
    parser.add_argument("--instrument", default="MGC", help="Instrument symbol")
    parser.add_argument("--start", type=date.fromisoformat, help="Start date")
    parser.add_argument("--end", type=date.fromisoformat, help="End date")
    parser.add_argument(
        "--orb-minutes", type=int, nargs="+", default=[15, 30],
        help="ORB duration(s) (default: 15 30)",
    )
    parser.add_argument("--dry-run", action="store_true", help="No DB writes")
    args = parser.parse_args()

    run_nested_discovery(
        instrument=args.instrument,
        start_date=args.start,
        end_date=args.end,
        orb_minutes_list=args.orb_minutes,
        dry_run=args.dry_run,
    )

if __name__ == "__main__":
    main()
