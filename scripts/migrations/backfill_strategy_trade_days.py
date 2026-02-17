"""
Backfill strategy_trade_days from validated_setups using canonical filter logic.

Reuses EXACT functions from strategy_discovery to ensure zero divergence
from how discovery computes trade days during grid search.

Usage:
    python scripts/backfill_strategy_trade_days.py --instrument MGC --db-path C:/db/gold.db
    python scripts/backfill_strategy_trade_days.py --all --db-path C:/db/gold.db
"""

import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

import duckdb

from trading_app.config import ALL_FILTERS, VolumeFilter
from trading_app.strategy_discovery import (
    _build_filter_day_sets,
    _compute_relative_volumes,
    _load_daily_features,
)
from trading_app.db_manager import init_trading_app_schema

# Force unbuffered stdout (Windows cp1252 buffering issue)
sys.stdout.reconfigure(line_buffering=True)

def backfill_trade_days(db_path: str, instrument: str) -> int:
    """
    Backfill strategy_trade_days for one instrument.

    Returns total rows inserted.
    """
    con = duckdb.connect(str(db_path))
    try:
        # Ensure table exists
        init_trading_app_schema(db_path=Path(db_path))

        # 1. Load validated strategies for this instrument
        rows = con.execute("""
            SELECT strategy_id, instrument, orb_label, orb_minutes,
                   rr_target, confirm_bars, entry_model, filter_type
            FROM validated_setups
            WHERE instrument = ?
            ORDER BY strategy_id
        """, [instrument]).fetchall()
        cols = [d[0] for d in con.description]
        strategies = [dict(zip(cols, r)) for r in rows]

        print(f"Backfilling {len(strategies)} strategies for {instrument}")

        if not strategies:
            print(f"No validated strategies for {instrument}")
            return 0

        # 2. Determine unique orb_labels, filter_types, entry_models
        orb_labels = sorted(set(s["orb_label"] for s in strategies))
        filter_types = set(s["filter_type"] for s in strategies)
        entry_models = sorted(set(s["entry_model"] for s in strategies))
        needed_filters = {k: v for k, v in ALL_FILTERS.items() if k in filter_types}

        print(f"  ORB labels: {orb_labels}")
        print(f"  Filter types: {sorted(filter_types)}")
        print(f"  Entry models: {entry_models}")

        # 3. Load daily_features (canonical: orb_minutes=5, full date range)
        features = _load_daily_features(con, instrument, 5, None, None)
        print(f"  {len(features)} daily_features rows loaded")

        # 4. Compute relative volumes if any strategy uses VolumeFilter
        has_vol_filter = any(isinstance(f, VolumeFilter) for f in needed_filters.values())
        if has_vol_filter:
            print("  Computing relative volumes (VolumeFilter detected)...")
            _compute_relative_volumes(con, features, instrument, orb_labels, needed_filters)

        # 5. Build filter day sets (CANONICAL -- same as discovery line 413)
        filter_days = _build_filter_day_sets(features, orb_labels, needed_filters)

        # 6. Load outcome trading_days grouped by (orb, em, rr, cb)
        #    Uses outcome IS NOT NULL -- same as discovery _load_outcomes_bulk line 350
        outcome_days = defaultdict(list)
        for orb_label in orb_labels:
            for em in entry_models:
                orows = con.execute("""
                    SELECT rr_target, confirm_bars, trading_day
                    FROM orb_outcomes
                    WHERE symbol = ? AND orb_minutes = 5
                      AND orb_label = ? AND entry_model = ?
                      AND outcome IS NOT NULL
                    ORDER BY rr_target, confirm_bars, trading_day
                """, [instrument, orb_label, em]).fetchall()
                for rr, cb, td in orows:
                    outcome_days[(orb_label, em, rr, cb)].append(td)

        total_outcome_rows = sum(len(v) for v in outcome_days.values())
        print(f"  {total_outcome_rows} outcome rows loaded")

        # 7. Delete existing rows for this instrument's strategies (idempotent)
        strategy_ids = [s["strategy_id"] for s in strategies]
        chunk_size = 500
        deleted = 0
        for i in range(0, len(strategy_ids), chunk_size):
            chunk = strategy_ids[i : i + chunk_size]
            placeholders = ",".join(["?"] * len(chunk))
            result = con.execute(
                f"DELETE FROM strategy_trade_days WHERE strategy_id IN ({placeholders})",
                chunk,
            )
            deleted += result.fetchone()[0] if result.description else 0
        if deleted:
            print(f"  Cleared {deleted} existing rows")

        # 8. For each strategy, compute trade days and batch insert
        insert_batch = []
        total_trade_days = 0

        for idx, strat in enumerate(strategies):
            sid = strat["strategy_id"]
            orb = strat["orb_label"]
            em = strat["entry_model"]
            rr = strat["rr_target"]
            cb = strat["confirm_bars"]
            filt = strat["filter_type"]

            # CANONICAL: same as discovery line 429
            matching_day_set = filter_days.get((filt, orb), set())

            # CANONICAL: same as discovery _load_outcomes_bulk grouping
            all_days = outcome_days.get((orb, em, rr, cb), [])

            # CANONICAL: same as discovery line 443
            trade_days = [d for d in all_days if d in matching_day_set]

            for td in trade_days:
                insert_batch.append((sid, td))
            total_trade_days += len(trade_days)

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(strategies)} strategies...")

            # Flush batch every 10K rows
            if len(insert_batch) >= 10_000:
                con.executemany(
                    "INSERT INTO strategy_trade_days (strategy_id, trading_day) VALUES (?, ?)",
                    insert_batch,
                )
                insert_batch = []

        # Final flush
        if insert_batch:
            con.executemany(
                "INSERT INTO strategy_trade_days (strategy_id, trading_day) VALUES (?, ?)",
                insert_batch,
            )

        con.commit()

        # 9. Verify
        count = con.execute("""
            SELECT COUNT(*) FROM strategy_trade_days
            WHERE strategy_id IN (
                SELECT strategy_id FROM validated_setups WHERE instrument = ?
            )
        """, [instrument]).fetchone()[0]

        unique_days = con.execute("""
            SELECT COUNT(DISTINCT trading_day) FROM strategy_trade_days
            WHERE strategy_id IN (
                SELECT strategy_id FROM validated_setups WHERE instrument = ?
            )
        """, [instrument]).fetchone()[0]

        print(
            f"Done: {count:,} trade-day rows for {len(strategies)} "
            f"{instrument} strategies ({unique_days} unique days)"
        )
        return count

    finally:
        con.close()

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Backfill strategy_trade_days from validated_setups"
    )
    parser.add_argument(
        "--instrument",
        help="Instrument symbol (MGC, MNQ, MES, MCL)",
    )
    parser.add_argument(
        "--db-path",
        default="C:/db/gold.db",
        help="Database path (default: C:/db/gold.db)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run for all instruments",
    )
    args = parser.parse_args()

    if not args.all and not args.instrument:
        parser.error("Either --instrument or --all is required")

    if args.all:
        total = 0
        for inst in ["MGC", "MNQ", "MES", "MCL"]:
            total += backfill_trade_days(args.db_path, inst)
            print()
        print(f"Grand total: {total:,} trade-day rows across all instruments")
    else:
        backfill_trade_days(args.db_path, args.instrument)

if __name__ == "__main__":
    main()
