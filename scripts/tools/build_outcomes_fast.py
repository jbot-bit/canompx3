#!/usr/bin/env python3
"""
Fast parallel outcome builder for MNQ (or any instrument).

Same logic as trading_app/outcome_builder.py but parallelized across
trading days using multiprocessing. Each day is fully independent.

Usage:
    python scripts/build_outcomes_fast.py --instrument MNQ
    python scripts/build_outcomes_fast.py --instrument MNQ --workers 16
    python scripts/build_outcomes_fast.py --instrument MNQ --dry-run
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import date, datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

sys.stdout.reconfigure(line_buffering=True)

import duckdb
import numpy as np
import pandas as pd

from pipeline.paths import GOLD_DB_PATH
from pipeline.cost_model import get_cost_spec
from pipeline.init_db import ORB_LABELS
from pipeline.build_daily_features import compute_trading_day_utc_range
from trading_app.config import ENTRY_MODELS
from trading_app.db_manager import init_trading_app_schema
from trading_app.outcome_builder import (
    compute_single_outcome,
    RR_TARGETS,
    CONFIRM_BARS_OPTIONS,
)

DB_PATH = Path(r"C:\db\gold.db")

def process_single_day(args_tuple):
    """Process all outcomes for a single trading day. Runs in worker process."""
    (trading_day, symbol, orb_minutes, row_dict, instrument) = args_tuple

    cost_spec = get_cost_spec(instrument)

    # Each worker opens its own read-only connection
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        td_start, td_end = compute_trading_day_utc_range(trading_day)
        bars_df = con.execute("""
            SELECT ts_utc, open, high, low, close, volume
            FROM bars_1m
            WHERE symbol = ?
            AND ts_utc >= ?::TIMESTAMPTZ
            AND ts_utc < ?::TIMESTAMPTZ
            ORDER BY ts_utc ASC
        """, [symbol, td_start.isoformat(), td_end.isoformat()]).fetchdf()

        if bars_df.empty:
            return []

        bars_df["ts_utc"] = pd.to_datetime(bars_df["ts_utc"], utc=True)

        day_batch = []

        for orb_label in ORB_LABELS:
            break_dir = row_dict.get(f"orb_{orb_label}_break_dir")
            break_ts = row_dict.get(f"orb_{orb_label}_break_ts")
            orb_high = row_dict.get(f"orb_{orb_label}_high")
            orb_low = row_dict.get(f"orb_{orb_label}_low")

            if break_dir is None or break_ts is None:
                continue
            if orb_high is None or orb_low is None:
                continue

            for rr_target in RR_TARGETS:
                for cb in CONFIRM_BARS_OPTIONS:
                    for em in ENTRY_MODELS:
                        if em == "E3" and cb > 1:
                            continue

                        outcome = compute_single_outcome(
                            bars_df=bars_df,
                            break_ts=break_ts,
                            orb_high=orb_high,
                            orb_low=orb_low,
                            break_dir=break_dir,
                            rr_target=rr_target,
                            confirm_bars=cb,
                            trading_day_end=td_end,
                            cost_spec=cost_spec,
                            entry_model=em,
                        )

                        day_batch.append((
                            symbol, trading_day, orb_label,
                            orb_minutes, rr_target, cb, em,
                            outcome["entry_ts"],
                            outcome["entry_price"],
                            outcome["stop_price"],
                            outcome["target_price"],
                            outcome["outcome"],
                            outcome["exit_ts"],
                            outcome["exit_price"],
                            outcome["pnl_r"],
                            outcome["mae_r"],
                            outcome["mfe_r"],
                        ))

        return day_batch
    finally:
        con.close()

def main():
    parser = argparse.ArgumentParser(description="Fast parallel outcome builder")
    parser.add_argument("--instrument", type=str, default="MNQ")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--orb-minutes", type=int, default=5)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    instrument = args.instrument.upper()
    cost_spec = get_cost_spec(instrument)
    orb_minutes = args.orb_minutes

    print("=" * 60, flush=True)
    print(f"FAST PARALLEL OUTCOME BUILDER ({instrument})", flush=True)
    print(f"  DB: {DB_PATH}", flush=True)
    print(f"  Workers: {args.workers}", flush=True)
    print(f"  ORB minutes: {orb_minutes}", flush=True)
    print(f"  Cost model: {instrument} (friction={cost_spec.total_friction})", flush=True)
    print("=" * 60, flush=True)

    t0 = time.time()

    # Ensure schema exists
    if not args.dry_run:
        init_trading_app_schema(db_path=DB_PATH)

    # Fetch all daily_features rows
    con = duckdb.connect(str(DB_PATH), read_only=True)
    date_clauses = []
    params = [instrument, orb_minutes]
    if args.start:
        date_clauses.append("AND trading_day >= ?")
        params.append(date.fromisoformat(args.start))
    if args.end:
        date_clauses.append("AND trading_day <= ?")
        params.append(date.fromisoformat(args.end))
    date_filter = " ".join(date_clauses)

    query = f"""
        SELECT trading_day, symbol, orb_minutes,
               {', '.join(
                   f'orb_{lbl}_high, orb_{lbl}_low, orb_{lbl}_break_dir, orb_{lbl}_break_ts'
                   for lbl in ORB_LABELS
               )}
        FROM daily_features
        WHERE symbol = ? AND orb_minutes = ?
        {date_filter}
        ORDER BY trading_day
    """
    rows = con.execute(query, params).fetchall()
    col_names = [desc[0] for desc in con.description]
    con.close()

    total_days = len(rows)
    print(f"Trading days to process: {total_days}", flush=True)

    if args.dry_run:
        combos_per_day = len(ORB_LABELS) * len(RR_TARGETS) * (
            len(CONFIRM_BARS_OPTIONS) * 1 + 1 * 1  # E1*CBs + E3*CB1
        )
        print(f"Max outcomes: ~{total_days * combos_per_day:,}", flush=True)
        print("DRY RUN -- no writes", flush=True)
        return

    # Build work items
    work_items = []
    for row in rows:
        row_dict = dict(zip(col_names, row))
        trading_day = row_dict["trading_day"]
        symbol = row_dict["symbol"]
        work_items.append((trading_day, symbol, orb_minutes, row_dict, instrument))

    # Process in parallel
    all_outcomes = []
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_day, item): item for item in work_items}

        for future in as_completed(futures):
            batch = future.result()
            all_outcomes.extend(batch)
            completed += 1

            if completed % 50 == 0 or completed == total_days:
                elapsed = time.time() - t0
                rate = completed / elapsed
                remaining = (total_days - completed) / rate if rate > 0 else 0
                print(
                    f"  {completed}/{total_days} days "
                    f"({len(all_outcomes):,} outcomes, "
                    f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)",
                    flush=True,
                )

    print(f"\nTotal outcomes computed: {len(all_outcomes):,}", flush=True)

    # Bulk write to DB
    print("Writing to database...", flush=True)
    t_write = time.time()

    con = duckdb.connect(str(DB_PATH))
    try:
        # Clear existing outcomes for this instrument in range
        del_clauses = ["symbol = ?"]
        del_params = [instrument]
        if args.start:
            del_clauses.append("trading_day >= ?")
            del_params.append(date.fromisoformat(args.start))
        if args.end:
            del_clauses.append("trading_day <= ?")
            del_params.append(date.fromisoformat(args.end))

        del_query = f"DELETE FROM orb_outcomes WHERE {' AND '.join(del_clauses)}"
        con.execute(del_query, del_params)
        print(f"  Cleared existing {instrument} outcomes", flush=True)

        # Bulk insert
        if all_outcomes:
            con.executemany("""
                INSERT INTO orb_outcomes (
                    symbol, trading_day, orb_label, orb_minutes,
                    rr_target, confirm_bars, entry_model,
                    entry_ts, entry_price, stop_price, target_price,
                    outcome, exit_ts, exit_price, pnl_r, mae_r, mfe_r
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, all_outcomes)

        print(f"  Inserted {len(all_outcomes):,} rows ({time.time()-t_write:.1f}s)", flush=True)

        # Verify
        count = con.execute(
            "SELECT COUNT(*) FROM orb_outcomes WHERE symbol = ?", [instrument]
        ).fetchone()[0]
        print(f"  Verified: {count:,} {instrument} outcomes in DB", flush=True)

    finally:
        con.close()

    total_time = time.time() - t0
    print(f"\n{'=' * 60}", flush=True)
    print(f"DONE: {len(all_outcomes):,} outcomes in {total_time:.0f}s", flush=True)
    print(f"{'=' * 60}", flush=True)

if __name__ == "__main__":
    main()
