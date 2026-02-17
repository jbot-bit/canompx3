#!/usr/bin/env python3
"""
Fast MES outcome builder -- parallel, preloaded bars.

Same logic as trading_app/outcome_builder.py but:
  1. Preloads ALL bars_1m into memory (707K rows, ~200MB)
  2. Partitions by trading day in pandas
  3. Processes days in parallel via multiprocessing

Usage:
    python scripts/build_mes_outcomes_fast.py
    python scripts/build_mes_outcomes_fast.py --orb-minutes 15
    python scripts/build_mes_outcomes_fast.py --dry-run
"""

import sys
import time
from pathlib import Path
from datetime import date
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import duckdb

sys.stdout.reconfigure(line_buffering=True)

from pipeline.cost_model import get_cost_spec
from pipeline.init_db import ORB_LABELS
from pipeline.build_daily_features import compute_trading_day_utc_range
from trading_app.outcome_builder import compute_single_outcome
from trading_app.config import ENTRY_MODELS

# =========================================================================
# CONFIG
# =========================================================================
INSTRUMENT = "MES"
DB_PATH = Path(r"C:\db\mes.db")
START_DATE = date(2024, 2, 12)
END_DATE = date(2026, 2, 11)
RR_TARGETS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
CONFIRM_BARS_OPTIONS = [1, 2, 3, 4, 5]
EARLY_EXIT_MINUTES = {"0900": 15, "1000": 30}


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def process_single_day(args):
    """Process one trading day -- called in worker process."""
    (day_features, bars_day_df, cost_spec, orb_minutes) = args

    trading_day = day_features["trading_day"]
    symbol = day_features["symbol"]
    _, td_end = compute_trading_day_utc_range(trading_day)

    day_batch = []
    for orb_label in ORB_LABELS:
        break_dir = day_features.get(f"orb_{orb_label}_break_dir")
        break_ts = day_features.get(f"orb_{orb_label}_break_ts")
        orb_high = day_features.get(f"orb_{orb_label}_high")
        orb_low = day_features.get(f"orb_{orb_label}_low")

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
                        bars_df=bars_day_df,
                        break_ts=break_ts,
                        orb_high=orb_high,
                        orb_low=orb_low,
                        break_dir=break_dir,
                        rr_target=rr_target,
                        confirm_bars=cb,
                        trading_day_end=td_end,
                        cost_spec=cost_spec,
                        entry_model=em,
                        orb_label=orb_label,
                    )
                    day_batch.append([
                        trading_day, symbol, orb_label, orb_minutes,
                        rr_target, cb, em,
                        outcome["entry_ts"], outcome["entry_price"],
                        outcome["stop_price"], outcome["target_price"],
                        outcome["outcome"], outcome["exit_ts"],
                        outcome["exit_price"], outcome["pnl_r"],
                        outcome["mae_r"], outcome["mfe_r"],
                    ])

    return day_batch


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fast MES outcome builder (parallel)")
    parser.add_argument("--orb-minutes", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 2))
    args = parser.parse_args()

    orb_minutes = args.orb_minutes
    cost_spec = get_cost_spec(INSTRUMENT)

    log("=" * 60)
    log(f"FAST MES OUTCOME BUILDER (PARALLEL)")
    log(f"  DB:          {DB_PATH}")
    log(f"  ORB minutes: {orb_minutes}")
    log(f"  Workers:     {args.workers}")
    log(f"  Dry run:     {args.dry_run}")
    log("=" * 60)

    # =====================================================================
    # STEP 1: Preload ALL bars_1m into memory
    # =====================================================================
    t0 = time.time()
    con = duckdb.connect(str(DB_PATH), read_only=True)

    log("Loading bars_1m...")
    all_bars = con.execute("""
        SELECT ts_utc, open, high, low, close, volume
        FROM bars_1m WHERE symbol = ?
        ORDER BY ts_utc
    """, [INSTRUMENT]).fetchdf()
    all_bars["ts_utc"] = pd.to_datetime(all_bars["ts_utc"], utc=True)
    log(f"  {len(all_bars):,} bars loaded ({time.time()-t0:.1f}s)")

    # =====================================================================
    # STEP 2: Load daily_features
    # =====================================================================
    orb_cols = ', '.join(
        f'orb_{lbl}_high, orb_{lbl}_low, orb_{lbl}_break_dir, orb_{lbl}_break_ts'
        for lbl in ORB_LABELS
    )
    features = con.execute(f"""
        SELECT trading_day, symbol, orb_minutes, {orb_cols}
        FROM daily_features
        WHERE symbol = ? AND orb_minutes = ?
        AND trading_day BETWEEN ? AND ?
        ORDER BY trading_day
    """, [INSTRUMENT, orb_minutes, START_DATE, END_DATE]).fetchall()
    col_names = [desc[0] for desc in con.description]
    con.close()

    log(f"  {len(features)} trading days loaded")

    # =====================================================================
    # STEP 3: Partition bars by trading day (vectorized)
    # =====================================================================
    t1 = time.time()
    from pipeline.build_daily_features import compute_trading_day_utc_range

    # Build (td_start, td_end) for each trading day
    trading_days = [dict(zip(col_names, row))["trading_day"] for row in features]
    day_ranges = {td: compute_trading_day_utc_range(td) for td in trading_days}

    # Partition bars
    bars_by_day = {}
    for td, (td_start, td_end) in day_ranges.items():
        mask = (all_bars["ts_utc"] >= pd.Timestamp(td_start)) & \
               (all_bars["ts_utc"] < pd.Timestamp(td_end))
        day_bars = all_bars[mask].copy()
        if not day_bars.empty:
            bars_by_day[td] = day_bars

    log(f"  Partitioned into {len(bars_by_day)} day chunks ({time.time()-t1:.1f}s)")

    # =====================================================================
    # STEP 4: Build task list
    # =====================================================================
    tasks = []
    for row in features:
        day_features = dict(zip(col_names, row))
        td = day_features["trading_day"]
        if td not in bars_by_day:
            continue
        tasks.append((day_features, bars_by_day[td], cost_spec, orb_minutes))

    log(f"  {len(tasks)} days to process")

    # =====================================================================
    # STEP 5: Process in parallel
    # =====================================================================
    t2 = time.time()
    all_outcomes = []

    with Pool(processes=args.workers) as pool:
        for i, batch in enumerate(pool.imap_unordered(process_single_day, tasks, chunksize=10)):
            all_outcomes.extend(batch)
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t2
                rate = (i + 1) / elapsed
                remaining = (len(tasks) - i - 1) / rate
                log(f"  {i+1}/{len(tasks)} days ({len(all_outcomes):,} outcomes, "
                    f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    log(f"  {len(all_outcomes):,} total outcomes ({time.time()-t2:.1f}s)")

    if args.dry_run:
        log("DRY RUN -- no DB writes")
        log("DONE")
        return

    # =====================================================================
    # STEP 6: Bulk write to DB
    # =====================================================================
    t3 = time.time()
    con = duckdb.connect(str(DB_PATH))

    # Clear existing outcomes for this instrument + orb_minutes
    con.execute("""
        DELETE FROM orb_outcomes
        WHERE symbol = ? AND orb_minutes = ?
        AND trading_day BETWEEN ? AND ?
    """, [INSTRUMENT, orb_minutes, START_DATE, END_DATE])

    # Bulk insert
    BATCH_SIZE = 10_000
    for i in range(0, len(all_outcomes), BATCH_SIZE):
        batch = all_outcomes[i:i + BATCH_SIZE]
        con.executemany("""
            INSERT INTO orb_outcomes
            (trading_day, symbol, orb_label, orb_minutes,
             rr_target, confirm_bars, entry_model,
             entry_ts, entry_price, stop_price, target_price,
             outcome, exit_ts, exit_price, pnl_r,
             mae_r, mfe_r)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, batch)

    con.commit()

    count = con.execute(
        "SELECT COUNT(*) FROM orb_outcomes WHERE symbol = ? AND orb_minutes = ?",
        [INSTRUMENT, orb_minutes]
    ).fetchone()[0]
    con.close()

    log(f"  Written {count:,} outcomes to DB ({time.time()-t3:.1f}s)")
    log("=" * 60)
    log(f"DONE: {count:,} outcomes, total {time.time()-t0:.0f}s")
    log("=" * 60)


if __name__ == "__main__":
    main()
