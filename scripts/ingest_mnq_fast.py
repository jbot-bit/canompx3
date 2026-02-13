#!/usr/bin/env python3
"""
Fast MNQ Ingestion -- vectorized, same honesty gates as ingest_dbn.py.

Bottleneck in ingest_dbn.py: iterrows() over every bar (Python-level loop).
This script does the same work vectorized with pandas groupby + bulk INSERT.

SAME GATES AS CANONICAL PIPELINE:
  1. Schema verification (ohlcv-1m)
  2. Timestamp UTC proof
  3. Outright filter (regex, exclude spreads)
  4. OHLCV validation (NaN, inf, non-positive, high/low)
  5. Front contract selection (highest volume, deterministic tiebreak)
  6. PK safety (no duplicate ts_utc per trading day)
  7. Merge integrity (no dupes/NULLs after write)
  8. Final honesty gates (global checks)

Usage:
    python scripts/ingest_mnq_fast.py
    python scripts/ingest_mnq_fast.py --dry-run
"""

import sys
import time
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import duckdb
import databento as db

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.asset_configs import get_asset_config
from pipeline.ingest_dbn_mgc import (
    validate_chunk,
    validate_timestamp_utc,
    choose_front_contract,
    run_final_gates,
)

# =========================================================================
# CONFIG
# =========================================================================
INSTRUMENT = "MNQ"
DB_PATH = Path(r"C:\db\gold.db")
START_DATE = date(2024, 2, 4)
END_DATE = date(2026, 2, 3)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compute_trading_days_fast(ts_index: pd.DatetimeIndex) -> np.ndarray:
    """Vectorized trading day computation (09:00 Brisbane boundary)."""
    import zoneinfo
    tz_local = zoneinfo.ZoneInfo("Australia/Brisbane")
    ts_local = ts_index.tz_convert(tz_local)
    hours = ts_local.hour
    dates = ts_local.date  # numpy array of date objects
    result = np.array(dates)
    mask = hours < 9
    result[mask] = np.array([d - timedelta(days=1) for d in result[mask]])
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fast MNQ ingestion (vectorized)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = get_asset_config(INSTRUMENT)
    symbol = config["symbol"]
    dbn_path = config["dbn_path"]
    outright_pattern = config["outright_pattern"]
    prefix_len = config["prefix_len"]

    log("=" * 60)
    log(f"FAST MNQ INGESTION (VECTORIZED, SAME HONESTY GATES)")
    log(f"  DBN: {dbn_path}")
    log(f"  DB:  {DB_PATH}")
    log(f"  Range: {START_DATE} to {END_DATE}")
    log(f"  Dry run: {args.dry_run}")
    log("=" * 60)

    # =====================================================================
    # STEP 1: Read entire DBN into DataFrame (20MB compressed, fits in RAM)
    # =====================================================================
    t0 = time.time()
    log("Reading DBN file...")
    store = db.DBNStore.from_file(dbn_path)

    if store.schema != "ohlcv-1m":
        log(f"FATAL: Schema is '{store.schema}', expected 'ohlcv-1m'")
        sys.exit(1)
    log(f"  Schema: {store.schema} [OK]")

    # Read all at once (small file, ~1.2M rows)
    all_chunks = []
    for chunk in store.to_df(count=200_000):
        chunk = chunk.reset_index()
        all_chunks.append(chunk)
    df = pd.concat(all_chunks, ignore_index=True)
    log(f"  Loaded {len(df):,} raw rows ({time.time()-t0:.1f}s)")

    # =====================================================================
    # STEP 2: Timestamp UTC proof (FAIL-CLOSED)
    # =====================================================================
    t1 = time.time()
    df_indexed = df.set_index("ts_event")
    ts_ok, ts_reason = validate_timestamp_utc(df_indexed)
    if not ts_ok:
        log(f"FATAL: Timestamp validation failed: {ts_reason}")
        sys.exit(1)
    log(f"  Timestamps are UTC [OK] ({time.time()-t1:.1f}s)")

    # =====================================================================
    # STEP 3: Filter to outrights only (vectorized regex)
    # =====================================================================
    t2 = time.time()
    outright_mask = df["symbol"].str.match(outright_pattern.pattern)
    n_before = len(df)
    df = df[outright_mask].copy()
    n_spreads = n_before - len(df)
    log(f"  Outrights: {len(df):,} kept, {n_spreads:,} spreads removed ({time.time()-t2:.1f}s)")

    # =====================================================================
    # STEP 4: OHLCV validation (FAIL-CLOSED, vectorized)
    # =====================================================================
    t3 = time.time()
    df_check = df.set_index("ts_event")
    valid, reason, bad_rows = validate_chunk(df_check)
    if not valid:
        log(f"FATAL: OHLCV validation failed: {reason}")
        if bad_rows is not None:
            log(f"  Bad rows:\n{bad_rows.head()}")
        sys.exit(1)
    log(f"  OHLCV validation [OK] ({time.time()-t3:.1f}s)")

    # =====================================================================
    # STEP 5: Compute trading days (vectorized)
    # =====================================================================
    t4 = time.time()
    df["trading_day"] = compute_trading_days_fast(
        pd.DatetimeIndex(df["ts_event"])
    )
    # Filter to date range
    df = df[(df["trading_day"] >= START_DATE) & (df["trading_day"] <= END_DATE)]
    log(f"  Trading days computed, {len(df):,} rows in range ({time.time()-t4:.1f}s)")

    # =====================================================================
    # STEP 6: Front contract selection per trading day (vectorized groupby)
    # =====================================================================
    t5 = time.time()
    # Compute daily volumes per contract
    daily_vols = df.groupby(["trading_day", "symbol"])["volume"].sum().reset_index()

    # Choose front contract per day
    front_contracts = {}
    for tday, grp in daily_vols.groupby("trading_day"):
        vols = dict(zip(grp["symbol"], grp["volume"]))
        front = choose_front_contract(
            vols, outright_pattern=outright_pattern, prefix_len=prefix_len,
            log_func=None,
        )
        if front:
            front_contracts[tday] = front

    log(f"  Front contracts: {len(front_contracts)} trading days ({time.time()-t5:.1f}s)")
    log(f"  Unique contracts: {len(set(front_contracts.values()))}")

    # =====================================================================
    # STEP 7: Filter to front contract bars only (vectorized)
    # =====================================================================
    t6 = time.time()
    # Build (trading_day, front_symbol) lookup
    front_df = pd.DataFrame(
        list(front_contracts.items()), columns=["trading_day", "front_symbol"]
    )
    df = df.merge(front_df, on="trading_day", how="inner")
    df = df[df["symbol"] == df["front_symbol"]].copy()
    df.drop(columns=["front_symbol"], inplace=True)
    log(f"  Front contract bars: {len(df):,} rows ({time.time()-t6:.1f}s)")

    # =====================================================================
    # STEP 8: PK safety -- no duplicate ts_event per (symbol, ts_event)
    # =====================================================================
    t7 = time.time()
    dupes = df.duplicated(subset=["ts_event"], keep=False)
    if dupes.any():
        n_dupes = dupes.sum()
        log(f"WARNING: {n_dupes} duplicate ts_event rows, keeping first")
        df = df.drop_duplicates(subset=["ts_event"], keep="first")
    log(f"  PK safety [OK] -- {len(df):,} unique rows ({time.time()-t7:.1f}s)")

    # =====================================================================
    # STEP 9: Prepare final DataFrame for INSERT
    # =====================================================================
    insert_df = pd.DataFrame({
        "ts_utc": df["ts_event"],
        "symbol": symbol,
        "source_symbol": df["symbol"],  # actual MNQ contract
        "open": df["open"].astype(float),
        "high": df["high"].astype(float),
        "low": df["low"].astype(float),
        "close": df["close"].astype(float),
        "volume": df["volume"].astype(int),
    })

    log(f"  Final: {len(insert_df):,} rows ready for INSERT")

    if args.dry_run:
        log("DRY RUN -- no DB writes")
        # Print summary stats
        log(f"  Date range: {insert_df['ts_utc'].min()} to {insert_df['ts_utc'].max()}")
        log(f"  Unique source_symbols: {insert_df['source_symbol'].nunique()}")
        log(f"  Source symbols: {sorted(insert_df['source_symbol'].unique())}")
        log("DRY RUN COMPLETE [OK]")
        sys.exit(0)

    # =====================================================================
    # STEP 10: Write to DB (bulk INSERT OR REPLACE)
    # =====================================================================
    t8 = time.time()
    log(f"Opening DB: {DB_PATH}")
    con = duckdb.connect(str(DB_PATH))

    try:
        # Delete existing MNQ data in range (clean slate, then bulk insert)
        con.execute("""
            DELETE FROM bars_1m
            WHERE symbol = ?
            AND DATE(ts_utc) BETWEEN ? AND ?
        """, [symbol, str(START_DATE), str(END_DATE)])
        log(f"  Cleared existing MNQ data in range")

        # Bulk insert from DataFrame -- DuckDB can read pandas directly
        con.execute("BEGIN TRANSACTION")
        con.execute("""
            INSERT INTO bars_1m (ts_utc, symbol, source_symbol, open, high, low, close, volume)
            SELECT * FROM insert_df
        """)
        con.execute("COMMIT")
        log(f"  Bulk INSERT: {len(insert_df):,} rows ({time.time()-t8:.1f}s)")

        # =================================================================
        # STEP 11: Merge integrity check (FAIL-CLOSED)
        # =================================================================
        t9 = time.time()
        dupe_check = con.execute("""
            SELECT symbol, ts_utc, COUNT(*) as cnt
            FROM bars_1m
            WHERE symbol = ? AND DATE(ts_utc) BETWEEN ? AND ?
            GROUP BY symbol, ts_utc
            HAVING COUNT(*) > 1
            LIMIT 5
        """, [symbol, str(START_DATE), str(END_DATE)]).fetchall()

        if dupe_check:
            log(f"FATAL: Duplicate rows found after insert: {dupe_check}")
            sys.exit(1)

        null_check = con.execute("""
            SELECT COUNT(*) FROM bars_1m
            WHERE symbol = ? AND DATE(ts_utc) BETWEEN ? AND ?
            AND source_symbol IS NULL
        """, [symbol, str(START_DATE), str(END_DATE)]).fetchone()[0]

        if null_check > 0:
            log(f"FATAL: {null_check} NULL source_symbol rows")
            sys.exit(1)

        log(f"  Merge integrity [OK] ({time.time()-t9:.1f}s)")

        # =================================================================
        # STEP 12: Final honesty gates (global)
        # =================================================================
        t10 = time.time()
        gates_ok, failures = run_final_gates(con)
        if not gates_ok:
            log("FATAL: Final honesty gates FAILED:")
            for f in failures:
                log(f"  - {f}")
            sys.exit(1)
        log(f"  Final honesty gates [OK] ({time.time()-t10:.1f}s)")

        # =================================================================
        # SUMMARY
        # =================================================================
        count = con.execute(
            "SELECT COUNT(*) FROM bars_1m WHERE symbol = ?", [symbol]
        ).fetchone()[0]
        date_range = con.execute(
            "SELECT MIN(DATE(ts_utc)), MAX(DATE(ts_utc)) FROM bars_1m WHERE symbol = ?",
            [symbol],
        ).fetchone()

        log("=" * 60)
        log(f"SUCCESS: {count:,} MNQ rows in DB")
        log(f"  Date range: {date_range[0]} to {date_range[1]}")
        log(f"  Total time: {time.time()-t0:.1f}s")
        log("=" * 60)

    finally:
        con.close()

    sys.exit(0)


if __name__ == "__main__":
    main()
