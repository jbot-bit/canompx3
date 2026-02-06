#!/usr/bin/env python3
"""
Ingest MGC DBN file into bars_1m table.

SPEC COMPLIANCE (backtestfix.txt):
- Bar timestamp: ts_event = bar OPEN time (Databento convention)
- Trading day: 09:00 Australia/Brisbane -> 09:00 next day
- Sorted: All bars sorted by ts_utc before insert
- Dedupe: PRIMARY KEY (symbol, ts_utc), INSERT OR REPLACE
- Price validation: high >= max(o,c), low <= min(o,c), high >= low
- Volume validation: volume >= 0 and integer
- Symbol filtering: Explicit regex for MGC outright contracts
- Storage: UTC only, raw prices (no back-adjustment)

Usage:
    python pipeline/ingest_dbn_mgc.py [--dry-run] [--start YYYY-MM-DD] [--end YYYY-MM-DD]
    python pipeline/ingest_dbn_mgc.py --resume  # Auto-detect and continue from last date

Options:
    --dry-run       Validate data without writing to DB
    --start         Start date (inclusive) for partial backfill
    --end           End date (inclusive) for partial backfill
    --resume        Auto-detect last date in DB and continue from there
    --chunk-size    Trading days per commit (default: 100)

Resumability:
    If interrupted, re-run with --resume to continue from where you left off.
    The script uses INSERT OR REPLACE, so re-running is safe (idempotent).
"""

import os
import sys
import re
import argparse
from pathlib import Path
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from collections import defaultdict

import duckdb
import databento as db
import pandas as pd

# Database path (standalone - no pipeline dependency)
GOLD_DB_PATH = Path(__file__).parent / "gold.db"

# =============================================================================
# CONFIGURATION (EXPLICIT PER backtestfix.txt)
# =============================================================================

# DBN file location
DBN_PATH = Path(r"C:\Users\sydne\OneDrive\Desktop\CANONICAL TRADING\OHLCV_MGC_FULL\glbx-mdp3-20100912-20260203.ohlcv-1m.dbn.zst")

# Database
DB_PATH = GOLD_DB_PATH

# Symbol
SYMBOL = "MGC"  # Logical symbol stored in DB

# Timezone for trading day boundary
TZ_LOCAL = ZoneInfo("Australia/Brisbane")
TZ_UTC = ZoneInfo("UTC")

# CRITICAL: Bar timestamp convention
# Databento ohlcv-1m ts_event = bar OPEN time (start of the minute)
# A bar at 09:00:00 represents the interval [09:00:00, 09:01:00)
BAR_TIMESTAMP_CONVENTION = "OPEN"

# =============================================================================
# SYMBOL FILTERING (backtestfix.txt requirement)
# =============================================================================

# CME Gold Micro futures outright contract pattern
# Format: MGC + month_code + year (1 or 2 digits)
# Month codes: F(Jan) G(Feb) H(Mar) J(Apr) K(May) M(Jun) N(Jul) Q(Aug) U(Sep) V(Oct) X(Nov) Z(Dec)
MGC_OUTRIGHT_PATTERN = re.compile(r'^MGC[FGHJKMNQUVXZ]\d{1,2}$')


def is_outright_contract(symbol: str) -> bool:
    """Check if symbol is a valid MGC outright contract (not a spread)."""
    if not symbol:
        return False
    return bool(MGC_OUTRIGHT_PATTERN.match(str(symbol)))


# =============================================================================
# TRADING DAY LOGIC (backtestfix.txt requirement)
# =============================================================================

def get_trading_day(ts_utc: datetime) -> date:
    """
    Determine trading day using Brisbane 09:00 boundary.

    Trading day = 09:00 local -> 09:00 next day local
    A bar at 08:59 Brisbane belongs to PREVIOUS trading day.
    A bar at 09:00 Brisbane belongs to CURRENT trading day.
    """
    ts_local = ts_utc.astimezone(TZ_LOCAL)
    if ts_local.hour < 9:
        return (ts_local - timedelta(days=1)).date()
    return ts_local.date()


# =============================================================================
# DATA VALIDATION (backtestfix.txt requirement)
# =============================================================================

def validate_bar(row: pd.Series) -> tuple[bool, str]:
    """
    Validate a single bar for price/volume sanity.

    Returns (is_valid, reason_if_invalid)
    """
    o, h, l, c = row['open'], row['high'], row['low'], row['close']
    v = row['volume']

    # Check for NaN/None in prices
    if pd.isna(o) or pd.isna(h) or pd.isna(l) or pd.isna(c):
        return False, "NaN price"

    # Price physics
    if h < max(o, c):
        return False, f"high ({h}) < max(open,close) ({max(o,c)})"

    if l > min(o, c):
        return False, f"low ({l}) > min(open,close) ({min(o,c)})"

    if h < l:
        return False, f"high ({h}) < low ({l})"

    # Volume validation
    if pd.isna(v):
        return False, "NaN volume"

    try:
        v_int = int(v)
        if v_int < 0:
            return False, f"negative volume ({v_int})"
    except (ValueError, TypeError):
        return False, f"non-integer volume ({v})"

    return True, ""


# =============================================================================
# CONTRACT SELECTION
# =============================================================================

def choose_front_contract(daily_volumes: dict) -> str | None:
    """
    Choose front-month contract (highest volume outright).

    Only considers symbols matching MGC_OUTRIGHT_PATTERN.
    """
    outrights = {s: v for s, v in daily_volumes.items() if is_outright_contract(s)}
    if not outrights:
        return None
    return max(outrights, key=outrights.get)


# =============================================================================
# MAIN INGESTION
# =============================================================================

def get_last_date_in_db(db_path: str, symbol: str) -> date | None:
    """Get the last trading day already in the database for this symbol."""
    try:
        con = duckdb.connect(db_path, read_only=True)
        result = con.execute(
            "SELECT MAX(DATE(ts_utc)) FROM bars_1m WHERE symbol = ?",
            [symbol]
        ).fetchone()
        con.close()
        if result and result[0]:
            return result[0]
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Ingest MGC DBN file into bars_1m")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, no DB write")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD (inclusive)")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD (inclusive)")
    parser.add_argument("--resume", action="store_true", help="Auto-resume from last date in DB")
    parser.add_argument("--chunk-size", type=int, default=100, help="Trading days per commit")
    args = parser.parse_args()

    # Handle --resume flag
    start_filter = None
    if args.resume:
        last_date = get_last_date_in_db(str(DB_PATH), SYMBOL)
        if last_date:
            start_filter = last_date + timedelta(days=1)
            print(f"RESUME MODE: Last date in DB is {last_date}, starting from {start_filter}")
        else:
            print("RESUME MODE: No existing data found, starting from beginning")
    elif args.start:
        start_filter = date.fromisoformat(args.start)

    end_filter = date.fromisoformat(args.end) if args.end else None

    print("=" * 70)
    print("MGC DBN INGESTION")
    print("=" * 70)
    print()
    print("SPEC COMPLIANCE:")
    print(f"  Bar timestamp convention: {BAR_TIMESTAMP_CONVENTION} (ts_event = bar open)")
    print(f"  Trading day boundary: 09:00 Australia/Brisbane")
    print(f"  Symbol filter: MGC outright regex")
    print(f"  Write mode: INSERT OR REPLACE (upsert)")
    print(f"  Dry run: {args.dry_run}")
    print()

    # Verify DBN file exists
    if not DBN_PATH.exists():
        print(f"ERROR: DBN file not found: {DBN_PATH}")
        sys.exit(1)

    file_size_gb = DBN_PATH.stat().st_size / (1024**3)
    print(f"DBN file: {DBN_PATH.name}")
    print(f"File size: {file_size_gb:.2f} GB")
    print(f"Database: {DB_PATH}")
    print(f"Symbol: {SYMBOL}")
    if start_filter or end_filter:
        print(f"Date filter: {start_filter or 'beginning'} to {end_filter or 'end'}")
    print()

    # Open database (unless dry run)
    con = None
    if not args.dry_run:
        try:
            con = duckdb.connect(str(DB_PATH))
            # Create tables if they don't exist
            con.execute("""
                CREATE TABLE IF NOT EXISTS bars_1m (
                    ts_utc TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR NOT NULL,
                    source_symbol VARCHAR,
                    open DOUBLE NOT NULL,
                    high DOUBLE NOT NULL,
                    low DOUBLE NOT NULL,
                    close DOUBLE NOT NULL,
                    volume BIGINT NOT NULL,
                    PRIMARY KEY (symbol, ts_utc)
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS bars_5m (
                    ts_utc TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR NOT NULL,
                    source_symbol VARCHAR,
                    open DOUBLE NOT NULL,
                    high DOUBLE NOT NULL,
                    low DOUBLE NOT NULL,
                    close DOUBLE NOT NULL,
                    volume BIGINT NOT NULL,
                    PRIMARY KEY (symbol, ts_utc)
                )
            """)
            con.commit()
        except Exception as e:
            print(f"ERROR: Could not open database: {e}")
            sys.exit(1)

    # Read DBN file
    print("Reading DBN file (this may take several minutes for 10+ years of data)...")
    sys.stdout.flush()

    store = db.DBNStore.from_file(DBN_PATH)
    print(f"Schema: {store.schema}")
    print(f"Dataset: {store.dataset}")
    print()

    # Convert to DataFrame
    print("Converting to DataFrame...")
    sys.stdout.flush()

    df = store.to_df()
    total_records = len(df)
    print(f"Total records in file: {total_records:,}")
    print(f"Columns: {list(df.columns)}")
    print()

    # Reset index to access ts_event as column
    df = df.reset_index()

    # Get unique symbols before filtering
    all_symbols = df['symbol'].unique()
    print(f"Unique symbols in file: {len(all_symbols)}")

    # Count outrights vs spreads
    outright_symbols = [s for s in all_symbols if is_outright_contract(s)]
    spread_symbols = [s for s in all_symbols if not is_outright_contract(s)]
    print(f"  Outright contracts: {len(outright_symbols)}")
    print(f"  Spreads/other (will be ignored): {len(spread_symbols)}")
    if spread_symbols[:5]:
        print(f"    Sample spreads: {spread_symbols[:5]}")
    print()

    # Compute trading day for each bar
    print("Computing trading days...")
    df['ts_utc_dt'] = pd.to_datetime(df['ts_event'], utc=True)
    df['ts_local'] = df['ts_utc_dt'].dt.tz_convert(str(TZ_LOCAL))
    df['hour'] = df['ts_local'].dt.hour
    df['base_date'] = df['ts_local'].dt.date
    df['trading_day'] = df.apply(
        lambda row: row['base_date'] - timedelta(days=1) if row['hour'] < 9 else row['base_date'],
        axis=1
    )
    print("  Done")
    print()

    # Get unique trading days
    trading_days = sorted(df['trading_day'].unique())
    print(f"Trading days in file: {len(trading_days)}")
    print(f"  Range: {trading_days[0]} to {trading_days[-1]}")

    # Apply date filter
    if start_filter:
        trading_days = [d for d in trading_days if d >= start_filter]
    if end_filter:
        trading_days = [d for d in trading_days if d <= end_filter]

    if start_filter or end_filter:
        print(f"  After filter: {len(trading_days)} days")
    print()

    # Process each trading day
    print("Processing trading days...")
    print()

    stats = {
        'days_processed': 0,
        'days_skipped_no_contract': 0,
        'bars_inserted': 0,
        'bars_rejected': 0,
        'rejection_reasons': defaultdict(int),
        'contracts_used': set(),
    }

    rows_buffer = []

    for i, trading_day in enumerate(trading_days):
        day_df = df[df['trading_day'] == trading_day]

        # Calculate volumes per symbol (only outrights)
        volumes = day_df[day_df['symbol'].apply(is_outright_contract)].groupby('symbol')['volume'].sum().to_dict()

        # Choose front-month
        front = choose_front_contract(volumes)
        if not front:
            stats['days_skipped_no_contract'] += 1
            continue

        stats['contracts_used'].add(front)

        # Filter to front-month only
        front_df = day_df[day_df['symbol'] == front].copy()

        # CRITICAL: Sort by timestamp (backtestfix.txt requirement)
        front_df = front_df.sort_values('ts_utc_dt').reset_index(drop=True)

        # Validate and prepare rows
        for _, row in front_df.iterrows():
            is_valid, reason = validate_bar(row)

            if not is_valid:
                stats['bars_rejected'] += 1
                stats['rejection_reasons'][reason] += 1
                continue

            rows_buffer.append((
                row['ts_utc_dt'].isoformat(),
                SYMBOL,
                front,
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                int(row['volume']),
            ))

        stats['days_processed'] += 1

        # Commit in chunks
        if con and len(rows_buffer) >= args.chunk_size * 1000:
            con.executemany(
                """
                INSERT OR REPLACE INTO bars_1m
                (ts_utc, symbol, source_symbol, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows_buffer
            )
            stats['bars_inserted'] += len(rows_buffer)
            con.commit()
            rows_buffer = []

        # Progress indicator
        if (i + 1) % 500 == 0 or i == len(trading_days) - 1:
            pct = (i + 1) / len(trading_days) * 100
            print(f"  {i+1}/{len(trading_days)} days ({pct:.1f}%) - {stats['bars_inserted'] + len(rows_buffer):,} bars")
            sys.stdout.flush()

    # Final insert
    if con and rows_buffer:
        con.executemany(
            """
            INSERT OR REPLACE INTO bars_1m
            (ts_utc, symbol, source_symbol, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows_buffer
        )
        stats['bars_inserted'] += len(rows_buffer)
        con.commit()
    elif args.dry_run:
        stats['bars_inserted'] = len(rows_buffer)

    print()

    # Summary
    print("=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)
    print()
    print(f"Days processed: {stats['days_processed']}")
    print(f"Days skipped (no outright contract): {stats['days_skipped_no_contract']}")
    print(f"Bars inserted: {stats['bars_inserted']:,}")
    print(f"Bars rejected: {stats['bars_rejected']:,}")
    print(f"Unique contracts used: {len(stats['contracts_used'])}")

    if stats['rejection_reasons']:
        print()
        print("Rejection reasons:")
        for reason, count in sorted(stats['rejection_reasons'].items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    # Skip 5m rebuild and verification if dry run
    if args.dry_run:
        print()
        print("DRY RUN - No data written to database")
        return

    # Rebuild bars_5m
    print()
    print("Rebuilding bars_5m from bars_1m...")

    # Delete existing 5m bars for MGC
    con.execute("DELETE FROM bars_5m WHERE symbol = ?", [SYMBOL])

    # Aggregate 1m -> 5m
    con.execute(
        """
        INSERT INTO bars_5m (ts_utc, symbol, source_symbol, open, high, low, close, volume)
        SELECT
            CAST(to_timestamp(floor(epoch(ts_utc) / 300) * 300) AS TIMESTAMPTZ) AS ts_5m,
            symbol,
            NULL AS source_symbol,
            arg_min(open, ts_utc)  AS open,
            max(high)              AS high,
            min(low)               AS low,
            arg_max(close, ts_utc) AS close,
            sum(volume)            AS volume
        FROM bars_1m
        WHERE symbol = ?
        GROUP BY 1, 2
        ORDER BY 1
        """,
        [SYMBOL]
    )
    con.commit()

    count_5m = con.execute("SELECT COUNT(*) FROM bars_5m WHERE symbol = ?", [SYMBOL]).fetchone()[0]
    print(f"  Created {count_5m:,} 5-minute bars")

    # Verify data
    print()
    print("=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    print()

    count_1m = con.execute("SELECT COUNT(*) FROM bars_1m WHERE symbol = ?", [SYMBOL]).fetchone()[0]
    date_range = con.execute(
        "SELECT MIN(DATE(ts_utc)), MAX(DATE(ts_utc)) FROM bars_1m WHERE symbol = ?",
        [SYMBOL]
    ).fetchone()

    print(f"bars_1m (MGC): {count_1m:,} rows")
    print(f"bars_5m (MGC): {count_5m:,} rows")
    print(f"Date range: {date_range[0]} to {date_range[1]}")
    print()
    print("Next steps:")
    print(f"  1. Run audit: python scripts/backfill_audit1.py --symbol MGC")
    print(f"  2. Build features: python pipeline/build_daily_features.py {date_range[0]} {date_range[1]} --instrument MGC")

    con.close()


if __name__ == "__main__":
    main()
