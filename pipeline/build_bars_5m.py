#!/usr/bin/env python3
"""
Build bars_5m from bars_1m (deterministic aggregation).

Bucket: floor(epoch(ts_utc) / 300) * 300  (UTC, 5-minute aligned)

For each (symbol, bucket_ts):
  open   = first open   (by ts_utc ASC)
  high   = max high
  low    = min low
  close  = last close   (by ts_utc ASC)
  volume = sum volume
  source_symbol = mode (most common); tie -> lexicographic smallest

Idempotent: DELETE existing bars_5m rows for the date range, then INSERT rebuilt rows.
BARS_5M ONLY: Does NOT touch bars_1m or any other table.

NOTE: --start/--end are UTC calendar dates, NOT Brisbane trading days.
      The query window is [start 00:00 UTC, end+1 00:00 UTC).
      Trading day logic (Brisbane 09:00 boundary) belongs in daily_features, not here.

Usage:
    python pipeline/build_bars_5m.py --instrument MGC --start 2024-01-01 --end 2024-01-31
    python pipeline/build_bars_5m.py --instrument MGC --start 2024-01-01 --end 2024-01-31 --dry-run
"""

import sys
import argparse
from datetime import datetime, date, timedelta

import duckdb

# Add project root to path

from pipeline.paths import GOLD_DB_PATH
from pipeline.asset_configs import get_asset_config, list_instruments

from pipeline.log import get_logger
logger = get_logger(__name__)

def build_5m_bars(con: duckdb.DuckDBPyConnection, symbol: str,
                  start_date: date, end_date: date, dry_run: bool) -> int:
    """
    Build bars_5m from bars_1m for the given symbol and date range.

    Returns number of rows written.

    The query is fully deterministic:
    - Bucket alignment: floor(epoch/300)*300 in UTC
    - open/close ordering: by ts_utc ASC (first/last)
    - source_symbol: mode with lexicographic tiebreak
    """

    # Convert dates to UTC timestamps for filtering bars_1m
    # start_date 00:00 UTC to end_date+1 00:00 UTC (exclusive upper bound)
    start_ts = f"{start_date}T00:00:00+00:00"
    end_ts = f"{end_date + timedelta(days=1)}T00:00:00+00:00"

    # Count source rows first
    count_query = """
        SELECT COUNT(*) FROM bars_1m
        WHERE symbol = ?
        AND ts_utc >= ?::TIMESTAMPTZ
        AND ts_utc < ?::TIMESTAMPTZ
    """
    source_count = con.execute(count_query, [symbol, start_ts, end_ts]).fetchone()[0]
    logger.info(f"  Source bars_1m rows: {source_count:,}")

    if source_count == 0:
        logger.info("  No source data found. Nothing to build.")
        return 0

    # Build 5m bars using SQL (fully deterministic, no Python loops)
    #
    # The CTE approach:
    # 1. bucketed: assign each 1m bar to its 5m bucket
    # 2. source_counts: count source_symbol occurrences per bucket for mode calculation
    # 3. source_mode: pick mode source_symbol (highest count, lex smallest on tie)
    # 4. Final SELECT: aggregate OHLCV with first/last by ts_utc ordering
    build_query = """
        WITH bucketed AS (
            SELECT
                time_bucket(INTERVAL '5 minutes', ts_utc) AS bucket_ts,
                symbol,
                source_symbol,
                ts_utc,
                open,
                high,
                low,
                close,
                volume
            FROM bars_1m
            WHERE symbol = ?
            AND ts_utc >= ?::TIMESTAMPTZ
            AND ts_utc < ?::TIMESTAMPTZ
        ),
        source_counts AS (
            SELECT
                bucket_ts,
                symbol,
                source_symbol,
                COUNT(*) AS cnt
            FROM bucketed
            GROUP BY bucket_ts, symbol, source_symbol
        ),
        source_mode AS (
            SELECT DISTINCT ON (bucket_ts, symbol)
                bucket_ts,
                symbol,
                source_symbol
            FROM source_counts
            ORDER BY bucket_ts, symbol, cnt DESC, source_symbol ASC
        ),
        agg AS (
            SELECT
                b.bucket_ts,
                b.symbol,
                FIRST(b.open ORDER BY b.ts_utc ASC) AS open,
                MAX(b.high) AS high,
                MIN(b.low) AS low,
                LAST(b.close ORDER BY b.ts_utc ASC) AS close,
                SUM(b.volume) AS volume
            FROM bucketed b
            GROUP BY b.bucket_ts, b.symbol
        )
        SELECT
            a.bucket_ts AS ts_utc,
            a.symbol,
            m.source_symbol,
            a.open,
            a.high,
            a.low,
            a.close,
            a.volume
        FROM agg a
        JOIN source_mode m ON a.bucket_ts = m.bucket_ts AND a.symbol = m.symbol
        ORDER BY a.bucket_ts ASC
    """

    if dry_run:
        # Count how many 5m bars would be built
        count_result = con.execute("""
            SELECT COUNT(DISTINCT (EXTRACT(EPOCH FROM ts_utc)::BIGINT / 300 * 300))
            FROM bars_1m
            WHERE symbol = ?
            AND ts_utc >= ?::TIMESTAMPTZ
            AND ts_utc < ?::TIMESTAMPTZ
        """, [symbol, start_ts, end_ts]).fetchone()[0]
        logger.info(f"  DRY RUN: Would build {count_result:,} bars_5m rows")
        return count_result

    # IDEMPOTENT: Delete existing rows for this range, then insert
    con.execute("BEGIN TRANSACTION")

    try:
        # Delete existing 5m bars in range
        delete_count = con.execute("""
            SELECT COUNT(*) FROM bars_5m
            WHERE symbol = ?
            AND ts_utc >= ?::TIMESTAMPTZ
            AND ts_utc < ?::TIMESTAMPTZ
        """, [symbol, start_ts, end_ts]).fetchone()[0]

        con.execute("""
            DELETE FROM bars_5m
            WHERE symbol = ?
            AND ts_utc >= ?::TIMESTAMPTZ
            AND ts_utc < ?::TIMESTAMPTZ
        """, [symbol, start_ts, end_ts])

        if delete_count > 0:
            logger.info(f"  Deleted {delete_count:,} existing bars_5m rows")

        # Insert rebuilt rows
        con.execute(f"""
            INSERT INTO bars_5m (ts_utc, symbol, source_symbol, open, high, low, close, volume)
            {build_query}
        """, [symbol, start_ts, end_ts])

        # Count inserted
        new_count = con.execute("""
            SELECT COUNT(*) FROM bars_5m
            WHERE symbol = ?
            AND ts_utc >= ?::TIMESTAMPTZ
            AND ts_utc < ?::TIMESTAMPTZ
        """, [symbol, start_ts, end_ts]).fetchone()[0]

        con.execute("COMMIT")

        logger.info(f"  Inserted {new_count:,} bars_5m rows")
        return new_count

    except Exception as e:
        con.execute("ROLLBACK")
        logger.error(f"FATAL: Exception during bars_5m build: {e}")
        raise

def verify_5m_integrity(con: duckdb.DuckDBPyConnection, symbol: str,
                        start_date: date, end_date: date) -> tuple[bool, list[str]]:
    """
    Verify bars_5m integrity after build.

    Checks:
    1. No duplicate (symbol, ts_utc) in bars_5m
    2. All ts_utc are 5-minute aligned (epoch % 300 == 0)
    3. OHLCV sanity: high >= low, high >= open, high >= close, low <= open, low <= close
    4. Volume >= 0
    """
    failures = []

    start_ts = f"{start_date}T00:00:00+00:00"
    end_ts = f"{end_date + timedelta(days=1)}T00:00:00+00:00"

    # Check 1: duplicates
    dupe_count = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT symbol, ts_utc FROM bars_5m
            WHERE symbol = ?
            AND ts_utc >= ?::TIMESTAMPTZ
            AND ts_utc < ?::TIMESTAMPTZ
            GROUP BY symbol, ts_utc
            HAVING COUNT(*) > 1
        )
    """, [symbol, start_ts, end_ts]).fetchone()[0]

    if dupe_count > 0:
        failures.append(f"Duplicate (symbol, ts_utc) in bars_5m: {dupe_count}")

    # Check 2: 5-minute alignment
    misaligned = con.execute("""
        SELECT COUNT(*) FROM bars_5m
        WHERE symbol = ?
        AND ts_utc >= ?::TIMESTAMPTZ
        AND ts_utc < ?::TIMESTAMPTZ
        AND EXTRACT(EPOCH FROM ts_utc)::BIGINT % 300 != 0
    """, [symbol, start_ts, end_ts]).fetchone()[0]

    if misaligned > 0:
        failures.append(f"Misaligned timestamps (not 5m boundary): {misaligned}")

    # Check 3: OHLCV sanity
    ohlcv_bad = con.execute("""
        SELECT COUNT(*) FROM bars_5m
        WHERE symbol = ?
        AND ts_utc >= ?::TIMESTAMPTZ
        AND ts_utc < ?::TIMESTAMPTZ
        AND (high < low OR high < open OR high < close OR low > open OR low > close)
    """, [symbol, start_ts, end_ts]).fetchone()[0]

    if ohlcv_bad > 0:
        failures.append(f"OHLCV sanity failures (high/low violations): {ohlcv_bad}")

    # Check 4: volume
    vol_bad = con.execute("""
        SELECT COUNT(*) FROM bars_5m
        WHERE symbol = ?
        AND ts_utc >= ?::TIMESTAMPTZ
        AND ts_utc < ?::TIMESTAMPTZ
        AND volume < 0
    """, [symbol, start_ts, end_ts]).fetchone()[0]

    if vol_bad > 0:
        failures.append(f"Negative volume: {vol_bad}")

    return len(failures) == 0, failures

def main():
    parser = argparse.ArgumentParser(
        description="Build bars_5m from bars_1m (deterministic aggregation)"
    )
    parser.add_argument(
        "--instrument", type=str, required=True,
        help=f"Instrument ({', '.join(list_instruments())})"
    )
    parser.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true", help="Count only, no DB writes")
    args = parser.parse_args()

    # Validate instrument (fail-closed)
    config = get_asset_config(args.instrument)
    symbol = config["symbol"]

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)

    start_time = datetime.now()

    logger.info("=" * 60)
    logger.info(f"BUILD BARS_5M ({symbol})")
    logger.info("=" * 60)
    logger.info(f"Instrument: {symbol}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Database: {GOLD_DB_PATH}")
    logger.info(f"Dry run: {args.dry_run}")

    if not GOLD_DB_PATH.exists():
        logger.error(f"FATAL: Database not found: {GOLD_DB_PATH}")
        sys.exit(1)

    with duckdb.connect(str(GOLD_DB_PATH)) as con:
        from pipeline.db_config import configure_connection
        configure_connection(con, writing=True)

        # Build
        logger.info("Building 5-minute bars...")
        row_count = build_5m_bars(con, symbol, start_date, end_date, args.dry_run)

        # Verify (skip for dry run)
        if not args.dry_run and row_count > 0:
            logger.info("Verifying integrity...")
            ok, failures = verify_5m_integrity(con, symbol, start_date, end_date)

            if not ok:
                logger.error("FATAL: Integrity verification FAILED:")
                for f in failures:
                    logger.info(f"  - {f}")
                sys.exit(1)

            logger.info("  No duplicates: PASSED [OK]")
            logger.info("  5-minute alignment: PASSED [OK]")
            logger.info("  OHLCV sanity: PASSED [OK]")
            logger.info("  Volume non-negative: PASSED [OK]")
            logger.info("ALL INTEGRITY CHECKS PASSED [OK]")
        elif args.dry_run:
            logger.info("Integrity check skipped (dry run)")

        elapsed = datetime.now() - start_time

        logger.info("=" * 60)
        logger.info(f"SUMMARY: {row_count:,} bars_5m rows {'(would be) ' if args.dry_run else ''}built")
        logger.info(f"Wall time: {elapsed}")
        logger.info("=" * 60)

        if args.dry_run:
            logger.info("DRY RUN COMPLETE. No changes made.")

        sys.exit(0)

if __name__ == "__main__":
    main()
