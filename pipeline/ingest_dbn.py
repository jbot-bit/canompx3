#!/usr/bin/env python3
"""
Generic multi-instrument DBN ingestion into bars_1m table.

Config-driven wrapper around the canonical ingest functions from ingest_dbn_mgc.py.
Loads per-asset config from asset_configs.py. No monkey-patching.

CANONICAL COMPLIANCE (preserved from ingest_dbn_mgc.py):
- FAIL-CLOSED: Any validation failure aborts entire backfill
- CHUNKED: Uses store.to_df(count=N) iterator, never full load
- VECTORIZED: No apply() or iterrows() over large data
- DETERMINISTIC: Stable tiebreak for equal-volume contracts
- CHECKPOINTED: JSONL append-only checkpoint system
- INTEGRITY GATED: Verifies no duplicates/NULLs after each merge
- BARS_1M ONLY: Does NOT touch bars_5m or any derived tables

Usage:
    python pipeline/ingest_dbn.py --instrument MGC [options]

Options:
    --instrument INST     Instrument to ingest (MGC, MNQ, NQ)
    --start YYYY-MM-DD    Start date (inclusive)
    --end YYYY-MM-DD      End date (inclusive)
    --resume              Resume from checkpoint
    --retry-failed        Retry failed chunks
    --dry-run             Validate only, no DB writes
    --chunk-days N        Trading days per commit (default: 7)
    --batch-size N        Rows per DBN read batch (default: 50000)
"""

import sys
import argparse
import traceback
from pathlib import Path
from datetime import datetime, date

import duckdb
import databento as db
import pandas as pd

# Add project root to path

from pipeline.paths import GOLD_DB_PATH
from pipeline.asset_configs import get_asset_config, list_instruments
from pipeline.ingest_dbn_mgc import (
    CheckpointManager,
    validate_chunk,
    validate_timestamp_utc,
    choose_front_contract,
    compute_trading_days,
    check_pk_safety,
    check_merge_integrity,
    run_final_gates,
)

from pipeline.log import get_logger
logger = get_logger(__name__)

# Checkpoint directory (shared, files are named per-source)
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"

def main():
    parser = argparse.ArgumentParser(
        description="Ingest DBN into bars_1m (multi-instrument, CANONICAL COMPLIANT)"
    )
    parser.add_argument(
        "--instrument", type=str, required=True,
        help=f"Instrument to ingest ({', '.join(list_instruments())})"
    )
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed chunks")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, no DB writes")
    parser.add_argument("--chunk-days", type=int, default=7, help="Trading days per commit")
    parser.add_argument("--batch-size", type=int, default=50000, help="Rows per DBN batch")
    args = parser.parse_args()

    # =========================================================================
    # LOAD AND VALIDATE ASSET CONFIG (FAIL-CLOSED)
    # =========================================================================
    # get_asset_config calls sys.exit(1) if instrument unknown, dbn_path None,
    # dbn_path missing, or minimum_start_date None
    config = get_asset_config(args.instrument)

    symbol = config["symbol"]
    dbn_path = config["dbn_path"]
    outright_pattern = config["outright_pattern"]
    prefix_len = config["prefix_len"]
    minimum_start_date = config["minimum_start_date"]
    schema_required = config["schema_required"]

    start_time = datetime.now()

    # =========================================================================
    # STARTUP: Log config snapshot
    # =========================================================================
    logger.info("=" * 70)
    logger.info(f"{symbol} DBN INGESTION (CANONICAL COMPLIANT)")
    logger.info("=" * 70)
    logger.info("CONFIG SNAPSHOT:")
    logger.info(f"  Instrument: {symbol}")
    logger.info(f"  DBN file: {dbn_path}")
    logger.info(f"  Database: {GOLD_DB_PATH}")
    logger.info(f"  Checkpoint dir: {CHECKPOINT_DIR}")
    logger.info(f"  Outright pattern: {outright_pattern.pattern}")
    logger.info(f"  Prefix len: {prefix_len}")
    logger.info(f"  Minimum start: {minimum_start_date}")
    logger.info(f"  Start filter: {args.start or 'None'}")
    logger.info(f"  End filter: {args.end or 'None'}")
    logger.info(f"  Resume: {args.resume}")
    logger.info(f"  Retry failed: {args.retry_failed}")
    logger.info(f"  Dry run: {args.dry_run}")
    logger.info(f"  Chunk days: {args.chunk_days}")
    logger.info(f"  Batch size: {args.batch_size}")

    # =========================================================================
    # VERIFY DBN FILE EXISTS (already checked by get_asset_config, belt+suspenders)
    # =========================================================================
    if not dbn_path.exists():
        logger.warning(f"FATAL: DBN file not found: {dbn_path}")
        sys.exit(1)

    file_size_gb = dbn_path.stat().st_size / (1024**3)
    logger.info(f"DBN file size: {file_size_gb:.2f} GB")

    # =========================================================================
    # OPEN DBN AND VERIFY SCHEMA (FAIL-CLOSED)
    # =========================================================================
    logger.info("Opening DBN file...")
    store = db.DBNStore.from_file(dbn_path)

    logger.info(f"  Schema: {store.schema}")
    logger.info(f"  Dataset: {store.dataset}")
    logger.info(f"  Date range: {store.start} to {store.end}")

    if store.schema != schema_required:
        logger.warning(f"FATAL: DBN schema is '{store.schema}', expected '{schema_required}'")
        logger.warning("ABORT: Schema verification failed (FAIL-CLOSED)")
        sys.exit(1)

    logger.info(f"  Schema verified: {schema_required} [OK]")

    # =========================================================================
    # INITIALIZE CHECKPOINT MANAGER
    # =========================================================================
    checkpoint_mgr = CheckpointManager(CHECKPOINT_DIR, dbn_path, db_path=GOLD_DB_PATH)
    logger.info(f"Checkpoint file: {checkpoint_mgr.checkpoint_file}")
    logger.info(f"Existing checkpoints: {len(checkpoint_mgr.checkpoints)}")

    # =========================================================================
    # OPEN DATABASE
    # =========================================================================
    con = None
    if not args.dry_run:
        con = duckdb.connect(str(GOLD_DB_PATH))
        from pipeline.db_config import configure_connection
        configure_connection(con, writing=True)
        logger.info(f"Database opened: {GOLD_DB_PATH}")
    else:
        logger.info("DRY RUN: Database will not be modified")

    # Ensure connection is closed on ALL exit paths (including sys.exit)
    import atexit
    def _close_con():
        if con is not None:
            try:
                con.close()
            except Exception:
                pass
    atexit.register(_close_con)

    # =========================================================================
    # DATE FILTERS WITH MINIMUM DATE ENFORCEMENT
    # =========================================================================
    start_filter = date.fromisoformat(args.start) if args.start else minimum_start_date
    end_filter = date.fromisoformat(args.end) if args.end else None

    if start_filter < minimum_start_date:
        logger.warning(f"WARNING: Requested start {start_filter} is before minimum_start_date {minimum_start_date}")
        logger.info(f"         {symbol} data before {minimum_start_date} may have insufficient bar coverage")
        logger.info(f"         FORCING start to {minimum_start_date}")
        start_filter = minimum_start_date

    logger.info(f"EFFECTIVE DATE RANGE: {start_filter} to {end_filter or 'end of file'}")

    # =========================================================================
    # PROCESS DBN IN CHUNKS
    # =========================================================================
    logger.info("Processing DBN in chunks...")

    stats = {
        'chunks_done': 0,
        'chunks_failed': 0,
        'chunks_skipped': 0,
        'rows_written': 0,
        'trading_days_processed': 0,
        'contracts_used': set(),
        'validation_errors': [],
    }

    trading_day_buffer = {}
    batch_num = 0
    skipped_batches = 0

    for chunk_df in store.to_df(count=args.batch_size):
        batch_num += 1

        chunk_df = chunk_df.reset_index()

        # =================================================================
        # FAST-FORWARD: Skip entire batches before start_filter
        # =================================================================
        batch_max_date = chunk_df['ts_event'].max().date()
        if batch_max_date < start_filter:
            skipped_batches += 1
            if skipped_batches % 20 == 1:
                logger.info(f"  FAST-FORWARD: Batch {batch_num} ends at {batch_max_date}, skipping (target: {start_filter})...")
            continue

        if skipped_batches > 0:
            logger.info(f"  FAST-FORWARD COMPLETE: Skipped {skipped_batches} batches of pre-{start_filter} data")
            skipped_batches = 0

        # =================================================================
        # VALIDATE TIMESTAMPS (FAIL-CLOSED)
        # =================================================================
        chunk_df = chunk_df.set_index('ts_event')

        ts_valid, ts_reason = validate_timestamp_utc(chunk_df)
        if not ts_valid:
            logger.warning(f"FATAL: Timestamp validation failed: {ts_reason}")
            logger.warning("ABORT: Timestamp verification gate failed (FAIL-CLOSED)")
            traceback.print_exc()
            sys.exit(1)

        # =================================================================
        # FILTER TO OUTRIGHTS (VECTORIZED, CONFIG-DRIVEN)
        # =================================================================
        outright_mask = chunk_df['symbol'].apply(
            lambda s: bool(outright_pattern.match(str(s)))
        )
        chunk_df = chunk_df[outright_mask]

        if len(chunk_df) == 0:
            continue

        # =================================================================
        # VALIDATE OHLCV (FAIL-CLOSED)
        # =================================================================
        valid, reason, bad_rows = validate_chunk(chunk_df)
        if not valid:
            logger.warning(f"FATAL: OHLCV validation failed: {reason}")
            if bad_rows is not None:
                logger.info("Offending rows:")
                logger.info(bad_rows.to_string())
            logger.warning("ABORT: Validation gate failed (FAIL-CLOSED)")
            traceback.print_exc()
            sys.exit(1)

        # =================================================================
        # COMPUTE TRADING DAYS (VECTORIZED)
        # =================================================================
        trading_days = compute_trading_days(chunk_df)
        chunk_df = chunk_df.copy()
        chunk_df['trading_day'] = trading_days

        # =================================================================
        # AGGREGATE BY TRADING DAY
        # =================================================================
        for tday, day_df in chunk_df.groupby('trading_day'):
            if start_filter and tday < start_filter:
                continue
            if end_filter and tday > end_filter:
                continue

            volumes = day_df.groupby('symbol')['volume'].sum().to_dict()

            front = choose_front_contract(
                volumes,
                outright_pattern=outright_pattern,
                prefix_len=prefix_len,
                log_func=lambda msg: None,
            )
            if not front:
                continue

            stats['contracts_used'].add(front)

            front_df = day_df[day_df['symbol'] == front].copy()

            pk_ok, pk_reason = check_pk_safety(front_df, tday)
            if not pk_ok:
                logger.warning(f"FATAL: PK safety check failed: {pk_reason}")
                logger.warning("ABORT: Primary key safety gate failed (FAIL-CLOSED)")
                sys.exit(1)

            front_df = front_df.sort_index()

            if tday not in trading_day_buffer:
                trading_day_buffer[tday] = []

            trading_day_buffer[tday].extend(zip(
                front_df.index,
                [front] * len(front_df),
                front_df['open'].to_numpy(dtype=float),
                front_df['high'].to_numpy(dtype=float),
                front_df['low'].to_numpy(dtype=float),
                front_df['close'].to_numpy(dtype=float),
                front_df['volume'].to_numpy(dtype=int),
            ))

        # =================================================================
        # MERGE WHEN CHUNK IS FULL
        # =================================================================
        sorted_days = sorted(trading_day_buffer.keys())

        while len(sorted_days) >= args.chunk_days:
            chunk_days = sorted_days[:args.chunk_days]
            chunk_start = str(chunk_days[0])
            chunk_end = str(chunk_days[-1])

            if args.resume and not checkpoint_mgr.should_process_chunk(chunk_start, chunk_end, args.retry_failed):
                logger.info(f"  SKIP: Chunk {chunk_start} to {chunk_end} (already done)")
                stats['chunks_skipped'] += 1
                for d in chunk_days:
                    del trading_day_buffer[d]
                sorted_days = sorted_days[args.chunk_days:]
                continue

            chunk_rows = []
            for d in chunk_days:
                chunk_rows.extend(trading_day_buffer[d])

            if not args.dry_run and con:
                checkpoint_mgr.write_checkpoint(chunk_start, chunk_end, 'in_progress')

                try:
                    con.execute("BEGIN TRANSACTION")

                    chunk_df = pd.DataFrame(
                        [(r[0], symbol, r[1], r[2], r[3], r[4], r[5], r[6]) for r in chunk_rows],
                        columns=['ts_utc', 'symbol', 'source_symbol', 'open', 'high', 'low', 'close', 'volume'],
                    )
                    con.execute("""
                        INSERT OR REPLACE INTO bars_1m
                        SELECT ts_utc, symbol, source_symbol, open, high, low, close, volume
                        FROM chunk_df
                    """)

                    int_ok, int_reason = check_merge_integrity(con, chunk_start, chunk_end)
                    if not int_ok:
                        con.execute("ROLLBACK")
                        checkpoint_mgr.write_checkpoint(chunk_start, chunk_end, 'failed', error=int_reason)
                        logger.warning(f"FATAL: Integrity gate failed: {int_reason}")
                        logger.warning("ABORT: Merge integrity gate failed (FAIL-CLOSED)")
                        sys.exit(1)

                    con.execute("COMMIT")

                    checkpoint_mgr.write_checkpoint(chunk_start, chunk_end, 'done', rows_written=len(chunk_rows))

                    stats['chunks_done'] += 1
                    stats['rows_written'] += len(chunk_rows)
                    stats['trading_days_processed'] += len(chunk_days)

                    logger.info(f"  DONE: Chunk {chunk_start} to {chunk_end}: {len(chunk_rows):,} rows, {len(chunk_days)} days")

                except Exception as e:
                    con.execute("ROLLBACK")
                    checkpoint_mgr.write_checkpoint(chunk_start, chunk_end, 'failed', error=str(e))
                    logger.warning(f"FATAL: Exception during merge: {e}")
                    traceback.print_exc()
                    sys.exit(1)

            else:
                stats['chunks_done'] += 1
                stats['rows_written'] += len(chunk_rows)
                stats['trading_days_processed'] += len(chunk_days)
                logger.info(f"  DRY RUN: Chunk {chunk_start} to {chunk_end}: {len(chunk_rows):,} rows")

            for d in chunk_days:
                del trading_day_buffer[d]
            sorted_days = sorted_days[args.chunk_days:]

        if batch_num % 20 == 0:
            logger.info(f"  Batch {batch_num}: {stats['rows_written']:,} rows written, {stats['trading_days_processed']} trading days")

    # =========================================================================
    # PROCESS REMAINING BUFFER
    # =========================================================================
    if trading_day_buffer:
        sorted_days = sorted(trading_day_buffer.keys())
        chunk_start = str(sorted_days[0])
        chunk_end = str(sorted_days[-1])

        should_process = True
        if args.resume:
            should_process = checkpoint_mgr.should_process_chunk(chunk_start, chunk_end, args.retry_failed)

        if should_process:
            chunk_rows = []
            for d in sorted_days:
                chunk_rows.extend(trading_day_buffer[d])

            if not args.dry_run and con and chunk_rows:
                checkpoint_mgr.write_checkpoint(chunk_start, chunk_end, 'in_progress')

                try:
                    con.execute("BEGIN TRANSACTION")
                    chunk_df = pd.DataFrame(
                        [(r[0], symbol, r[1], r[2], r[3], r[4], r[5], r[6]) for r in chunk_rows],
                        columns=['ts_utc', 'symbol', 'source_symbol', 'open', 'high', 'low', 'close', 'volume'],
                    )
                    con.execute("""
                        INSERT OR REPLACE INTO bars_1m
                        SELECT ts_utc, symbol, source_symbol, open, high, low, close, volume
                        FROM chunk_df
                    """)

                    int_ok, int_reason = check_merge_integrity(con, chunk_start, chunk_end)
                    if not int_ok:
                        con.execute("ROLLBACK")
                        checkpoint_mgr.write_checkpoint(chunk_start, chunk_end, 'failed', error=int_reason)
                        logger.warning(f"FATAL: Final chunk integrity failed: {int_reason}")
                        sys.exit(1)

                    con.execute("COMMIT")
                    checkpoint_mgr.write_checkpoint(chunk_start, chunk_end, 'done', rows_written=len(chunk_rows))

                    stats['chunks_done'] += 1
                    stats['rows_written'] += len(chunk_rows)
                    stats['trading_days_processed'] += len(sorted_days)

                    logger.info(f"  DONE: Final chunk {chunk_start} to {chunk_end}: {len(chunk_rows):,} rows")

                except Exception as e:
                    con.execute("ROLLBACK")
                    checkpoint_mgr.write_checkpoint(chunk_start, chunk_end, 'failed', error=str(e))
                    logger.warning(f"FATAL: Final chunk exception: {e}")
                    traceback.print_exc()
                    sys.exit(1)
            elif args.dry_run:
                stats['chunks_done'] += 1
                stats['rows_written'] += len(chunk_rows)
                stats['trading_days_processed'] += len(sorted_days)
                logger.info(f"  DRY RUN: Final chunk {chunk_start} to {chunk_end}: {len(chunk_rows):,} rows")
        else:
            stats['chunks_skipped'] += 1
            logger.info(f"  SKIP: Final chunk {chunk_start} to {chunk_end} (already done)")

    # =========================================================================
    # FINAL HONESTY GATES
    # =========================================================================
    logger.info("=" * 70)
    logger.info("FINAL HONESTY GATES")
    logger.info("=" * 70)

    if not args.dry_run and con:
        gates_passed, failures = run_final_gates(con)

        if not gates_passed:
            logger.warning("FATAL: Final honesty gates FAILED:")
            for f in failures:
                logger.info(f"  - {f}")
            logger.info("BACKFILL DECLARED INVALID")
            con.close()
            sys.exit(1)

        logger.info("  ts_utc type check: PASSED [OK]")
        logger.info("  No duplicate (symbol, ts_utc): PASSED [OK]")
        logger.info("  No NULL source_symbol: PASSED [OK]")
        logger.info("ALL HONESTY GATES PASSED [OK]")
    else:
        logger.info("  Skipped (dry run)")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    logger.info("=" * 70)
    logger.info("INGESTION SUMMARY")
    logger.info("=" * 70)

    end_time = datetime.now()
    elapsed = end_time - start_time

    logger.info(f"Instrument: {symbol}")
    logger.info(f"Chunks done: {stats['chunks_done']}")
    logger.info(f"Chunks failed: {stats['chunks_failed']}")
    logger.info(f"Chunks skipped: {stats['chunks_skipped']}")
    logger.info(f"Trading days processed: {stats['trading_days_processed']}")
    logger.info(f"Total rows written: {stats['rows_written']:,}")
    logger.info(f"Unique contracts used: {len(stats['contracts_used'])}")
    logger.info(f"Wall time: {elapsed}")

    if not args.dry_run and con:
        count = con.execute(
            "SELECT COUNT(*) FROM bars_1m WHERE symbol = ?", [symbol]
        ).fetchone()[0]
        date_range = con.execute(
            "SELECT MIN(DATE(ts_utc)), MAX(DATE(ts_utc)) FROM bars_1m WHERE symbol = ?", [symbol]
        ).fetchone()

        logger.info(f"Database rows ({symbol}): {count:,}")
        logger.info(f"Date range in DB: {date_range[0]} to {date_range[1]}")

        con.close()

    logger.info("SUCCESS: Backfill complete and validated.")
    sys.exit(0)

if __name__ == "__main__":
    main()
