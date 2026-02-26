#!/usr/bin/env python3
"""
Ingest daily DBN files into bars_1m table (multi-instrument).

Handles the case where data is split into individual daily .dbn.zst files
(as downloaded from Databento with split_duration=day).

Key difference from ingest_dbn_mgc.py:
- Iterates over individual daily files instead of one monolithic file
- Handles stype_out=instrument_id by loading symbology.json mapping
- Supports multiple instruments via --instrument flag
- Reuses all validation functions from ingest_dbn_mgc.py

Special case: MGC data contains GC (full-size Gold) outrights, stored as
symbol='MGC'. This is because GC has better bar coverage. Other instruments
(MCL, MNQ) use their own outrights directly.

Usage:
    python pipeline/ingest_dbn_daily.py --instrument MGC --start 2021-02-05 --end 2026-02-04
    python pipeline/ingest_dbn_daily.py --instrument MCL --start 2021-07-11 --end 2026-02-10
    python pipeline/ingest_dbn_daily.py --resume
    python pipeline/ingest_dbn_daily.py --dry-run
"""

import sys
import re
import json
import argparse
from pathlib import Path
from datetime import datetime, date
from zoneinfo import ZoneInfo

# Force unbuffered stdout so progress prints appear immediately
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import duckdb
import databento as db

# Add project root to path

from pipeline.paths import DAILY_DBN_DIR, GOLD_DB_PATH
from pipeline.asset_configs import get_asset_config
from pipeline.ingest_dbn_mgc import (
    validate_chunk,
    validate_timestamp_utc,
    choose_front_contract,
    compute_trading_days,
    check_pk_safety,
    check_merge_integrity,
    run_final_gates,
    CheckpointManager,
    GC_OUTRIGHT_PATTERN,
    MINIMUM_START_DATE,
)

from pipeline.log import get_logger
logger = get_logger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

TZ_LOCAL = ZoneInfo("Australia/Brisbane")
TZ_UTC = ZoneInfo("UTC")

# Checkpoint directory
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"

# Daily file pattern
DAILY_FILE_PATTERN = re.compile(r'glbx-mdp3-(\d{8})\.ohlcv-1m\.dbn\.zst')

# =============================================================================
# INSTRUMENT-SPECIFIC CONFIG
# =============================================================================

def get_ingest_config(instrument: str) -> dict:
    """
    Return ingestion-specific config for an instrument.

    MGC is a special case: data contains GC outrights (better bar coverage),
    stored under symbol='MGC'. All other instruments use their own outrights.

    Returns dict with keys: symbol, outright_pattern, prefix_len,
    minimum_start_date, data_dir.
    """
    config = get_asset_config(instrument)

    if instrument.upper() == "MGC":
        # MGC data source is GC outrights (2-char prefix: GC)
        return {
            "symbol": "MGC",
            "outright_pattern": GC_OUTRIGHT_PATTERN,
            "prefix_len": 2,
            "minimum_start_date": MINIMUM_START_DATE,
            "data_dir": DAILY_DBN_DIR,
        }
    else:
        return {
            "symbol": config["symbol"],
            "outright_pattern": config["outright_pattern"],
            "prefix_len": config["prefix_len"],
            "minimum_start_date": config["minimum_start_date"],
            "data_dir": config["dbn_path"],
        }

# =============================================================================
# SYMBOLOGY MAPPING
# =============================================================================

def load_symbology(data_dir: Path) -> dict:
    """
    Load symbology.json and build instrument_id -> contract_name mapping.

    The symbology maps contract names to instrument IDs with date ranges.
    We invert it to map IDs back to names.

    Returns dict: {instrument_id_str: contract_name}
    """
    symb_path = data_dir / "symbology.json"
    if not symb_path.exists():
        logger.warning(f"FATAL: symbology.json not found in {data_dir}")
        sys.exit(1)

    with open(symb_path, 'r') as f:
        data = json.load(f)

    # Build reverse mapping: instrument_id -> contract_name
    id_to_name = {}
    for contract_name, entries in data.get("result", {}).items():
        for entry in entries:
            inst_id = str(entry["s"])
            if inst_id not in id_to_name:
                id_to_name[inst_id] = contract_name

    return id_to_name

def discover_daily_files(data_dir: Path, start_date: date, end_date: date) -> list[tuple[date, Path]]:
    """
    Discover and sort daily DBN files within date range.

    Returns list of (file_date, file_path) sorted by date.
    """
    files = []

    for fpath in data_dir.iterdir():
        match = DAILY_FILE_PATTERN.match(fpath.name)
        if not match:
            continue

        file_date = date.fromisoformat(f"{match.group(1)[:4]}-{match.group(1)[4:6]}-{match.group(1)[6:8]}")

        if file_date < start_date or file_date > end_date:
            continue

        files.append((file_date, fpath))

    files.sort(key=lambda x: x[0])
    return files

# =============================================================================
# MAIN INGESTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Ingest daily DBN files into bars_1m")
    parser.add_argument("--instrument", type=str, default="MGC",
                        help="Instrument to ingest (default: MGC)")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory (default: from asset config)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed days")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, no DB writes")
    parser.add_argument("--chunk-days", type=int, default=50,
                        help="Trading days per commit (default: 50)")
    parser.add_argument("--db", type=str, default=None,
                        help="Database path (default: gold.db)")
    args = parser.parse_args()

    start_time = datetime.now()

    # Load instrument config
    ingest_cfg = get_ingest_config(args.instrument)
    symbol = ingest_cfg["symbol"]
    outright_pattern = ingest_cfg["outright_pattern"]
    prefix_len = ingest_cfg["prefix_len"]
    minimum_start_date = ingest_cfg["minimum_start_date"]
    data_dir = Path(args.data_dir) if args.data_dir else ingest_cfg["data_dir"]

    # Resolve DB path
    db_path = Path(args.db) if args.db else GOLD_DB_PATH

    # Build regex string for vectorized str.match
    outright_regex = outright_pattern.pattern

    # =========================================================================
    # STARTUP
    # =========================================================================
    logger.info("=" * 70)
    logger.info(f"{symbol} DAILY DBN INGESTION (CANONICAL COMPLIANT)")
    logger.info("=" * 70)
    logger.info("CONFIG SNAPSHOT:")
    logger.info(f"  Instrument: {symbol}")
    logger.info(f"  Data dir: {data_dir}")
    logger.info(f"  Database: {db_path}")
    logger.info(f"  Outright pattern: {outright_regex}")
    logger.info(f"  Prefix len: {prefix_len}")
    logger.info(f"  Minimum start: {minimum_start_date}")
    logger.info(f"  Start: {args.start or 'earliest'}")
    logger.info(f"  End: {args.end or 'latest'}")
    logger.info(f"  Resume: {args.resume}")
    logger.info(f"  Dry run: {args.dry_run}")
    logger.info(f"  Chunk days: {args.chunk_days}")

    # =========================================================================
    # VERIFY DATA DIRECTORY
    # =========================================================================
    if not data_dir.exists():
        logger.warning(f"FATAL: Data directory not found: {data_dir}")
        sys.exit(1)

    # NOTE: Symbology mapping is NOT needed -- Databento's to_df() already
    # resolves instrument_ids to readable contract names (e.g. "MCLX5").
    # The outright_pattern filter handles everything downstream.

    # =========================================================================
    # DISCOVER FILES
    # =========================================================================
    start_filter = date.fromisoformat(args.start) if args.start else minimum_start_date
    end_filter = date.fromisoformat(args.end) if args.end else date(2030, 12, 31)

    # Enforce minimum date
    if start_filter < minimum_start_date:
        logger.warning(f"WARNING: Start {start_filter} before minimum_start_date {minimum_start_date}")
        logger.info(f"         Forcing start to {minimum_start_date}")
        start_filter = minimum_start_date

    daily_files = discover_daily_files(data_dir, start_filter, end_filter)
    logger.info(f"Found {len(daily_files)} daily files in range {start_filter} to {end_filter}")
    if daily_files:
        logger.info(f"  First: {daily_files[0][0]} ({daily_files[0][1].name})")
        logger.info(f"  Last:  {daily_files[-1][0]} ({daily_files[-1][1].name})")

    if not daily_files:
        logger.info("No files to process.")
        sys.exit(0)

    # =========================================================================
    # INITIALIZE CHECKPOINT (keyed by first file for identity)
    # =========================================================================
    checkpoint_mgr = CheckpointManager(CHECKPOINT_DIR, daily_files[0][1], db_path=db_path)
    logger.info(f"Checkpoint file: {checkpoint_mgr.checkpoint_file}")

    # =========================================================================
    # OPEN DATABASE
    # =========================================================================
    con = None
    if not args.dry_run:
        con = duckdb.connect(str(db_path))
        from pipeline.db_config import configure_connection
        configure_connection(con, writing=True)
        logger.info(f"Database opened: {db_path}")
    else:
        logger.info("DRY RUN: Database will not be modified")

    import atexit
    def _close_con():
        if con is not None:
            try:
                con.close()
            except Exception:
                pass
    atexit.register(_close_con)

    # =========================================================================
    # PROCESS DAILY FILES
    # =========================================================================
    stats = {
        'files_processed': 0,
        'files_skipped': 0,
        'files_failed': 0,
        'chunks_done': 0,
        'rows_written': 0,
        'trading_days_processed': 0,
        'contracts_used': set(),
    }

    # Buffer: accumulate DataFrames per trading day (vectorized, no iterrows)
    trading_day_buffer = {}  # trading_day -> list of DataFrames

    logger.info(f"Processing {len(daily_files)} files...")
    for file_idx, (file_date, fpath) in enumerate(daily_files):
        # Progress every 10 files
        if (file_idx + 1) % 10 == 0 or file_idx == 0:
            pct = (file_idx + 1) / len(daily_files) * 100
            logger.info(f"  [{pct:5.1f}%] File {file_idx + 1}/{len(daily_files)} ({file_date}) -- {stats['rows_written']:,} rows written")

        try:
            # Open daily DBN file
            store = db.DBNStore.from_file(fpath)

            # Verify schema
            if store.schema != 'ohlcv-1m':
                logger.warning(f"FATAL: {fpath.name} schema is '{store.schema}', expected 'ohlcv-1m'")
                sys.exit(1)

            # Read all bars from this daily file (daily files are small, ~1K-3K rows)
            chunk_df = store.to_df()
            if len(chunk_df) == 0:
                continue

            chunk_df = chunk_df.reset_index()

            # =================================================================
            # FILTER TO OUTRIGHTS (vectorized str.match)
            # =================================================================
            outright_mask = chunk_df['symbol'].astype(str).str.match(outright_regex)
            chunk_df = chunk_df[outright_mask]

            if len(chunk_df) == 0:
                continue

            # =================================================================
            # SET INDEX AND VALIDATE
            # =================================================================
            chunk_df = chunk_df.set_index('ts_event')

            ts_valid, ts_reason = validate_timestamp_utc(chunk_df)
            if not ts_valid:
                logger.warning(f"FATAL: {fpath.name} timestamp validation failed: {ts_reason}")
                sys.exit(1)

            valid, reason, bad_rows = validate_chunk(chunk_df)
            if not valid:
                logger.warning(f"FATAL: {fpath.name} OHLCV validation failed: {reason}")
                if bad_rows is not None:
                    logger.info(bad_rows.to_string())
                sys.exit(1)

            # =================================================================
            # COMPUTE TRADING DAYS AND AGGREGATE
            # =================================================================
            trading_days = compute_trading_days(chunk_df)
            chunk_df = chunk_df.copy()
            chunk_df['trading_day'] = trading_days

            for tday, day_df in chunk_df.groupby('trading_day'):
                if tday < start_filter or tday > end_filter:
                    continue

                volumes = day_df.groupby('symbol')['volume'].sum().to_dict()
                front = choose_front_contract(
                    volumes, outright_pattern=outright_pattern,
                    prefix_len=prefix_len, log_func=lambda msg: None
                )
                if not front:
                    continue

                stats['contracts_used'].add(front)
                front_df = day_df[day_df['symbol'] == front].copy()

                pk_ok, pk_reason = check_pk_safety(front_df, tday)
                if not pk_ok:
                    logger.warning(f"FATAL: PK safety failed for {fpath.name}: {pk_reason}")
                    sys.exit(1)

                # Build insert-ready DataFrame (vectorized, no iterrows)
                insert_df = front_df[['open', 'high', 'low', 'close', 'volume']].copy()
                insert_df = insert_df.reset_index()
                insert_df.rename(columns={'ts_event': 'ts_utc'}, inplace=True)
                insert_df['symbol'] = symbol
                insert_df['source_symbol'] = front

                if tday not in trading_day_buffer:
                    trading_day_buffer[tday] = []
                trading_day_buffer[tday].append(insert_df)

            stats['files_processed'] += 1

        except Exception as e:
            logger.warning(f"ERROR processing {fpath.name}: {e}")
            stats['files_failed'] += 1
            continue

        # =====================================================================
        # FLUSH BUFFER WHEN FULL
        # =====================================================================
        sorted_days = sorted(trading_day_buffer.keys())

        while len(sorted_days) >= args.chunk_days:
            chunk_days_list = sorted_days[:args.chunk_days]
            chunk_start = str(chunk_days_list[0])
            chunk_end = str(chunk_days_list[-1])

            if args.resume and not checkpoint_mgr.should_process_chunk(
                chunk_start, chunk_end, args.retry_failed
            ):
                stats['files_skipped'] += len(chunk_days_list)
                for d in chunk_days_list:
                    del trading_day_buffer[d]
                sorted_days = sorted_days[args.chunk_days:]
                continue

            # Concat all buffered DataFrames for this chunk (vectorized)
            chunk_dfs = []
            for d in chunk_days_list:
                chunk_dfs.extend(trading_day_buffer[d])
            if not chunk_dfs:
                for d in chunk_days_list:
                    del trading_day_buffer[d]
                sorted_days = sorted_days[args.chunk_days:]
                continue
            chunk_frame = pd.concat(chunk_dfs, ignore_index=True)
            n_rows = len(chunk_frame)

            if not args.dry_run and con:
                checkpoint_mgr.write_checkpoint(chunk_start, chunk_end, 'in_progress')
                try:
                    con.execute("BEGIN TRANSACTION")
                    # Bulk insert from DataFrame
                    con.execute("""
                        INSERT OR REPLACE INTO bars_1m
                        (ts_utc, symbol, source_symbol, open, high, low, close, volume)
                        SELECT ts_utc, symbol, source_symbol, open, high, low, close, volume
                        FROM chunk_frame
                    """)

                    int_ok, int_reason = check_merge_integrity(con, chunk_start, chunk_end)
                    if not int_ok:
                        con.execute("ROLLBACK")
                        checkpoint_mgr.write_checkpoint(
                            chunk_start, chunk_end, 'failed', error=int_reason
                        )
                        logger.warning(f"FATAL: Integrity failed: {int_reason}")
                        sys.exit(1)

                    con.execute("COMMIT")
                    checkpoint_mgr.write_checkpoint(
                        chunk_start, chunk_end, 'done', rows_written=n_rows
                    )

                    stats['chunks_done'] += 1
                    stats['rows_written'] += n_rows
                    stats['trading_days_processed'] += len(chunk_days_list)

                    logger.info(f"  DONE: {chunk_start} to {chunk_end}: {n_rows:,} rows, {len(chunk_days_list)} days")

                except Exception as e:
                    con.execute("ROLLBACK")
                    checkpoint_mgr.write_checkpoint(
                        chunk_start, chunk_end, 'failed', error=str(e)
                    )
                    logger.warning(f"FATAL: Exception during merge: {e}")
                    sys.exit(1)
            else:
                stats['chunks_done'] += 1
                stats['rows_written'] += n_rows
                stats['trading_days_processed'] += len(chunk_days_list)
                if n_rows:
                    logger.info(f"  DRY RUN: {chunk_start} to {chunk_end}: {n_rows:,} rows")

            for d in chunk_days_list:
                del trading_day_buffer[d]
            sorted_days = sorted_days[args.chunk_days:]

    # =========================================================================
    # FLUSH REMAINING BUFFER
    # =========================================================================
    if trading_day_buffer:
        sorted_days = sorted(trading_day_buffer.keys())
        chunk_start = str(sorted_days[0])
        chunk_end = str(sorted_days[-1])

        chunk_dfs = []
        for d in sorted_days:
            chunk_dfs.extend(trading_day_buffer[d])

        if not chunk_dfs:
            pass  # nothing to flush
        else:
            chunk_frame = pd.concat(chunk_dfs, ignore_index=True)
            n_rows = len(chunk_frame)

            if not args.dry_run and con:
                checkpoint_mgr.write_checkpoint(chunk_start, chunk_end, 'in_progress')
                try:
                    con.execute("BEGIN TRANSACTION")
                    con.execute("""
                        INSERT OR REPLACE INTO bars_1m
                        (ts_utc, symbol, source_symbol, open, high, low, close, volume)
                        SELECT ts_utc, symbol, source_symbol, open, high, low, close, volume
                        FROM chunk_frame
                    """)

                    int_ok, int_reason = check_merge_integrity(con, chunk_start, chunk_end)
                    if not int_ok:
                        con.execute("ROLLBACK")
                        checkpoint_mgr.write_checkpoint(
                            chunk_start, chunk_end, 'failed', error=int_reason
                        )
                        logger.warning(f"FATAL: Final chunk integrity failed: {int_reason}")
                        sys.exit(1)

                    con.execute("COMMIT")
                    checkpoint_mgr.write_checkpoint(
                        chunk_start, chunk_end, 'done', rows_written=n_rows
                    )

                    stats['chunks_done'] += 1
                    stats['rows_written'] += n_rows
                    stats['trading_days_processed'] += len(sorted_days)

                    logger.info(f"  DONE: Final {chunk_start} to {chunk_end}: {n_rows:,} rows")

                except Exception as e:
                    con.execute("ROLLBACK")
                    checkpoint_mgr.write_checkpoint(
                        chunk_start, chunk_end, 'failed', error=str(e)
                    )
                    logger.warning(f"FATAL: Final chunk exception: {e}")
                    sys.exit(1)
            elif args.dry_run:
                stats['chunks_done'] += 1
                stats['rows_written'] += n_rows
                stats['trading_days_processed'] += len(sorted_days)
                logger.info(f"  DRY RUN: Final {chunk_start} to {chunk_end}: {n_rows:,} rows")

    # =========================================================================
    # FINAL GATES
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
            con.close()
            sys.exit(1)

        logger.info("  ts_utc type check: PASSED [OK]")
        logger.info("  No duplicate (symbol, ts_utc): PASSED [OK]")
        logger.info("  No NULL source_symbol: PASSED [OK]")
        logger.info("ALL HONESTY GATES PASSED [OK]")
    else:
        logger.info("  Skipped (dry run)")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    logger.info("=" * 70)
    logger.info("INGESTION SUMMARY")
    logger.info("=" * 70)

    elapsed = datetime.now() - start_time
    logger.info(f"Instrument: {symbol}")
    logger.info(f"Files processed: {stats['files_processed']}")
    logger.info(f"Files skipped: {stats['files_skipped']}")
    logger.info(f"Files failed: {stats['files_failed']}")
    logger.info(f"Chunks committed: {stats['chunks_done']}")
    logger.info(f"Trading days: {stats['trading_days_processed']}")
    logger.info(f"Total rows: {stats['rows_written']:,}")
    logger.info(f"Unique contracts: {len(stats['contracts_used'])}")
    logger.info(f"Wall time: {elapsed}")

    if not args.dry_run and con:
        count = con.execute(
            "SELECT COUNT(*) FROM bars_1m WHERE symbol = ?", [symbol]
        ).fetchone()[0]
        date_range = con.execute(
            "SELECT MIN(DATE(ts_utc)), MAX(DATE(ts_utc)) FROM bars_1m WHERE symbol = ?",
            [symbol]
        ).fetchone()
        logger.info(f"Database rows ({symbol}): {count:,}")
        logger.info(f"Date range: {date_range[0]} to {date_range[1]}")
        con.close()

    if stats['files_failed'] > 0:
        logger.error(
            f"FAIL: {stats['files_failed']} file(s) failed during ingestion. "
            "Data may be incomplete — do NOT proceed with downstream pipeline stages."
        )
        sys.exit(1)

    logger.info(f"SUCCESS: {symbol} daily ingestion complete and validated.")
    sys.exit(0)

if __name__ == "__main__":
    main()
