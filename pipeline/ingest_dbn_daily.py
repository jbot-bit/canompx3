#!/usr/bin/env python3
"""
Ingest daily DBN files into bars_1m table.

Handles the case where data is split into individual daily .dbn.zst files
(as downloaded from Databento with split_duration=day).

Key difference from ingest_dbn_mgc.py:
- Iterates over individual daily files instead of one monolithic file
- Handles stype_out=instrument_id by loading symbology.json mapping
- Filters to MGC contracts only (data includes GC.FUT too)
- Reuses all validation functions from ingest_dbn_mgc.py

Usage:
    python pipeline/ingest_dbn_daily.py --start 2021-02-05 --end 2026-02-04
    python pipeline/ingest_dbn_daily.py --resume
    python pipeline/ingest_dbn_daily.py --dry-run
"""

import os
import sys
import re
import json
import argparse
from pathlib import Path
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import duckdb
import databento as db

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.paths import GOLD_DB_PATH, DAILY_DBN_DIR
from pipeline.ingest_dbn_mgc import (
    validate_chunk,
    validate_timestamp_utc,
    choose_front_contract,
    compute_trading_days,
    check_pk_safety,
    check_merge_integrity,
    run_final_gates,
    CheckpointManager,
    MGC_OUTRIGHT_PATTERN,
    MINIMUM_START_DATE,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

SYMBOL = "MGC"
TZ_LOCAL = ZoneInfo("Australia/Brisbane")
TZ_UTC = ZoneInfo("UTC")

# Checkpoint directory
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"

# Daily file pattern
DAILY_FILE_PATTERN = re.compile(r'glbx-mdp3-(\d{8})\.ohlcv-1m\.dbn\.zst')


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
        print(f"FATAL: symbology.json not found in {data_dir}")
        sys.exit(1)

    with open(symb_path, 'r') as f:
        data = json.load(f)

    # Build reverse mapping: instrument_id -> contract_name
    id_to_name = {}
    for contract_name, entries in data.get("result", {}).items():
        for entry in entries:
            inst_id = str(entry["s"])
            # If multiple contracts map to same ID, keep the one already there
            # (shouldn't happen but be safe)
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
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD")
    parser.add_argument("--data-dir", type=str, default=str(DAILY_DBN_DIR),
                        help=f"Directory with daily .dbn.zst files (default: {DAILY_DBN_DIR})")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed days")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, no DB writes")
    parser.add_argument("--chunk-days", type=int, default=7,
                        help="Trading days per commit (default: 7)")
    args = parser.parse_args()

    start_time = datetime.now()
    data_dir = Path(args.data_dir)

    # =========================================================================
    # STARTUP
    # =========================================================================
    print("=" * 70)
    print("DAILY DBN INGESTION (CANONICAL COMPLIANT)")
    print("=" * 70)
    print()
    print("CONFIG SNAPSHOT:")
    print(f"  Data dir: {data_dir}")
    print(f"  Database: {GOLD_DB_PATH}")
    print(f"  Start: {args.start or 'earliest'}")
    print(f"  End: {args.end or 'latest'}")
    print(f"  Resume: {args.resume}")
    print(f"  Dry run: {args.dry_run}")
    print(f"  Chunk days: {args.chunk_days}")
    print()

    # =========================================================================
    # VERIFY DATA DIRECTORY
    # =========================================================================
    if not data_dir.exists():
        print(f"FATAL: Data directory not found: {data_dir}")
        sys.exit(1)

    # =========================================================================
    # LOAD SYMBOLOGY (instrument_id -> contract_name)
    # =========================================================================
    print("Loading symbology mapping...")
    id_to_name = load_symbology(data_dir)
    mgc_ids = {k for k, v in id_to_name.items() if MGC_OUTRIGHT_PATTERN.match(v)}
    print(f"  Total mappings: {len(id_to_name)}")
    print(f"  MGC contract IDs: {len(mgc_ids)}")
    print()

    # =========================================================================
    # DISCOVER FILES
    # =========================================================================
    start_filter = date.fromisoformat(args.start) if args.start else MINIMUM_START_DATE
    end_filter = date.fromisoformat(args.end) if args.end else date(2030, 12, 31)

    # Enforce minimum date
    if start_filter < MINIMUM_START_DATE:
        print(f"WARNING: Start {start_filter} before MINIMUM_START_DATE {MINIMUM_START_DATE}")
        print(f"         Forcing start to {MINIMUM_START_DATE}")
        start_filter = MINIMUM_START_DATE

    daily_files = discover_daily_files(data_dir, start_filter, end_filter)
    print(f"Found {len(daily_files)} daily files in range {start_filter} to {end_filter}")
    if daily_files:
        print(f"  First: {daily_files[0][0]} ({daily_files[0][1].name})")
        print(f"  Last:  {daily_files[-1][0]} ({daily_files[-1][1].name})")
    print()

    if not daily_files:
        print("No files to process.")
        sys.exit(0)

    # =========================================================================
    # INITIALIZE CHECKPOINT (keyed by first file for identity)
    # =========================================================================
    checkpoint_mgr = CheckpointManager(CHECKPOINT_DIR, daily_files[0][1])
    print(f"Checkpoint file: {checkpoint_mgr.checkpoint_file}")
    print()

    # =========================================================================
    # OPEN DATABASE
    # =========================================================================
    con = None
    if not args.dry_run:
        con = duckdb.connect(str(GOLD_DB_PATH))
        print(f"Database opened: {GOLD_DB_PATH}")
    else:
        print("DRY RUN: Database will not be modified")
    print()

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

    trading_day_buffer = {}  # trading_day -> list of (ts_utc, source_symbol, o, h, l, c, v)

    for file_idx, (file_date, fpath) in enumerate(daily_files):
        # Progress
        if (file_idx + 1) % 50 == 0:
            print(f"  Progress: {file_idx + 1}/{len(daily_files)} files, "
                  f"{stats['rows_written']:,} rows written")

        try:
            # Open daily DBN file
            store = db.DBNStore.from_file(fpath)

            # Verify schema
            if store.schema != 'ohlcv-1m':
                print(f"FATAL: {fpath.name} schema is '{store.schema}', expected 'ohlcv-1m'")
                sys.exit(1)

            # Read all bars from this daily file (daily files are small, ~1K-3K rows)
            chunk_df = store.to_df()
            if len(chunk_df) == 0:
                continue

            chunk_df = chunk_df.reset_index()

            # =================================================================
            # MAP INSTRUMENT IDS TO CONTRACT NAMES
            # =================================================================
            chunk_df['symbol'] = chunk_df['symbol'].astype(str).map(id_to_name)

            # Drop rows where mapping failed (unknown instruments)
            unmapped = chunk_df['symbol'].isna()
            if unmapped.any():
                chunk_df = chunk_df[~unmapped]

            if len(chunk_df) == 0:
                continue

            # =================================================================
            # FILTER TO MGC OUTRIGHTS ONLY (exclude GC.FUT and spreads)
            # =================================================================
            outright_mask = chunk_df['symbol'].apply(
                lambda s: bool(MGC_OUTRIGHT_PATTERN.match(str(s)))
            )
            chunk_df = chunk_df[outright_mask]

            if len(chunk_df) == 0:
                continue

            # =================================================================
            # SET INDEX AND VALIDATE
            # =================================================================
            chunk_df = chunk_df.set_index('ts_event')

            ts_valid, ts_reason = validate_timestamp_utc(chunk_df)
            if not ts_valid:
                print(f"FATAL: {fpath.name} timestamp validation failed: {ts_reason}")
                sys.exit(1)

            valid, reason, bad_rows = validate_chunk(chunk_df)
            if not valid:
                print(f"FATAL: {fpath.name} OHLCV validation failed: {reason}")
                if bad_rows is not None:
                    print(bad_rows.to_string())
                sys.exit(1)

            # =================================================================
            # COMPUTE TRADING DAYS AND AGGREGATE
            # =================================================================
            trading_days = compute_trading_days(chunk_df)
            chunk_df = chunk_df.copy()
            chunk_df['trading_day'] = trading_days

            for tday, day_df in chunk_df.groupby('trading_day'):
                if tday < start_filter:
                    continue
                if tday > end_filter:
                    continue

                volumes = day_df.groupby('symbol')['volume'].sum().to_dict()
                front = choose_front_contract(
                    volumes, outright_pattern=MGC_OUTRIGHT_PATTERN,
                    prefix_len=3, log_func=lambda msg: None
                )
                if not front:
                    continue

                stats['contracts_used'].add(front)
                front_df = day_df[day_df['symbol'] == front].copy()

                pk_ok, pk_reason = check_pk_safety(front_df, tday)
                if not pk_ok:
                    print(f"FATAL: PK safety failed for {fpath.name}: {pk_reason}")
                    sys.exit(1)

                front_df = front_df.sort_index()

                if tday not in trading_day_buffer:
                    trading_day_buffer[tday] = []

                for ts_utc, row in front_df.iterrows():
                    trading_day_buffer[tday].append((
                        ts_utc, front,
                        float(row['open']), float(row['high']),
                        float(row['low']), float(row['close']),
                        int(row['volume']),
                    ))

            stats['files_processed'] += 1

        except Exception as e:
            print(f"ERROR processing {fpath.name}: {e}")
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

            chunk_rows = []
            for d in chunk_days_list:
                chunk_rows.extend(trading_day_buffer[d])

            if not args.dry_run and con and chunk_rows:
                checkpoint_mgr.write_checkpoint(chunk_start, chunk_end, 'in_progress')
                try:
                    con.execute("BEGIN TRANSACTION")
                    con.executemany(
                        """INSERT OR REPLACE INTO bars_1m
                        (ts_utc, symbol, source_symbol, open, high, low, close, volume)
                        VALUES (?, 'MGC', ?, ?, ?, ?, ?, ?)""",
                        chunk_rows
                    )

                    int_ok, int_reason = check_merge_integrity(con, chunk_start, chunk_end)
                    if not int_ok:
                        con.execute("ROLLBACK")
                        checkpoint_mgr.write_checkpoint(
                            chunk_start, chunk_end, 'failed', error=int_reason
                        )
                        print(f"FATAL: Integrity failed: {int_reason}")
                        sys.exit(1)

                    con.execute("COMMIT")
                    checkpoint_mgr.write_checkpoint(
                        chunk_start, chunk_end, 'done', rows_written=len(chunk_rows)
                    )

                    stats['chunks_done'] += 1
                    stats['rows_written'] += len(chunk_rows)
                    stats['trading_days_processed'] += len(chunk_days_list)

                    print(f"  DONE: {chunk_start} to {chunk_end}: "
                          f"{len(chunk_rows):,} rows, {len(chunk_days_list)} days")

                except Exception as e:
                    con.execute("ROLLBACK")
                    checkpoint_mgr.write_checkpoint(
                        chunk_start, chunk_end, 'failed', error=str(e)
                    )
                    print(f"FATAL: Exception during merge: {e}")
                    sys.exit(1)
            else:
                stats['chunks_done'] += 1
                stats['rows_written'] += len(chunk_rows)
                stats['trading_days_processed'] += len(chunk_days_list)
                if chunk_rows:
                    print(f"  DRY RUN: {chunk_start} to {chunk_end}: {len(chunk_rows):,} rows")

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

        chunk_rows = []
        for d in sorted_days:
            chunk_rows.extend(trading_day_buffer[d])

        if not args.dry_run and con and chunk_rows:
            checkpoint_mgr.write_checkpoint(chunk_start, chunk_end, 'in_progress')
            try:
                con.execute("BEGIN TRANSACTION")
                con.executemany(
                    """INSERT OR REPLACE INTO bars_1m
                    (ts_utc, symbol, source_symbol, open, high, low, close, volume)
                    VALUES (?, 'MGC', ?, ?, ?, ?, ?, ?)""",
                    chunk_rows
                )

                int_ok, int_reason = check_merge_integrity(con, chunk_start, chunk_end)
                if not int_ok:
                    con.execute("ROLLBACK")
                    checkpoint_mgr.write_checkpoint(
                        chunk_start, chunk_end, 'failed', error=int_reason
                    )
                    print(f"FATAL: Final chunk integrity failed: {int_reason}")
                    sys.exit(1)

                con.execute("COMMIT")
                checkpoint_mgr.write_checkpoint(
                    chunk_start, chunk_end, 'done', rows_written=len(chunk_rows)
                )

                stats['chunks_done'] += 1
                stats['rows_written'] += len(chunk_rows)
                stats['trading_days_processed'] += len(sorted_days)

                print(f"  DONE: Final {chunk_start} to {chunk_end}: "
                      f"{len(chunk_rows):,} rows")

            except Exception as e:
                con.execute("ROLLBACK")
                checkpoint_mgr.write_checkpoint(
                    chunk_start, chunk_end, 'failed', error=str(e)
                )
                print(f"FATAL: Final chunk exception: {e}")
                sys.exit(1)
        elif args.dry_run and chunk_rows:
            stats['chunks_done'] += 1
            stats['rows_written'] += len(chunk_rows)
            stats['trading_days_processed'] += len(sorted_days)
            print(f"  DRY RUN: Final {chunk_start} to {chunk_end}: {len(chunk_rows):,} rows")

    # =========================================================================
    # FINAL GATES
    # =========================================================================
    print()
    print("=" * 70)
    print("FINAL HONESTY GATES")
    print("=" * 70)

    if not args.dry_run and con:
        gates_passed, failures = run_final_gates(con)
        if not gates_passed:
            print("FATAL: Final honesty gates FAILED:")
            for f in failures:
                print(f"  - {f}")
            con.close()
            sys.exit(1)

        print("  ts_utc type check: PASSED [OK]")
        print("  No duplicate (symbol, ts_utc): PASSED [OK]")
        print("  No NULL source_symbol: PASSED [OK]")
        print()
        print("ALL HONESTY GATES PASSED [OK]")
    else:
        print("  Skipped (dry run)")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print()
    print("=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)
    print()

    elapsed = datetime.now() - start_time
    print(f"Files processed: {stats['files_processed']}")
    print(f"Files skipped: {stats['files_skipped']}")
    print(f"Files failed: {stats['files_failed']}")
    print(f"Chunks committed: {stats['chunks_done']}")
    print(f"Trading days: {stats['trading_days_processed']}")
    print(f"Total rows: {stats['rows_written']:,}")
    print(f"Unique contracts: {len(stats['contracts_used'])}")
    print(f"Wall time: {elapsed}")
    print()

    if not args.dry_run and con:
        count = con.execute(
            "SELECT COUNT(*) FROM bars_1m WHERE symbol = 'MGC'"
        ).fetchone()[0]
        date_range = con.execute(
            "SELECT MIN(DATE(ts_utc)), MAX(DATE(ts_utc)) FROM bars_1m WHERE symbol = 'MGC'"
        ).fetchone()
        print(f"Database rows (MGC): {count:,}")
        print(f"Date range: {date_range[0]} to {date_range[1]}")
        con.close()

    print()
    print("SUCCESS: Daily ingestion complete and validated.")
    sys.exit(0)


if __name__ == "__main__":
    main()
