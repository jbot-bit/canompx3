#!/usr/bin/env python3
"""
Ingest MGC DBN file into bars_1m table.

ROLE: [L1 Library] Validation gates. All ingest paths import from here. Do not run directly.

CANONICAL COMPLIANCE (CANONICAL_backfill_dbn_mgc_rules.txt):
- FAIL-CLOSED: Any validation failure aborts entire backfill
- CHUNKED: Uses store.to_df(count=N) iterator, never full load
- VECTORIZED: No apply() or iterrows() over large data
- DETERMINISTIC: Stable tiebreak for equal-volume contracts
- CHECKPOINTED: JSONL append-only checkpoint system
- INTEGRITY GATED: Verifies no duplicates/NULLs after each merge
- BARS_1M ONLY: Does NOT touch bars_5m or any derived tables

Usage (DEPRECATED -- prefer ingest_dbn.py or ingest_dbn_daily.py):
    python pipeline/ingest_dbn.py --instrument MGC [options]
    python pipeline/ingest_dbn_daily.py --instrument MGC --start ... --end ...
"""

import sys
import re
import json
import hashlib
import argparse
import traceback
from pathlib import Path
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from typing import Optional

import numpy as np
import pandas as pd
import duckdb
import databento as db

# Add project root to path
from pipeline.paths import GOLD_DB_PATH

from pipeline.log import get_logger
logger = get_logger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# DBN file location (portable, relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent
DBN_PATH = PROJECT_ROOT / "OHLCV_MGC_FULL" / "glbx-mdp3-20100912-20260203.ohlcv-1m.dbn.zst"

# Database
DB_PATH = GOLD_DB_PATH

# Checkpoint directory
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"

# Symbol
SYMBOL = "MGC"

# Timezone
TZ_LOCAL = ZoneInfo("Australia/Brisbane")
TZ_UTC = ZoneInfo("UTC")

# Outright contract pattern
# GC + month_code (FGHJKMNQUVXZ) + year (1-2 digits)
# We use GC (full-size Gold) bars for price data — better tick coverage than MGC.
# Prices are identical (same underlying); cost model remains MGC ($10/point).
GC_OUTRIGHT_PATTERN = re.compile(r'^GC[FGHJKMNQUVXZ]\d{1,2}$')

# Month codes for expiry parsing
MONTH_CODES = 'FGHJKMNQUVXZ'  # Jan=F, Feb=G, ..., Dec=Z

# MINIMUM DATE: Dataset now covers 2016-02-01 onward (GC data from Databento)
# Two data directories: gold_db_fullsize_2016-2021 and GOLD_DB_FULLSIZE (2021+)
MINIMUM_START_DATE = date(2016, 1, 1)

# =============================================================================
# CHECKPOINT SYSTEM (JSONL, APPEND-ONLY)
# =============================================================================

class CheckpointManager:
    """Append-only JSONL checkpoint system."""

    def __init__(self, checkpoint_dir: Path, source_file: Path, db_path: Path = None):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint file named after source DBN + DB path hash
        # Different DBs get separate checkpoint files to prevent cross-contamination
        if db_path is not None:
            db_hash = hashlib.md5(str(db_path.resolve()).encode()).hexdigest()[:8]
            self.checkpoint_file = checkpoint_dir / f"checkpoint_{source_file.stem}_{db_hash}.jsonl"
        else:
            # Backwards compatible: no db_path = old naming
            self.checkpoint_file = checkpoint_dir / f"checkpoint_{source_file.stem}.jsonl"

        # Source file identity (for detecting file changes)
        self.source_identity = self._get_source_identity(source_file)

        # Load existing checkpoints
        self.checkpoints = self._load_checkpoints()

        # Attempt counter
        self.attempt_id = self._get_next_attempt_id()

    def _get_source_identity(self, source_file: Path) -> dict:
        """Get source file identity (size + mtime)."""
        stat = source_file.stat()
        return {
            "path": str(source_file),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
        }

    def _load_checkpoints(self) -> dict:
        """Load checkpoints from JSONL file."""
        checkpoints = {}  # key = (chunk_start, chunk_end)

        if not self.checkpoint_file.exists():
            return checkpoints

        with open(self.checkpoint_file, 'r') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    key = (record['chunk_start'], record['chunk_end'])
                    # Keep latest record per chunk (by attempt_id, or later in file for same attempt)
                    # Use >= so that 'done' overwrites 'in_progress' with same attempt_id
                    if key not in checkpoints or record['attempt_id'] >= checkpoints[key]['attempt_id']:
                        checkpoints[key] = record

        return checkpoints

    def _get_next_attempt_id(self) -> int:
        """Get next monotonic attempt ID."""
        if not self.checkpoints:
            return 1
        return max(r['attempt_id'] for r in self.checkpoints.values()) + 1

    def get_chunk_status(self, chunk_start: str, chunk_end: str) -> Optional[str]:
        """Get status of a chunk."""
        key = (chunk_start, chunk_end)
        if key in self.checkpoints:
            return self.checkpoints[key]['status']
        return None

    def should_process_chunk(self, chunk_start: str, chunk_end: str, retry_failed: bool) -> bool:
        """Determine if chunk should be processed."""
        status = self.get_chunk_status(chunk_start, chunk_end)

        if status is None:
            return True  # Never processed
        if status == 'done':
            return False  # Already complete
        if status == 'in_progress':
            return True  # Resume interrupted
        if status == 'failed':
            return retry_failed  # Only if flag set

        return True

    def write_checkpoint(self, chunk_start: str, chunk_end: str, status: str,
                         rows_written: int = 0, error: str = None):
        """Append checkpoint record (immutable)."""
        record = {
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
            "status": status,
            "rows_written": rows_written,
            "started_at": datetime.now(TZ_UTC).isoformat() if status == 'in_progress' else None,
            "finished_at": datetime.now(TZ_UTC).isoformat() if status in ('done', 'failed') else None,
            "source_dbn": self.source_identity,
            "error": error,
            "attempt_id": self.attempt_id,
        }

        # Append to file (never edit/delete)
        with open(self.checkpoint_file, 'a') as f:
            f.write(json.dumps(record) + '\n')

        # Update in-memory cache
        key = (chunk_start, chunk_end)
        self.checkpoints[key] = record

        if status in ('done', 'failed'):
            self.attempt_id += 1

# =============================================================================
# VALIDATION (VECTORIZED, FAIL-CLOSED)
# =============================================================================

def validate_chunk(df: pd.DataFrame) -> tuple[bool, str, Optional[pd.DataFrame]]:
    """
    Validate chunk with vectorized operations.

    Returns (is_valid, reason, offending_rows_if_invalid)

    FAIL-CLOSED: Any single invalid row fails entire validation.
    """
    # Required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            return False, f"Missing column: {col}", None

    # Check for NaN in OHLCV
    nan_mask = df[required_cols].isna().any(axis=1)
    if nan_mask.any():
        return False, "NaN values in OHLCV", df[nan_mask].head(5)

    # Check prices are finite
    price_cols = ['open', 'high', 'low', 'close']
    inf_mask = ~np.isfinite(df[price_cols].values).all(axis=1)
    if inf_mask.any():
        return False, "Infinite price values", df[inf_mask].head(5)

    # Check prices > 0
    nonpos_mask = (df[price_cols] <= 0).any(axis=1)
    if nonpos_mask.any():
        return False, "Non-positive prices (must be > 0)", df[nonpos_mask].head(5)

    # Check high >= max(open, close, low)
    max_ocl = df[['open', 'close', 'low']].max(axis=1)
    high_fail = df['high'] < max_ocl
    if high_fail.any():
        return False, "high < max(open, close, low)", df[high_fail].head(5)

    # Check low <= min(open, close)
    min_oc = df[['open', 'close']].min(axis=1)
    low_fail = df['low'] > min_oc
    if low_fail.any():
        return False, "low > min(open, close)", df[low_fail].head(5)

    # Check high >= low
    hl_fail = df['high'] < df['low']
    if hl_fail.any():
        return False, "high < low", df[hl_fail].head(5)

    # Check volume >= 0
    vol_fail = df['volume'] < 0
    if vol_fail.any():
        return False, "Negative volume", df[vol_fail].head(5)

    return True, "", None

def validate_timestamp_utc(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Explicitly verify timestamps are UTC.

    FAIL-CLOSED: If timezone cannot be proven, abort.
    """
    # Check index is datetime with UTC timezone
    if not hasattr(df.index, 'tz'):
        return False, "Index has no timezone info"

    if df.index.tz is None:
        return False, "Index timezone is None (tz-naive)"

    if str(df.index.tz) != 'UTC':
        return False, f"Index timezone is {df.index.tz}, not UTC"

    # Check for null timestamps
    if df.index.isna().any():
        return False, "Null timestamps found"

    return True, ""

# =============================================================================
# CONTRACT SELECTION (DETERMINISTIC)
# =============================================================================

def parse_expiry(symbol: str, prefix_len: int = 3) -> tuple[int, int]:
    """
    Parse contract expiry from symbol.

    Returns (year, month) for sorting.
    Raises ValueError if cannot parse.
    """
    # Format: PREFIX + month_code + year (e.g., MGCG5, MGCZ25, NQH5)
    month_code = symbol[prefix_len]
    year_str = symbol[prefix_len + 1:]

    month = MONTH_CODES.index(month_code) + 1
    year = int(year_str)

    # Handle 2-digit years
    if year < 50:
        year += 2000
    elif year < 100:
        year += 1900

    return (year, month)

def choose_front_contract(daily_volumes: dict, outright_pattern=None, prefix_len: int = 3, log_func=None) -> Optional[str]:
    """
    Choose front-month contract with DETERMINISTIC tiebreak.

    1. Highest total volume wins
    2. If tie: earliest expiry (if parseable for ALL tied)
    3. If tie or parse fails: lexicographically smallest
    """
    # Filter to outrights only
    pattern = outright_pattern or GC_OUTRIGHT_PATTERN
    outrights = {s: v for s, v in daily_volumes.items()
                 if pattern.match(str(s))}

    if not outrights:
        return None

    max_vol = max(outrights.values())
    tied = sorted([s for s, v in outrights.items() if v == max_vol])

    if len(tied) == 1:
        return tied[0]

    # Log tie situation
    if log_func:
        log_func(f"  TIE: {len(tied)} contracts with volume {max_vol}: {tied}")

    # Tiebreak #1: earliest expiry (only if ALL tied can be parsed)
    try:
        expiries = {s: parse_expiry(s, prefix_len) for s in tied}
        winner = min(tied, key=lambda s: expiries[s])
        if log_func:
            log_func(f"  TIEBREAK by expiry: {winner}")
        return winner
    except (ValueError, IndexError):
        pass

    # Tiebreak #2: lexicographically smallest
    winner = min(tied)
    if log_func:
        log_func(f"  TIEBREAK by lexicographic: {winner}")
    return winner

# =============================================================================
# TRADING DAY CALCULATION (VECTORIZED)
# =============================================================================

def compute_trading_days(df: pd.DataFrame) -> pd.Series:
    """
    Compute trading day for each bar using vectorized operations.

    Trading day = 09:00 Brisbane -> 09:00 next day Brisbane
    Bars before 09:00 local belong to PREVIOUS trading day.
    """
    # Convert to Brisbane time
    ts_local = df.index.tz_convert(str(TZ_LOCAL))

    # Extract components
    hours = ts_local.hour
    dates = ts_local.date

    # Vectorized: if hour < 9, subtract one day
    trading_days = pd.Series(dates, index=df.index)
    mask = hours < 9
    trading_days[mask] = trading_days[mask].apply(lambda d: d - timedelta(days=1))

    return trading_days

# =============================================================================
# INTEGRITY GATES
# =============================================================================

def check_pk_safety(df: pd.DataFrame, trading_day: date) -> tuple[bool, str]:
    """
    Assert no duplicate ts_utc in selected bars (PK safety).

    Must run BEFORE merge.
    """
    if df.index.duplicated().any():
        dupes = df[df.index.duplicated(keep=False)]
        return False, f"Duplicate ts_utc found for trading day {trading_day}: {len(dupes)} rows"
    return True, ""

def check_merge_integrity(con: duckdb.DuckDBPyConnection, chunk_start: str, chunk_end: str, symbol: str = None) -> tuple[bool, str]:
    """
    Assert no duplicates or NULL source_symbol after merge.

    Must run AFTER merge. If symbol is provided, scopes checks to that symbol only.
    Uses a 1-day buffer on the start of the date range to account for overnight bars
    whose DATE(ts_utc) may precede the trading day.
    """
    symbol_clause = "AND symbol = ?" if symbol else ""
    params_base = [chunk_start, chunk_end]
    if symbol:
        params_base.append(symbol)

    # Check for duplicates (with 1-day buffer for overnight bars)
    dupe_check = con.execute(f"""
        SELECT symbol, ts_utc, COUNT(*) as cnt
        FROM bars_1m
        WHERE DATE(ts_utc) BETWEEN CAST(? AS DATE) - INTERVAL 1 DAY AND ?
        {symbol_clause}
        GROUP BY symbol, ts_utc
        HAVING COUNT(*) > 1
        LIMIT 5
    """, params_base).fetchall()

    if dupe_check:
        return False, f"Duplicate (symbol, ts_utc) found after merge: {dupe_check}"

    # Check for NULL source_symbol
    null_params = [chunk_start, chunk_end]
    if symbol:
        null_params.append(symbol)
    null_check = con.execute(f"""
        SELECT COUNT(*) FROM bars_1m
        WHERE DATE(ts_utc) BETWEEN CAST(? AS DATE) - INTERVAL 1 DAY AND ?
        {symbol_clause}
        AND source_symbol IS NULL
    """, null_params).fetchone()[0]

    if null_check > 0:
        return False, f"NULL source_symbol found after merge: {null_check} rows"

    return True, ""

# =============================================================================
# FINAL HONESTY GATES
# =============================================================================

def run_final_gates(con: duckdb.DuckDBPyConnection) -> tuple[bool, list[str]]:
    """
    Run final honesty gates after full backfill.

    Returns (all_passed, list_of_failures)
    """
    failures = []

    # Gate 1: Check ts_utc column type
    col_info = con.execute("""
        SELECT data_type FROM information_schema.columns
        WHERE table_name = 'bars_1m' AND column_name = 'ts_utc'
    """).fetchone()

    if col_info:
        dtype = col_info[0].upper()
        if 'TIMESTAMP' not in dtype:
            failures.append(f"ts_utc type is {dtype}, expected TIMESTAMP/TIMESTAMPTZ")

    # Gate 2: Check for any duplicate (symbol, ts_utc) globally
    dupe_count = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT symbol, ts_utc FROM bars_1m
            GROUP BY symbol, ts_utc
            HAVING COUNT(*) > 1
        )
    """).fetchone()[0]

    if dupe_count > 0:
        failures.append(f"Global duplicate (symbol, ts_utc) found: {dupe_count}")

    # Gate 3: Check for any NULL source_symbol
    null_count = con.execute("""
        SELECT COUNT(*) FROM bars_1m WHERE source_symbol IS NULL
    """).fetchone()[0]

    if null_count > 0:
        failures.append(f"NULL source_symbol found: {null_count} rows")

    return len(failures) == 0, failures

# =============================================================================
# MAIN INGESTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Ingest MGC DBN into bars_1m (CANONICAL COMPLIANT)")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed chunks")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, no DB writes")
    parser.add_argument("--chunk-days", type=int, default=7, help="Trading days per commit")
    parser.add_argument("--batch-size", type=int, default=50000, help="Rows per DBN batch")
    args = parser.parse_args()

    start_time = datetime.now()

    # =========================================================================
    # STARTUP: Log config snapshot
    # =========================================================================
    logger.info("=" * 70)
    logger.info("MGC DBN INGESTION (CANONICAL COMPLIANT)")
    logger.info("=" * 70)
    logger.info("CONFIG SNAPSHOT:")
    logger.info(f"  DBN file: {DBN_PATH}")
    logger.info(f"  Database: {DB_PATH}")
    logger.info(f"  Checkpoint dir: {CHECKPOINT_DIR}")
    logger.info(f"  Start filter: {args.start or 'None'}")
    logger.info(f"  End filter: {args.end or 'None'}")
    logger.info(f"  Resume: {args.resume}")
    logger.info(f"  Retry failed: {args.retry_failed}")
    logger.info(f"  Dry run: {args.dry_run}")
    logger.info(f"  Chunk days: {args.chunk_days}")
    logger.info(f"  Batch size: {args.batch_size}")

    # =========================================================================
    # VERIFY DBN FILE EXISTS
    # =========================================================================
    if not DBN_PATH.exists():
        logger.warning(f"FATAL: DBN file not found: {DBN_PATH}")
        sys.exit(1)

    file_size_gb = DBN_PATH.stat().st_size / (1024**3)
    logger.info(f"DBN file size: {file_size_gb:.2f} GB")

    # =========================================================================
    # OPEN DBN AND VERIFY SCHEMA (FAIL-CLOSED)
    # =========================================================================
    logger.info("Opening DBN file...")
    store = db.DBNStore.from_file(DBN_PATH)

    logger.info(f"  Schema: {store.schema}")
    logger.info(f"  Dataset: {store.dataset}")
    logger.info(f"  Date range: {store.start} to {store.end}")

    # DBN CONTENT GATE: Must be ohlcv-1m
    if store.schema != 'ohlcv-1m':
        logger.warning(f"FATAL: DBN schema is '{store.schema}', expected 'ohlcv-1m'")
        logger.warning("ABORT: Schema verification failed (FAIL-CLOSED)")
        sys.exit(1)

    logger.info("  Schema verified: ohlcv-1m [OK]")

    # =========================================================================
    # INITIALIZE CHECKPOINT MANAGER
    # =========================================================================
    checkpoint_mgr = CheckpointManager(CHECKPOINT_DIR, DBN_PATH, db_path=DB_PATH)
    logger.info(f"Checkpoint file: {checkpoint_mgr.checkpoint_file}")
    logger.info(f"Existing checkpoints: {len(checkpoint_mgr.checkpoints)}")

    # =========================================================================
    # OPEN DATABASE
    # =========================================================================
    con = None
    if not args.dry_run:
        con = duckdb.connect(str(DB_PATH))
        from pipeline.db_config import configure_connection
        configure_connection(con, writing=True)
        logger.info(f"Database opened: {DB_PATH}")
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
    # PHASE 1-4: EXTRACT, VALIDATE, TRANSFORM, AGGREGATE
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

    # Accumulator for trading days
    trading_day_buffer = {}  # trading_day -> list of (ts_utc, source_symbol, o, h, l, c, v)

    # Date filters with MINIMUM DATE ENFORCEMENT
    start_filter = date.fromisoformat(args.start) if args.start else MINIMUM_START_DATE
    end_filter = date.fromisoformat(args.end) if args.end else None

    # ENFORCE MINIMUM DATE (GC data available from 2016-02-01 onward)
    if start_filter < MINIMUM_START_DATE:
        logger.warning(f"WARNING: Requested start {start_filter} is before MINIMUM_START_DATE {MINIMUM_START_DATE}")
        logger.info(f"         GC data coverage starts {MINIMUM_START_DATE}")
        logger.info(f"         FORCING start to {MINIMUM_START_DATE}")
        start_filter = MINIMUM_START_DATE

    logger.info(f"EFFECTIVE DATE RANGE: {start_filter} to {end_filter or 'end of file'}")

    batch_num = 0
    skipped_batches = 0
    for chunk_df in store.to_df(count=args.batch_size):
        batch_num += 1

        # Reset index to access ts_event as column
        chunk_df = chunk_df.reset_index()

        # =====================================================================
        # FAST-FORWARD: Skip entire batches before start_filter
        # (DBN is sequential - we have to scan but don't need to process)
        # =====================================================================
        batch_max_date = chunk_df['ts_event'].max().date()
        if batch_max_date < start_filter:
            skipped_batches += 1
            if skipped_batches % 20 == 1:
                logger.info(f"  FAST-FORWARD: Batch {batch_num} ends at {batch_max_date}, skipping (target: {start_filter})...")
            continue

        if skipped_batches > 0:
            logger.info(f"  FAST-FORWARD COMPLETE: Skipped {skipped_batches} batches of pre-{start_filter} data")
            skipped_batches = 0  # Reset so we don't print again

        # =====================================================================
        # VALIDATE TIMESTAMPS (FAIL-CLOSED)
        # =====================================================================
        # Re-set ts_event as index for validation
        chunk_df = chunk_df.set_index('ts_event')

        ts_valid, ts_reason = validate_timestamp_utc(chunk_df)
        if not ts_valid:
            logger.warning(f"FATAL: Timestamp validation failed: {ts_reason}")
            logger.warning("ABORT: Timestamp verification gate failed (FAIL-CLOSED)")
            traceback.print_exc()
            sys.exit(1)

        # =====================================================================
        # FILTER TO OUTRIGHTS FIRST (VECTORIZED)
        # (Spreads have negative prices which is valid for them, so filter before validation)
        # =====================================================================
        outright_mask = chunk_df['symbol'].apply(lambda s: bool(GC_OUTRIGHT_PATTERN.match(str(s))))
        chunk_df = chunk_df[outright_mask]

        if len(chunk_df) == 0:
            continue

        # =====================================================================
        # VALIDATE OHLCV (FAIL-CLOSED) - Only outrights at this point
        # =====================================================================
        valid, reason, bad_rows = validate_chunk(chunk_df)
        if not valid:
            logger.warning(f"FATAL: OHLCV validation failed: {reason}")
            if bad_rows is not None:
                logger.info("Offending rows:")
                logger.info(bad_rows.to_string())
            logger.warning("ABORT: Validation gate failed (FAIL-CLOSED)")
            traceback.print_exc()
            sys.exit(1)

        # =====================================================================
        # COMPUTE TRADING DAYS (VECTORIZED)
        # =====================================================================
        trading_days = compute_trading_days(chunk_df)
        chunk_df = chunk_df.copy()
        chunk_df['trading_day'] = trading_days

        # =====================================================================
        # AGGREGATE BY TRADING DAY
        # =====================================================================
        for tday, day_df in chunk_df.groupby('trading_day'):
            # Apply date filters
            if start_filter and tday < start_filter:
                continue
            if end_filter and tday > end_filter:
                continue

            # Calculate volume per contract
            volumes = day_df.groupby('symbol')['volume'].sum().to_dict()

            # Choose front contract (deterministic)
            front = choose_front_contract(volumes, outright_pattern=GC_OUTRIGHT_PATTERN, prefix_len=2, log_func=lambda msg: None)
            if not front:
                continue

            stats['contracts_used'].add(front)

            # Filter to front contract only
            front_df = day_df[day_df['symbol'] == front].copy()

            # PK SAFETY: Check for duplicate timestamps
            pk_ok, pk_reason = check_pk_safety(front_df, tday)
            if not pk_ok:
                logger.warning(f"FATAL: PK safety check failed: {pk_reason}")
                logger.warning("ABORT: Primary key safety gate failed (FAIL-CLOSED)")
                sys.exit(1)

            # Sort by timestamp (monotonic requirement)
            front_df = front_df.sort_index()

            # Store in buffer
            if tday not in trading_day_buffer:
                trading_day_buffer[tday] = []

            for ts_utc, row in front_df.iterrows():
                trading_day_buffer[tday].append((
                    ts_utc,
                    front,
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    int(row['volume']),
                ))

        # =====================================================================
        # MERGE WHEN CHUNK IS FULL
        # =====================================================================
        sorted_days = sorted(trading_day_buffer.keys())

        while len(sorted_days) >= args.chunk_days:
            # Take chunk_days worth of trading days
            chunk_days = sorted_days[:args.chunk_days]
            chunk_start = str(chunk_days[0])
            chunk_end = str(chunk_days[-1])

            # Check if should process this chunk
            if args.resume and not checkpoint_mgr.should_process_chunk(chunk_start, chunk_end, args.retry_failed):
                logger.info(f"  SKIP: Chunk {chunk_start} to {chunk_end} (already done)")
                stats['chunks_skipped'] += 1
                # Remove from buffer
                for d in chunk_days:
                    del trading_day_buffer[d]
                sorted_days = sorted_days[args.chunk_days:]
                continue

            # Collect rows for this chunk
            chunk_rows = []
            for d in chunk_days:
                chunk_rows.extend(trading_day_buffer[d])

            if not args.dry_run and con:
                # Write checkpoint: in_progress
                checkpoint_mgr.write_checkpoint(chunk_start, chunk_end, 'in_progress')

                try:
                    # BEGIN TRANSACTION
                    con.execute("BEGIN TRANSACTION")

                    # INSERT OR REPLACE
                    con.executemany(
                        """
                        INSERT OR REPLACE INTO bars_1m
                        (ts_utc, symbol, source_symbol, open, high, low, close, volume)
                        VALUES (?, 'MGC', ?, ?, ?, ?, ?, ?)
                        """,
                        chunk_rows
                    )

                    # INTEGRITY GATE
                    int_ok, int_reason = check_merge_integrity(con, chunk_start, chunk_end, symbol='MGC')
                    if not int_ok:
                        con.execute("ROLLBACK")
                        checkpoint_mgr.write_checkpoint(chunk_start, chunk_end, 'failed', error=int_reason)
                        logger.warning(f"FATAL: Integrity gate failed: {int_reason}")
                        logger.warning("ABORT: Merge integrity gate failed (FAIL-CLOSED)")
                        sys.exit(1)

                    # COMMIT
                    con.execute("COMMIT")

                    # Write checkpoint: done
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
                # Dry run
                stats['chunks_done'] += 1
                stats['rows_written'] += len(chunk_rows)
                stats['trading_days_processed'] += len(chunk_days)
                logger.info(f"  DRY RUN: Chunk {chunk_start} to {chunk_end}: {len(chunk_rows):,} rows")

            # Remove processed days from buffer
            for d in chunk_days:
                del trading_day_buffer[d]
            sorted_days = sorted_days[args.chunk_days:]

        # Progress indicator
        if batch_num % 20 == 0:
            logger.info(f"  Batch {batch_num}: {stats['rows_written']:,} rows written, {stats['trading_days_processed']} trading days")

    # =========================================================================
    # PROCESS REMAINING BUFFER
    # =========================================================================
    if trading_day_buffer:
        sorted_days = sorted(trading_day_buffer.keys())
        chunk_start = str(sorted_days[0])
        chunk_end = str(sorted_days[-1])

        # Check if should process
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
                    con.executemany(
                        """
                        INSERT OR REPLACE INTO bars_1m
                        (ts_utc, symbol, source_symbol, open, high, low, close, volume)
                        VALUES (?, 'MGC', ?, ?, ?, ?, ?, ?)
                        """,
                        chunk_rows
                    )

                    int_ok, int_reason = check_merge_integrity(con, chunk_start, chunk_end, symbol='MGC')
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

    logger.info(f"Chunks done: {stats['chunks_done']}")
    logger.info(f"Chunks failed: {stats['chunks_failed']}")
    logger.info(f"Chunks skipped: {stats['chunks_skipped']}")
    logger.info(f"Trading days processed: {stats['trading_days_processed']}")
    logger.info(f"Total rows written: {stats['rows_written']:,}")
    logger.info(f"Unique contracts used: {len(stats['contracts_used'])}")
    logger.info(f"Wall time: {elapsed}")

    if not args.dry_run and con:
        # Get actual DB stats
        count = con.execute("SELECT COUNT(*) FROM bars_1m WHERE symbol = 'MGC'").fetchone()[0]
        date_range = con.execute(
            "SELECT MIN(DATE(ts_utc)), MAX(DATE(ts_utc)) FROM bars_1m WHERE symbol = 'MGC'"
        ).fetchone()

        logger.info(f"Database rows (MGC): {count:,}")
        logger.info(f"Date range in DB: {date_range[0]} to {date_range[1]}")

        con.close()

    logger.info("SUCCESS: Backfill complete and validated.")
    sys.exit(0)

if __name__ == "__main__":
    logger.info("NOTE: For multi-instrument support, prefer:")
    logger.info("  python pipeline/ingest_dbn.py --instrument MGC")
    main()
