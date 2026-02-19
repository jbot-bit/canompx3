#!/usr/bin/env python3
"""
Initialize the DuckDB database schema for MGC data pipeline.

Creates tables:
- bars_1m: Primary 1-minute OHLCV data (raw data from Databento)
- bars_5m: Derived 5-minute OHLCV data (aggregated from bars_1m)
- daily_features: One row per trading day per instrument (ORBs, session stats, RSI)

Usage:
    python pipeline/init_db.py [--force]

Options:
    --force    Drop existing tables and recreate (WARNING: destroys data)
"""

import argparse
from pathlib import Path

import duckdb

# Add project root to path
from pipeline.paths import GOLD_DB_PATH

from pipeline.log import get_logger
logger = get_logger(__name__)

# =============================================================================
# SCHEMA DEFINITIONS (CANONICAL - matches CLAUDE.md)
# =============================================================================

BARS_1M_SCHEMA = """
CREATE TABLE IF NOT EXISTS bars_1m (
    ts_utc        TIMESTAMPTZ NOT NULL,
    symbol        TEXT        NOT NULL,
    source_symbol TEXT        NOT NULL,
    open          DOUBLE      NOT NULL,
    high          DOUBLE      NOT NULL,
    low           DOUBLE      NOT NULL,
    close         DOUBLE      NOT NULL,
    volume        BIGINT      NOT NULL,
    PRIMARY KEY (symbol, ts_utc)
);
"""

BARS_5M_SCHEMA = """
CREATE TABLE IF NOT EXISTS bars_5m (
    ts_utc        TIMESTAMPTZ NOT NULL,
    symbol        TEXT        NOT NULL,
    source_symbol TEXT,
    open          DOUBLE      NOT NULL,
    high          DOUBLE      NOT NULL,
    low           DOUBLE      NOT NULL,
    close         DOUBLE      NOT NULL,
    volume        BIGINT      NOT NULL,
    PRIMARY KEY (symbol, ts_utc)
);
"""

# ORB labels: Opening Range Breakout windows at these local times.
# All times are Australia/Brisbane (UTC+10, no DST).
#
# Fixed session mapping (data-proven market event identities):
#   0900 Brisbane = 23:00 UTC — CME open in winter / 1hr after in summer
#   1000 Brisbane = 00:00 UTC — Tokyo 9AM JST (no DST)
#   1100 Brisbane = 01:00 UTC — ~Singapore/Shanghai open (no DST)
#   1130 Brisbane = 01:30 UTC — HK/SG equity open 9:30 AM HKT (no DST)
#   1800 Brisbane = 08:00 UTC — London metals in winter / 1hr after summer
#   2300 Brisbane = 13:00 UTC — ~8AM ET winter / ~9AM ET summer
#   0030 Brisbane = 14:30 UTC — NYSE 9:30 ET in winter / 1hr after summer
#
# The ORB is the high-low range of the first N minutes (configurable).
# A "break" occurs when a 1-min bar closes above orb_high (long) or
# below orb_low (short). See pipeline/build_daily_features.py for logic.
ORB_LABELS_FIXED = ["0900", "1000", "1100", "1130", "1800", "2300", "0030"]

# Dynamic sessions: DST-aware windows that track specific market events
# regardless of daylight saving time changes. Resolved per-day by
# pipeline/dst.py resolvers. See pipeline/dst.py SESSION_CATALOG for
# the master registry including aliases and event descriptions.
#
#   CME_OPEN        - CME Globex electronic open at 5:00 PM CT
#   US_EQUITY_OPEN  - NYSE cash open at 09:30 ET (MES, MNQ)
#   US_DATA_OPEN    - US economic data release at 08:30 ET (MGC)
#   LONDON_OPEN     - London metals open at 08:00 London time (MGC)
#   CME_CLOSE       - CME equity futures pre-close at 2:45 PM CT (MNQ, MES)
ORB_LABELS_DYNAMIC = ["CME_OPEN", "US_EQUITY_OPEN", "US_DATA_OPEN", "LONDON_OPEN", "US_POST_EQUITY", "CME_CLOSE"]

# Combined label list — used by schema generation and feature builders
ORB_LABELS = ORB_LABELS_FIXED + ORB_LABELS_DYNAMIC

def _build_daily_features_ddl() -> str:
    """Generate CREATE TABLE DDL for daily_features.

    Columns per ORB (9 each):
      orb_{label}_high         - ORB range high
      orb_{label}_low          - ORB range low
      orb_{label}_size         - high - low (points)
      orb_{label}_break_dir    - 'long', 'short', or NULL (no break)
      orb_{label}_break_ts     - timestamp of first break (1m close outside range)
      orb_{label}_outcome      - outcome at RR=1.0 ('win', 'loss', 'scratch', NULL)
      orb_{label}_mae_r        - max adverse excursion in R (NULL until cost model)
      orb_{label}_mfe_r        - max favorable excursion in R (NULL until cost model)
      orb_{label}_double_break - True if BOTH ORB high and low were breached
    """
    orb_cols = []
    for label in ORB_LABELS:
        orb_cols.extend([
            f"    orb_{label}_high         DOUBLE,",
            f"    orb_{label}_low          DOUBLE,",
            f"    orb_{label}_size         DOUBLE,",
            f"    orb_{label}_break_dir    TEXT,",
            f"    orb_{label}_break_ts     TIMESTAMPTZ,",
            f"    orb_{label}_outcome      TEXT,",
            f"    orb_{label}_mae_r        DOUBLE,",
            f"    orb_{label}_mfe_r        DOUBLE,",
            f"    orb_{label}_double_break BOOLEAN,",
        ])
    orb_block = "\n".join(orb_cols)

    return f"""
CREATE TABLE IF NOT EXISTS daily_features (
    trading_day       DATE    NOT NULL,
    symbol            TEXT    NOT NULL,
    orb_minutes       INTEGER NOT NULL,
    bar_count_1m      INTEGER,

    -- Session stats (local Brisbane times)
    session_asia_high   DOUBLE,
    session_asia_low    DOUBLE,
    session_london_high DOUBLE,
    session_london_low  DOUBLE,
    session_ny_high     DOUBLE,
    session_ny_low      DOUBLE,

    -- RSI (Wilder's 14-period on 5m closes, computed at first ORB)
    rsi_14_at_0900    DOUBLE,

    -- Daily OHLC (from all 1m bars in the trading day)
    daily_open        DOUBLE,
    daily_high        DOUBLE,
    daily_low         DOUBLE,
    daily_close       DOUBLE,

    -- Overnight gap: today's open - previous day's close (positive = gap up)
    gap_open_points   DOUBLE,

    -- ATR(20): 20-day simple moving average of True Range
    -- True Range = max(H-L, |H-prevClose|, |L-prevClose|)
    -- Used as regime filter (vol expansion/contraction detection)
    atr_20            DOUBLE,

    -- DST flags: whether US/UK was in daylight saving time on this trading day.
    -- Used by dynamic sessions (US_EQUITY_OPEN, US_DATA_OPEN, LONDON_OPEN)
    -- to verify correct window resolution.
    us_dst            BOOLEAN,
    uk_dst            BOOLEAN,

    -- Calendar skip flags (deterministic from date, no parameter search)
    -- Used as portfolio overlay filters, not in discovery grid.
    is_nfp_day        BOOLEAN,
    is_opex_day       BOOLEAN,
    is_friday         BOOLEAN,

    -- Day-of-week flags (Feb 2026 DOW research)
    -- Used by DayOfWeekSkipFilter in discovery grid (session-specific composites).
    is_monday         BOOLEAN,
    is_tuesday        BOOLEAN,
    day_of_week       INTEGER,    -- 0=Mon, 1=Tue, ..., 4=Fri (Python weekday convention)

    -- ORB columns (7 fixed + 6 dynamic = 13 sessions x 9 columns = 117)
{orb_block}

    PRIMARY KEY (symbol, trading_day, orb_minutes)
);
"""

DAILY_FEATURES_SCHEMA = _build_daily_features_ddl()

def init_db(db_path: Path, force: bool = False):
    """Initialize database with schema."""

    print("=" * 60)
    logger.info("DATABASE INITIALIZATION")
    print("=" * 60)
    print()
    logger.info(f"Database path: {db_path}")
    logger.info(f"Force recreate: {force}")
    print()

    # Connect to database (creates file if doesn't exist)
    with duckdb.connect(str(db_path)) as con:

        if force:
            logger.info("FORCE MODE: Dropping ALL tables...")
            # Drop trading_app tables first (FK dependencies on daily_features)
            for t in ["validated_setups_archive", "validated_setups",
                       "experimental_strategies", "orb_outcomes"]:
                con.execute(f"DROP TABLE IF EXISTS {t}")
            # Drop pipeline tables
            con.execute("DROP TABLE IF EXISTS daily_features")
            con.execute("DROP TABLE IF EXISTS bars_5m")
            con.execute("DROP TABLE IF EXISTS bars_1m")
            logger.info("  All tables dropped (pipeline + trading_app).")
            print()

        # Create tables
        logger.info("Creating tables...")

        con.execute(BARS_1M_SCHEMA)
        logger.info("  bars_1m: created (or already exists)")

        con.execute(BARS_5M_SCHEMA)
        logger.info("  bars_5m: created (or already exists)")

        con.execute(DAILY_FEATURES_SCHEMA)
        logger.info("  daily_features: created (or already exists)")

        # Migration: add calendar skip flag columns (Feb 2026)
        for col in ["is_nfp_day", "is_opex_day", "is_friday"]:
            try:
                con.execute(f"ALTER TABLE daily_features ADD COLUMN {col} BOOLEAN")
                logger.info(f"  Migration: added {col} column to daily_features")
            except duckdb.CatalogException:
                pass  # column already exists

        # Migration: add DOW columns (Feb 2026 DOW research)
        for col, typedef in [("is_monday", "BOOLEAN"), ("is_tuesday", "BOOLEAN"), ("day_of_week", "INTEGER")]:
            try:
                con.execute(f"ALTER TABLE daily_features ADD COLUMN {col} {typedef}")
                logger.info(f"  Migration: added {col} column to daily_features")
            except duckdb.CatalogException:
                pass  # column already exists

        con.commit()
        print()

        # Verify schema
        logger.info("Verifying schema...")

        tables = con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
        table_names = [t[0] for t in tables]

        logger.info(f"  Tables found: {table_names}")

        # Check columns for each table
        for table_name in ['bars_1m', 'bars_5m', 'daily_features']:
            if table_name in table_names:
                cols = con.execute(
                    f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}'"
                ).fetchall()
                logger.info(f"  {table_name} columns ({len(cols)}): {[c[0] for c in cols]}")


    print()
    print("=" * 60)
    logger.info("INITIALIZATION COMPLETE")
    print("=" * 60)
    print()
    logger.info("Next steps:")
    logger.info("  1. Run ingestion: python OHLCV_MGC_FULL/ingest_dbn_mgc.py")
    logger.info("  2. Check database: python pipeline/check_db.py")

def main():
    parser = argparse.ArgumentParser(description="Initialize DuckDB schema")
    parser.add_argument("--force", action="store_true", help="Drop and recreate tables")
    args = parser.parse_args()

    init_db(GOLD_DB_PATH, force=args.force)

if __name__ == "__main__":
    main()
