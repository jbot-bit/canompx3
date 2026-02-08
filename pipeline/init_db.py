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

import sys
import argparse
from pathlib import Path

import duckdb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.paths import GOLD_DB_PATH


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

# ORB labels: 5-minute Opening Range Breakout windows at these local times.
# All times are Australia/Brisbane (UTC+10, no DST).
#
# Session mapping and UTC equivalents:
#   0900 Brisbane = 23:00 UTC (prev day) — US equity close / CME open
#   1000 Brisbane = 00:00 UTC             — 1hr after CME open
#   1100 Brisbane = 01:00 UTC             — 2hr after CME open
#   1800 Brisbane = 08:00 UTC             — GLOBEX/London open
#   2300 Brisbane = 13:00 UTC             — Overnight/NY afternoon
#   0030 Brisbane = 14:30 UTC             — Late overnight
#
# The ORB is the high-low range of the first 5 minutes (configurable).
# A "break" occurs when a 1-min bar closes above orb_high (long) or
# below orb_low (short). See pipeline/build_daily_features.py for logic.
ORB_LABELS = ["0900", "1000", "1100", "1800", "2300", "0030"]


def _build_daily_features_ddl() -> str:
    """Generate CREATE TABLE DDL for daily_features.

    Columns per ORB (8 each):
      orb_{label}_high      - ORB range high
      orb_{label}_low       - ORB range low
      orb_{label}_size      - high - low (points)
      orb_{label}_break_dir - 'long', 'short', or NULL (no break)
      orb_{label}_break_ts  - timestamp of first break (1m close outside range)
      orb_{label}_outcome   - outcome at RR=1.0 ('win', 'loss', 'scratch', NULL)
      orb_{label}_mae_r     - max adverse excursion in R (NULL until cost model)
      orb_{label}_mfe_r     - max favorable excursion in R (NULL until cost model)
    """
    orb_cols = []
    for label in ORB_LABELS:
        orb_cols.extend([
            f"    orb_{label}_high      DOUBLE,",
            f"    orb_{label}_low       DOUBLE,",
            f"    orb_{label}_size      DOUBLE,",
            f"    orb_{label}_break_dir TEXT,",
            f"    orb_{label}_break_ts  TIMESTAMPTZ,",
            f"    orb_{label}_outcome   TEXT,",
            f"    orb_{label}_mae_r     DOUBLE,",
            f"    orb_{label}_mfe_r     DOUBLE,",
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

    -- ORB columns (6 ORBs x 8 columns = 48)
{orb_block}

    PRIMARY KEY (symbol, trading_day, orb_minutes)
);
"""


DAILY_FEATURES_SCHEMA = _build_daily_features_ddl()


def init_db(db_path: Path, force: bool = False):
    """Initialize database with schema."""

    print("=" * 60)
    print("DATABASE INITIALIZATION")
    print("=" * 60)
    print()
    print(f"Database path: {db_path}")
    print(f"Force recreate: {force}")
    print()

    # Connect to database (creates file if doesn't exist)
    con = duckdb.connect(str(db_path))

    if force:
        print("FORCE MODE: Dropping ALL tables...")
        # Drop trading_app tables first (FK dependencies on daily_features)
        for t in ["validated_setups_archive", "validated_setups",
                   "experimental_strategies", "orb_outcomes"]:
            con.execute(f"DROP TABLE IF EXISTS {t}")
        # Drop pipeline tables
        con.execute("DROP TABLE IF EXISTS daily_features")
        con.execute("DROP TABLE IF EXISTS bars_5m")
        con.execute("DROP TABLE IF EXISTS bars_1m")
        print("  All tables dropped (pipeline + trading_app).")
        print()

    # Create tables
    print("Creating tables...")

    con.execute(BARS_1M_SCHEMA)
    print("  bars_1m: created (or already exists)")

    con.execute(BARS_5M_SCHEMA)
    print("  bars_5m: created (or already exists)")

    con.execute(DAILY_FEATURES_SCHEMA)
    print("  daily_features: created (or already exists)")

    con.commit()
    print()

    # Verify schema
    print("Verifying schema...")

    tables = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()
    table_names = [t[0] for t in tables]

    print(f"  Tables found: {table_names}")

    # Check columns for each table
    for table_name in ['bars_1m', 'bars_5m', 'daily_features']:
        if table_name in table_names:
            cols = con.execute(
                f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}'"
            ).fetchall()
            print(f"  {table_name} columns ({len(cols)}): {[c[0] for c in cols]}")

    con.close()

    print()
    print("=" * 60)
    print("INITIALIZATION COMPLETE")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Run ingestion: python OHLCV_MGC_FULL/ingest_dbn_mgc.py")
    print("  2. Check database: python pipeline/check_db.py")


def main():
    parser = argparse.ArgumentParser(description="Initialize DuckDB schema")
    parser.add_argument("--force", action="store_true", help="Drop and recreate tables")
    args = parser.parse_args()

    init_db(GOLD_DB_PATH, force=args.force)


if __name__ == "__main__":
    main()
