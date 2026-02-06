#!/usr/bin/env python3
"""
Initialize the DuckDB database schema for MGC data pipeline.

Creates tables:
- bars_1m: Primary 1-minute OHLCV data (raw data from Databento)
- bars_5m: Derived 5-minute OHLCV data (aggregated from bars_1m)

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
        print("FORCE MODE: Dropping existing tables...")
        con.execute("DROP TABLE IF EXISTS bars_1m")
        con.execute("DROP TABLE IF EXISTS bars_5m")
        print("  Tables dropped.")
        print()

    # Create tables
    print("Creating tables...")

    con.execute(BARS_1M_SCHEMA)
    print("  bars_1m: created (or already exists)")

    con.execute(BARS_5M_SCHEMA)
    print("  bars_5m: created (or already exists)")

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
    for table_name in ['bars_1m', 'bars_5m']:
        if table_name in table_names:
            cols = con.execute(
                f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}'"
            ).fetchall()
            print(f"  {table_name} columns: {[c[0] for c in cols]}")

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
