#!/usr/bin/env python3
"""
Check database contents and integrity.

Usage:
    python pipeline/check_db.py
"""

import sys
from pathlib import Path

import duckdb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.paths import GOLD_DB_PATH


def check_db():
    """Display database summary statistics."""

    print("=" * 60)
    print("DATABASE CHECK")
    print("=" * 60)
    print()
    print(f"Database: {GOLD_DB_PATH}")
    print()

    if not GOLD_DB_PATH.exists():
        print("ERROR: Database does not exist!")
        print("Run: python pipeline/init_db.py")
        sys.exit(1)

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    # List tables
    tables = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()
    table_names = [t[0] for t in tables]
    print(f"Tables: {table_names}")
    print()

    # Check bars_1m
    if 'bars_1m' in table_names:
        print("bars_1m:")
        count = con.execute("SELECT COUNT(*) FROM bars_1m").fetchone()[0]
        print(f"  Row count: {count:,}")

        if count > 0:
            date_range = con.execute(
                "SELECT MIN(ts_utc), MAX(ts_utc) FROM bars_1m"
            ).fetchone()
            print(f"  Date range: {date_range[0]} to {date_range[1]}")

            symbols = con.execute(
                "SELECT DISTINCT symbol FROM bars_1m"
            ).fetchall()
            print(f"  Symbols: {[s[0] for s in symbols]}")

            contracts = con.execute(
                "SELECT source_symbol, COUNT(*) as cnt FROM bars_1m GROUP BY source_symbol ORDER BY cnt DESC LIMIT 10"
            ).fetchall()
            print(f"  Top contracts: {[(c[0], c[1]) for c in contracts[:5]]}")
        print()

    # Check bars_5m
    if 'bars_5m' in table_names:
        print("bars_5m:")
        count = con.execute("SELECT COUNT(*) FROM bars_5m").fetchone()[0]
        print(f"  Row count: {count:,}")

        if count > 0:
            date_range = con.execute(
                "SELECT MIN(ts_utc), MAX(ts_utc) FROM bars_5m"
            ).fetchone()
            print(f"  Date range: {date_range[0]} to {date_range[1]}")
        print()

    con.close()


if __name__ == "__main__":
    check_db()
