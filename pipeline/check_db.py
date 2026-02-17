#!/usr/bin/env python3
"""
Check database contents and integrity.

Usage:
    python pipeline/check_db.py
"""

import sys

import duckdb

# Add project root to path
from pipeline.paths import GOLD_DB_PATH

from pipeline.log import get_logger
logger = get_logger(__name__)

def check_db():
    """Display database summary statistics."""

    print("=" * 60)
    logger.info("DATABASE CHECK")
    print("=" * 60)
    print()
    logger.info(f"Database: {GOLD_DB_PATH}")
    print()

    if not GOLD_DB_PATH.exists():
        logger.error("ERROR: Database does not exist!")
        logger.info("Run: python pipeline/init_db.py")
        sys.exit(1)

    with duckdb.connect(str(GOLD_DB_PATH), read_only=True) as con:

        # List tables
        tables = con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
        table_names = [t[0] for t in tables]
        logger.info(f"Tables: {table_names}")
        print()

        # Check bars_1m
        if 'bars_1m' in table_names:
            logger.info("bars_1m:")
            count = con.execute("SELECT COUNT(*) FROM bars_1m").fetchone()[0]
            logger.info(f"  Row count: {count:,}")

            if count > 0:
                date_range = con.execute(
                    "SELECT MIN(ts_utc), MAX(ts_utc) FROM bars_1m"
                ).fetchone()
                logger.info(f"  Date range: {date_range[0]} to {date_range[1]}")

                symbols = con.execute(
                    "SELECT DISTINCT symbol FROM bars_1m"
                ).fetchall()
                logger.info(f"  Symbols: {[s[0] for s in symbols]}")

                contracts = con.execute(
                    "SELECT source_symbol, COUNT(*) as cnt FROM bars_1m GROUP BY source_symbol ORDER BY cnt DESC LIMIT 10"
                ).fetchall()
                logger.info(f"  Top contracts: {[(c[0], c[1]) for c in contracts[:5]]}")
            print()

        # Check bars_5m
        if 'bars_5m' in table_names:
            logger.info("bars_5m:")
            count = con.execute("SELECT COUNT(*) FROM bars_5m").fetchone()[0]
            logger.info(f"  Row count: {count:,}")

            if count > 0:
                date_range = con.execute(
                    "SELECT MIN(ts_utc), MAX(ts_utc) FROM bars_5m"
                ).fetchone()
                logger.info(f"  Date range: {date_range[0]} to {date_range[1]}")
            print()

        # Check daily_features
        if 'daily_features' in table_names:
            logger.info("daily_features:")
            count = con.execute("SELECT COUNT(*) FROM daily_features").fetchone()[0]
            logger.info(f"  Row count: {count:,}")

            if count > 0:
                date_range = con.execute(
                    "SELECT MIN(trading_day), MAX(trading_day) FROM daily_features"
                ).fetchone()
                logger.info(f"  Date range: {date_range[0]} to {date_range[1]}")

                orb_minutes = con.execute(
                    "SELECT DISTINCT orb_minutes FROM daily_features"
                ).fetchall()
                logger.info(f"  ORB durations: {[o[0] for o in orb_minutes]}")
            print()


if __name__ == "__main__":
    check_db()
