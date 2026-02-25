#!/usr/bin/env python3
"""
Initialize the DuckDB database schema for multi-instrument data pipeline.

Creates tables:
- bars_1m: Primary 1-minute OHLCV data (raw data from Databento)
- bars_5m: Derived 5-minute OHLCV data (aggregated from bars_1m)
- daily_features: One row per (trading_day, symbol, orb_minutes) (ORBs, session stats, RSI)

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

# ORB labels: Opening Range Breakout windows at event-based session times.
# All sessions are dynamic (DST-aware), resolved per-day by pipeline/dst.py.
#
# The ORB is the high-low range of the first N minutes (configurable).
# A "break" occurs when a 1-min bar closes above orb_high (long) or
# below orb_low (short). See pipeline/build_daily_features.py for logic.
# All 10 sessions are dynamic (DST-aware, resolver per-day).
# See pipeline/dst.py SESSION_CATALOG for the master registry.
#
#   CME_REOPEN      - CME Globex electronic reopen at 5:00 PM CT
#   TOKYO_OPEN      - Tokyo Stock Exchange open at 9:00 AM JST
#   SINGAPORE_OPEN  - SGX/HKEX open at 9:00 AM SGT
#   LONDON_METALS   - London metals AM session at 8:00 AM London
#   US_DATA_830     - US economic data release at 8:30 AM ET
#   NYSE_OPEN       - NYSE cash open at 9:30 AM ET
#   US_DATA_1000    - US 10:00 AM data (ISM/CC) + post-equity-open flow
#   COMEX_SETTLE    - COMEX gold settlement at 1:30 PM ET (MGC)
#   CME_PRECLOSE    - CME equity futures pre-settlement at 2:45 PM CT
#   NYSE_CLOSE      - NYSE closing bell at 4:00 PM ET
ORB_LABELS_DYNAMIC = [
    "CME_REOPEN", "TOKYO_OPEN", "SINGAPORE_OPEN", "LONDON_METALS",
    "US_DATA_830", "NYSE_OPEN", "US_DATA_1000", "COMEX_SETTLE",
    "CME_PRECLOSE", "NYSE_CLOSE",
]

# Combined label list — used by schema generation and feature builders
ORB_LABELS = ORB_LABELS_DYNAMIC

PROSPECTIVE_SIGNALS_SCHEMA = """
CREATE TABLE IF NOT EXISTS prospective_signals (
    signal_id        VARCHAR NOT NULL,
    trading_day      DATE NOT NULL,
    symbol           VARCHAR NOT NULL,
    session          VARCHAR NOT NULL,
    prev_day_outcome VARCHAR NOT NULL,
    orb_size         DOUBLE,
    entry_model      VARCHAR NOT NULL,
    confirm_bars     INTEGER NOT NULL,
    rr_target        DOUBLE NOT NULL,
    outcome          VARCHAR,
    pnl_r            DOUBLE,
    is_prospective   BOOLEAN NOT NULL,
    freeze_date      DATE NOT NULL,
    created_at       TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (signal_id, trading_day)
);
"""

def _build_daily_features_ddl() -> str:
    """Generate CREATE TABLE DDL for daily_features.

    Columns per ORB (14 each):
      orb_{label}_high              - ORB range high
      orb_{label}_low               - ORB range low
      orb_{label}_size              - high - low (points)
      orb_{label}_volume            - total contracts traded during ORB window
      orb_{label}_break_dir         - 'long', 'short', or NULL (no break)
      orb_{label}_break_ts          - timestamp of first break (1m close outside range)
      orb_{label}_break_delay_min   - minutes from ORB end to first break (NULL if no break)
      orb_{label}_break_bar_continues - break bar closes in break direction (True/False/NULL)
      orb_{label}_break_bar_volume  - volume on the bar that broke the ORB (NULL if no break)
      orb_{label}_outcome           - outcome at RR=1.0 ('win', 'loss', 'scratch', NULL)
      orb_{label}_mae_r             - max adverse excursion in R (NULL until cost model)
      orb_{label}_mfe_r             - max favorable excursion in R (NULL until cost model)
      orb_{label}_double_break      - True if BOTH ORB high and low were breached
      rel_vol_{label}               - break_bar_volume / rolling 20-session median (NULL if no break
                                      or < 5 prior break-days). Relative volume signal.
    """
    orb_cols = []
    for label in ORB_LABELS:
        orb_cols.extend([
            f"    orb_{label}_high              DOUBLE,",
            f"    orb_{label}_low               DOUBLE,",
            f"    orb_{label}_size              DOUBLE,",
            f"    orb_{label}_volume            BIGINT,",
            f"    orb_{label}_break_dir         TEXT,",
            f"    orb_{label}_break_ts          TIMESTAMPTZ,",
            f"    orb_{label}_break_delay_min   DOUBLE,",
            f"    orb_{label}_break_bar_continues BOOLEAN,",
            f"    orb_{label}_break_bar_volume  BIGINT,",
            f"    orb_{label}_outcome           TEXT,",
            f"    orb_{label}_mae_r             DOUBLE,",
            f"    orb_{label}_mfe_r             DOUBLE,",
            f"    orb_{label}_double_break      BOOLEAN,",
            f"    rel_vol_{label}               DOUBLE,",
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

    -- RSI (Wilder's 14-period on 5m closes, computed at CME_REOPEN time)
    rsi_14_at_0900    DOUBLE,  -- column name historical; value is at CME_REOPEN

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

    -- ATR Velocity regime (Feb 2026 — research_avoid_crosscheck.py)
    -- atr_vel_ratio  = atr_20 / avg(prior 5 days atr_20). Prior-days only, no look-ahead.
    -- atr_vel_regime = 'Expanding' (>1.05), 'Contracting' (<0.95), 'Stable' (else), NULL (<5 prior days)
    -- Used by ATRVelocityFilter to skip sessions when vol is actively de-volatilizing.
    atr_vel_ratio     DOUBLE,
    atr_vel_regime    TEXT,

    -- Per-session ORB compression tier (Feb 2026 — research_avoid_crosscheck.py)
    -- z-score of (orb_size/atr_20) vs rolling prior-20-day mean/std. Prior-days only.
    -- 'Compressed' (z < -0.5), 'Neutral' (-0.5..0.5), 'Expanded' (z > 0.5), NULL (<5 prior days)
    -- Used by ATRVelocityFilter: Contracting+Neutral/Compressed = skip.
    orb_CME_REOPEN_compression_z    DOUBLE,
    orb_CME_REOPEN_compression_tier TEXT,
    orb_TOKYO_OPEN_compression_z    DOUBLE,
    orb_TOKYO_OPEN_compression_tier TEXT,
    orb_LONDON_METALS_compression_z    DOUBLE,
    orb_LONDON_METALS_compression_tier TEXT,

    -- DST flags: whether US/UK was in daylight saving time on this trading day.
    -- Used by dynamic sessions (NYSE_OPEN, US_DATA_830, LONDON_METALS)
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

    -- Prior day reference levels (post-pass: requires prior row)
    prev_day_high       DOUBLE,
    prev_day_low        DOUBLE,
    prev_day_close      DOUBLE,
    prev_day_range      DOUBLE,
    prev_day_direction  TEXT,
    gap_type            TEXT,

    -- Pre-session activity (same-day: Asia window + pre-1000 window)
    overnight_high           DOUBLE,
    overnight_low            DOUBLE,
    overnight_range          DOUBLE,
    pre_1000_high            DOUBLE,
    pre_1000_low             DOUBLE,

    -- Liquidity sweep labels (post-pass: needs prev_day_high/low)
    took_pdh_before_1000     BOOLEAN,
    took_pdl_before_1000     BOOLEAN,
    overnight_took_pdh       BOOLEAN,
    overnight_took_pdl       BOOLEAN,

    -- Retrospective day type (post-pass: needs atr_20)
    -- 'TREND_UP' / 'TREND_DOWN' / 'BALANCED' / 'REVERSAL_UP' / 'REVERSAL_DOWN' / 'NON_TREND'
    -- NOTE: look-ahead relative to intraday entry — research only, not a live filter
    day_type            TEXT,

    -- GARCH(1,1) forward volatility forecast (Feb 2026)
    -- 1-step-ahead conditional vol from trailing 252 daily close-to-close log returns.
    -- garch_atr_ratio = garch_forecast_vol / atr_20 (regime comparison signal).
    -- NULL during warm-up (<252 prior closes) or fit failure. Research only.
    garch_forecast_vol  DOUBLE,
    garch_atr_ratio     DOUBLE,

    -- ORB columns (10 dynamic sessions x 14 columns = 140)
{orb_block}

    PRIMARY KEY (symbol, trading_day, orb_minutes)
);
"""

DAILY_FEATURES_SCHEMA = _build_daily_features_ddl()

def init_db(db_path: Path, force: bool = False):
    """Initialize database with schema."""

    logger.info("=" * 60)
    logger.info("DATABASE INITIALIZATION")
    logger.info("=" * 60)
    logger.info(f"Database path: {db_path}")
    logger.info(f"Force recreate: {force}")

    # Connect to database (creates file if doesn't exist)
    with duckdb.connect(str(db_path)) as con:

        if force:
            logger.info("FORCE MODE: Dropping ALL tables...")
            # Drop trading_app tables first (FK dependencies on daily_features)
            for t in ["validated_setups_archive", "validated_setups",
                       "experimental_strategies", "orb_outcomes",
                       "prospective_signals"]:
                con.execute(f"DROP TABLE IF EXISTS {t}")
            # Drop pipeline tables
            con.execute("DROP TABLE IF EXISTS daily_features")
            con.execute("DROP TABLE IF EXISTS bars_5m")
            con.execute("DROP TABLE IF EXISTS bars_1m")
            logger.info("  All tables dropped (pipeline + trading_app).")

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

        # Migration: add ATR velocity + compression tier columns (Feb 2026)
        for col, typedef in [
            ("atr_vel_ratio",             "DOUBLE"),
            ("atr_vel_regime",            "TEXT"),
            ("orb_CME_REOPEN_compression_z",    "DOUBLE"),
            ("orb_CME_REOPEN_compression_tier", "TEXT"),
            ("orb_TOKYO_OPEN_compression_z",    "DOUBLE"),
            ("orb_TOKYO_OPEN_compression_tier", "TEXT"),
            ("orb_LONDON_METALS_compression_z",    "DOUBLE"),
            ("orb_LONDON_METALS_compression_tier", "TEXT"),
        ]:
            try:
                con.execute(f"ALTER TABLE daily_features ADD COLUMN {col} {typedef}")
                logger.info(f"  Migration: added {col} column to daily_features")
            except duckdb.CatalogException:
                pass  # column already exists

        # Migration: add break_delay_min + break_bar_continues (were in DDL but never migrated)
        # 10 sessions × 2 columns = 20 columns
        for label in ORB_LABELS:
            for col, typedef in [
                (f"orb_{label}_break_delay_min",   "DOUBLE"),
                (f"orb_{label}_break_bar_continues", "BOOLEAN"),
            ]:
                try:
                    con.execute(f"ALTER TABLE daily_features ADD COLUMN {col} {typedef}")
                    logger.info(f"  Migration: added {col} column to daily_features")
                except duckdb.CatalogException:
                    pass  # column already exists

        # Migration: add per-session volume + relative volume columns (Feb 2026)
        # 10 sessions × 3 columns = 30 new columns
        for label in ORB_LABELS:
            for col, typedef in [
                (f"orb_{label}_volume",           "BIGINT"),
                (f"orb_{label}_break_bar_volume", "BIGINT"),
                (f"rel_vol_{label}",              "DOUBLE"),
            ]:
                try:
                    con.execute(f"ALTER TABLE daily_features ADD COLUMN {col} {typedef}")
                    logger.info(f"  Migration: added {col} column to daily_features")
                except duckdb.CatalogException:
                    pass  # column already exists

        # Migration: add Market Profile context columns (Feb 2026)
        for col, typedef in [
            ("prev_day_high",           "DOUBLE"),
            ("prev_day_low",            "DOUBLE"),
            ("prev_day_close",          "DOUBLE"),
            ("prev_day_range",          "DOUBLE"),
            ("prev_day_direction",      "TEXT"),
            ("gap_type",                "TEXT"),
            ("overnight_high",          "DOUBLE"),
            ("overnight_low",           "DOUBLE"),
            ("overnight_range",         "DOUBLE"),
            ("pre_1000_high",           "DOUBLE"),
            ("pre_1000_low",            "DOUBLE"),
            ("took_pdh_before_1000",    "BOOLEAN"),
            ("took_pdl_before_1000",    "BOOLEAN"),
            ("overnight_took_pdh",      "BOOLEAN"),
            ("overnight_took_pdl",      "BOOLEAN"),
            ("day_type",                "TEXT"),
        ]:
            try:
                con.execute(f"ALTER TABLE daily_features ADD COLUMN {col} {typedef}")
                logger.info(f"  Migration: added {col} column to daily_features")
            except duckdb.CatalogException:
                pass  # column already exists

        # Migration: add GARCH forecast columns (Feb 2026)
        for col, typedef in [
            ("garch_forecast_vol",  "DOUBLE"),
            ("garch_atr_ratio",     "DOUBLE"),
        ]:
            try:
                con.execute(f"ALTER TABLE daily_features ADD COLUMN {col} {typedef}")
                logger.info(f"  Migration: added {col} column to daily_features")
            except duckdb.CatalogException:
                pass  # column already exists

        # Migration: add all core ORB columns for any new sessions (Feb 2026)
        # Handles COMEX_SETTLE, NYSE_CLOSE, and any future additions to ORB_LABELS.
        for label in ORB_LABELS:
            for col, typedef in [
                (f"orb_{label}_high",         "DOUBLE"),
                (f"orb_{label}_low",          "DOUBLE"),
                (f"orb_{label}_size",         "DOUBLE"),
                (f"orb_{label}_break_dir",    "TEXT"),
                (f"orb_{label}_break_ts",     "TIMESTAMPTZ"),
                (f"orb_{label}_outcome",      "TEXT"),
                (f"orb_{label}_mae_r",        "DOUBLE"),
                (f"orb_{label}_mfe_r",        "DOUBLE"),
                (f"orb_{label}_double_break", "BOOLEAN"),
            ]:
                try:
                    con.execute(f"ALTER TABLE daily_features ADD COLUMN {col} {typedef}")
                    logger.info(f"  Migration: added {col} column to daily_features")
                except duckdb.CatalogException:
                    pass  # column already exists

        con.execute(PROSPECTIVE_SIGNALS_SCHEMA)
        logger.info("  prospective_signals: created (or already exists)")

        con.commit()

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


    logger.info("=" * 60)
    logger.info("INITIALIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info("Next steps:")
    logger.info("  1. Run ingestion: python pipeline/ingest_dbn.py --instrument MGC")
    logger.info("  2. Check database: python pipeline/health_check.py")

def main():
    parser = argparse.ArgumentParser(description="Initialize DuckDB schema")
    parser.add_argument("--force", action="store_true", help="Drop and recreate tables")
    args = parser.parse_args()

    init_db(GOLD_DB_PATH, force=args.force)

if __name__ == "__main__":
    main()
