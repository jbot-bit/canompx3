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

from pipeline.log import get_logger

# Add project root to path
from pipeline.paths import GOLD_DB_PATH

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
# All 12 sessions are dynamic (DST-aware, resolver per-day).
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
#   BRISBANE_1025   - Fixed 10:25 AM Brisbane (not event-relative)
#   EUROPE_FLOW    - European flow adjacent to London metals
ORB_LABELS_DYNAMIC = [
    "CME_REOPEN",
    "TOKYO_OPEN",
    "SINGAPORE_OPEN",
    "LONDON_METALS",
    "EUROPE_FLOW",
    "US_DATA_830",
    "NYSE_OPEN",
    "US_DATA_1000",
    "COMEX_SETTLE",
    "CME_PRECLOSE",
    "NYSE_CLOSE",
    "BRISBANE_1025",
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

FAMILY_RR_LOCKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS family_rr_locks (
    instrument    TEXT    NOT NULL,
    orb_label     TEXT    NOT NULL,
    filter_type   TEXT    NOT NULL,
    entry_model   TEXT    NOT NULL,
    orb_minutes   INTEGER NOT NULL,
    confirm_bars  INTEGER NOT NULL,
    locked_rr     REAL    NOT NULL,
    method        TEXT    NOT NULL,
    sharpe_at_rr  REAL,
    maxdd_at_rr   REAL,
    n_at_rr       INTEGER,
    expr_at_rr    REAL,
    tpy_at_rr     REAL,
    updated_at    TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (instrument, orb_label, filter_type, entry_model, orb_minutes, confirm_bars)
);
"""

REBUILD_MANIFEST_SCHEMA = """
CREATE TABLE IF NOT EXISTS rebuild_manifest (
    rebuild_id      TEXT        PRIMARY KEY,
    instrument      TEXT        NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL,
    completed_at    TIMESTAMPTZ,
    status          TEXT        NOT NULL,
    failed_step     TEXT,
    steps_completed TEXT[],
    trigger         TEXT        NOT NULL
);
"""

PIPELINE_AUDIT_LOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS pipeline_audit_log (
    log_id        TEXT        PRIMARY KEY,
    timestamp     TIMESTAMPTZ NOT NULL,
    operation     TEXT        NOT NULL,
    table_name    TEXT        NOT NULL,
    instrument    TEXT,
    date_start    DATE,
    date_end      DATE,
    rows_before   INTEGER,
    rows_after    INTEGER,
    duration_s    DOUBLE,
    git_sha       TEXT,
    rebuild_id    TEXT,
    status        TEXT        NOT NULL
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
      rel_vol_{label}               - break_bar_volume / median(prior 20 bars_1m at same UTC
                                      minute-of-day). Matches strategy_discovery._compute_relative_volumes().
                                      NULL if no break or < 5 prior bars at that minute.
    """
    orb_cols = []
    for label in ORB_LABELS:
        orb_cols.extend(
            [
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
                f"    orb_{label}_vwap              DOUBLE,",
                f"    orb_{label}_pre_velocity      DOUBLE,",
            ]
        )
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
    rsi_14_at_CME_REOPEN  DOUBLE,

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

    -- ATR percentile: rolling rank of atr_20 among prior 252 trading days (0-100).
    -- Used by CombinedATRVolumeFilter (ATR70+VOL): trade only when atr_20_pct >= 70
    -- AND rel_vol >= 1.2. Prior-days only, no look-ahead.
    -- @research-source research/research_vol_regime_filter.py
    atr_20_pct        DOUBLE,

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

    -- Prior-week HTF level fields (post-pass: Monday-anchor via DATE_TRUNC('week'))
    -- Aggregated over fully-closed prior Mon-Sun calendar week (Sunday trading_day
    -- rows group with the ENDING Mon-Sun week per DuckDB DATE_TRUNC semantics).
    -- NULL until a fully-completed prior week exists. Price-safe (no volume input).
    prev_week_high      DOUBLE,
    prev_week_low       DOUBLE,
    prev_week_open      DOUBLE,
    prev_week_close     DOUBLE,
    prev_week_range     DOUBLE,
    prev_week_mid       DOUBLE,

    -- Prior-month HTF level fields (post-pass: calendar-month anchor).
    -- Aggregated over fully-closed prior calendar month. NULL until prior month exists.
    -- Price-safe (no volume input).
    prev_month_high     DOUBLE,
    prev_month_low      DOUBLE,
    prev_month_open     DOUBLE,
    prev_month_close    DOUBLE,
    prev_month_range    DOUBLE,
    prev_month_mid      DOUBLE,

    -- Pre-session activity (same-day: Asia window + pre-1000 window)
    overnight_high           DOUBLE,
    overnight_low            DOUBLE,
    overnight_range          DOUBLE,
    overnight_range_pct      DOUBLE,
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

    -- Exchange pit range / ATR (Apr 2026 — exchange_range_t2t8.py)
    -- (prev_day pit_session_high - pit_session_low) / atr_20
    -- From CME exchange statistics (Databento). Zero look-ahead: pit closes 21:00 UTC,
    -- CME_REOPEN starts 23:00 UTC. Used by PitRangeFilter at CME_REOPEN.
    -- @research-source scripts/research/exchange_range_t2t8.py
    pit_range_atr     DOUBLE,

    -- GARCH(1,1) forward volatility forecast (Feb 2026)
    -- 1-step-ahead conditional vol from trailing 252 daily close-to-close log returns.
    -- garch_atr_ratio = garch_forecast_vol / atr_20 (regime comparison signal).
    -- NULL during warm-up (<252 prior closes) or fit failure. Research only.
    garch_forecast_vol  DOUBLE,
    garch_atr_ratio     DOUBLE,

    -- GARCH forecast vol rolling percentile (Apr 2026 — Wave 5 G5 deployment).
    -- Rank of today's garch_forecast_vol among prior 252 trading days (0-100).
    -- Prior-only window, no look-ahead. Min 60 prior days for stable ranking.
    -- Same pattern as atr_20_pct / overnight_range_pct — regime-adaptive percentile
    -- avoids the cross-instrument threshold contamination that absolute GARCH cutoffs
    -- would cause (MNQ Q20 ~0.16 vs MES Q20 ~0.11, etc.).
    -- @research-source scripts/research/wave4_presession_t2t8.py (2026-04-11): Phase B
    -- MNQ NYSE_OPEN RR1.5 LOW garch_forecast_vol in_ExpR +0.240 WFE 1.00 p=0.042.
    garch_forecast_vol_pct  DOUBLE,

    -- ORB columns (12 dynamic sessions x 14 columns = 168)
{orb_block}

    PRIMARY KEY (symbol, trading_day, orb_minutes)
);
"""


DAILY_FEATURES_SCHEMA = _build_daily_features_ddl()


EXCHANGE_STATISTICS_SCHEMA = """
CREATE TABLE IF NOT EXISTS exchange_statistics (
    cal_date             DATE    NOT NULL,
    symbol               TEXT    NOT NULL,
    session_high         DOUBLE,
    session_low          DOUBLE,
    settlement           DOUBLE,
    opening_price        DOUBLE,
    indicative_open      DOUBLE,
    cleared_volume       BIGINT,
    open_interest        BIGINT,
    total_cleared_volume BIGINT,
    front_contract       TEXT,
    PRIMARY KEY (symbol, cal_date)
);
"""


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
            # Query for all user tables — avoids hardcoded list going stale
            all_tables = [
                row[0]
                for row in con.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
                ).fetchall()
            ]
            for t in all_tables:
                con.execute(f"DROP TABLE IF EXISTS {t}")
                logger.info(f"  Dropped: {t}")
            logger.info(f"  All {len(all_tables)} tables dropped.")

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
            ("atr_vel_ratio", "DOUBLE"),
            ("atr_vel_regime", "TEXT"),
            ("orb_CME_REOPEN_compression_z", "DOUBLE"),
            ("orb_CME_REOPEN_compression_tier", "TEXT"),
            ("orb_TOKYO_OPEN_compression_z", "DOUBLE"),
            ("orb_TOKYO_OPEN_compression_tier", "TEXT"),
            ("orb_LONDON_METALS_compression_z", "DOUBLE"),
            ("orb_LONDON_METALS_compression_tier", "TEXT"),
        ]:
            try:
                con.execute(f"ALTER TABLE daily_features ADD COLUMN {col} {typedef}")
                logger.info(f"  Migration: added {col} column to daily_features")
            except duckdb.CatalogException:
                pass  # column already exists

        # Migration: add break_delay_min + break_bar_continues (were in DDL but never migrated)
        # 12 sessions × 2 columns = 24 columns
        for label in ORB_LABELS:
            for col, typedef in [
                (f"orb_{label}_break_delay_min", "DOUBLE"),
                (f"orb_{label}_break_bar_continues", "BOOLEAN"),
            ]:
                try:
                    con.execute(f"ALTER TABLE daily_features ADD COLUMN {col} {typedef}")
                    logger.info(f"  Migration: added {col} column to daily_features")
                except duckdb.CatalogException:
                    pass  # column already exists

        # Migration: add per-session volume + relative volume columns (Feb 2026)
        # 12 sessions × 3 columns = 36 new columns
        for label in ORB_LABELS:
            for col, typedef in [
                (f"orb_{label}_volume", "BIGINT"),
                (f"orb_{label}_break_bar_volume", "BIGINT"),
                (f"rel_vol_{label}", "DOUBLE"),
            ]:
                try:
                    con.execute(f"ALTER TABLE daily_features ADD COLUMN {col} {typedef}")
                    logger.info(f"  Migration: added {col} column to daily_features")
                except duckdb.CatalogException:
                    pass  # column already exists

        # Migration: add Market Profile context columns (Feb 2026)
        for col, typedef in [
            ("prev_day_high", "DOUBLE"),
            ("prev_day_low", "DOUBLE"),
            ("prev_day_close", "DOUBLE"),
            ("prev_day_range", "DOUBLE"),
            ("prev_day_direction", "TEXT"),
            ("gap_type", "TEXT"),
            ("overnight_high", "DOUBLE"),
            ("overnight_low", "DOUBLE"),
            ("overnight_range", "DOUBLE"),
            ("overnight_range_pct", "DOUBLE"),
            ("pre_1000_high", "DOUBLE"),
            ("pre_1000_low", "DOUBLE"),
            ("took_pdh_before_1000", "BOOLEAN"),
            ("took_pdl_before_1000", "BOOLEAN"),
            ("overnight_took_pdh", "BOOLEAN"),
            ("overnight_took_pdl", "BOOLEAN"),
            ("day_type", "TEXT"),
        ]:
            try:
                con.execute(f"ALTER TABLE daily_features ADD COLUMN {col} {typedef}")
                logger.info(f"  Migration: added {col} column to daily_features")
            except duckdb.CatalogException:
                pass  # column already exists

        # Migration: add ATR percentile column (Mar 2026)
        try:
            con.execute("ALTER TABLE daily_features ADD COLUMN atr_20_pct DOUBLE")
            logger.info("  Migration: added atr_20_pct column to daily_features")
        except duckdb.CatalogException:
            pass  # column already exists

        # Migration: add GARCH forecast columns (Feb 2026)
        # + garch_forecast_vol_pct (Apr 2026 — Wave 5 G5 deployment).
        for col, typedef in [
            ("garch_forecast_vol", "DOUBLE"),
            ("garch_atr_ratio", "DOUBLE"),
            ("garch_forecast_vol_pct", "DOUBLE"),
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
                (f"orb_{label}_high", "DOUBLE"),
                (f"orb_{label}_low", "DOUBLE"),
                (f"orb_{label}_size", "DOUBLE"),
                (f"orb_{label}_break_dir", "TEXT"),
                (f"orb_{label}_break_ts", "TIMESTAMPTZ"),
                (f"orb_{label}_outcome", "TEXT"),
                (f"orb_{label}_mae_r", "DOUBLE"),
                (f"orb_{label}_mfe_r", "DOUBLE"),
                (f"orb_{label}_double_break", "BOOLEAN"),
            ]:
                try:
                    con.execute(f"ALTER TABLE daily_features ADD COLUMN {col} {typedef}")
                    logger.info(f"  Migration: added {col} column to daily_features")
                except duckdb.CatalogException:
                    pass  # column already exists

        # Migration: add VWAP + pre-velocity columns (Mar 2026)
        for label in ORB_LABELS:
            for col, typedef in [
                (f"orb_{label}_vwap", "DOUBLE"),
                (f"orb_{label}_pre_velocity", "DOUBLE"),
            ]:
                try:
                    con.execute(f"ALTER TABLE daily_features ADD COLUMN {col} {typedef}")
                    logger.info(f"  Migration: added {col} column to daily_features")
                except duckdb.CatalogException:
                    pass  # column already exists

        con.execute(EXCHANGE_STATISTICS_SCHEMA)
        logger.info("  exchange_statistics: created (or already exists)")

        # Migration: add pit_range_atr column (Apr 2026 — F5 exchange pit range)
        try:
            con.execute("ALTER TABLE daily_features ADD COLUMN pit_range_atr DOUBLE")
            logger.info("  Migration: added pit_range_atr column to daily_features")
        except duckdb.CatalogException:
            pass  # column already exists

        # Migration: add HTF prev-week / prev-month level columns (Apr 2026 — Path A)
        for col in [
            "prev_week_high",
            "prev_week_low",
            "prev_week_open",
            "prev_week_close",
            "prev_week_range",
            "prev_week_mid",
            "prev_month_high",
            "prev_month_low",
            "prev_month_open",
            "prev_month_close",
            "prev_month_range",
            "prev_month_mid",
        ]:
            try:
                con.execute(f"ALTER TABLE daily_features ADD COLUMN {col} DOUBLE")
                logger.info(f"  Migration: added {col} column to daily_features")
            except duckdb.CatalogException:
                pass  # column already exists

        con.execute(PROSPECTIVE_SIGNALS_SCHEMA)
        logger.info("  prospective_signals: created (or already exists)")

        con.execute(FAMILY_RR_LOCKS_SCHEMA)
        logger.info("  family_rr_locks: created (or already exists)")

        con.execute(REBUILD_MANIFEST_SCHEMA)
        logger.info("  rebuild_manifest: created (or already exists)")

        con.execute(PIPELINE_AUDIT_LOG_SCHEMA)
        logger.info("  pipeline_audit_log: created (or already exists)")

        con.commit()

        # Verify schema
        logger.info("Verifying schema...")

        tables = con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()
        table_names = [t[0] for t in tables]

        logger.info(f"  Tables found: {table_names}")

        # Check columns for each table
        for table_name in ["bars_1m", "bars_5m", "daily_features"]:
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
