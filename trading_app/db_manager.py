"""
Database schema manager for trading_app tables.

Creates and manages:
- orb_outcomes: Pre-computed outcomes for all RR targets × confirm_bars combinations
- experimental_strategies: Backtest results awaiting validation
- validated_setups: Production-ready strategies
- validated_setups_archive: Historical audit trail
- strategy_trade_days: Ground truth post-filter trade days
- edge_families: Strategy clustering by trade-day hash
"""

import sys
from pathlib import Path

from pipeline.log import get_logger
logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import duckdb
from pipeline.paths import GOLD_DB_PATH

def compute_trade_day_hash(days: list) -> str:
    """Compute deterministic MD5 hash of sorted trade-day list.

    Canonical implementation — used by strategy_discovery and build_edge_families.
    """
    import hashlib
    day_str = ",".join(str(d) for d in sorted(days))
    return hashlib.md5(day_str.encode()).hexdigest()

def init_trading_app_schema(db_path: Path | None = None, force: bool = False) -> None:
    """
    Create trading_app tables if they don't exist.

    Args:
        db_path: Path to DuckDB database (default: from paths.py)
        force: If True, drop existing tables first (WARNING: destroys data)
    """
    if db_path is None:
        db_path = GOLD_DB_PATH

    with duckdb.connect(str(db_path)) as con:
        if force:
            logger.warning("WARN: Force mode: Dropping existing trading_app tables...")
            con.execute("DROP TABLE IF EXISTS edge_families")
            con.execute("DROP TABLE IF EXISTS strategy_trade_days")
            con.execute("DROP TABLE IF EXISTS validated_setups_archive")
            con.execute("DROP TABLE IF EXISTS validated_setups")
            con.execute("DROP TABLE IF EXISTS experimental_strategies")
            con.execute("DROP TABLE IF EXISTS orb_outcomes")

        # Table 1: orb_outcomes
        con.execute("""
            CREATE TABLE IF NOT EXISTS orb_outcomes (
                trading_day       DATE        NOT NULL,
                symbol            TEXT        NOT NULL,
                orb_label         TEXT        NOT NULL,
                orb_minutes       INTEGER     NOT NULL,
                rr_target         DOUBLE      NOT NULL,
                confirm_bars      INTEGER     NOT NULL,
                entry_model       TEXT        NOT NULL,

                -- Entry details (NULL if no entry signal)
                entry_ts          TIMESTAMPTZ,
                entry_price       DOUBLE,
                stop_price        DOUBLE,
                target_price      DOUBLE,

                -- Outcome
                outcome           TEXT,
                exit_ts           TIMESTAMPTZ,
                exit_price        DOUBLE,
                pnl_r             DOUBLE,

                -- Dollar amounts (per-contract)
                risk_dollars      DOUBLE,
                pnl_dollars       DOUBLE,

                -- Excursions
                mae_r             DOUBLE,
                mfe_r             DOUBLE,

                PRIMARY KEY (symbol, trading_day, orb_label, orb_minutes, rr_target, confirm_bars, entry_model),
                FOREIGN KEY (symbol, trading_day, orb_minutes)
                    REFERENCES daily_features(symbol, trading_day, orb_minutes)
            )
        """)

        # Table 2: experimental_strategies
        con.execute("""
            CREATE TABLE IF NOT EXISTS experimental_strategies (
                strategy_id       TEXT        PRIMARY KEY,
                created_at        TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

                -- Strategy parameters
                instrument        TEXT        NOT NULL,
                orb_label         TEXT        NOT NULL,
                orb_minutes       INTEGER     NOT NULL,
                rr_target         DOUBLE      NOT NULL,
                confirm_bars      INTEGER     NOT NULL,
                entry_model       TEXT        NOT NULL,

                -- Filters
                filter_type       TEXT,
                filter_params     TEXT,

                -- Backtest results
                sample_size       INTEGER,
                win_rate          DOUBLE,
                avg_win_r         DOUBLE,
                avg_loss_r        DOUBLE,
                expectancy_r      DOUBLE,
                sharpe_ratio      DOUBLE,
                max_drawdown_r    DOUBLE,
                median_risk_points DOUBLE,
                avg_risk_points   DOUBLE,

                -- Dollar aggregates (per-contract)
                median_risk_dollars DOUBLE,
                avg_risk_dollars  DOUBLE,
                avg_win_dollars   DOUBLE,
                avg_loss_dollars  DOUBLE,

                -- Annualized metrics
                trades_per_year   DOUBLE,
                sharpe_ann        DOUBLE,

                -- Yearly breakdown (JSON)
                yearly_results    TEXT,

                -- Entry signal counts (sample_size = wins + losses only)
                entry_signals     INTEGER,
                scratch_count     INTEGER,
                early_exit_count  INTEGER,

                -- Trade-day hash dedup
                trade_day_hash    TEXT,
                is_canonical      BOOLEAN     DEFAULT TRUE,
                canonical_strategy_id TEXT,

                -- DST regime split (Feb 2026)
                dst_winter_n      INTEGER,
                dst_winter_avg_r  DOUBLE,
                dst_summer_n      INTEGER,
                dst_summer_avg_r  DOUBLE,
                dst_verdict       TEXT,

                -- Validation status
                validation_status TEXT,
                validation_notes  TEXT
            )
        """)

        # Table 3: validated_setups
        con.execute("""
            CREATE TABLE IF NOT EXISTS validated_setups (
                strategy_id       TEXT        PRIMARY KEY,
                promoted_at       TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                promoted_from     TEXT,

                -- Strategy parameters (denormalized)
                instrument        TEXT        NOT NULL,
                orb_label         TEXT        NOT NULL,
                orb_minutes       INTEGER     NOT NULL,
                rr_target         DOUBLE      NOT NULL,
                confirm_bars      INTEGER     NOT NULL,
                entry_model       TEXT        NOT NULL,
                filter_type       TEXT        NOT NULL,
                filter_params     TEXT,

                -- Validation results
                sample_size       INTEGER     NOT NULL,
                win_rate          DOUBLE      NOT NULL,
                expectancy_r      DOUBLE      NOT NULL,
                years_tested      INTEGER     NOT NULL,
                all_years_positive BOOLEAN    NOT NULL,
                stress_test_passed BOOLEAN    NOT NULL,

                -- Performance
                sharpe_ratio      DOUBLE,
                max_drawdown_r    DOUBLE,
                trades_per_year   DOUBLE,
                sharpe_ann        DOUBLE,
                yearly_results    TEXT,

                -- Dollar aggregates (per-contract)
                median_risk_dollars DOUBLE,
                avg_risk_dollars  DOUBLE,
                avg_win_dollars   DOUBLE,
                avg_loss_dollars  DOUBLE,

                -- Execution spec (JSON)
                execution_spec    TEXT,

                -- Edge family membership
                family_hash       TEXT,
                is_family_head    BOOLEAN     DEFAULT FALSE,

                -- Regime waivers (Phase 3)
                regime_waivers    TEXT,
                regime_waiver_count INTEGER DEFAULT 0,

                -- DST regime split (Feb 2026)
                dst_winter_n      INTEGER,
                dst_winter_avg_r  DOUBLE,
                dst_summer_n      INTEGER,
                dst_summer_avg_r  DOUBLE,
                dst_verdict       TEXT,

                -- Status
                status            TEXT        NOT NULL,
                retired_at        TIMESTAMPTZ,
                retirement_reason TEXT,

                FOREIGN KEY (promoted_from)
                    REFERENCES experimental_strategies(strategy_id)
            )
        """)

        # Table 4: validated_setups_archive
        con.execute("""
            CREATE TABLE IF NOT EXISTS validated_setups_archive (
                archive_id        TEXT        PRIMARY KEY,
                archived_at       TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                original_strategy_id TEXT     NOT NULL,
                archive_reason    TEXT,

                -- Full snapshot (JSON)
                setup_snapshot    TEXT        NOT NULL,

                FOREIGN KEY (original_strategy_id)
                    REFERENCES validated_setups(strategy_id)
            )
        """)

        # Table 5: strategy_trade_days (ground truth post-filter trade days)
        con.execute("""
            CREATE TABLE IF NOT EXISTS strategy_trade_days (
                strategy_id       TEXT        NOT NULL,
                trading_day       DATE        NOT NULL,
                PRIMARY KEY (strategy_id, trading_day)
            )
        """)
        con.execute("""
            CREATE INDEX IF NOT EXISTS idx_std_strategy
            ON strategy_trade_days(strategy_id)
        """)
        con.execute("""
            CREATE INDEX IF NOT EXISTS idx_std_day
            ON strategy_trade_days(trading_day)
        """)

        # Table 6: edge_families (strategy clustering by trade-day hash)
        con.execute("""
            CREATE TABLE IF NOT EXISTS edge_families (
                family_hash       TEXT        PRIMARY KEY,
                instrument        TEXT        NOT NULL,
                member_count      INTEGER     NOT NULL,
                trade_day_count   INTEGER     NOT NULL,
                head_strategy_id  TEXT        NOT NULL,
                head_expectancy_r DOUBLE,
                head_sharpe_ann   DOUBLE,

                -- Robustness metrics
                robustness_status   TEXT      DEFAULT 'PENDING',
                cv_expectancy       DOUBLE,
                median_expectancy_r DOUBLE,
                avg_sharpe_ann      DOUBLE,
                min_member_trades   INTEGER,
                trade_tier          TEXT      DEFAULT 'PENDING',

                created_at        TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (head_strategy_id)
                    REFERENCES validated_setups(strategy_id)
            )
        """)

        # Migration: add regime waiver columns (for existing DBs)
        for col, typedef in [
            ("regime_waivers", "TEXT"),
            ("regime_waiver_count", "INTEGER DEFAULT 0"),
        ]:
            try:
                con.execute(f"ALTER TABLE validated_setups ADD COLUMN {col} {typedef}")
            except duckdb.CatalogException:
                pass  # column already exists

        # Migration: add DST regime split columns (Feb 2026)
        dst_cols = [
            ("dst_winter_n", "INTEGER"),
            ("dst_winter_avg_r", "DOUBLE"),
            ("dst_summer_n", "INTEGER"),
            ("dst_summer_avg_r", "DOUBLE"),
            ("dst_verdict", "TEXT"),
        ]
        for table in ["experimental_strategies", "validated_setups"]:
            for col, typedef in dst_cols:
                try:
                    con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typedef}")
                except duckdb.CatalogException:
                    pass  # column already exists

        # Migration: add DOW columns on daily_features (Feb 2026 DOW research)
        for col, typedef in [("is_monday", "BOOLEAN"), ("is_tuesday", "BOOLEAN"), ("day_of_week", "INTEGER")]:
            try:
                con.execute(f"ALTER TABLE daily_features ADD COLUMN {col} {typedef}")
            except duckdb.CatalogException:
                pass
        # Backfill day_of_week from trading_day for existing rows
        con.execute("""
            UPDATE daily_features
            SET day_of_week = EXTRACT(ISODOW FROM trading_day::DATE) - 1,
                is_monday = (EXTRACT(ISODOW FROM trading_day::DATE) = 1),
                is_tuesday = (EXTRACT(ISODOW FROM trading_day::DATE) = 2)
            WHERE day_of_week IS NULL
        """)

        # Migration: add dollar columns (Feb 2026)
        for col, typedef in [
            ("risk_dollars", "DOUBLE"),
            ("pnl_dollars", "DOUBLE"),
        ]:
            try:
                con.execute(f"ALTER TABLE orb_outcomes ADD COLUMN {col} {typedef}")
            except duckdb.CatalogException:
                pass  # column already exists

        dollar_agg_cols = [
            ("median_risk_dollars", "DOUBLE"),
            ("avg_risk_dollars", "DOUBLE"),
            ("avg_win_dollars", "DOUBLE"),
            ("avg_loss_dollars", "DOUBLE"),
        ]
        for table in ["experimental_strategies", "validated_setups"]:
            for col, typedef in dollar_agg_cols:
                try:
                    con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typedef}")
                except duckdb.CatalogException:
                    pass  # column already exists

        # Migration: add statistical honesty columns (Feb 2026 audit fixes)
        # F-04: p_value on experimental_strategies (t-test H0: mean_pnl_r = 0)
        # F-11: sharpe_ann_adj + autocorr_lag1 on experimental_strategies
        # F-01: fdr_significant + fdr_adjusted_p on validated_setups
        audit_exp_cols = [
            ("p_value", "DOUBLE"),
            ("sharpe_ann_adj", "DOUBLE"),
            ("autocorr_lag1", "DOUBLE"),
        ]
        for col, typedef in audit_exp_cols:
            try:
                con.execute(f"ALTER TABLE experimental_strategies ADD COLUMN {col} {typedef}")
            except duckdb.CatalogException:
                pass  # column already exists

        audit_val_cols = [
            ("fdr_significant", "BOOLEAN"),
            ("fdr_adjusted_p", "DOUBLE"),
            ("p_value", "DOUBLE"),
            ("sharpe_ann_adj", "DOUBLE"),
        ]
        for col, typedef in audit_val_cols:
            try:
                con.execute(f"ALTER TABLE validated_setups ADD COLUMN {col} {typedef}")
            except duckdb.CatalogException:
                pass  # column already exists

        # Migration: add ambiguous_bar tracking (F-05 audit fix)
        # Flags outcomes where both target and stop hit in same 1m bar
        try:
            con.execute("ALTER TABLE orb_outcomes ADD COLUMN ambiguous_bar BOOLEAN DEFAULT FALSE")
        except duckdb.CatalogException:
            pass  # column already exists

        con.commit()
        logger.info("Trading app schema initialized successfully")

def verify_trading_app_schema(db_path: Path | None = None) -> tuple[bool, list[str]]:
    """
    Verify all trading_app tables exist with correct schema.

    Returns:
        (all_valid, violations): violations is empty if all_valid=True
    """
    if db_path is None:
        db_path = GOLD_DB_PATH

    with duckdb.connect(str(db_path), read_only=True) as con:
        violations = []

        expected_tables = [
            "orb_outcomes",
            "experimental_strategies",
            "validated_setups",
            "validated_setups_archive",
            "strategy_trade_days",
            "edge_families",
        ]

        # Check tables exist
        result = con.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
        """).fetchall()
        existing_tables = {row[0] for row in result}

        for table in expected_tables:
            if table not in existing_tables:
                violations.append(f"Missing table: {table}")

        # Check orb_outcomes schema
        if "orb_outcomes" in existing_tables:
            result = con.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'orb_outcomes'
                ORDER BY ordinal_position
            """).fetchall()

            expected_cols = {
                "trading_day", "symbol", "orb_label", "orb_minutes",
                "rr_target", "confirm_bars", "entry_model", "entry_ts",
                "entry_price", "stop_price", "target_price", "outcome",
                "exit_ts", "exit_price", "pnl_r", "mae_r", "mfe_r",
                "risk_dollars", "pnl_dollars"
            }
            actual_cols = {row[0] for row in result}

            missing = expected_cols - actual_cols
            if missing:
                violations.append(f"orb_outcomes missing columns: {missing}")

        # Check experimental_strategies schema
        if "experimental_strategies" in existing_tables:
            result = con.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'experimental_strategies'
            """).fetchall()

            expected_cols = {
                "strategy_id", "created_at", "instrument", "orb_label",
                "orb_minutes", "rr_target", "confirm_bars", "entry_model",
                "filter_type", "filter_params", "sample_size", "win_rate",
                "avg_win_r", "avg_loss_r", "expectancy_r", "sharpe_ratio",
                "max_drawdown_r", "median_risk_points", "avg_risk_points",
                "trades_per_year", "sharpe_ann",
                "yearly_results",
                "entry_signals", "scratch_count", "early_exit_count",
                "trade_day_hash", "is_canonical", "canonical_strategy_id",
                "dst_winter_n", "dst_winter_avg_r",
                "dst_summer_n", "dst_summer_avg_r", "dst_verdict",
                "validation_status", "validation_notes"
            }
            actual_cols = {row[0] for row in result}

            missing = expected_cols - actual_cols
            if missing:
                violations.append(f"experimental_strategies missing columns: {missing}")

        # Check validated_setups schema
        if "validated_setups" in existing_tables:
            result = con.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'validated_setups'
            """).fetchall()

            expected_cols = {
                "strategy_id", "promoted_at", "promoted_from",
                "instrument", "orb_label", "orb_minutes", "rr_target",
                "confirm_bars", "entry_model", "filter_type", "filter_params",
                "sample_size", "win_rate", "expectancy_r",
                "years_tested", "all_years_positive", "stress_test_passed",
                "sharpe_ratio", "max_drawdown_r",
                "trades_per_year", "sharpe_ann",
                "yearly_results", "execution_spec",
                "family_hash", "is_family_head",
                "regime_waivers", "regime_waiver_count",
                "dst_winter_n", "dst_winter_avg_r",
                "dst_summer_n", "dst_summer_avg_r", "dst_verdict",
                "status", "retired_at", "retirement_reason"
            }
            actual_cols = {row[0] for row in result}

            missing = expected_cols - actual_cols
            if missing:
                violations.append(f"validated_setups missing columns: {missing}")

        all_valid = len(violations) == 0
        return all_valid, violations

def get_family_head_ids(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
    exclude_purged: bool = True,
) -> set[str]:
    """Return strategy_ids of family heads from edge_families (source of truth).

    Uses edge_families.head_strategy_id rather than validated_setups.is_family_head
    to avoid denormalization drift.

    Args:
        exclude_purged: If True (default), excludes PURGED families.
    """
    purge_filter = " AND robustness_status != 'PURGED'" if exclude_purged else ""
    rows = con.execute(f"""
        SELECT head_strategy_id FROM edge_families
        WHERE instrument = ?{purge_filter}
    """, [instrument]).fetchall()
    return {r[0] for r in rows}

def has_edge_families(con: duckdb.DuckDBPyConnection) -> bool:
    """Check if edge_families table exists (for graceful degradation)."""
    tables = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()
    return "edge_families" in {r[0] for r in tables}

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Initialize trading_app database schema"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop existing tables (WARNING: destroys data)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify schema instead of creating",
    )
    args = parser.parse_args()

    if args.verify:
        all_valid, violations = verify_trading_app_schema()
        if all_valid:
            logger.info("All trading_app tables verified")
            sys.exit(0)
        else:
            logger.info("Schema verification failed:")
            for v in violations:
                logger.info(f"  - {v}")
            sys.exit(1)
    else:
        init_trading_app_schema(force=args.force)

if __name__ == "__main__":
    main()
