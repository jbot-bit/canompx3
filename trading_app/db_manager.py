"""
Database schema manager for trading_app tables.

Creates and manages:
- orb_outcomes: Pre-computed outcomes for all RR targets × confirm_bars combinations
- experimental_strategies: Backtest results awaiting validation
- validated_setups: Production-ready strategies
- validated_setups_archive: Historical audit trail
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
from pipeline.paths import GOLD_DB_PATH


def init_trading_app_schema(db_path: Path | None = None, force: bool = False) -> None:
    """
    Create trading_app tables if they don't exist.

    Args:
        db_path: Path to DuckDB database (default: from paths.py)
        force: If True, drop existing tables first (WARNING: destroys data)
    """
    if db_path is None:
        db_path = GOLD_DB_PATH

    con = duckdb.connect(str(db_path))
    try:
        if force:
            print("WARN: Force mode: Dropping existing trading_app tables...")
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

                -- Annualized metrics
                trades_per_year   DOUBLE,
                sharpe_ann        DOUBLE,

                -- Yearly breakdown (JSON)
                yearly_results    TEXT,

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

                -- Execution spec (JSON)
                execution_spec    TEXT,

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

        con.commit()
        print("Trading app schema initialized successfully")

    finally:
        con.close()


def verify_trading_app_schema(db_path: Path | None = None) -> tuple[bool, list[str]]:
    """
    Verify all trading_app tables exist with correct schema.

    Returns:
        (all_valid, violations): violations is empty if all_valid=True
    """
    if db_path is None:
        db_path = GOLD_DB_PATH

    con = duckdb.connect(str(db_path), read_only=True)
    violations = []

    try:
        expected_tables = [
            "orb_outcomes",
            "experimental_strategies",
            "validated_setups",
            "validated_setups_archive",
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
                "exit_ts", "exit_price", "pnl_r", "mae_r", "mfe_r"
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
                "yearly_results", "validation_status",
                "validation_notes"
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
                "yearly_results", "execution_spec", "status",
                "retired_at", "retirement_reason"
            }
            actual_cols = {row[0] for row in result}

            missing = expected_cols - actual_cols
            if missing:
                violations.append(f"validated_setups missing columns: {missing}")

        all_valid = len(violations) == 0
        return all_valid, violations

    finally:
        con.close()


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
            print("All trading_app tables verified")
            sys.exit(0)
        else:
            print("Schema verification failed:")
            for v in violations:
                print(f"  - {v}")
            sys.exit(1)
    else:
        init_trading_app_schema(force=args.force)


if __name__ == "__main__":
    main()
