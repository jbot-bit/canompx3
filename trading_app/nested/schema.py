"""
Schema for nested ORB tables (isolated from production tables).

Creates 3 new tables:
  - nested_outcomes: Pre-computed outcomes with wider ORB + 5m entry bars
  - nested_strategies: Strategy discovery results
  - nested_validated: Validated strategies

Does NOT call or modify init_trading_app_schema().
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
from pipeline.paths import GOLD_DB_PATH


def init_nested_schema(
    db_path: Path | None = None,
    force: bool = False,
    con: "duckdb.DuckDBPyConnection | None" = None,
) -> None:
    """Create nested ORB tables if they don't exist.

    Args:
        db_path: Path to DuckDB file (ignored if con is provided).
        force: Drop existing tables first.
        con: Existing DuckDB connection to reuse.  When provided the caller
             owns the connection lifecycle (no close here).
    """
    if db_path is None:
        db_path = GOLD_DB_PATH

    owns_con = con is None
    if owns_con:
        con = duckdb.connect(str(db_path))
    try:
        if force:
            print("WARN: Force mode: Dropping existing nested tables...")
            con.execute("DROP TABLE IF EXISTS nested_validated")
            con.execute("DROP TABLE IF EXISTS nested_strategies")
            con.execute("DROP TABLE IF EXISTS nested_outcomes")

        # Table 1: nested_outcomes
        # Same schema as orb_outcomes + entry_resolution
        # PK adds entry_resolution to distinguish 5m entry from 1m entry
        con.execute("""
            CREATE TABLE IF NOT EXISTS nested_outcomes (
                trading_day       DATE        NOT NULL,
                symbol            TEXT        NOT NULL,
                orb_label         TEXT        NOT NULL,
                orb_minutes       INTEGER     NOT NULL,
                entry_resolution  INTEGER     NOT NULL,
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

                PRIMARY KEY (symbol, trading_day, orb_label, orb_minutes,
                             rr_target, confirm_bars, entry_model, entry_resolution),
                FOREIGN KEY (symbol, trading_day, orb_minutes)
                    REFERENCES daily_features(symbol, trading_day, orb_minutes)
            )
        """)

        # Table 2: nested_strategies
        con.execute("""
            CREATE TABLE IF NOT EXISTS nested_strategies (
                strategy_id       TEXT        PRIMARY KEY,
                created_at        TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

                -- Strategy parameters
                instrument        TEXT        NOT NULL,
                orb_label         TEXT        NOT NULL,
                orb_minutes       INTEGER     NOT NULL,
                entry_resolution  INTEGER     NOT NULL,
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

                -- Yearly breakdown (JSON)
                yearly_results    TEXT,

                -- Validation status
                validation_status TEXT,
                validation_notes  TEXT
            )
        """)

        # Table 3: nested_validated
        con.execute("""
            CREATE TABLE IF NOT EXISTS nested_validated (
                strategy_id       TEXT        PRIMARY KEY,
                promoted_at       TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                promoted_from     TEXT,

                -- Strategy parameters (denormalized)
                instrument        TEXT        NOT NULL,
                orb_label         TEXT        NOT NULL,
                orb_minutes       INTEGER     NOT NULL,
                entry_resolution  INTEGER     NOT NULL,
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
                yearly_results    TEXT,

                -- Execution spec (JSON)
                execution_spec    TEXT,

                -- Status
                status            TEXT        NOT NULL,
                retired_at        TIMESTAMPTZ,
                retirement_reason TEXT,

                FOREIGN KEY (promoted_from)
                    REFERENCES nested_strategies(strategy_id)
            )
        """)

        con.commit()
        print("Nested ORB schema initialized successfully")

    finally:
        if owns_con:
            con.close()


def verify_nested_schema(db_path: Path | None = None) -> tuple[bool, list[str]]:
    """Verify all nested tables exist with correct schema."""
    if db_path is None:
        db_path = GOLD_DB_PATH

    con = duckdb.connect(str(db_path), read_only=True)
    violations = []

    try:
        result = con.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
        """).fetchall()
        existing_tables = {row[0] for row in result}

        expected_tables = ["nested_outcomes", "nested_strategies", "nested_validated"]
        for table in expected_tables:
            if table not in existing_tables:
                violations.append(f"Missing table: {table}")

        # Check nested_outcomes columns
        if "nested_outcomes" in existing_tables:
            result = con.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'nested_outcomes'
            """).fetchall()
            actual_cols = {row[0] for row in result}

            expected_cols = {
                "trading_day", "symbol", "orb_label", "orb_minutes",
                "entry_resolution", "rr_target", "confirm_bars", "entry_model",
                "entry_ts", "entry_price", "stop_price", "target_price",
                "outcome", "exit_ts", "exit_price", "pnl_r", "mae_r", "mfe_r",
            }
            missing = expected_cols - actual_cols
            if missing:
                violations.append(f"nested_outcomes missing columns: {missing}")

        # Check nested_strategies columns
        if "nested_strategies" in existing_tables:
            result = con.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'nested_strategies'
            """).fetchall()
            actual_cols = {row[0] for row in result}

            expected_cols = {
                "strategy_id", "created_at", "instrument", "orb_label",
                "orb_minutes", "entry_resolution", "rr_target", "confirm_bars",
                "entry_model", "filter_type", "filter_params", "sample_size",
                "win_rate", "avg_win_r", "avg_loss_r", "expectancy_r",
                "sharpe_ratio", "max_drawdown_r", "median_risk_points",
                "avg_risk_points", "yearly_results", "validation_status",
                "validation_notes",
            }
            missing = expected_cols - actual_cols
            if missing:
                violations.append(f"nested_strategies missing columns: {missing}")

        # Check nested_validated columns
        if "nested_validated" in existing_tables:
            result = con.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'nested_validated'
            """).fetchall()
            actual_cols = {row[0] for row in result}

            expected_cols = {
                "strategy_id", "promoted_at", "promoted_from", "instrument",
                "orb_label", "orb_minutes", "entry_resolution", "rr_target",
                "confirm_bars", "entry_model", "filter_type", "filter_params",
                "sample_size", "win_rate", "expectancy_r", "years_tested",
                "all_years_positive", "stress_test_passed", "sharpe_ratio",
                "max_drawdown_r", "yearly_results", "execution_spec",
                "status", "retired_at", "retirement_reason",
            }
            missing = expected_cols - actual_cols
            if missing:
                violations.append(f"nested_validated missing columns: {missing}")

        return len(violations) == 0, violations

    finally:
        con.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Initialize nested ORB schema")
    parser.add_argument("--force", action="store_true", help="Drop existing tables")
    parser.add_argument("--verify", action="store_true", help="Verify schema only")
    args = parser.parse_args()

    if args.verify:
        ok, violations = verify_nested_schema()
        if ok:
            print("All nested tables verified")
            sys.exit(0)
        else:
            print("Schema verification failed:")
            for v in violations:
                print(f"  - {v}")
            sys.exit(1)
    else:
        init_nested_schema(force=args.force)


if __name__ == "__main__":
    main()
