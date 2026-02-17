"""
Schema for regime discovery tables (isolated from production tables).

Creates 2 new tables:
  - regime_strategies: Strategy discovery results for a date-bounded run
  - regime_validated: Validated strategies from a regime run

Does NOT call or modify init_trading_app_schema().
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

import duckdb
from contextlib import nullcontext
from pipeline.paths import GOLD_DB_PATH

def init_regime_schema(
    db_path: Path | None = None,
    force: bool = False,
    con: "duckdb.DuckDBPyConnection | None" = None,
) -> None:
    """Create regime tables if they don't exist.

    Args:
        db_path: Path to DuckDB file (ignored if con is provided).
        force: Drop existing tables first.
        con: Existing DuckDB connection to reuse.  When provided the caller
             owns the connection lifecycle (no close here).
    """
    if db_path is None:
        db_path = GOLD_DB_PATH

    cm = duckdb.connect(str(db_path)) if con is None else nullcontext(con)
    with cm as con:
        if force:
            print("WARN: Force mode: Dropping existing regime tables...")
            con.execute("DROP TABLE IF EXISTS regime_validated")
            con.execute("DROP TABLE IF EXISTS regime_strategies")

        # Table 1: regime_strategies
        # Mirrors experimental_strategies + run_label, start_date, end_date
        con.execute("""
            CREATE TABLE IF NOT EXISTS regime_strategies (
                run_label         TEXT        NOT NULL,
                strategy_id       TEXT        NOT NULL,
                created_at        TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

                -- Run bounds
                start_date        DATE,
                end_date          DATE,

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

                -- Yearly breakdown (JSON)
                yearly_results    TEXT,

                -- Validation status
                validation_status TEXT,
                validation_notes  TEXT,

                PRIMARY KEY (run_label, strategy_id)
            )
        """)

        # Table 2: regime_validated
        con.execute("""
            CREATE TABLE IF NOT EXISTS regime_validated (
                run_label         TEXT        NOT NULL,
                strategy_id       TEXT        NOT NULL,
                promoted_at       TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

                -- Run bounds
                start_date        DATE,
                end_date          DATE,

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
                yearly_results    TEXT,

                -- Status
                status            TEXT        NOT NULL,

                PRIMARY KEY (run_label, strategy_id)
            )
        """)

        con.commit()
        print("Regime schema initialized successfully")


def verify_regime_schema(db_path: Path | None = None) -> tuple[bool, list[str]]:
    """Verify all regime tables exist with correct schema."""
    if db_path is None:
        db_path = GOLD_DB_PATH

    with duckdb.connect(str(db_path), read_only=True) as con:
        violations = []

        result = con.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
        """).fetchall()
        existing_tables = {row[0] for row in result}

        expected_tables = ["regime_strategies", "regime_validated"]
        for table in expected_tables:
            if table not in existing_tables:
                violations.append(f"Missing table: {table}")

        # Check regime_strategies columns
        if "regime_strategies" in existing_tables:
            result = con.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'regime_strategies'
            """).fetchall()
            actual_cols = {row[0] for row in result}

            expected_cols = {
                "run_label", "strategy_id", "created_at", "start_date", "end_date",
                "instrument", "orb_label", "orb_minutes", "rr_target", "confirm_bars",
                "entry_model", "filter_type", "filter_params", "sample_size",
                "win_rate", "avg_win_r", "avg_loss_r", "expectancy_r",
                "sharpe_ratio", "max_drawdown_r", "median_risk_points",
                "avg_risk_points", "yearly_results", "validation_status",
                "validation_notes",
            }
            missing = expected_cols - actual_cols
            if missing:
                violations.append(f"regime_strategies missing columns: {missing}")

        # Check regime_validated columns
        if "regime_validated" in existing_tables:
            result = con.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'regime_validated'
            """).fetchall()
            actual_cols = {row[0] for row in result}

            expected_cols = {
                "run_label", "strategy_id", "promoted_at", "start_date", "end_date",
                "instrument", "orb_label", "orb_minutes", "rr_target", "confirm_bars",
                "entry_model", "filter_type", "filter_params",
                "sample_size", "win_rate", "expectancy_r", "years_tested",
                "all_years_positive", "stress_test_passed", "sharpe_ratio",
                "max_drawdown_r", "yearly_results", "status",
            }
            missing = expected_cols - actual_cols
            if missing:
                violations.append(f"regime_validated missing columns: {missing}")

        return len(violations) == 0, violations

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Initialize regime schema")
    parser.add_argument("--force", action="store_true", help="Drop existing tables")
    parser.add_argument("--verify", action="store_true", help="Verify schema only")
    args = parser.parse_args()

    if args.verify:
        ok, violations = verify_regime_schema()
        if ok:
            print("All regime tables verified")
            sys.exit(0)
        else:
            print("Schema verification failed:")
            for v in violations:
                print(f"  - {v}")
            sys.exit(1)
    else:
        init_regime_schema(force=args.force)

if __name__ == "__main__":
    main()
