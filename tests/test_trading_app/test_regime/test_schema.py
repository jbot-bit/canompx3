"""
Tests for trading_app.regime.schema -- table creation and column verification.
"""

import sys
from pathlib import Path

import pytest
import duckdb

from trading_app.regime.schema import init_regime_schema, verify_regime_schema

@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary DuckDB (no parent table needed -- regime has no FK)."""
    db_path = tmp_path / "test.db"
    con = duckdb.connect(str(db_path))
    con.commit()
    con.close()
    return db_path

class TestInitRegimeSchema:
    """Tests for init_regime_schema()."""

    def test_creates_both_tables(self, tmp_db):
        init_regime_schema(db_path=tmp_db)

        con = duckdb.connect(str(tmp_db), read_only=True)
        try:
            tables = {
                r[0] for r in con.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
                ).fetchall()
            }
            assert "regime_strategies" in tables
            assert "regime_validated" in tables
        finally:
            con.close()

    def test_idempotent(self, tmp_db):
        """Calling init twice does not error."""
        init_regime_schema(db_path=tmp_db)
        init_regime_schema(db_path=tmp_db)

        con = duckdb.connect(str(tmp_db), read_only=True)
        try:
            tables = {
                r[0] for r in con.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
                ).fetchall()
            }
            assert "regime_strategies" in tables
        finally:
            con.close()

    def test_force_drops_and_recreates(self, tmp_db):
        """Force mode drops existing tables."""
        init_regime_schema(db_path=tmp_db)

        # Insert a row
        con = duckdb.connect(str(tmp_db))
        con.execute("""
            INSERT INTO regime_strategies
            (run_label, strategy_id, instrument, orb_label, orb_minutes,
             rr_target, confirm_bars, entry_model)
            VALUES ('test', 'S1', 'MGC', '0900', 5, 2.0, 2, 'E1')
        """)
        con.commit()
        count = con.execute("SELECT COUNT(*) FROM regime_strategies").fetchone()[0]
        assert count == 1
        con.close()

        # Force recreate
        init_regime_schema(db_path=tmp_db, force=True)

        con = duckdb.connect(str(tmp_db), read_only=True)
        try:
            count = con.execute("SELECT COUNT(*) FROM regime_strategies").fetchone()[0]
            assert count == 0
        finally:
            con.close()

    def test_con_parameter(self, tmp_db):
        """Passing an existing connection works and caller retains ownership."""
        con = duckdb.connect(str(tmp_db))
        init_regime_schema(con=con)
        # Connection should still be open
        tables = {
            r[0] for r in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
        }
        assert "regime_strategies" in tables
        con.close()

class TestRegimeStrategiesColumns:
    """Verify regime_strategies has expected columns."""

    def test_has_run_label_and_strategy_params(self, tmp_db):
        init_regime_schema(db_path=tmp_db)

        con = duckdb.connect(str(tmp_db), read_only=True)
        try:
            cols = {
                r[0] for r in con.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'regime_strategies'"
                ).fetchall()
            }
            expected = {
                "run_label", "strategy_id", "start_date", "end_date",
                "instrument", "orb_label", "orb_minutes", "rr_target",
                "confirm_bars", "entry_model", "filter_type",
                "sample_size", "win_rate", "expectancy_r", "sharpe_ratio",
                "max_drawdown_r", "yearly_results", "validation_status",
            }
            assert expected <= cols
        finally:
            con.close()

class TestRegimeValidatedColumns:
    """Verify regime_validated has expected columns."""

    def test_has_run_label_and_validation_fields(self, tmp_db):
        init_regime_schema(db_path=tmp_db)

        con = duckdb.connect(str(tmp_db), read_only=True)
        try:
            cols = {
                r[0] for r in con.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'regime_validated'"
                ).fetchall()
            }
            expected = {
                "run_label", "strategy_id", "start_date", "end_date",
                "instrument", "orb_label", "sample_size", "win_rate",
                "expectancy_r", "years_tested", "all_years_positive",
                "stress_test_passed", "sharpe_ratio", "status",
            }
            assert expected <= cols
        finally:
            con.close()

class TestVerifyRegimeSchema:
    """Tests for verify_regime_schema()."""

    def test_passes_when_tables_exist(self, tmp_db):
        init_regime_schema(db_path=tmp_db)
        ok, violations = verify_regime_schema(db_path=tmp_db)
        assert ok
        assert violations == []

    def test_fails_when_tables_missing(self, tmp_db):
        ok, violations = verify_regime_schema(db_path=tmp_db)
        assert not ok
        assert len(violations) >= 2  # Both tables missing

class TestRegimeDoesNotTouchProduction:
    """Verify regime schema does not modify production tables."""

    def test_production_tables_unchanged(self, tmp_db):
        """Creating regime schema does not create orb_outcomes or experimental_strategies."""
        init_regime_schema(db_path=tmp_db)

        con = duckdb.connect(str(tmp_db), read_only=True)
        try:
            tables = {
                r[0] for r in con.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
                ).fetchall()
            }
            assert "orb_outcomes" not in tables
            assert "experimental_strategies" not in tables
            assert "validated_setups" not in tables
        finally:
            con.close()
