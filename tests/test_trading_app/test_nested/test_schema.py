"""
Tests for trading_app.nested.schema — table creation and column verification.
"""

import sys
import tempfile
from pathlib import Path

import pytest
import duckdb

from trading_app.nested.schema import init_nested_schema, verify_nested_schema

@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary DuckDB with required parent tables."""
    db_path = tmp_path / "test.db"
    con = duckdb.connect(str(db_path))
    # Create the parent table that nested_outcomes FK references
    con.execute("""
        CREATE TABLE daily_features (
            symbol TEXT NOT NULL,
            trading_day DATE NOT NULL,
            orb_minutes INTEGER NOT NULL,
            bar_count_1m INTEGER,
            PRIMARY KEY (symbol, trading_day, orb_minutes)
        )
    """)
    con.commit()
    con.close()
    return db_path

class TestInitNestedSchema:
    """Tests for init_nested_schema()."""

    def test_creates_all_three_tables(self, tmp_db):
        init_nested_schema(db_path=tmp_db)

        con = duckdb.connect(str(tmp_db), read_only=True)
        try:
            tables = {
                r[0] for r in con.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
                ).fetchall()
            }
            assert "nested_outcomes" in tables
            assert "nested_strategies" in tables
            assert "nested_validated" in tables
        finally:
            con.close()

    def test_idempotent(self, tmp_db):
        """Calling init twice does not error or duplicate tables."""
        init_nested_schema(db_path=tmp_db)
        init_nested_schema(db_path=tmp_db)

        con = duckdb.connect(str(tmp_db), read_only=True)
        try:
            tables = {
                r[0] for r in con.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
                ).fetchall()
            }
            assert "nested_outcomes" in tables
        finally:
            con.close()

    def test_force_drops_and_recreates(self, tmp_db):
        """Force mode drops existing tables."""
        init_nested_schema(db_path=tmp_db)

        # Insert a row
        con = duckdb.connect(str(tmp_db))
        con.execute("""
            INSERT INTO daily_features VALUES ('MGC', '2024-01-05', 15, 100)
        """)
        con.execute("""
            INSERT INTO nested_outcomes
            (trading_day, symbol, orb_label, orb_minutes, entry_resolution,
             rr_target, confirm_bars, entry_model)
            VALUES ('2024-01-05', 'MGC', 'CME_REOPEN', 15, 5, 2.0, 2, 'E1')
        """)
        con.commit()
        count = con.execute("SELECT COUNT(*) FROM nested_outcomes").fetchone()[0]
        assert count == 1
        con.close()

        # Force recreate
        init_nested_schema(db_path=tmp_db, force=True)

        con = duckdb.connect(str(tmp_db), read_only=True)
        try:
            count = con.execute("SELECT COUNT(*) FROM nested_outcomes").fetchone()[0]
            assert count == 0
        finally:
            con.close()

class TestNestedOutcomesColumns:
    """Verify nested_outcomes has the expected columns including entry_resolution."""

    def test_has_entry_resolution(self, tmp_db):
        init_nested_schema(db_path=tmp_db)

        con = duckdb.connect(str(tmp_db), read_only=True)
        try:
            cols = {
                r[0] for r in con.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'nested_outcomes'"
                ).fetchall()
            }
            assert "entry_resolution" in cols
            assert "orb_minutes" in cols
            assert "trading_day" in cols
            assert "pnl_r" in cols
            assert "mae_r" in cols
            assert "mfe_r" in cols
        finally:
            con.close()

    def test_has_all_outcome_columns(self, tmp_db):
        init_nested_schema(db_path=tmp_db)

        con = duckdb.connect(str(tmp_db), read_only=True)
        try:
            cols = {
                r[0] for r in con.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'nested_outcomes'"
                ).fetchall()
            }
            expected = {
                "trading_day", "symbol", "orb_label", "orb_minutes",
                "entry_resolution", "rr_target", "confirm_bars", "entry_model",
                "entry_ts", "entry_price", "stop_price", "target_price",
                "outcome", "exit_ts", "exit_price", "pnl_r", "mae_r", "mfe_r",
            }
            assert expected <= cols
        finally:
            con.close()

class TestNestedStrategiesColumns:
    """Verify nested_strategies has entry_resolution."""

    def test_has_entry_resolution(self, tmp_db):
        init_nested_schema(db_path=tmp_db)

        con = duckdb.connect(str(tmp_db), read_only=True)
        try:
            cols = {
                r[0] for r in con.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'nested_strategies'"
                ).fetchall()
            }
            assert "entry_resolution" in cols
            assert "strategy_id" in cols
            assert "orb_minutes" in cols
        finally:
            con.close()

class TestVerifyNestedSchema:
    """Tests for verify_nested_schema()."""

    def test_passes_when_tables_exist(self, tmp_db):
        init_nested_schema(db_path=tmp_db)
        ok, violations = verify_nested_schema(db_path=tmp_db)
        assert ok
        assert violations == []

    def test_fails_when_tables_missing(self, tmp_db):
        # Don't init nested schema
        ok, violations = verify_nested_schema(db_path=tmp_db)
        assert not ok
        assert len(violations) >= 3  # All 3 tables missing

class TestNestedDoesNotTouchProduction:
    """Verify nested schema does not modify production tables."""

    def test_production_tables_unchanged(self, tmp_db):
        """Creating nested schema does not create orb_outcomes or experimental_strategies."""
        init_nested_schema(db_path=tmp_db)

        con = duckdb.connect(str(tmp_db), read_only=True)
        try:
            tables = {
                r[0] for r in con.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
                ).fetchall()
            }
            # Should NOT have production tables (only daily_features from fixture)
            assert "orb_outcomes" not in tables
            assert "experimental_strategies" not in tables
            assert "validated_setups" not in tables
        finally:
            con.close()
