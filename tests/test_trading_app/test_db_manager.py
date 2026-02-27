"""
Tests for trading_app.db_manager schema management.
"""

import pytest
import duckdb
from pathlib import Path

from trading_app.db_manager import (
    init_trading_app_schema,
    verify_trading_app_schema,
)


@pytest.fixture
def db_path(tmp_path):
    """Create a temp DuckDB with base pipeline schema (daily_features)."""
    path = tmp_path / "test.db"
    con = duckdb.connect(str(path))
    # Create the base daily_features table that orb_outcomes references
    con.execute("""
        CREATE TABLE daily_features (
            trading_day DATE NOT NULL,
            symbol TEXT NOT NULL,
            orb_minutes INTEGER NOT NULL,
            bar_count_1m INTEGER,
            PRIMARY KEY (symbol, trading_day, orb_minutes)
        )
    """)
    con.commit()
    con.close()
    return path


class TestInitSchema:
    """Test schema creation."""

    def test_creates_all_tables(self, db_path):
        init_trading_app_schema(db_path=db_path)

        con = duckdb.connect(str(db_path), read_only=True)
        tables = con.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'main'
        """).fetchall()
        table_names = {t[0] for t in tables}
        con.close()

        assert "orb_outcomes" in table_names
        assert "experimental_strategies" in table_names
        assert "validated_setups" in table_names
        assert "validated_setups_archive" in table_names
        assert "strategy_trade_days" in table_names
        assert "edge_families" in table_names

    def test_idempotent(self, db_path):
        """Running twice doesn't crash."""
        init_trading_app_schema(db_path=db_path)
        init_trading_app_schema(db_path=db_path)  # Should not raise

    def test_force_recreates(self, db_path):
        init_trading_app_schema(db_path=db_path)

        # Insert a row into experimental_strategies
        con = duckdb.connect(str(db_path))
        con.execute("""
            INSERT INTO experimental_strategies (strategy_id, instrument, orb_label, orb_minutes, rr_target, confirm_bars, entry_model)
            VALUES ('test', 'MGC', 'CME_REOPEN', 5, 2.0, 3, 'E1')
        """)
        con.commit()
        con.close()

        # Force recreate — data should be gone
        init_trading_app_schema(db_path=db_path, force=True)

        con = duckdb.connect(str(db_path), read_only=True)
        count = con.execute("SELECT COUNT(*) FROM experimental_strategies").fetchone()[0]
        con.close()
        assert count == 0

    def test_orb_outcomes_columns(self, db_path):
        init_trading_app_schema(db_path=db_path)

        con = duckdb.connect(str(db_path), read_only=True)
        cols = con.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'orb_outcomes'
        """).fetchall()
        col_names = {c[0] for c in cols}
        con.close()

        expected = {
            "trading_day", "symbol", "orb_label", "orb_minutes",
            "rr_target", "confirm_bars", "entry_model", "entry_ts",
            "entry_price", "stop_price", "target_price", "outcome",
            "exit_ts", "exit_price", "pnl_r", "mae_r", "mfe_r",
            "ambiguous_bar", "ts_outcome", "ts_pnl_r", "ts_exit_ts",
        }
        assert expected.issubset(col_names)

    def test_orb_outcomes_pk(self, db_path):
        """Primary key prevents duplicate outcomes."""
        init_trading_app_schema(db_path=db_path)

        # Insert daily_features row first (FK reference)
        con = duckdb.connect(str(db_path))
        con.execute("""
            INSERT INTO daily_features (symbol, trading_day, orb_minutes)
            VALUES ('MGC', '2024-01-15', 5)
        """)

        con.execute("""
            INSERT INTO orb_outcomes (symbol, trading_day, orb_label, orb_minutes, rr_target, confirm_bars, entry_model)
            VALUES ('MGC', '2024-01-15', 'CME_REOPEN', 5, 2.0, 3, 'E1')
        """)
        con.commit()

        # Duplicate should fail
        with pytest.raises(duckdb.ConstraintException):
            con.execute("""
                INSERT INTO orb_outcomes (symbol, trading_day, orb_label, orb_minutes, rr_target, confirm_bars, entry_model)
                VALUES ('MGC', '2024-01-15', 'CME_REOPEN', 5, 2.0, 3, 'E1')
            """)

        con.close()


class TestVerifySchema:
    """Test schema verification."""

    def test_all_valid(self, db_path):
        init_trading_app_schema(db_path=db_path)
        valid, violations = verify_trading_app_schema(db_path=db_path)
        assert valid is True
        assert violations == []

    def test_missing_table(self, db_path):
        """Verify detects missing tables."""
        valid, violations = verify_trading_app_schema(db_path=db_path)
        assert valid is False
        assert any("Missing table" in v for v in violations)

    def test_partial_schema(self, db_path):
        """Only some tables exist."""
        con = duckdb.connect(str(db_path))
        con.execute("""
            CREATE TABLE orb_outcomes (
                trading_day DATE, symbol TEXT, orb_label TEXT,
                orb_minutes INTEGER, rr_target DOUBLE, confirm_bars INTEGER,
                entry_model TEXT, entry_ts TIMESTAMPTZ, entry_price DOUBLE,
                stop_price DOUBLE, target_price DOUBLE,
                outcome TEXT, exit_ts TIMESTAMPTZ,
                exit_price DOUBLE, pnl_r DOUBLE,
                mae_r DOUBLE, mfe_r DOUBLE,
                ambiguous_bar BOOLEAN DEFAULT FALSE,
                ts_outcome TEXT, ts_pnl_r DOUBLE, ts_exit_ts TIMESTAMPTZ,
                PRIMARY KEY (symbol, trading_day, orb_label, orb_minutes, rr_target, confirm_bars, entry_model)
            )
        """)
        con.commit()
        con.close()

        valid, violations = verify_trading_app_schema(db_path=db_path)
        assert valid is False
        # Should find missing tables
        assert any("experimental_strategies" in v for v in violations)


class TestEdgeFamiliesSchema:
    """Test edge_families table and family columns on validated_setups."""

    def test_edge_families_columns(self, db_path):
        init_trading_app_schema(db_path=db_path)

        con = duckdb.connect(str(db_path), read_only=True)
        cols = con.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'edge_families'
        """).fetchall()
        col_names = {c[0] for c in cols}
        con.close()

        expected = {
            "family_hash", "instrument", "member_count",
            "trade_day_count", "head_strategy_id",
            "head_expectancy_r", "head_sharpe_ann",
        }
        assert expected.issubset(col_names)

    def test_validated_setups_family_columns(self, db_path):
        init_trading_app_schema(db_path=db_path)

        con = duckdb.connect(str(db_path), read_only=True)
        cols = con.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'validated_setups'
        """).fetchall()
        col_names = {c[0] for c in cols}
        con.close()

        assert "family_hash" in col_names
        assert "is_family_head" in col_names

    def test_edge_families_pk(self, db_path):
        init_trading_app_schema(db_path=db_path)

        con = duckdb.connect(str(db_path))
        con.execute("""
            INSERT INTO validated_setups
            (strategy_id, instrument, orb_label, orb_minutes, rr_target,
             confirm_bars, entry_model, filter_type, sample_size,
             win_rate, expectancy_r, years_tested, all_years_positive,
             stress_test_passed, status)
            VALUES ('test_s1', 'MGC', 'CME_REOPEN', 5, 2.0, 2, 'E1', 'ORB_G5',
                    100, 0.55, 0.30, 3, TRUE, TRUE, 'active')
        """)
        con.execute("""
            INSERT INTO edge_families
            (family_hash, instrument, member_count, trade_day_count,
             head_strategy_id, head_expectancy_r, head_sharpe_ann)
            VALUES ('abc123', 'MGC', 3, 100, 'test_s1', 0.30, 1.2)
        """)
        con.commit()

        with pytest.raises(duckdb.ConstraintException):
            con.execute("""
                INSERT INTO edge_families
                (family_hash, instrument, member_count, trade_day_count,
                 head_strategy_id, head_expectancy_r, head_sharpe_ann)
                VALUES ('abc123', 'MGC', 1, 50, 'test_s1', 0.20, 0.8)
            """)
        con.close()
