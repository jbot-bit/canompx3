"""
Tests for pipeline.init_db schema initialization.

Requires tmp_db fixture (creates temp DuckDB).
"""

import pytest
import duckdb
from pathlib import Path

from pipeline.init_db import init_db


class TestInitDb:
    """Tests for database schema creation."""

    def test_creates_bars_1m_table(self, tmp_path):
        db_path = tmp_path / "test.db"
        init_db(db_path, force=False)

        con = duckdb.connect(str(db_path))
        tables = [t[0] for t in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()]
        con.close()

        assert "bars_1m" in tables

    def test_creates_bars_5m_table(self, tmp_path):
        db_path = tmp_path / "test.db"
        init_db(db_path, force=False)

        con = duckdb.connect(str(db_path))
        tables = [t[0] for t in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()]
        con.close()

        assert "bars_5m" in tables

    def test_bars_1m_has_correct_columns(self, tmp_path):
        db_path = tmp_path / "test.db"
        init_db(db_path, force=False)

        con = duckdb.connect(str(db_path))
        cols = [c[0] for c in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'bars_1m'"
        ).fetchall()]
        con.close()

        expected = ['ts_utc', 'symbol', 'source_symbol', 'open', 'high', 'low', 'close', 'volume']
        for col in expected:
            assert col in cols, f"Missing column: {col}"

    def test_bars_1m_ts_utc_is_timestamptz(self, tmp_path):
        db_path = tmp_path / "test.db"
        init_db(db_path, force=False)

        con = duckdb.connect(str(db_path))
        dtype = con.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = 'bars_1m' AND column_name = 'ts_utc'"
        ).fetchone()[0]
        con.close()

        assert "TIMESTAMP" in dtype.upper()

    def test_force_recreates(self, tmp_path):
        db_path = tmp_path / "test.db"

        # Create first time
        init_db(db_path, force=False)

        # Insert a row
        con = duckdb.connect(str(db_path))
        con.execute("""
            INSERT INTO bars_1m VALUES
            ('2024-01-01T00:00:00+00:00', 'MGC', 'MGCG4', 2350, 2352, 2349, 2351, 100)
        """)
        count_before = con.execute("SELECT COUNT(*) FROM bars_1m").fetchone()[0]
        con.close()
        assert count_before == 1

        # Force recreate
        init_db(db_path, force=True)

        # Should be empty
        con = duckdb.connect(str(db_path))
        count_after = con.execute("SELECT COUNT(*) FROM bars_1m").fetchone()[0]
        con.close()
        assert count_after == 0

    def test_idempotent_without_force(self, tmp_path):
        db_path = tmp_path / "test.db"

        # Create twice without force — should not error
        init_db(db_path, force=False)
        init_db(db_path, force=False)

        con = duckdb.connect(str(db_path))
        tables = [t[0] for t in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()]
        con.close()

        assert "bars_1m" in tables
        assert "bars_5m" in tables
