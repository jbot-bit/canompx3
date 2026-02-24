"""Tests for pipeline.init_db — database schema initialization."""

import pytest
import duckdb
from pathlib import Path

from pipeline.init_db import (
    BARS_1M_SCHEMA,
    BARS_5M_SCHEMA,
    DAILY_FEATURES_SCHEMA,
    ORB_LABELS,
    ORB_LABELS_FIXED,
    ORB_LABELS_DYNAMIC,
    init_db,
)


class TestSchemaConstants:
    def test_bars_1m_schema_creates_table(self, tmp_path):
        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))
        con.execute(BARS_1M_SCHEMA)
        tables = [r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()]
        con.close()
        assert "bars_1m" in tables

    def test_bars_5m_schema_creates_table(self, tmp_path):
        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))
        con.execute(BARS_5M_SCHEMA)
        tables = [r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()]
        con.close()
        assert "bars_5m" in tables

    def test_daily_features_schema_creates_table(self, tmp_path):
        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))
        con.execute(DAILY_FEATURES_SCHEMA)
        tables = [r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()]
        con.close()
        assert "daily_features" in tables

    def test_orb_labels_combined(self):
        assert ORB_LABELS == ORB_LABELS_FIXED + ORB_LABELS_DYNAMIC

    def test_orb_labels_fixed_is_empty(self):
        """After event-based rename, all sessions are dynamic."""
        assert ORB_LABELS_FIXED == []

    def test_orb_labels_dynamic_contains_all_sessions(self):
        for session in [
            "CME_REOPEN", "TOKYO_OPEN", "SINGAPORE_OPEN", "LONDON_METALS",
            "US_DATA_830", "NYSE_OPEN", "US_DATA_1000", "COMEX_SETTLE",
            "CME_PRECLOSE", "NYSE_CLOSE",
        ]:
            assert session in ORB_LABELS_DYNAMIC


class TestInitDb:
    def test_fresh_db_creates_tables(self, tmp_path):
        db_path = tmp_path / "fresh.db"
        init_db(db_path, force=False)
        con = duckdb.connect(str(db_path), read_only=True)
        tables = [r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()]
        con.close()
        assert "bars_1m" in tables
        assert "bars_5m" in tables
        assert "daily_features" in tables

    def test_idempotent_no_force(self, tmp_path):
        """Running init_db twice without force should not error."""
        db_path = tmp_path / "idem.db"
        init_db(db_path, force=False)
        init_db(db_path, force=False)  # Should not raise

    def test_force_recreates(self, tmp_path):
        """Force mode drops and recreates tables."""
        db_path = tmp_path / "force.db"
        init_db(db_path, force=False)

        # Insert a row
        con = duckdb.connect(str(db_path))
        con.execute("""
            INSERT INTO bars_1m VALUES (
                '2024-01-01T00:00:00+00:00', 'MGC', 'GCG24',
                2350.0, 2352.0, 2349.0, 2351.0, 100
            )
        """)
        count = con.execute("SELECT COUNT(*) FROM bars_1m").fetchone()[0]
        assert count == 1
        con.close()

        # Force recreate
        init_db(db_path, force=True)

        # Data should be gone
        con = duckdb.connect(str(db_path), read_only=True)
        count = con.execute("SELECT COUNT(*) FROM bars_1m").fetchone()[0]
        con.close()
        assert count == 0

    def test_daily_features_has_orb_columns(self, tmp_path):
        """Daily features should have columns for each ORB label."""
        db_path = tmp_path / "cols.db"
        init_db(db_path, force=False)
        con = duckdb.connect(str(db_path), read_only=True)
        cols = [r[0] for r in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name='daily_features'"
        ).fetchall()]
        con.close()
        # Check a few ORB-label-derived columns
        assert "orb_CME_REOPEN_high" in cols
        assert "orb_TOKYO_OPEN_outcome" in cols
        assert "orb_minutes" in cols
