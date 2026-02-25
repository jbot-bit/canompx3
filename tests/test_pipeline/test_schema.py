"""
Tests for pipeline.init_db schema initialization.

Requires tmp_db fixture (creates temp DuckDB).
"""

import pytest
import duckdb
from pathlib import Path

from pipeline.init_db import init_db, ORB_LABELS


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

    def test_creates_daily_features_table(self, tmp_path):
        db_path = tmp_path / "test.db"
        init_db(db_path, force=False)

        con = duckdb.connect(str(db_path))
        tables = [t[0] for t in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()]
        con.close()

        assert "daily_features" in tables

    def test_daily_features_has_orb_columns(self, tmp_path):
        db_path = tmp_path / "test.db"
        init_db(db_path, force=False)

        con = duckdb.connect(str(db_path))
        cols = [c[0] for c in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'daily_features'"
        ).fetchall()]
        con.close()

        # Each ORB label should have 8 columns
        for label in ORB_LABELS:
            for suffix in ['high', 'low', 'size', 'break_dir', 'break_ts', 'outcome', 'mae_r', 'mfe_r']:
                col_name = f"orb_{label}_{suffix}"
                assert col_name in cols, f"Missing ORB column: {col_name}"

    def test_daily_features_has_session_columns(self, tmp_path):
        db_path = tmp_path / "test.db"
        init_db(db_path, force=False)

        con = duckdb.connect(str(db_path))
        cols = [c[0] for c in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'daily_features'"
        ).fetchall()]
        con.close()

        for col in ['session_asia_high', 'session_asia_low',
                     'session_london_high', 'session_london_low',
                     'session_ny_high', 'session_ny_low',
                     'rsi_14_at_CME_REOPEN', 'atr_20']:
            assert col in cols, f"Missing session column: {col}"

    def test_daily_features_has_garch_columns(self, tmp_path):
        db_path = tmp_path / "test.db"
        init_db(db_path, force=False)

        con = duckdb.connect(str(db_path))
        cols = [c[0] for c in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'daily_features'"
        ).fetchall()]
        con.close()

        assert "garch_forecast_vol" in cols, "Missing garch_forecast_vol column"
        assert "garch_atr_ratio" in cols, "Missing garch_atr_ratio column"

    def test_creates_prospective_signals_table(self, tmp_path):
        db_path = tmp_path / "test.db"
        init_db(db_path, force=False)

        con = duckdb.connect(str(db_path))
        tables = [t[0] for t in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()]
        con.close()

        assert "prospective_signals" in tables

    def test_prospective_signals_columns(self, tmp_path):
        db_path = tmp_path / "test.db"
        init_db(db_path, force=False)

        con = duckdb.connect(str(db_path))
        cols = [c[0] for c in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'prospective_signals'"
        ).fetchall()]
        con.close()

        expected = ["signal_id", "trading_day", "symbol", "session",
                    "prev_day_outcome", "orb_size", "entry_model",
                    "confirm_bars", "rr_target", "outcome", "pnl_r",
                    "is_prospective", "freeze_date", "created_at"]
        for col in expected:
            assert col in cols, f"Missing column: {col}"

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
        assert "daily_features" in tables
