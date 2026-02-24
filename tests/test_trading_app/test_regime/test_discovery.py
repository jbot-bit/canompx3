"""
Tests for trading_app.regime.discovery -- regime strategy discovery.
"""

import sys
import json
from pathlib import Path
from datetime import date

import pytest
import duckdb

from trading_app.regime.schema import init_regime_schema
from trading_app.regime.discovery import run_regime_discovery

@pytest.fixture
def regime_db(tmp_path):
    """Create a temporary DuckDB with daily_features and orb_outcomes tables."""
    db_path = tmp_path / "test.db"
    con = duckdb.connect(str(db_path))

    # Use production schema for daily_features
    from pipeline.init_db import DAILY_FEATURES_SCHEMA
    con.execute(DAILY_FEATURES_SCHEMA)

    # Minimal orb_outcomes
    con.execute("""
        CREATE TABLE orb_outcomes (
            trading_day DATE NOT NULL,
            symbol TEXT NOT NULL,
            orb_label TEXT NOT NULL,
            orb_minutes INTEGER NOT NULL,
            rr_target DOUBLE NOT NULL,
            confirm_bars INTEGER NOT NULL,
            entry_model TEXT NOT NULL,
            entry_ts TIMESTAMPTZ,
            entry_price DOUBLE,
            stop_price DOUBLE,
            target_price DOUBLE,
            outcome TEXT,
            exit_ts TIMESTAMPTZ,
            exit_price DOUBLE,
            pnl_r DOUBLE,
            mae_r DOUBLE,
            mfe_r DOUBLE,
            ambiguous_bar BOOLEAN DEFAULT FALSE,
            ts_outcome TEXT,
            ts_pnl_r DOUBLE,
            ts_exit_ts TIMESTAMPTZ,
            PRIMARY KEY (symbol, trading_day, orb_label, orb_minutes,
                         rr_target, confirm_bars, entry_model),
            FOREIGN KEY (symbol, trading_day, orb_minutes)
                REFERENCES daily_features(symbol, trading_day, orb_minutes)
        )
    """)

    # Insert test data: 5 days in 2025
    for i, day in enumerate([
        date(2025, 1, 6), date(2025, 1, 7), date(2025, 1, 8),
        date(2025, 1, 9), date(2025, 1, 10),
    ]):
        orb_size = 5.0 + i  # 5, 6, 7, 8, 9 -- all pass G4+ filter
        orb_high = 2700.0 + orb_size / 2
        orb_low = 2700.0 - orb_size / 2
        con.execute(
            """INSERT INTO daily_features
               (symbol, trading_day, orb_minutes, bar_count_1m,
                orb_CME_REOPEN_high, orb_CME_REOPEN_low, orb_CME_REOPEN_size, orb_CME_REOPEN_break_dir)
               VALUES (?, ?, 5, 1400, ?, ?, ?, 'long')""",
            ['MGC', day, orb_high, orb_low, orb_size],
        )
        # Outcomes: E1 RR2.0 CB2 -> win half the time
        outcome = "win" if i % 2 == 0 else "loss"
        pnl_r = 1.5 if outcome == "win" else -1.0
        con.execute(
            """INSERT INTO orb_outcomes
               (trading_day, symbol, orb_label, orb_minutes,
                rr_target, confirm_bars, entry_model,
                entry_price, stop_price, target_price,
                outcome, pnl_r, mae_r, mfe_r)
               VALUES (?, 'MGC', 'CME_REOPEN', 5, 2.0, 2, 'E1',
                       2701.0, 2698.0, 2707.0, ?, ?, -0.3, 1.2)""",
            [day, outcome, pnl_r],
        )

    con.commit()
    con.close()
    return db_path

class TestRegimeDiscovery:

    def test_populates_regime_strategies(self, regime_db):
        """Discovery writes strategies to regime_strategies with correct run_label."""
        count = run_regime_discovery(
            db_path=regime_db,
            instrument="MGC",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 12, 31),
            run_label="test_2025",
            orb_minutes=5,
        )
        assert count > 0

        con = duckdb.connect(str(regime_db), read_only=True)
        try:
            rows = con.execute(
                "SELECT run_label, strategy_id, start_date, end_date, sample_size "
                "FROM regime_strategies WHERE run_label = 'test_2025'"
            ).fetchall()
            assert len(rows) > 0
            # All rows should have correct run_label and date bounds
            for run_label, sid, start, end, n in rows:
                assert run_label == "test_2025"
                assert start == date(2025, 1, 1)
                assert end == date(2025, 12, 31)
                assert n > 0
        finally:
            con.close()

    def test_idempotent_rerun(self, regime_db):
        """Running discovery twice with same label replaces previous results."""
        count1 = run_regime_discovery(
            db_path=regime_db, instrument="MGC",
            start_date=date(2025, 1, 1), end_date=date(2025, 12, 31),
            run_label="rerun_test",
        )
        count2 = run_regime_discovery(
            db_path=regime_db, instrument="MGC",
            start_date=date(2025, 1, 1), end_date=date(2025, 12, 31),
            run_label="rerun_test",
        )
        assert count1 == count2

        con = duckdb.connect(str(regime_db), read_only=True)
        try:
            total = con.execute(
                "SELECT COUNT(*) FROM regime_strategies WHERE run_label = 'rerun_test'"
            ).fetchone()[0]
            assert total == count2  # No duplicates
        finally:
            con.close()

    def test_different_labels_coexist(self, regime_db):
        """Multiple run_labels can coexist in the same table."""
        run_regime_discovery(
            db_path=regime_db, instrument="MGC",
            start_date=date(2025, 1, 1), end_date=date(2025, 6, 30),
            run_label="h1_2025",
        )
        run_regime_discovery(
            db_path=regime_db, instrument="MGC",
            start_date=date(2025, 1, 1), end_date=date(2025, 12, 31),
            run_label="full_2025",
        )

        con = duckdb.connect(str(regime_db), read_only=True)
        try:
            labels = {
                r[0] for r in con.execute(
                    "SELECT DISTINCT run_label FROM regime_strategies"
                ).fetchall()
            }
            assert "h1_2025" in labels
            assert "full_2025" in labels
        finally:
            con.close()

    def test_dry_run_no_writes(self, regime_db):
        """Dry run does not write to database."""
        count = run_regime_discovery(
            db_path=regime_db, instrument="MGC",
            start_date=date(2025, 1, 1), end_date=date(2025, 12, 31),
            run_label="dry_test", dry_run=True,
        )
        assert count > 0

        con = duckdb.connect(str(regime_db), read_only=True)
        try:
            tables = {
                r[0] for r in con.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
                ).fetchall()
            }
            # regime_strategies may not even exist in dry_run mode
            if "regime_strategies" in tables:
                total = con.execute(
                    "SELECT COUNT(*) FROM regime_strategies WHERE run_label = 'dry_test'"
                ).fetchone()[0]
                assert total == 0
        finally:
            con.close()

    def test_uses_same_strategy_id_format(self, regime_db):
        """Strategy IDs in regime_strategies match production format (no prefix)."""
        run_regime_discovery(
            db_path=regime_db, instrument="MGC",
            start_date=date(2025, 1, 1), end_date=date(2025, 12, 31),
            run_label="id_test",
        )

        con = duckdb.connect(str(regime_db), read_only=True)
        try:
            sids = [
                r[0] for r in con.execute(
                    "SELECT strategy_id FROM regime_strategies WHERE run_label = 'id_test'"
                ).fetchall()
            ]
            for sid in sids:
                # Should match production format: MGC_XXXX_EX_RRXX_CBXX_FILTER
                assert sid.startswith("MGC_")
                assert "NESTED_" not in sid
        finally:
            con.close()
