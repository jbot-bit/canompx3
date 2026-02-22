"""Tests for trading_app.cascade_table — cross-session probability lookup."""

import pytest
import duckdb
from pathlib import Path

from trading_app.cascade_table import build_cascade_table, lookup_cascade


@pytest.fixture
def cascade_db(tmp_path):
    """Create a temp DB with daily_features containing session outcome data."""
    db_path = tmp_path / "cascade_test.db"
    con = duckdb.connect(str(db_path))

    # Minimal schema for cascade table (only columns it queries)
    con.execute("""
        CREATE TABLE daily_features (
            trading_day DATE,
            symbol VARCHAR,
            orb_minutes INTEGER,
            orb_0900_outcome VARCHAR,
            orb_0900_break_dir VARCHAR,
            orb_1000_outcome VARCHAR,
            orb_1000_break_dir VARCHAR,
            orb_1100_outcome VARCHAR,
            orb_1100_break_dir VARCHAR,
            orb_1800_outcome VARCHAR,
            orb_1800_break_dir VARCHAR,
            orb_2300_outcome VARCHAR,
            orb_2300_break_dir VARCHAR,
            orb_0030_outcome VARCHAR,
            orb_0030_break_dir VARCHAR
        )
    """)

    # Insert enough rows for 0900->1000 pair to pass the n>=5 threshold
    # 6 rows: 0900 loss + same dir -> 1000 outcomes
    for i in range(6):
        outcome_b = "win" if i < 4 else "loss"  # 4 wins, 2 losses = 66.7% WR
        con.execute("""
            INSERT INTO daily_features VALUES (
                ?, 'MGC', 5,
                'loss', 'long', ?, 'long', NULL, NULL,
                NULL, NULL, NULL, NULL, NULL, NULL
            )
        """, [f"2024-01-{i+1:02d}", outcome_b])

    # 3 rows: not enough for threshold (will be filtered)
    for i in range(3):
        con.execute("""
            INSERT INTO daily_features VALUES (
                ?, 'MGC', 5,
                'win', 'short', 'loss', 'short', NULL, NULL,
                NULL, NULL, NULL, NULL, NULL, NULL
            )
        """, [f"2024-02-{i+1:02d}"])

    con.close()
    return db_path


class TestBuildCascadeTable:
    def test_returns_dict(self, cascade_db):
        table = build_cascade_table(cascade_db)
        assert isinstance(table, dict)

    def test_sufficient_sample_included(self, cascade_db):
        """Group with n>=5 should be in table."""
        table = build_cascade_table(cascade_db)
        key = ("0900", "loss", "same")
        assert key in table
        assert table[key]["n"] == 6
        assert table[key]["1000_wr"] == pytest.approx(4 / 6, abs=1e-4)

    def test_small_n_filtered(self, cascade_db):
        """Group with n<5 should NOT be in table."""
        table = build_cascade_table(cascade_db)
        key = ("0900", "win", "same")
        assert key not in table

    def test_empty_db(self, tmp_path):
        """Empty daily_features returns empty table."""
        db_path = tmp_path / "empty.db"
        con = duckdb.connect(str(db_path))
        con.execute("""
            CREATE TABLE daily_features (
                trading_day DATE, symbol VARCHAR, orb_minutes INTEGER,
                orb_0900_outcome VARCHAR, orb_0900_break_dir VARCHAR,
                orb_1000_outcome VARCHAR, orb_1000_break_dir VARCHAR,
                orb_1100_outcome VARCHAR, orb_1100_break_dir VARCHAR,
                orb_1800_outcome VARCHAR, orb_1800_break_dir VARCHAR,
                orb_2300_outcome VARCHAR, orb_2300_break_dir VARCHAR,
                orb_0030_outcome VARCHAR, orb_0030_break_dir VARCHAR
            )
        """)
        con.close()
        table = build_cascade_table(db_path)
        assert table == {}


class TestLookupCascade:
    def test_found(self):
        table = {("0900", "loss", "opposite"): {"1000_wr": 0.52, "n": 148}}
        result = lookup_cascade(table, "0900", "loss", "opposite")
        assert result is not None
        assert result["n"] == 148

    def test_not_found(self):
        table = {("0900", "loss", "opposite"): {"1000_wr": 0.52, "n": 148}}
        result = lookup_cascade(table, "1000", "win", "same")
        assert result is None

    def test_empty_table(self):
        result = lookup_cascade({}, "0900", "loss", "same")
        assert result is None
