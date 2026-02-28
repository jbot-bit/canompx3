"""
Tests for walk-forward column backfill migration.

Tests use file-based DuckDB + temp JSONL to validate the backfill logic.
"""

import json
import duckdb
import pytest
from pathlib import Path

from scripts.migrations.backfill_wf_columns import backfill_wf


def _setup_test_db_and_jsonl(tmp_path):
    """Create DB with 3 rows (S1, S2, S3) and JSONL with 2 records.

    JSONL: S1 passed with 4 windows, S2 failed with 2 windows.
    S3 has no JSONL entry (untested).
    """
    db_path = tmp_path / "test.db"
    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE validated_setups (
            strategy_id VARCHAR PRIMARY KEY,
            instrument VARCHAR,
            status VARCHAR,
            wf_tested BOOLEAN,
            wf_passed BOOLEAN,
            wf_windows INTEGER
        )
    """)
    con.execute("""
        INSERT INTO validated_setups VALUES
            ('S1', 'MGC', 'active', NULL, NULL, NULL),
            ('S2', 'MES', 'active', NULL, NULL, NULL),
            ('S3', 'MNQ', 'active', NULL, NULL, NULL)
    """)
    con.close()

    jsonl_path = tmp_path / "wf_results.jsonl"
    records = [
        {"strategy_id": "S1", "passed": True, "n_valid_windows": 4,
         "instrument": "MGC"},
        {"strategy_id": "S2", "passed": False, "n_valid_windows": 2,
         "instrument": "MES"},
    ]
    with open(jsonl_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    return db_path, jsonl_path


class TestBackfillWf:
    def test_backfills_from_jsonl(self, tmp_path):
        """S1=(True, True, 4), S2=(True, False, 2), S3=(None, None, None)."""
        db_path, jsonl_path = _setup_test_db_and_jsonl(tmp_path)
        count = backfill_wf(db_path, jsonl_path)
        assert count == 2

        con = duckdb.connect(str(db_path), read_only=True)
        rows = con.execute(
            "SELECT strategy_id, wf_tested, wf_passed, wf_windows "
            "FROM validated_setups ORDER BY strategy_id"
        ).fetchall()
        con.close()

        # S1: tested=True, passed=True, windows=4
        assert rows[0] == ("S1", True, True, 4)
        # S2: tested=True, passed=False, windows=2
        assert rows[1] == ("S2", True, False, 2)
        # S3: untested — all NULL
        assert rows[2] == ("S3", None, None, None)

    def test_dry_run_no_changes(self, tmp_path):
        """All 3 rows still NULL after dry_run."""
        db_path, jsonl_path = _setup_test_db_and_jsonl(tmp_path)
        count = backfill_wf(db_path, jsonl_path, dry_run=True)
        assert count == 0

        con = duckdb.connect(str(db_path), read_only=True)
        rows = con.execute(
            "SELECT wf_tested, wf_passed, wf_windows "
            "FROM validated_setups ORDER BY strategy_id"
        ).fetchall()
        con.close()

        for wf_tested, wf_passed, wf_windows in rows:
            assert wf_tested is None
            assert wf_passed is None
            assert wf_windows is None

    def test_last_entry_wins(self, tmp_path):
        """When JSONL has multiple entries for same strategy_id, last wins."""
        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))
        con.execute("""
            CREATE TABLE validated_setups (
                strategy_id VARCHAR PRIMARY KEY,
                instrument VARCHAR,
                status VARCHAR,
                wf_tested BOOLEAN,
                wf_passed BOOLEAN,
                wf_windows INTEGER
            )
        """)
        con.execute("""
            INSERT INTO validated_setups VALUES
                ('S1', 'MGC', 'active', NULL, NULL, NULL)
        """)
        con.close()

        jsonl_path = tmp_path / "wf_results.jsonl"
        # First entry: failed with 2 windows
        # Second entry: passed with 5 windows (should win)
        records = [
            {"strategy_id": "S1", "passed": False, "n_valid_windows": 2},
            {"strategy_id": "S1", "passed": True, "n_valid_windows": 5},
        ]
        with open(jsonl_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        count = backfill_wf(db_path, jsonl_path)
        assert count == 1

        con = duckdb.connect(str(db_path), read_only=True)
        row = con.execute(
            "SELECT wf_tested, wf_passed, wf_windows "
            "FROM validated_setups WHERE strategy_id = 'S1'"
        ).fetchone()
        con.close()

        assert row == (True, True, 5)
