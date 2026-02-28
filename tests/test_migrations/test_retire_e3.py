"""
Tests for E3 soft retirement migration.

Tests use in-memory DuckDB with minimal validated_setups schema.
"""

import duckdb
import pytest
from pathlib import Path

from scripts.migrations.retire_e3_strategies import retire_e3


def _create_test_db():
    """Create in-memory DuckDB with minimal validated_setups table."""
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE validated_setups (
            strategy_id VARCHAR PRIMARY KEY,
            instrument VARCHAR,
            entry_model VARCHAR,
            status VARCHAR,
            retired_at TIMESTAMPTZ,
            retirement_reason VARCHAR,
            fdr_significant BOOLEAN
        )
    """)
    # 3 E3 rows (active), 2 non-E3 rows (active)
    con.execute("""
        INSERT INTO validated_setups VALUES
            ('E3_MGC_1', 'MGC', 'E3', 'active', NULL, NULL, FALSE),
            ('E3_MNQ_1', 'MNQ', 'E3', 'active', NULL, NULL, FALSE),
            ('E3_MES_1', 'MES', 'E3', 'active', NULL, NULL, FALSE),
            ('E1_MGC_1', 'MGC', 'E1', 'active', NULL, NULL, TRUE),
            ('E2_MNQ_1', 'MNQ', 'E2', 'active', NULL, NULL, TRUE)
    """)
    return con


class TestRetireE3:
    def test_retires_all_e3_rows(self):
        """Sets up 3 E3 + 2 non-E3, runs retire_e3, asserts 3 retired with correct status/reason."""
        con = _create_test_db()
        try:
            count = retire_e3(con=con)
            assert count == 3

            rows = con.execute(
                "SELECT strategy_id, status, retirement_reason, retired_at "
                "FROM validated_setups WHERE entry_model = 'E3' ORDER BY strategy_id"
            ).fetchall()
            assert len(rows) == 3
            for sid, status, reason, retired_at in rows:
                assert status == "RETIRED"
                assert reason == "PASS2: 0/50 FDR-sig, no timeout mechanism"
                assert retired_at is not None
        finally:
            con.close()

    def test_does_not_touch_non_e3(self):
        """Asserts 2 non-E3 rows still active after retirement."""
        con = _create_test_db()
        try:
            retire_e3(con=con)

            rows = con.execute(
                "SELECT strategy_id, status, retired_at, retirement_reason "
                "FROM validated_setups WHERE entry_model != 'E3' ORDER BY strategy_id"
            ).fetchall()
            assert len(rows) == 2
            for sid, status, retired_at, reason in rows:
                assert status == "active"
                assert retired_at is None
                assert reason is None
        finally:
            con.close()

    def test_dry_run_no_changes(self):
        """Asserts all 5 rows still active after dry_run=True."""
        con = _create_test_db()
        try:
            count = retire_e3(con=con, dry_run=True)
            assert count == 0

            active = con.execute(
                "SELECT COUNT(*) FROM validated_setups WHERE status = 'active'"
            ).fetchone()[0]
            assert active == 5
        finally:
            con.close()

    def test_idempotent_rerun(self):
        """Runs twice, second run returns 0."""
        con = _create_test_db()
        try:
            first = retire_e3(con=con)
            assert first == 3
            second = retire_e3(con=con)
            assert second == 0
        finally:
            con.close()
