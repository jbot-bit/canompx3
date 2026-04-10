"""Tests for non-active instrument shelf retirement migration."""

import duckdb

from scripts.migrations.retire_non_active_validated import RETIREMENT_REASON, retire_non_active_validated


def _create_test_db():
    con = duckdb.connect(":memory:")
    con.execute(
        """
        CREATE TABLE validated_setups (
            strategy_id VARCHAR PRIMARY KEY,
            instrument VARCHAR,
            status VARCHAR,
            retired_at TIMESTAMPTZ,
            retirement_reason VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE edge_families (
            family_hash VARCHAR PRIMARY KEY,
            instrument VARCHAR
        )
        """
    )
    con.execute(
        """
        INSERT INTO validated_setups VALUES
            ('GC_1', 'GC', 'active', NULL, NULL),
            ('MCL_1', 'MCL', 'active', NULL, NULL),
            ('MNQ_1', 'MNQ', 'active', NULL, NULL),
            ('MES_1', 'MES', 'retired', CURRENT_TIMESTAMP, 'old')
        """
    )
    con.execute(
        """
        INSERT INTO edge_families VALUES
            ('gc_fam', 'GC'),
            ('mcl_fam', 'MCL'),
            ('mnq_fam', 'MNQ')
        """
    )
    return con


class TestRetireNonActiveValidated:
    def test_retires_non_active_rows_and_clears_edge_families(self):
        con = _create_test_db()
        try:
            count = retire_non_active_validated(con=con)
            assert count == 2

            retired_rows = con.execute(
                """
                SELECT strategy_id, status, retirement_reason, retired_at
                FROM validated_setups
                WHERE instrument IN ('GC', 'MCL')
                ORDER BY strategy_id
                """
            ).fetchall()
            assert len(retired_rows) == 2
            for _sid, status, reason, retired_at in retired_rows:
                assert status == "retired"
                assert reason == RETIREMENT_REASON
                assert retired_at is not None

            active_count = con.execute(
                "SELECT COUNT(*) FROM validated_setups WHERE instrument = 'MNQ' AND status = 'active'"
            ).fetchone()[0]
            assert active_count == 1

            edge_rows = con.execute("SELECT instrument FROM edge_families ORDER BY instrument").fetchall()
            assert edge_rows == [("MNQ",)]
        finally:
            con.close()

    def test_dry_run_leaves_rows_unchanged(self):
        con = _create_test_db()
        try:
            count = retire_non_active_validated(con=con, dry_run=True)
            assert count == 0

            statuses = con.execute(
                "SELECT instrument, status FROM validated_setups ORDER BY instrument, strategy_id"
            ).fetchall()
            assert ("GC", "active") in statuses
            assert ("MCL", "active") in statuses
        finally:
            con.close()
