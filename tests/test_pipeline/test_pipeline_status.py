"""Tests for rebuild_manifest table schema."""

import duckdb

from pipeline.init_db import init_db


class TestRebuildManifest:
    def test_rebuild_manifest_table_exists(self, tmp_path):
        """rebuild_manifest table is created by init_db."""
        db_path = tmp_path / "test.db"
        init_db(db_path, force=False)

        con = duckdb.connect(str(db_path), read_only=True)
        tables = [
            t[0]
            for t in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        ]
        con.close()

        assert "rebuild_manifest" in tables

    def test_rebuild_manifest_schema(self, tmp_path):
        """rebuild_manifest has all expected columns with correct types."""
        db_path = tmp_path / "test.db"
        init_db(db_path, force=False)

        con = duckdb.connect(str(db_path), read_only=True)
        cols = {
            r[0]: r[1]
            for r in con.execute(
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'rebuild_manifest'"
            ).fetchall()
        }
        con.close()

        expected_cols = [
            "rebuild_id",
            "instrument",
            "started_at",
            "completed_at",
            "status",
            "failed_step",
            "steps_completed",
            "trigger",
        ]
        for col in expected_cols:
            assert col in cols, f"Missing column: {col}"

        # Verify key type constraints
        assert "TIMESTAMP" in cols["started_at"].upper()
        assert "TIMESTAMP" in cols["completed_at"].upper()
        assert cols["steps_completed"].upper().endswith("[]")
