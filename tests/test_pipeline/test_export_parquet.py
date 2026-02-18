"""Tests for pipeline/export_parquet.py."""

import duckdb
import pytest
from pathlib import Path

from pipeline.export_parquet import export_table, export_all, EXPORT_CONFIG


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary DuckDB with test data."""
    db_path = tmp_path / "test.db"
    con = duckdb.connect(str(db_path))

    # Create a minimal table matching daily_features schema
    con.execute("""
        CREATE TABLE daily_features (
            trading_day DATE,
            symbol VARCHAR,
            orb_minutes INTEGER,
            bar_count_1m INTEGER
        )
    """)
    con.execute("""
        INSERT INTO daily_features VALUES
        ('2024-01-01', 'MGC', 15, 100),
        ('2024-01-02', 'MGC', 15, 110),
        ('2024-01-01', 'MES', 15, 95)
    """)

    # Create validated_setups (small, no partition)
    con.execute("""
        CREATE TABLE validated_setups (
            strategy_id VARCHAR,
            instrument VARCHAR
        )
    """)
    con.execute("INSERT INTO validated_setups VALUES ('s1', 'MGC')")

    con.close()
    return db_path


class TestExportTable:
    def test_export_unpartitioned(self, tmp_db, tmp_path):
        output_dir = tmp_path / "parquet_out"
        con = duckdb.connect(str(tmp_db), read_only=True)

        rows = export_table(con, "validated_setups", output_dir)
        assert rows == 1
        assert (output_dir / "validated_setups" / "validated_setups.parquet").exists()
        con.close()

    def test_export_partitioned(self, tmp_db, tmp_path):
        output_dir = tmp_path / "parquet_out"
        con = duckdb.connect(str(tmp_db), read_only=True)

        rows = export_table(con, "daily_features", output_dir, partition_by=["symbol"])
        assert rows == 3
        # Partitioned output creates subdirectories
        assert (output_dir / "daily_features").exists()
        con.close()

    def test_export_empty_table(self, tmp_db, tmp_path):
        output_dir = tmp_path / "parquet_out"
        con = duckdb.connect(str(tmp_db))
        con.execute("CREATE TABLE empty_table (id INTEGER)")

        rows = export_table(con, "empty_table", output_dir)
        assert rows == 0
        con.close()


class TestExportAll:
    def test_export_all_skips_missing_tables(self, tmp_db, tmp_path):
        """Tables not in the DB should be skipped, not crash."""
        output_dir = tmp_path / "parquet_out"
        results = export_all(db_path=tmp_db, output_dir=output_dir)

        # daily_features and validated_setups exist; orb_outcomes and edge_families don't
        assert results["daily_features"] == 3
        assert results["validated_setups"] == 1
        assert results["orb_outcomes"] == 0
        assert results["edge_families"] == 0

    def test_export_specific_table(self, tmp_db, tmp_path):
        output_dir = tmp_path / "parquet_out"
        results = export_all(
            db_path=tmp_db, output_dir=output_dir,
            tables=["validated_setups"],
        )
        assert len(results) == 1
        assert results["validated_setups"] == 1


class TestExportConfig:
    def test_config_has_expected_tables(self):
        assert "orb_outcomes" in EXPORT_CONFIG
        assert "daily_features" in EXPORT_CONFIG
        assert "validated_setups" in EXPORT_CONFIG
        assert "edge_families" in EXPORT_CONFIG

    def test_large_tables_partitioned(self):
        assert EXPORT_CONFIG["orb_outcomes"].get("partition_by") == ["symbol"]
        assert EXPORT_CONFIG["daily_features"].get("partition_by") == ["symbol"]

    def test_small_tables_unpartitioned(self):
        assert EXPORT_CONFIG["validated_setups"].get("partition_by") is None
        assert EXPORT_CONFIG["edge_families"].get("partition_by") is None
