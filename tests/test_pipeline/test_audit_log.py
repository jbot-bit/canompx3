"""Tests for pipeline.audit_log — immutable audit log."""

import duckdb
import pytest

from pipeline.audit_log import (
    ensure_audit_table,
    get_git_sha,
    get_previous_counts,
    get_table_row_count,
    log_operation,
)


@pytest.fixture
def audit_db(tmp_path):
    """DuckDB with audit log table + a bars_1m table for row count tests."""
    db_path = tmp_path / "test.db"
    con = duckdb.connect(str(db_path))
    ensure_audit_table(con)
    # Create a sample table for get_table_row_count
    con.execute("""
        CREATE TABLE bars_1m (
            ts_utc TIMESTAMPTZ NOT NULL,
            symbol TEXT NOT NULL,
            open DOUBLE NOT NULL,
            high DOUBLE NOT NULL,
            low DOUBLE NOT NULL,
            close DOUBLE NOT NULL,
            volume BIGINT NOT NULL,
            source_symbol TEXT NOT NULL,
            PRIMARY KEY (symbol, ts_utc)
        )
    """)
    con.execute("""
        CREATE TABLE validated_setups (
            strategy_id TEXT PRIMARY KEY,
            instrument TEXT NOT NULL,
            status TEXT NOT NULL
        )
    """)
    yield con
    con.close()


def test_ensure_audit_table_idempotent(audit_db):
    """Table creation is idempotent — safe to call multiple times."""
    ensure_audit_table(audit_db)
    ensure_audit_table(audit_db)
    # Should not raise
    audit_db.execute("SELECT COUNT(*) FROM pipeline_audit_log")


def test_log_operation_inserts_row(audit_db):
    """log_operation writes exactly one row with correct fields."""
    log_id = log_operation(
        audit_db,
        "INGEST",
        "bars_1m",
        instrument="MGC",
        rows_before=100,
        rows_after=200,
        duration_s=5.5,
        status="SUCCESS",
    )
    assert log_id  # non-empty string
    row = audit_db.execute(
        "SELECT operation, table_name, instrument, rows_before, rows_after, "
        "duration_s, status FROM pipeline_audit_log WHERE log_id = $1",
        [log_id],
    ).fetchone()
    assert row is not None
    assert row[0] == "INGEST"
    assert row[1] == "bars_1m"
    assert row[2] == "MGC"
    assert row[3] == 100
    assert row[4] == 200
    assert row[5] == pytest.approx(5.5, abs=0.01)
    assert row[6] == "SUCCESS"


def test_log_operation_captures_git_sha(audit_db):
    """log_operation auto-captures git SHA when not provided."""
    log_id = log_operation(audit_db, "TEST", "bars_1m", status="SUCCESS")
    row = audit_db.execute(
        "SELECT git_sha FROM pipeline_audit_log WHERE log_id = $1",
        [log_id],
    ).fetchone()
    # Should be non-empty if we're in a git repo (which we are during tests)
    assert row is not None
    sha = row[0]
    if sha is not None:
        assert len(sha) == 12  # truncated to 12 chars


def test_log_operation_with_rebuild_id(audit_db):
    """Rebuild ID links audit log to rebuild_manifest."""
    log_id = log_operation(
        audit_db,
        "OUTCOME_BUILDER",
        "orb_outcomes",
        instrument="MNQ",
        rebuild_id="test-rebuild-123",
        status="SUCCESS",
    )
    row = audit_db.execute(
        "SELECT rebuild_id FROM pipeline_audit_log WHERE log_id = $1",
        [log_id],
    ).fetchone()
    assert row[0] == "test-rebuild-123"


def test_log_operation_failed_status(audit_db):
    """FAILED operations are logged just like SUCCESS."""
    log_id = log_operation(
        audit_db,
        "VALIDATOR",
        "validated_setups",
        instrument="MGC",
        status="FAILED",
    )
    row = audit_db.execute(
        "SELECT status FROM pipeline_audit_log WHERE log_id = $1",
        [log_id],
    ).fetchone()
    assert row[0] == "FAILED"


def test_get_previous_counts_returns_latest(audit_db):
    """get_previous_counts returns the most recent rows_after for SUCCESS."""
    log_operation(audit_db, "INGEST", "bars_1m", instrument="MGC", rows_before=0, rows_after=100, status="SUCCESS")
    log_operation(audit_db, "INGEST", "bars_1m", instrument="MGC", rows_before=100, rows_after=250, status="SUCCESS")

    result = get_previous_counts(audit_db, "MGC", "bars_1m")
    assert result == 250


def test_get_previous_counts_ignores_failed(audit_db):
    """FAILED operations are excluded from get_previous_counts."""
    log_operation(audit_db, "INGEST", "bars_1m", instrument="MGC", rows_before=0, rows_after=100, status="SUCCESS")
    log_operation(audit_db, "INGEST", "bars_1m", instrument="MGC", rows_before=100, rows_after=50, status="FAILED")

    result = get_previous_counts(audit_db, "MGC", "bars_1m")
    assert result == 100  # Ignores the FAILED entry


def test_get_previous_counts_none_when_empty(audit_db):
    """Returns None when no prior log entries exist."""
    result = get_previous_counts(audit_db, "MGC", "bars_1m")
    assert result is None


def test_get_table_row_count_symbol_tables(audit_db):
    """Counts rows using 'symbol' column for pipeline tables."""
    audit_db.execute(
        "INSERT INTO bars_1m (ts_utc, symbol, source_symbol, open, high, low, close, volume) "
        "VALUES ('2024-01-01 00:00:00+00', 'MGC', 'GC', 100, 101, 99, 100.5, 50)"
    )
    audit_db.execute(
        "INSERT INTO bars_1m (ts_utc, symbol, source_symbol, open, high, low, close, volume) "
        "VALUES ('2024-01-01 00:01:00+00', 'MGC', 'GC', 100.5, 102, 100, 101, 60)"
    )
    assert get_table_row_count(audit_db, "bars_1m", "MGC") == 2
    assert get_table_row_count(audit_db, "bars_1m", "MNQ") == 0
    assert get_table_row_count(audit_db, "bars_1m") == 2


def test_get_table_row_count_instrument_tables(audit_db):
    """Counts rows using 'instrument' column for trading_app tables."""
    audit_db.execute("INSERT INTO validated_setups VALUES ('S1', 'MGC', 'active')")
    audit_db.execute("INSERT INTO validated_setups VALUES ('S2', 'MGC', 'active')")
    audit_db.execute("INSERT INTO validated_setups VALUES ('S3', 'MNQ', 'active')")
    assert get_table_row_count(audit_db, "validated_setups", "MGC") == 2
    assert get_table_row_count(audit_db, "validated_setups", "MNQ") == 1


def test_get_git_sha_returns_string():
    """get_git_sha returns a 12-char hex string in a git repo."""
    sha = get_git_sha()
    # We're in a git repo during tests
    assert sha is not None
    assert len(sha) == 12
    assert all(c in "0123456789abcdef" for c in sha)
