"""Tests for scripts/tools/assert_rebuild.py — post-rebuild data integrity assertions."""

import duckdb
import pytest

from scripts.tools.assert_rebuild import (
    EXPECTED_DAILY_FEATURES_COLUMNS,
    assert_cross_table_fk,
    assert_date_continuity,
    assert_outcome_coverage,
    assert_row_count_no_decrease,
    assert_schema_alignment,
    assert_strategy_count_stable,
    has_failures,
    has_warnings,
    run_assertions,
)


@pytest.fixture
def assert_db(tmp_path):
    """DuckDB with minimal schema for assertion tests."""
    db_path = tmp_path / "test.db"
    con = duckdb.connect(str(db_path))

    # Pipeline tables
    con.execute("""
        CREATE TABLE bars_1m (
            ts_utc TIMESTAMPTZ NOT NULL,
            symbol TEXT NOT NULL,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
            volume BIGINT, source_symbol TEXT,
            PRIMARY KEY (symbol, ts_utc)
        )
    """)
    con.execute("""
        CREATE TABLE bars_5m (
            ts_utc TIMESTAMPTZ NOT NULL,
            symbol TEXT NOT NULL,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
            volume BIGINT, source_symbol TEXT,
            PRIMARY KEY (symbol, ts_utc)
        )
    """)
    con.execute("""
        CREATE TABLE daily_features (
            trading_day DATE NOT NULL,
            symbol TEXT NOT NULL,
            orb_minutes INTEGER NOT NULL,
            PRIMARY KEY (symbol, trading_day, orb_minutes)
        )
    """)
    con.execute("""
        CREATE TABLE orb_outcomes (
            symbol TEXT NOT NULL,
            trading_day DATE NOT NULL,
            orb_minutes INTEGER NOT NULL,
            orb_label TEXT NOT NULL,
            break_dir TEXT,
            PRIMARY KEY (symbol, trading_day, orb_minutes, orb_label)
        )
    """)
    con.execute("""
        CREATE TABLE validated_setups (
            strategy_id TEXT PRIMARY KEY,
            instrument TEXT NOT NULL,
            status TEXT NOT NULL
        )
    """)

    # Audit log for A1/A4 (previous counts)
    from pipeline.audit_log import ensure_audit_table

    ensure_audit_table(con)

    yield con
    con.close()


# ---------------------------------------------------------------------------
# A1: Row count decrease
# ---------------------------------------------------------------------------


def test_a1_no_prior_log(assert_db):
    """A1 passes when there's no prior audit log entry to compare against."""
    results = assert_row_count_no_decrease(assert_db, "MGC")
    assert all(r.passed for r in results)


def test_a1_detects_decrease(assert_db):
    """A1 warns when current row count is less than logged previous count."""
    from pipeline.audit_log import log_operation

    # Log a previous count of 100
    log_operation(assert_db, "INGEST", "bars_1m", instrument="MGC", rows_before=0, rows_after=100, status="SUCCESS")
    # Current table has 0 rows
    results = assert_row_count_no_decrease(assert_db, "MGC")
    failed = [r for r in results if not r.passed]
    assert len(failed) >= 1
    assert failed[0].severity == "WARNING"
    assert "decreased" in failed[0].message


# ---------------------------------------------------------------------------
# A2: Date continuity
# ---------------------------------------------------------------------------


def test_a2_passes_no_gaps(assert_db):
    """A2 passes when bars_1m has continuous dates."""
    for day in range(1, 6):
        assert_db.execute(
            f"INSERT INTO bars_1m VALUES ('2024-01-{day:02d} 00:00:00+00', 'MGC', 100, 101, 99, 100.5, 50, 'GC')"
        )
    result = assert_date_continuity(assert_db, "MGC")
    assert result.passed


def test_a2_detects_gap(assert_db):
    """A2 fails when there's a gap > 3 calendar days in bars_1m."""
    assert_db.execute("INSERT INTO bars_1m VALUES ('2024-01-01 00:00:00+00', 'MGC', 100, 101, 99, 100.5, 50, 'GC')")
    assert_db.execute("INSERT INTO bars_1m VALUES ('2024-01-10 00:00:00+00', 'MGC', 100, 101, 99, 100.5, 50, 'GC')")
    result = assert_date_continuity(assert_db, "MGC")
    assert not result.passed
    assert result.severity == "FAIL"
    assert "gap" in result.message.lower()


# ---------------------------------------------------------------------------
# A3: Cross-table FK
# ---------------------------------------------------------------------------


def test_a3_passes_all_matched(assert_db):
    """A3 passes when all orb_outcomes have matching daily_features."""
    assert_db.execute("INSERT INTO daily_features VALUES ('2024-01-01', 'MGC', 5)")
    assert_db.execute("INSERT INTO orb_outcomes VALUES ('MGC', '2024-01-01', 5, 'CME_REOPEN', 'long')")
    result = assert_cross_table_fk(assert_db, "MGC")
    assert result.passed


def test_a3_detects_orphans(assert_db):
    """A3 fails when orb_outcomes references missing daily_features."""
    assert_db.execute("INSERT INTO orb_outcomes VALUES ('MGC', '2024-01-01', 5, 'CME_REOPEN', 'long')")
    result = assert_cross_table_fk(assert_db, "MGC")
    assert not result.passed
    assert result.severity == "FAIL"


# ---------------------------------------------------------------------------
# A4: Strategy count stability
# ---------------------------------------------------------------------------


def test_a4_passes_no_prior(assert_db):
    """A4 passes when there's no prior strategy count."""
    result = assert_strategy_count_stable(assert_db, "MGC")
    assert result.passed


def test_a4_detects_major_drop(assert_db):
    """A4 warns when strategy count drops below threshold."""
    from pipeline.audit_log import log_operation

    # Log previous count of 100
    log_operation(
        assert_db, "VALIDATOR", "validated_setups", instrument="MGC", rows_before=0, rows_after=100, status="SUCCESS"
    )
    # Current has only 10 (< 70% of 100)
    for i in range(10):
        assert_db.execute(f"INSERT INTO validated_setups VALUES ('S{i}', 'MGC', 'active')")
    result = assert_strategy_count_stable(assert_db, "MGC")
    assert not result.passed
    assert result.severity == "WARNING"


def test_a4_passes_acceptable_drop(assert_db):
    """A4 passes when strategy count is above threshold."""
    from pipeline.audit_log import log_operation

    log_operation(
        assert_db, "VALIDATOR", "validated_setups", instrument="MGC", rows_before=0, rows_after=100, status="SUCCESS"
    )
    for i in range(80):
        assert_db.execute(f"INSERT INTO validated_setups VALUES ('S{i}', 'MGC', 'active')")
    result = assert_strategy_count_stable(assert_db, "MGC")
    assert result.passed


def test_a4_counts_all_statuses(assert_db):
    """A4 counts total rows (not just active) to match audit log semantics."""
    from pipeline.audit_log import log_operation

    # Audit log recorded 100 total rows
    log_operation(
        assert_db, "VALIDATOR", "validated_setups", instrument="MGC", rows_before=0, rows_after=100, status="SUCCESS"
    )
    # Insert 80 total: 50 active + 30 retired — above 70% threshold
    for i in range(50):
        assert_db.execute(f"INSERT INTO validated_setups VALUES ('S{i}', 'MGC', 'active')")
    for i in range(50, 80):
        assert_db.execute(f"INSERT INTO validated_setups VALUES ('S{i}', 'MGC', 'retired')")
    result = assert_strategy_count_stable(assert_db, "MGC")
    # 80/100 = 80% >= 70% threshold → should pass (would fail if only counting active: 50/100 = 50%)
    assert result.passed


# ---------------------------------------------------------------------------
# A5: Outcome coverage
# ---------------------------------------------------------------------------


def test_a5_detects_missing_coverage(assert_db):
    """A5 fails when a session × aperture combo has no outcomes."""
    # Insert outcomes for only one session + one aperture
    assert_db.execute("INSERT INTO orb_outcomes VALUES ('MGC', '2024-01-01', 5, 'CME_REOPEN', 'long')")
    result = assert_outcome_coverage(assert_db, "MGC")
    assert not result.passed
    assert result.severity == "FAIL"
    assert "missing" in result.message.lower()


def test_a5_passes_full_coverage(assert_db):
    """A5 passes when all session × aperture combos have at least one outcome."""
    from pipeline.dst import SESSION_CATALOG

    for session in SESSION_CATALOG:
        for aperture in [5, 15, 30]:
            assert_db.execute(
                "INSERT INTO orb_outcomes VALUES ($1, '2024-01-01', $2, $3, 'long')",
                ["MGC", aperture, session],
            )
    result = assert_outcome_coverage(assert_db, "MGC")
    assert result.passed


# ---------------------------------------------------------------------------
# A6: Schema alignment
# ---------------------------------------------------------------------------


def test_a6_detects_mismatch(assert_db):
    """A6 fails when daily_features has wrong column count (test fixture has 3 cols)."""
    result = assert_schema_alignment(assert_db)
    assert not result.passed
    assert result.severity == "FAIL"
    assert "columns" in result.message


def test_a6_passes_correct_schema(tmp_path):
    """A6 passes when daily_features matches init_db.py schema exactly."""
    from pipeline.init_db import init_db

    db_path = tmp_path / "full_schema.db"
    init_db(db_path, force=False)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        result = assert_schema_alignment(con)
        assert result.passed, f"Expected {EXPECTED_DAILY_FEATURES_COLUMNS} columns: {result.message}"
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------


def test_has_failures_and_warnings():
    """has_failures and has_warnings correctly classify results."""
    from scripts.tools.assert_rebuild import AssertionResult

    results = [
        AssertionResult("A1", "WARNING", True, "OK"),
        AssertionResult("A2", "FAIL", True, "OK"),
        AssertionResult("A3", "FAIL", False, "orphans found"),
    ]
    assert has_failures(results)
    assert not has_warnings(results)

    results2 = [
        AssertionResult("A1", "WARNING", False, "rows decreased"),
        AssertionResult("A2", "FAIL", True, "OK"),
    ]
    assert not has_failures(results2)
    assert has_warnings(results2)
