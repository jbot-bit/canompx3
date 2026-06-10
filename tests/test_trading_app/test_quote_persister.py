"""Tests for QuotePersister (Defect B — live spread capture to live_quotes).

Verifies idempotent DELETE+INSERT, fail-open behavior on DB error, and that it
writes ONLY to live_quotes (never bars_1m). Uses the canonical LIVE_QUOTES_SCHEMA
so the DDL itself is exercised.
"""

import tempfile
from datetime import UTC, datetime
from unittest.mock import patch

import duckdb
import pytest

from pipeline.init_db import LIVE_QUOTES_SCHEMA
from trading_app.live.quote_persister import QuotePersister
from trading_app.live.spread_accumulator import QuoteMinute


@pytest.fixture
def tmp_db():
    """Temp DuckDB with the canonical live_quotes schema."""
    path = tempfile.mktemp(suffix=".duckdb")
    con = duckdb.connect(path)
    con.execute(LIVE_QUOTES_SCHEMA)
    con.close()
    return path


def _qm(minute: int, avg: float = 0.25, n: int = 10) -> QuoteMinute:
    return QuoteMinute(
        ts_utc=datetime(2026, 6, 10, 14, minute, 0, tzinfo=UTC),
        avg_spread=avg,
        close_spread=avg,
        min_spread=avg,
        max_spread=avg,
        n_quotes=n,
        symbol="MNQ",
    )


def _row_count(db_path: str) -> int:
    con = duckdb.connect(db_path, read_only=True)
    try:
        return con.execute("SELECT COUNT(*) FROM live_quotes").fetchone()[0]
    finally:
        con.close()


class TestFlush:
    def test_writes_rows(self, tmp_db):
        p = QuotePersister("MNQ", db_path=tmp_db)
        p.append(_qm(30))
        p.append(_qm(31))
        assert p.flush_to_db() == 2
        assert _row_count(tmp_db) == 2

    def test_no_quotes_returns_zero(self, tmp_db):
        p = QuotePersister("MNQ", db_path=tmp_db)
        assert p.flush_to_db() == 0
        assert _row_count(tmp_db) == 0

    def test_invalid_quotes_filtered(self, tmp_db):
        """A zero-quote minute must not be written (QuoteMinute.is_valid gate)."""
        p = QuotePersister("MNQ", db_path=tmp_db)
        p.append(_qm(30))  # valid
        p.append(_qm(31, n=0))  # invalid — n_quotes 0
        assert p.flush_to_db() == 1
        assert _row_count(tmp_db) == 1

    def test_values_persisted_correctly(self, tmp_db):
        p = QuotePersister("MNQ", db_path=tmp_db)
        p.append(
            QuoteMinute(
                ts_utc=datetime(2026, 6, 10, 14, 30, 0, tzinfo=UTC),
                avg_spread=0.5,
                close_spread=0.75,
                min_spread=0.25,
                max_spread=0.75,
                n_quotes=42,
                symbol="MNQ",
            )
        )
        p.flush_to_db()
        con = duckdb.connect(tmp_db, read_only=True)
        try:
            row = con.execute(
                "SELECT symbol, source_symbol, avg_spread, close_spread, min_spread, max_spread, n_quotes "
                "FROM live_quotes"
            ).fetchone()
        finally:
            con.close()
        assert row == ("MNQ", "MNQ", 0.5, 0.75, 0.25, 0.75, 42)


class TestIdempotency:
    def test_reflush_same_minute_no_duplicates(self, tmp_db):
        """Re-flushing the same minute range must not create duplicate rows (PK holds)."""
        p = QuotePersister("MNQ", db_path=tmp_db)
        p.append(_qm(30))
        p.append(_qm(31))
        p.flush_to_db()
        # Re-flush the same in-memory quotes (DELETE+INSERT over the range).
        p.flush_to_db()
        assert _row_count(tmp_db) == 2

    def test_overlapping_range_replaced(self, tmp_db):
        """A second flush overlapping the first replaces, not appends."""
        p1 = QuotePersister("MNQ", db_path=tmp_db)
        p1.append(_qm(30, avg=0.25))
        p1.flush_to_db()
        # New persister, same minute, different value — DELETE+INSERT replaces.
        p2 = QuotePersister("MNQ", db_path=tmp_db)
        p2.append(_qm(30, avg=0.99))
        p2.flush_to_db()
        con = duckdb.connect(tmp_db, read_only=True)
        try:
            rows = con.execute("SELECT avg_spread FROM live_quotes").fetchall()
        finally:
            con.close()
        assert rows == [(0.99,)]  # replaced, not duplicated


class TestFailOpen:
    def test_db_error_returns_zero_not_raise(self, tmp_db):
        """A duckdb.Error during flush must be swallowed (fail-open) → return 0."""
        p = QuotePersister("MNQ", db_path=tmp_db)
        p.append(_qm(30))
        with patch("trading_app.live.quote_persister.duckdb.connect", side_effect=duckdb.Error("boom")):
            assert p.flush_to_db() == 0  # did not raise

    def test_os_error_returns_zero(self, tmp_db):
        p = QuotePersister("MNQ", db_path=tmp_db)
        p.append(_qm(30))
        with patch("trading_app.live.quote_persister.duckdb.connect", side_effect=OSError("disk full")):
            assert p.flush_to_db() == 0
