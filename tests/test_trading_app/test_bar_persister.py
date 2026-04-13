"""Tests for BarPersister — broker-feed bar capture to bars_1m."""

import tempfile
from datetime import datetime, timezone

import duckdb
import pytest

from trading_app.live.bar_aggregator import Bar
from trading_app.live.bar_persister import BarPersister


@pytest.fixture
def tmp_db():
    """Create a temp DuckDB with bars_1m schema."""
    import os
    path = tempfile.mktemp(suffix=".duckdb")
    con = duckdb.connect(path)
    con.execute("""
        CREATE TABLE bars_1m (
            ts_utc TIMESTAMPTZ NOT NULL,
            symbol VARCHAR NOT NULL,
            source_symbol VARCHAR NOT NULL,
            open DOUBLE NOT NULL,
            high DOUBLE NOT NULL,
            low DOUBLE NOT NULL,
            close DOUBLE NOT NULL,
            volume BIGINT NOT NULL
        )
    """)
    con.close()
    return path


def _bar(minute: int, price: float = 20000.0, volume: int = 100) -> Bar:
    return Bar(
        ts_utc=datetime(2026, 4, 13, 10, minute, 0, tzinfo=timezone.utc),
        open=price,
        high=price + 1,
        low=price - 1,
        close=price + 0.5,
        volume=volume,
        symbol="MNQ",
    )


class TestBarPersister:
    def test_append_is_thread_safe(self):
        bp = BarPersister("MNQ")
        bp.append(_bar(0))
        bp.append(_bar(1))
        assert bp.bar_count == 2

    def test_flush_empty_returns_zero(self, tmp_db):
        bp = BarPersister("MNQ", db_path=tmp_db)
        assert bp.flush_to_db() == 0

    def test_flush_inserts_bars(self, tmp_db):
        bp = BarPersister("MNQ", db_path=tmp_db)
        bp.append(_bar(0))
        bp.append(_bar(1))
        bp.append(_bar(2))
        count = bp.flush_to_db()
        assert count == 3

        con = duckdb.connect(tmp_db, read_only=True)
        rows = con.sql("SELECT COUNT(*) FROM bars_1m WHERE symbol = 'MNQ'").fetchone()
        assert rows[0] == 3
        con.close()

    def test_flush_is_idempotent(self, tmp_db):
        bp = BarPersister("MNQ", db_path=tmp_db)
        bp.append(_bar(0))
        bp.append(_bar(1))

        # Flush twice — second should DELETE+INSERT same rows
        bp.flush_to_db()
        bp.flush_to_db()

        con = duckdb.connect(tmp_db, read_only=True)
        rows = con.sql("SELECT COUNT(*) FROM bars_1m WHERE symbol = 'MNQ'").fetchone()
        assert rows[0] == 2  # not 4
        con.close()

    def test_flush_preserves_bar_data(self, tmp_db):
        bp = BarPersister("MNQ", db_path=tmp_db)
        bp.append(_bar(5, price=19500.0, volume=250))
        bp.flush_to_db()

        con = duckdb.connect(tmp_db, read_only=True)
        row = con.sql("SELECT open, high, low, close, volume FROM bars_1m").fetchone()
        assert row[0] == 19500.0
        assert row[1] == 19501.0
        assert row[2] == 19499.0
        assert row[3] == 19500.5
        assert row[4] == 250
        con.close()

    def test_invalid_bar_skipped(self, tmp_db):
        bp = BarPersister("MNQ", db_path=tmp_db)
        good = _bar(0)
        bad = Bar(
            ts_utc=datetime(2026, 4, 13, 10, 1, 0, tzinfo=timezone.utc),
            open=-1.0, high=0.0, low=-2.0, close=-0.5, volume=100,
        )
        bp.append(good)
        bp.append(bad)
        count = bp.flush_to_db()
        assert count == 1  # bad bar filtered by is_valid()

    def test_clear_removes_bars(self):
        bp = BarPersister("MNQ")
        bp.append(_bar(0))
        bp.append(_bar(1))
        bp.clear()
        assert bp.bar_count == 0

    def test_flush_fail_open_on_bad_path(self):
        bp = BarPersister("MNQ", db_path="/nonexistent/path/db.db")
        bp.append(_bar(0))
        # Should NOT raise — fail-open
        count = bp.flush_to_db()
        assert count == 0

    def test_symbol_isolation(self, tmp_db):
        """Bars for different symbols don't interfere."""
        bp_mnq = BarPersister("MNQ", db_path=tmp_db)
        bp_mes = BarPersister("MES", db_path=tmp_db)
        bp_mnq.append(_bar(0))
        bp_mnq.append(_bar(1))
        bp_mes.append(_bar(0))
        bp_mnq.flush_to_db()
        bp_mes.flush_to_db()

        con = duckdb.connect(tmp_db, read_only=True)
        mnq = con.sql("SELECT COUNT(*) FROM bars_1m WHERE symbol = 'MNQ'").fetchone()[0]
        mes = con.sql("SELECT COUNT(*) FROM bars_1m WHERE symbol = 'MES'").fetchone()[0]
        assert mnq == 2
        assert mes == 1
        con.close()
