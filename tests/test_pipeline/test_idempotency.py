"""
Tests for idempotent 5-minute bar aggregation.

Verifies that running build_5m_bars twice on the same data produces
identical results (same rows, same values, same count).
"""

import pytest
import duckdb
from datetime import date, datetime
from zoneinfo import ZoneInfo


def _insert_sample_1m_bars(con, symbol="MGC", source_symbol="GCM4",
                           base_date=date(2024, 6, 3), n_bars=60):
    """Insert n_bars of 1m bars starting at midnight UTC on base_date."""
    utc = ZoneInfo("UTC")
    rows = []
    for i in range(n_bars):
        ts = datetime(base_date.year, base_date.month, base_date.day,
                      0, i, tzinfo=utc)
        price = 2350.0 + i * 0.1
        rows.append((ts, symbol, source_symbol,
                      price, price + 1.0, price - 0.5, price + 0.5, 100 + i))

    con.executemany("""
        INSERT OR REPLACE INTO bars_1m
        (ts_utc, symbol, source_symbol, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)


@pytest.fixture
def tmp_db_with_1m(tmp_path):
    """Temp DB with bars_1m schema and sample data."""
    db_path = tmp_path / "test.db"
    con = duckdb.connect(str(db_path))

    con.execute("""
        CREATE TABLE bars_1m (
            ts_utc        TIMESTAMPTZ NOT NULL,
            symbol        TEXT        NOT NULL,
            source_symbol TEXT        NOT NULL,
            open          DOUBLE      NOT NULL,
            high          DOUBLE      NOT NULL,
            low           DOUBLE      NOT NULL,
            close         DOUBLE      NOT NULL,
            volume        BIGINT      NOT NULL,
            PRIMARY KEY (symbol, ts_utc)
        )
    """)
    con.execute("""
        CREATE TABLE bars_5m (
            ts_utc        TIMESTAMPTZ NOT NULL,
            symbol        TEXT        NOT NULL,
            source_symbol TEXT,
            open          DOUBLE      NOT NULL,
            high          DOUBLE      NOT NULL,
            low           DOUBLE      NOT NULL,
            close         DOUBLE      NOT NULL,
            volume        BIGINT      NOT NULL,
            PRIMARY KEY (symbol, ts_utc)
        )
    """)

    _insert_sample_1m_bars(con)
    yield con
    con.close()


def _snapshot_5m(con, symbol="MGC"):
    """Return bars_5m as a sorted list of tuples for comparison."""
    rows = con.execute("""
        SELECT ts_utc, symbol, source_symbol, open, high, low, close, volume
        FROM bars_5m
        WHERE symbol = ?
        ORDER BY ts_utc
    """, [symbol]).fetchall()
    return rows


class TestIdempotency:
    """Verify build_5m_bars is idempotent: run twice, get identical result."""

    def test_double_run_identical(self, tmp_db_with_1m):
        """Running build_5m_bars twice produces identical bars_5m."""
        from pipeline.build_bars_5m import build_5m_bars

        con = tmp_db_with_1m
        start = date(2024, 6, 3)
        end = date(2024, 6, 3)

        # First run
        count1 = build_5m_bars(con, "MGC", start, end, dry_run=False)
        snapshot1 = _snapshot_5m(con)

        # Second run (should DELETE + re-INSERT identically)
        count2 = build_5m_bars(con, "MGC", start, end, dry_run=False)
        snapshot2 = _snapshot_5m(con)

        assert count1 > 0, "First run should produce rows"
        assert count1 == count2, "Row counts must match"
        assert len(snapshot1) == len(snapshot2), "Snapshot lengths must match"

        for i, (row1, row2) in enumerate(zip(snapshot1, snapshot2)):
            assert row1 == row2, f"Row {i} differs: {row1} vs {row2}"

    def test_no_duplicate_rows(self, tmp_db_with_1m):
        """After double run, no duplicate (symbol, ts_utc) in bars_5m."""
        from pipeline.build_bars_5m import build_5m_bars

        con = tmp_db_with_1m
        start = date(2024, 6, 3)
        end = date(2024, 6, 3)

        build_5m_bars(con, "MGC", start, end, dry_run=False)
        build_5m_bars(con, "MGC", start, end, dry_run=False)

        dupes = con.execute("""
            SELECT symbol, ts_utc, COUNT(*) AS cnt
            FROM bars_5m
            GROUP BY symbol, ts_utc
            HAVING COUNT(*) > 1
        """).fetchall()

        assert len(dupes) == 0, f"Duplicates found: {dupes}"

    def test_row_count_matches_buckets(self, tmp_db_with_1m):
        """Number of 5m bars matches expected bucket count from 1m data."""
        from pipeline.build_bars_5m import build_5m_bars

        con = tmp_db_with_1m
        start = date(2024, 6, 3)
        end = date(2024, 6, 3)

        count = build_5m_bars(con, "MGC", start, end, dry_run=False)

        # 60 1m bars at minute 0-59 -> 12 5-minute buckets
        assert count == 12
