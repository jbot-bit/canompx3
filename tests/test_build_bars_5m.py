"""
Tests for pipeline.build_bars_5m aggregation and integrity checks.

Requires tmp_db fixture (creates temp DuckDB with schema).
"""

import pytest
import duckdb
from datetime import date, datetime

from pipeline.build_bars_5m import build_5m_bars, verify_5m_integrity


class TestBuild5mBars:
    """Tests for deterministic 5-minute bar aggregation."""

    def _insert_1m_bars(self, con, bars):
        """Helper: insert 1m bars into tmp_db.

        bars: list of (ts_utc_str, symbol, source_symbol, o, h, l, c, vol)
        """
        con.executemany(
            "INSERT INTO bars_1m VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            bars
        )

    def test_basic_aggregation(self, tmp_db):
        """5 x 1m bars at 00:00-00:04 UTC → 1 x 5m bar."""
        self._insert_1m_bars(tmp_db, [
            ("2024-06-03T00:00:00+00:00", "MGC", "MGCM4", 2350, 2352, 2349, 2351, 100),
            ("2024-06-03T00:01:00+00:00", "MGC", "MGCM4", 2351, 2355, 2350, 2354, 150),
            ("2024-06-03T00:02:00+00:00", "MGC", "MGCM4", 2354, 2356, 2353, 2355, 80),
            ("2024-06-03T00:03:00+00:00", "MGC", "MGCM4", 2355, 2357, 2354, 2356, 120),
            ("2024-06-03T00:04:00+00:00", "MGC", "MGCM4", 2356, 2358, 2355, 2357, 200),
        ])

        count = build_5m_bars(tmp_db, "MGC", date(2024, 6, 3), date(2024, 6, 3), dry_run=False)
        assert count == 1

        row = tmp_db.execute("SELECT * FROM bars_5m").fetchone()
        assert row is not None

        # OHLCV checks
        ts, symbol, source, o, h, l, c, vol = row
        assert symbol == "MGC"
        assert o == 2350.0    # first open
        assert h == 2358.0    # max high
        assert l == 2349.0    # min low
        assert c == 2357.0    # last close
        assert vol == 650     # sum volume

    def test_idempotent(self, tmp_db):
        """Running build twice produces same result."""
        self._insert_1m_bars(tmp_db, [
            ("2024-06-03T00:00:00+00:00", "MGC", "MGCM4", 2350, 2352, 2349, 2351, 100),
            ("2024-06-03T00:01:00+00:00", "MGC", "MGCM4", 2351, 2353, 2350, 2352, 150),
        ])

        count1 = build_5m_bars(tmp_db, "MGC", date(2024, 6, 3), date(2024, 6, 3), dry_run=False)
        count2 = build_5m_bars(tmp_db, "MGC", date(2024, 6, 3), date(2024, 6, 3), dry_run=False)

        assert count1 == count2

        total = tmp_db.execute("SELECT COUNT(*) FROM bars_5m").fetchone()[0]
        assert total == count1  # No duplicates

    def test_dry_run_no_write(self, tmp_db):
        """Dry run doesn't write to database."""
        self._insert_1m_bars(tmp_db, [
            ("2024-06-03T00:00:00+00:00", "MGC", "MGCM4", 2350, 2352, 2349, 2351, 100),
        ])

        build_5m_bars(tmp_db, "MGC", date(2024, 6, 3), date(2024, 6, 3), dry_run=True)

        count = tmp_db.execute("SELECT COUNT(*) FROM bars_5m").fetchone()[0]
        assert count == 0

    def test_no_source_data(self, tmp_db):
        """No source data returns 0."""
        count = build_5m_bars(tmp_db, "MGC", date(2024, 6, 3), date(2024, 6, 3), dry_run=False)
        assert count == 0


class TestVerify5mIntegrity:
    """Tests for 5m integrity checks."""

    def test_valid_data_passes(self, tmp_db):
        """Valid 5m bars pass integrity."""
        tmp_db.execute("""
            INSERT INTO bars_5m VALUES
            ('2024-06-03T00:00:00+00:00', 'MGC', 'MGCM4', 2350, 2358, 2349, 2357, 650)
        """)

        ok, failures = verify_5m_integrity(tmp_db, "MGC", date(2024, 6, 3), date(2024, 6, 3))
        assert ok is True
        assert len(failures) == 0

    def test_misaligned_timestamp_fails(self, tmp_db):
        """Timestamp not on 5-minute boundary fails."""
        tmp_db.execute("""
            INSERT INTO bars_5m VALUES
            ('2024-06-03T00:01:00+00:00', 'MGC', 'MGCM4', 2350, 2352, 2349, 2351, 100)
        """)

        ok, failures = verify_5m_integrity(tmp_db, "MGC", date(2024, 6, 3), date(2024, 6, 3))
        assert ok is False
        assert any("aligned" in f.lower() or "misalign" in f.lower() for f in failures)

    def test_ohlcv_violation_fails(self, tmp_db):
        """high < low fails integrity."""
        tmp_db.execute("""
            INSERT INTO bars_5m VALUES
            ('2024-06-03T00:00:00+00:00', 'MGC', 'MGCM4', 2350, 2348, 2351, 2350, 100)
        """)

        ok, failures = verify_5m_integrity(tmp_db, "MGC", date(2024, 6, 3), date(2024, 6, 3))
        assert ok is False
        assert any("ohlcv" in f.lower() or "sanity" in f.lower() for f in failures)

    def test_negative_volume_fails(self, tmp_db):
        """Negative volume fails integrity."""
        tmp_db.execute("""
            INSERT INTO bars_5m VALUES
            ('2024-06-03T00:00:00+00:00', 'MGC', 'MGCM4', 2350, 2352, 2349, 2351, -1)
        """)

        ok, failures = verify_5m_integrity(tmp_db, "MGC", date(2024, 6, 3), date(2024, 6, 3))
        assert ok is False
        assert any("volume" in f.lower() for f in failures)
