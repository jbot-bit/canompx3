"""Tests for BarPersister — broker-feed bar capture to bars_1m."""

import tempfile
import time
from datetime import UTC, datetime, timezone

import duckdb
import pytest

from trading_app.live import bar_ring
from trading_app.live.bar_aggregator import Bar
from trading_app.live.bar_persister import BarPersister


@pytest.fixture
def isolated_ring_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(bar_ring, "RING_DIR", tmp_path / "live_bars")
    with bar_ring._writers_lock:
        leftover = list(bar_ring._writers.values())
        bar_ring._writers.clear()
    for w in leftover:
        try:
            w.drain_and_stop(timeout=2.0)
        except Exception:
            pass
    yield tmp_path / "live_bars"
    with bar_ring._writers_lock:
        leftover = list(bar_ring._writers.values())
        bar_ring._writers.clear()
    for w in leftover:
        try:
            w.drain_and_stop(timeout=2.0)
        except Exception:
            pass


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
        ts_utc=datetime(2026, 4, 13, 10, minute, 0, tzinfo=UTC),
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
            ts_utc=datetime(2026, 4, 13, 10, 1, 0, tzinfo=UTC),
            open=-1.0,
            high=0.0,
            low=-2.0,
            close=-0.5,
            volume=100,
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

    def test_append_enqueues_to_ring(self, isolated_ring_dir):
        bp = BarPersister("MNQ", session_id="sess-1")
        bp.append(_bar(0))
        # Wait for the ring writer thread to flush.
        deadline = time.monotonic() + 3.0
        snap = bar_ring.read_bar_ring("MNQ")
        while time.monotonic() < deadline and not snap.bars:
            time.sleep(0.02)
            snap = bar_ring.read_bar_ring("MNQ")
        assert len(snap.bars) == 1
        assert snap.session_id == "sess-1"

    def test_clear_ring_removes_file(self, isolated_ring_dir):
        bp = BarPersister("MNQ")
        bp.append(_bar(0))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not (isolated_ring_dir / "MNQ.json").exists():
            time.sleep(0.02)
        assert (isolated_ring_dir / "MNQ.json").exists()
        bp.clear_ring()
        assert not (isolated_ring_dir / "MNQ.json").exists()

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


class TestShutdownTraceBreadcrumb:
    """Forensics surface for post_session bar-persist branch decisions.

    Verifies the bar_ring.write_shutdown_trace helper used by
    session_orchestrator.post_session — every shutdown branch must leave a
    breadcrumb so the next live smoke can disambiguate "hook never ran" from
    "audit-fix #2 preserve-on-failure fired".
    """

    def test_reset_truncates_prior_session_trace(self, isolated_ring_dir):
        bar_ring.write_shutdown_trace("MNQ", "prior_session:exit", reset=True)
        bar_ring.write_shutdown_trace("MNQ", "prior_session:cleanup")
        # New session: reset=True must wipe the prior file.
        bar_ring.write_shutdown_trace("MNQ", "post_session:entry", reset=True)
        contents = bar_ring.read_shutdown_trace("MNQ")
        assert "prior_session:exit" not in contents
        assert "prior_session:cleanup" not in contents
        assert "post_session:entry" in contents
        assert contents.count("\n") == 1

    def test_append_mode_accumulates_branch_tags(self, isolated_ring_dir):
        bar_ring.write_shutdown_trace("MNQ", "post_session:entry", reset=True)
        bar_ring.write_shutdown_trace("MNQ", "drain_ok")
        bar_ring.write_shutdown_trace("MNQ", "flush_attempt:bars_captured=61")
        bar_ring.write_shutdown_trace("MNQ", "flush_returned:n_persisted=0")
        bar_ring.write_shutdown_trace("MNQ", "ring_preserved:bars_captured=61,n_persisted=0")
        contents = bar_ring.read_shutdown_trace("MNQ")
        # All five branches present, in order.
        lines = [ln for ln in contents.split("\n") if ln]
        assert len(lines) == 5
        assert "post_session:entry" in lines[0]
        assert "drain_ok" in lines[1]
        assert "flush_attempt:bars_captured=61" in lines[2]
        assert "flush_returned:n_persisted=0" in lines[3]
        assert "ring_preserved:bars_captured=61,n_persisted=0" in lines[4]

    def test_fail_open_when_dir_unwritable(self, isolated_ring_dir, monkeypatch):
        """OSError on disk must NOT propagate — fail-open per § 6.

        Patches the path resolver so writes target a path that cannot be
        created (a file masquerading as a parent directory).
        """
        from pathlib import Path

        blocker = isolated_ring_dir / "blocker"
        blocker.parent.mkdir(parents=True, exist_ok=True)
        blocker.write_text("not a directory", encoding="utf-8")
        # Force the trace path under the blocker file (which is not a dir).
        monkeypatch.setattr(
            bar_ring,
            "_shutdown_trace_path",
            lambda symbol: Path(blocker) / f"{symbol}.shutdown_trace.txt",
        )
        # Must not raise.
        bar_ring.write_shutdown_trace("MNQ", "post_session:entry", reset=True)
        # And read returns empty rather than raising.
        assert bar_ring.read_shutdown_trace("MNQ") == ""

    def test_timestamp_is_iso_utc(self, isolated_ring_dir):
        bar_ring.write_shutdown_trace("MNQ", "post_session:entry", reset=True)
        contents = bar_ring.read_shutdown_trace("MNQ")
        ts_token = contents.split(" ", 1)[0]
        parsed = datetime.fromisoformat(ts_token)
        assert parsed.tzinfo is not None
