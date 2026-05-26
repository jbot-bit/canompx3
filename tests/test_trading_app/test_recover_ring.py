"""Tests for scripts/tools/recover_ring.py — operator escape valve for
audit-fix #2 preserve-on-failure state.

Canonical delegation invariant (institutional-rigor § 4): the recovery path
MUST use BarPersister.flush_to_db rather than re-encoding bars_1m INSERT.
These tests exercise the round trip end-to-end to catch any drift.
"""

from __future__ import annotations

import importlib
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

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
    path = tempfile.mktemp(suffix=".duckdb")
    con = duckdb.connect(path)
    con.execute(
        """
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
        """
    )
    con.close()
    return path


def _bar(minute: int, price: float = 20000.0) -> Bar:
    return Bar(
        ts_utc=datetime(2026, 4, 13, 10, minute, 0, tzinfo=UTC),
        open=price,
        high=price + 1,
        low=price - 1,
        close=price + 0.5,
        volume=100,
        symbol="MNQ",
    )


def _populate_ring(symbol: str, bars: list[Bar]) -> None:
    """Drive bars through the canonical writer so the ring file matches
    what a live session would produce."""
    bp = BarPersister(symbol, session_id="recover-test")
    for b in bars:
        bp.append(b)
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        snap = bar_ring.read_bar_ring(symbol)
        if len(snap.bars) >= len(bars):
            return
        time.sleep(0.02)
    raise AssertionError(f"ring did not reach {len(bars)} bars within 3s")


@pytest.fixture
def recover_module():
    """Load scripts/tools/recover_ring.py without invoking main()."""
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root / "scripts" / "tools"))
    try:
        import recover_ring  # type: ignore[import-not-found]

        return importlib.reload(recover_ring)
    finally:
        sys.path.pop(0)


class TestRecoverRingRoundTrip:
    def test_empty_ring_returns_exit_1(self, isolated_ring_dir, tmp_db, recover_module):
        rc = recover_module.recover("MNQ", db_path=tmp_db)
        assert rc == 1
        # Ring file did not exist; still doesn't.
        assert not (isolated_ring_dir / "MNQ.json").exists()

    def test_recovery_writes_bars_and_clears_ring(self, isolated_ring_dir, tmp_db, recover_module):
        bars = [_bar(0), _bar(1), _bar(2)]
        _populate_ring("MNQ", bars)
        assert (isolated_ring_dir / "MNQ.json").exists()

        rc = recover_module.recover("MNQ", db_path=tmp_db)
        assert rc == 0

        # Bars present in bars_1m via canonical persister path.
        con = duckdb.connect(tmp_db, read_only=True)
        n = con.sql("SELECT COUNT(*) FROM bars_1m WHERE symbol = 'MNQ'").fetchone()
        assert n is not None and n[0] == 3
        con.close()
        # Ring cleared.
        assert not (isolated_ring_dir / "MNQ.json").exists()

    def test_dry_run_does_not_write_or_clear(self, isolated_ring_dir, tmp_db, recover_module):
        bars = [_bar(0), _bar(1)]
        _populate_ring("MNQ", bars)

        rc = recover_module.recover("MNQ", db_path=tmp_db, dry_run=True)
        assert rc == 0

        con = duckdb.connect(tmp_db, read_only=True)
        n = con.sql("SELECT COUNT(*) FROM bars_1m WHERE symbol = 'MNQ'").fetchone()
        assert n is not None and n[0] == 0
        con.close()
        # Ring preserved.
        assert (isolated_ring_dir / "MNQ.json").exists()

    def test_keep_ring_preserves_file_on_success(self, isolated_ring_dir, tmp_db, recover_module):
        bars = [_bar(0)]
        _populate_ring("MNQ", bars)

        rc = recover_module.recover("MNQ", db_path=tmp_db, keep_ring=True)
        assert rc == 0

        con = duckdb.connect(tmp_db, read_only=True)
        n = con.sql("SELECT COUNT(*) FROM bars_1m WHERE symbol = 'MNQ'").fetchone()
        assert n is not None and n[0] == 1
        con.close()
        assert (isolated_ring_dir / "MNQ.json").exists()

    def test_idempotent_re_run(self, isolated_ring_dir, tmp_db, recover_module):
        """Running recovery twice with --keep-ring against the same ring must
        not double-insert (delegates to flush_to_db's DELETE+INSERT)."""
        bars = [_bar(0), _bar(1), _bar(2)]
        _populate_ring("MNQ", bars)

        rc1 = recover_module.recover("MNQ", db_path=tmp_db, keep_ring=True)
        assert rc1 == 0
        rc2 = recover_module.recover("MNQ", db_path=tmp_db, keep_ring=True)
        assert rc2 == 0

        con = duckdb.connect(tmp_db, read_only=True)
        n = con.sql("SELECT COUNT(*) FROM bars_1m WHERE symbol = 'MNQ'").fetchone()
        assert n is not None and n[0] == 3
        con.close()

    def test_flush_failure_returns_3_and_preserves_ring(self, isolated_ring_dir, tmp_db, recover_module, monkeypatch):
        """If flush_to_db returns 0 (DB write failure), ring file MUST be preserved."""
        bars = [_bar(0)]
        _populate_ring("MNQ", bars)

        # Monkey-patch BarPersister.flush_to_db on the recover module's
        # import to simulate failure.
        from trading_app.live import bar_persister as _bp

        monkeypatch.setattr(_bp.BarPersister, "flush_to_db", lambda self: 0)

        rc = recover_module.recover("MNQ", db_path=tmp_db)
        assert rc == 3
        # Ring preserved for re-attempt.
        assert (isolated_ring_dir / "MNQ.json").exists()

    def test_flush_raising_non_duckdb_exception_returns_3_and_preserves_ring(
        self, isolated_ring_dir, tmp_db, recover_module, monkeypatch
    ):
        """flush_to_db catches only (duckdb.Error, OSError); any other exception
        class must still map to the fail-closed exit-3 contract (ring preserved),
        not escape as an uncontrolled traceback."""
        bars = [_bar(0)]
        _populate_ring("MNQ", bars)

        from trading_app.live import bar_persister as _bp

        def _raise(self):
            raise RuntimeError("connection pool misconfigured")

        monkeypatch.setattr(_bp.BarPersister, "flush_to_db", _raise)

        rc = recover_module.recover("MNQ", db_path=tmp_db)
        assert rc == 3
        # Ring preserved for re-attempt.
        assert (isolated_ring_dir / "MNQ.json").exists()
