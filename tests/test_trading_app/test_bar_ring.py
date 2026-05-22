"""Tests for trading_app.live.bar_ring — atomic-file IPC for live bars."""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from trading_app.live import bar_ring
from trading_app.live.bar_aggregator import Bar


@pytest.fixture
def isolated_ring_dir(tmp_path, monkeypatch):
    """Redirect bar_ring.RING_DIR + reset module state between tests."""
    monkeypatch.setattr(bar_ring, "RING_DIR", tmp_path / "live_bars")
    # Drain any writers left over from a prior test.
    with bar_ring._writers_lock:
        leftover = list(bar_ring._writers.values())
        bar_ring._writers.clear()
    for w in leftover:
        try:
            w.drain_and_stop(timeout=2.0)
        except Exception:
            pass
    yield tmp_path / "live_bars"
    # Tear-down: drain anything created during the test.
    with bar_ring._writers_lock:
        leftover = list(bar_ring._writers.values())
        bar_ring._writers.clear()
    for w in leftover:
        try:
            w.drain_and_stop(timeout=2.0)
        except Exception:
            pass


def _bar(minute: int, price: float = 20000.0, volume: int = 100) -> Bar:
    return Bar(
        ts_utc=datetime(2026, 4, 13, 10, minute % 60, 0, tzinfo=UTC)
        + timedelta(hours=minute // 60),
        open=price,
        high=price + 1,
        low=price - 1,
        close=price + 0.5,
        volume=volume,
        symbol="MNQ",
    )


def _wait_for_bars(symbol: str, expected_n: int, timeout: float = 3.0) -> bar_ring.RingSnapshot:
    """Poll the ring file until it has expected_n bars or timeout."""
    deadline = time.monotonic() + timeout
    snap = bar_ring.read_bar_ring(symbol)
    while time.monotonic() < deadline and len(snap.bars) < expected_n:
        time.sleep(0.02)
        snap = bar_ring.read_bar_ring(symbol)
    return snap


class TestBarRing:
    # 1. round-trip identity
    def test_round_trip_identity(self, isolated_ring_dir):
        b = _bar(0, price=19_750.5, volume=42)
        result = bar_ring.enqueue_bar("MNQ", b)
        assert result.enqueued is True
        snap = _wait_for_bars("MNQ", 1)
        assert len(snap.bars) == 1
        out = snap.bars[0]
        assert out["open"] == 19_750.5
        assert out["high"] == 19_751.5
        assert out["low"] == 19_749.5
        assert out["close"] == 19_751.0
        assert out["volume"] == 42

    # 2. ring cap at 240 drops oldest
    def test_ring_cap_drops_oldest(self, isolated_ring_dir):
        for i in range(bar_ring.RING_CAPACITY + 5):
            bar_ring.enqueue_bar("MNQ", _bar(i))
        bar_ring.drain_and_stop_writer("MNQ")
        snap = bar_ring.read_bar_ring("MNQ")
        assert len(snap.bars) == bar_ring.RING_CAPACITY
        # Oldest dropped — first remaining bar should NOT be minute 0.
        first_ts = snap.bars[0]["ts_utc"]
        first_minute = datetime.fromisoformat(first_ts)
        zero_minute = datetime(2026, 4, 13, 10, 0, 0, tzinfo=UTC)
        assert first_minute > zero_minute

    # 3. invalid bars rejected with counter bump
    def test_invalid_bar_rejected(self, isolated_ring_dir):
        bad = Bar(
            ts_utc=datetime(2026, 4, 13, 10, 0, 0, tzinfo=UTC),
            open=-1.0,
            high=0.0,
            low=-2.0,
            close=-0.5,
            volume=100,
            symbol="MNQ",
        )
        result = bar_ring.enqueue_bar("MNQ", bad)
        assert result.enqueued is False
        assert result.invalid_rejected is True
        # Subsequent valid bar still enqueues.
        good = _bar(1)
        result2 = bar_ring.enqueue_bar("MNQ", good)
        assert result2.enqueued is True
        snap = _wait_for_bars("MNQ", 1)
        assert snap.invalid_rejected_count == 1
        assert len(snap.bars) == 1

    # 4. concurrent reader/writer torn-read safety
    def test_concurrent_reader_writer_no_torn_reads(self, isolated_ring_dir):
        bar_ring.enqueue_bar("MNQ", _bar(0))
        _wait_for_bars("MNQ", 1)
        errors: list[Exception] = []
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                try:
                    snap = bar_ring.read_bar_ring("MNQ")
                    # Ensure the snapshot is structurally consistent.
                    assert isinstance(snap.bars, list)
                except (json.JSONDecodeError, OSError) as e:
                    errors.append(e)

        threads = [threading.Thread(target=reader, daemon=True) for _ in range(4)]
        for t in threads:
            t.start()
        for i in range(1, 101):
            bar_ring.enqueue_bar("MNQ", _bar(i))
        bar_ring.drain_and_stop_writer("MNQ")
        stop.set()
        for t in threads:
            t.join(timeout=2.0)
        assert errors == [], f"reader saw torn snapshots: {errors[:3]}"

    # 5. stale detection
    def test_is_stale(self, isolated_ring_dir):
        empty = bar_ring.RingSnapshot(symbol="MNQ")
        assert bar_ring.is_stale(empty) is True
        fresh = bar_ring.RingSnapshot(symbol="MNQ", updated_utc=datetime.now(UTC))
        assert bar_ring.is_stale(fresh) is False
        old = bar_ring.RingSnapshot(
            symbol="MNQ", updated_utc=datetime.now(UTC) - timedelta(seconds=200)
        )
        assert bar_ring.is_stale(old) is True

    # 6. PID round-trips and is readable
    def test_pid_metadata_round_trips(self, isolated_ring_dir):
        import os

        bar_ring.enqueue_bar("MNQ", _bar(0), session_id="sess-abc")
        snap = _wait_for_bars("MNQ", 1)
        assert snap.writer_pid == os.getpid()
        assert snap.session_id == "sess-abc"

    # 7. corrupt file fail-closed empty return
    def test_corrupt_file_returns_empty(self, isolated_ring_dir):
        isolated_ring_dir.mkdir(parents=True, exist_ok=True)
        ring_file = isolated_ring_dir / "MNQ.json"
        ring_file.write_text("this is not json {", encoding="utf-8")
        snap = bar_ring.read_bar_ring("MNQ")
        assert snap.is_empty()
        assert snap.symbol == "MNQ"

    # 8. clear_ring deletes file + subsequent read empty
    def test_clear_ring_deletes_file(self, isolated_ring_dir):
        bar_ring.enqueue_bar("MNQ", _bar(0))
        _wait_for_bars("MNQ", 1)
        assert (isolated_ring_dir / "MNQ.json").exists()
        bar_ring.clear_ring("MNQ")
        assert not (isolated_ring_dir / "MNQ.json").exists()
        snap = bar_ring.read_bar_ring("MNQ")
        assert snap.is_empty()

    # 9. bounded queue overflow drops oldest + WARNING
    def test_queue_overflow_drops_oldest(self, isolated_ring_dir, caplog):
        # Shrink queue to provoke overflow deterministically. Don't start the writer
        # thread so items accumulate in the queue.
        import queue as _queue
        symbol = "MNQOVF"
        writer = bar_ring._RingWriter.__new__(bar_ring._RingWriter)
        writer.symbol = symbol
        writer.session_id = None
        writer._pid = 0
        writer._q = _queue.Queue(maxsize=4)
        writer._bars = []
        writer._invalid_rejected = 0
        writer._consecutive_write_failures = 0
        writer._lock = threading.Lock()
        writer._stop_event = threading.Event()
        writer._thread = threading.Thread(target=lambda: None)  # never started
        with bar_ring._writers_lock:
            bar_ring._writers[symbol] = writer
        import logging
        with caplog.at_level(logging.WARNING, logger="trading_app.live.bar_ring"):
            for i in range(8):
                result = bar_ring.enqueue_bar(symbol, _bar(i))
                if i < 4:
                    assert result.dropped_oldest is False
                else:
                    assert result.dropped_oldest is True
        warnings = [r for r in caplog.records if "queue overflow" in r.getMessage()]
        assert warnings, "expected at least one queue-overflow WARNING"
        # Clean up — writer thread is fake; flush queue manually.
        with bar_ring._writers_lock:
            bar_ring._writers.pop(symbol, None)

    # 10. Mock-contamination refusal
    def test_mock_contamination_refused(self, isolated_ring_dir):
        mock_bar = MagicMock(spec=Bar)
        with pytest.raises(TypeError, match="bar_ring contamination"):
            bar_ring.enqueue_bar("MNQ", mock_bar)
        # Ring file must not exist as a result of the mock attempt.
        assert not (isolated_ring_dir / "MNQ.json").exists()

    # 11. consecutive-failure counter CRITICAL at ≥3
    def test_consecutive_write_failure_critical(self, isolated_ring_dir, caplog, monkeypatch):
        # Force write_text to raise; counter should bump and CRITICAL log after 3.
        import logging
        from pathlib import Path as _Path

        original_write_text = _Path.write_text
        fail_calls = {"n": 0}

        def failing_write_text(self, *args, **kwargs):  # type: ignore[override]
            if str(self).endswith(".json.tmp"):
                fail_calls["n"] += 1
                raise OSError("simulated disk error")
            return original_write_text(self, *args, **kwargs)

        monkeypatch.setattr(_Path, "write_text", failing_write_text)
        with caplog.at_level(logging.WARNING, logger="trading_app.live.bar_ring"):
            for i in range(4):
                bar_ring.enqueue_bar("MNQFAIL", _bar(i))
            bar_ring.drain_and_stop_writer("MNQFAIL")
        critical = [r for r in caplog.records if r.levelno >= logging.CRITICAL]
        assert critical, "expected CRITICAL log after >=3 consecutive write failures"
        assert any("MNQFAIL" in r.getMessage() for r in critical)

    # 13. AUDIT-FIX (defense-in-depth): Bar.is_valid() rejects a real Bar
    #     with a Mock OHLC field at enqueue time (first line of defense).
    def test_field_level_mock_rejected_at_enqueue(self, isolated_ring_dir):
        b = _bar(0)
        b.open = MagicMock()  # type: ignore[assignment]
        result = bar_ring.enqueue_bar("MNQ", b)
        # First-line defense: Bar.is_valid() checks isinstance(field, int|float),
        # MagicMock fails that → invalid_rejected counter bumps.
        assert result.enqueued is False
        assert result.invalid_rejected is True

    # 14. AUDIT-FIX: per-field guard in _serialize_bar is the SECOND-line
    #     defense — fires if a Bar bypasses is_valid() (e.g. an object that
    #     satisfies isinstance(x, int|float) via subclassing but is still a
    #     Mock). Verified by calling _serialize_bar directly.
    def test_serialize_bar_refuses_field_level_mock(self):
        b = _bar(0)
        b.open = MagicMock()  # type: ignore[assignment]
        with pytest.raises(TypeError, match="bar_ring contamination"):
            bar_ring._serialize_bar(b)

    # 15. AUDIT-FIX: writer thread survives contamination — subsequent
    #     valid bars still flow through (does not die silently on bad bar).
    def test_writer_thread_survives_contamination(self, isolated_ring_dir):
        # Construct a Bar that passes is_valid() but contains a mock that
        # the writer thread's per-field guard must reject. Easiest path:
        # subclass int so isinstance check passes but _is_mock_object also
        # returns True. Simulate via direct queue injection.
        symbol = "MNQSURVIVE"
        writer = bar_ring._get_or_create_writer(symbol)

        # Inject a Bar whose ts_utc is a MagicMock — bypasses is_valid()
        # (which only checks OHLC numeric fields) but the serializer's
        # ts_utc Mock check fires.
        bad = _bar(0)
        bad.ts_utc = MagicMock()  # type: ignore[assignment]
        writer._q.put_nowait(bad)
        # Follow with valid bars.
        for i in range(1, 5):
            bar_ring.enqueue_bar(symbol, _bar(i))
        snap = _wait_for_bars(symbol, 4)
        assert len(snap.bars) == 4  # writer survived, valid bars flowed through

    # 12. invalid_rejected count exposed through reader after a valid write
    def test_invalid_rejected_counter_exposed(self, isolated_ring_dir):
        # Two bad bars then a good one — counter persists in the ring payload.
        bad1 = Bar(
            ts_utc=datetime(2026, 4, 13, 10, 0, 0, tzinfo=UTC),
            open=float("nan"), high=1.0, low=1.0, close=1.0, volume=1, symbol="MNQ"
        )
        bad2 = Bar(
            ts_utc=datetime(2026, 4, 13, 10, 1, 0, tzinfo=UTC),
            open=1.0, high=0.5, low=1.0, close=1.0, volume=1, symbol="MNQ"  # high < low
        )
        bar_ring.enqueue_bar("MNQ", bad1)
        bar_ring.enqueue_bar("MNQ", bad2)
        bar_ring.enqueue_bar("MNQ", _bar(2))
        snap = _wait_for_bars("MNQ", 1)
        assert snap.invalid_rejected_count == 2
        assert len(snap.bars) == 1
