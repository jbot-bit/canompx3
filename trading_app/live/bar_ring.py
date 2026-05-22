"""Live-bars ring file — written by SessionOrchestrator, read by dashboard.

Solves the DuckDB-per-process-lock problem: BarPersister.flush_to_db() only
writes bars_1m at session end, so the dashboard chart shows stale historical
bars while the session is live. We cannot share gold.db read between writer
and reader during the session (Windows: per-process exclusive file lock — see
``memory/feedback_duckdb_windows_lock_is_per_process.md``).

Mirrors the atomic-replace IPC pattern from ``trading_app.live.bot_state``:
session writes ``data/live_bars/<SYMBOL>.json`` ring-buffered (≤240 bars =
~4h of 1m bars), dashboard reads the same file each poll tick. No DuckDB
during the session — flush_to_db() at session end keeps gold.db research
batched writes.

Architecture:
  BarPersister.append(bar) -> bar_ring.enqueue(symbol, bar)
                                |-> background writer thread
                                |     |-> serialize ring
                                |     |-> .tmp write
                                |     |-> os.replace -> data/live_bars/<SYM>.json
  Dashboard poll -> read_bar_ring(symbol) -> RingSnapshot
  Session end -> clear_ring(symbol) (after flush_to_db succeeds)

Concurrency:
  - One writer thread per symbol; bounded queue (capacity 480, drops oldest
    on overflow with WARNING) so the trading hot path never blocks on disk.
  - Reader uses atomic-replace semantics: each read returns either the prior
    snapshot or the new one — never a torn file.
  - Fail-open on disk errors (institutional-rigor § 6): consecutive-failure
    counter surfaces staleness; ≥3 -> CRITICAL log.

Mock-contamination guard mirrors ``bot_state._sanitize_for_state`` — refuses
to serialize ``unittest.mock.*`` objects (added 2026-05-17 after test mocks
leaked into ``data/bot_state.json``).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trading_app.live.bar_aggregator import Bar

log = logging.getLogger(__name__)

RING_DIR = Path(__file__).parent.parent.parent / "data" / "live_bars"

# Rationale: 240 minutes = 4 hours, which matches the dashboard chart's
# default lookback window. Sized for a 1m bar cadence so the operator sees a
# full intraday session without unbounded growth (Carver Ch. 4 — bounded
# state for live processes).
RING_CAPACITY = 240

# Rationale: 480 = 2 * RING_CAPACITY. The bounded queue absorbs any
# startup-flush burst (e.g., bar_aggregator emitting a backlog after a feed
# reconnect) without the trading hot path blocking on disk I/O via
# put_nowait. Overflow drops oldest so the freshest market state survives.
WRITER_QUEUE_CAPACITY = 480

# Rationale: 90s = 1.5 * bar cadence (60s). Two consecutive missed writes
# flip the snapshot to STALE, matching the dashboard's HEARTBEAT_STALE_AFTER_S
# semantics in bot_dashboard.py. Below this threshold a slow tick or
# scheduler jitter does not false-trip staleness.
DEFAULT_STALE_AFTER_SECS = 90.0

# Consecutive-write-failure threshold before CRITICAL log (mirrors
# ``bar_aggregator._BAD_BAR_ALERT_THRESHOLD``).
_WRITE_FAIL_CRITICAL_THRESHOLD = 3

_MOCK_CLASS_NAMES = frozenset(
    {"Mock", "MagicMock", "AsyncMock", "NonCallableMock", "NonCallableMagicMock"}
)


def _is_mock_object(value: Any) -> bool:
    for cls in type(value).__mro__:
        if cls.__name__ in _MOCK_CLASS_NAMES:
            return True
    return False


def _ring_path(symbol: str) -> Path:
    return RING_DIR / f"{symbol}.json"


@dataclass
class WriteResult:
    """Result of an enqueue call. Counter visible to caller for operator surfacing."""

    enqueued: bool
    dropped_oldest: bool = False
    invalid_rejected: bool = False
    consecutive_write_failures: int = 0


@dataclass
class RingSnapshot:
    """Reader-side view of a ring file."""

    symbol: str = ""
    bars: list[dict[str, Any]] = field(default_factory=list)
    writer_pid: int | None = None
    session_id: str | None = None
    updated_utc: datetime | None = None
    invalid_rejected_count: int = 0

    def is_empty(self) -> bool:
        return not self.bars


def _serialize_bar(bar: Bar) -> dict[str, Any]:
    """Project a Bar to a JSON-safe dict mirroring the SSE bar event shape.

    Field-level mock-contamination guard: a real ``Bar`` instance whose OHLC
    fields are mock objects (e.g. ``bar.open = MagicMock()``) bypasses the
    top-level ``_is_mock_object`` check in ``enqueue_bar``. Catch the
    smuggle here per-field — refusing keeps the ring snapshot free of mock
    artefacts that ``float(MagicMock())`` would otherwise silently coerce.
    """
    for field_name in ("open", "high", "low", "close", "volume"):
        v = getattr(bar, field_name)
        if _is_mock_object(v):
            raise TypeError(
                f"bar_ring contamination: bar.{field_name} = "
                f"{type(v).__name__} (unittest.mock object); refuse to serialize"
            )
        if not isinstance(v, (int, float)):
            raise TypeError(
                f"bar_ring contamination: bar.{field_name} is "
                f"{type(v).__name__}, expected int|float"
            )
    ts = bar.ts_utc
    if _is_mock_object(ts):
        raise TypeError(
            f"bar_ring contamination: bar.ts_utc = {type(ts).__name__} "
            f"(unittest.mock object); refuse to serialize"
        )
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return {
        "ts_utc": ts.astimezone(UTC).isoformat(),
        "time": int(ts.astimezone(UTC).timestamp()),
        "open": float(bar.open),
        "high": float(bar.high),
        "low": float(bar.low),
        "close": float(bar.close),
        "volume": int(bar.volume),
    }


class _RingWriter:
    """Per-symbol background writer thread + bounded queue.

    Owns one ring file. Serializes incoming bars onto an in-memory deque (cap
    RING_CAPACITY), then atomically replaces ``data/live_bars/<SYMBOL>.json``.
    """

    def __init__(self, symbol: str, session_id: str | None = None):
        self.symbol = symbol
        self.session_id = session_id
        self._pid = os.getpid()
        self._q: queue.Queue[Bar | None] = queue.Queue(maxsize=WRITER_QUEUE_CAPACITY)
        self._bars: list[dict[str, Any]] = []
        self._invalid_rejected = 0
        self._consecutive_write_failures = 0
        self._lock = threading.Lock()  # protects _bars / counters
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"bar-ring-{symbol}", daemon=True
        )
        self._thread.start()

    def enqueue(self, bar: Bar) -> WriteResult:
        if _is_mock_object(bar):
            raise TypeError(
                f"bar_ring contamination: bar for {self.symbol} is "
                f"{type(bar).__name__} (unittest.mock object); refuse to enqueue"
            )
        if not bar.is_valid():
            with self._lock:
                self._invalid_rejected += 1
                fails = self._consecutive_write_failures
            return WriteResult(
                enqueued=False, invalid_rejected=True, consecutive_write_failures=fails
            )
        dropped = False
        try:
            self._q.put_nowait(bar)
        except queue.Full:
            # Drop oldest: pull one out, push new. Never block the trading path.
            try:
                self._q.get_nowait()
                self._q.task_done()
            except queue.Empty:  # pragma: no cover — race with writer thread
                pass
            dropped = True
            try:
                self._q.put_nowait(bar)
            except queue.Full:  # pragma: no cover — defensive
                log.warning(
                    "bar_ring(%s): queue still full after drop-oldest; dropping new bar",
                    self.symbol,
                )
                with self._lock:
                    fails = self._consecutive_write_failures
                return WriteResult(
                    enqueued=False, dropped_oldest=True, consecutive_write_failures=fails
                )
            log.warning(
                "bar_ring(%s): queue overflow at cap=%d; dropped oldest bar",
                self.symbol,
                WRITER_QUEUE_CAPACITY,
            )
        with self._lock:
            fails = self._consecutive_write_failures
        return WriteResult(
            enqueued=True, dropped_oldest=dropped, consecutive_write_failures=fails
        )

    def drain_and_stop(self, timeout: float = 5.0) -> None:
        """Block until queue drains, then stop the writer thread.

        Called from session shutdown BEFORE flush_to_db / clear_ring so any
        bars still in the queue are persisted to the ring file. Bounded by
        timeout to avoid hanging session teardown on a stuck writer.
        """
        try:
            # Sentinel signals end-of-stream; writer flushes the deque then exits.
            self._q.put_nowait(None)
        except queue.Full:  # pragma: no cover — defensive
            # Force the sentinel through: drop one then retry.
            try:
                self._q.get_nowait()
                self._q.task_done()
                self._q.put_nowait(None)
            except queue.Empty:
                pass
        self._stop_event.set()
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while True:
            try:
                item = self._q.get(timeout=1.0)
            except queue.Empty:
                if self._stop_event.is_set():
                    return
                continue
            try:
                if item is None:
                    return
                try:
                    serialized = _serialize_bar(item)
                except TypeError as exc:
                    # Mock-contamination or non-numeric field smuggled in.
                    # Logged CRITICAL so it surfaces in server logs; the bad
                    # bar is dropped, counter bumped. Writer thread MUST
                    # survive — silent thread death would freeze the chart.
                    log.critical(
                        "bar_ring(%s): refused contaminated bar — %s",
                        self.symbol,
                        exc,
                    )
                    with self._lock:
                        self._invalid_rejected += 1
                    continue
                with self._lock:
                    self._bars.append(serialized)
                    if len(self._bars) > RING_CAPACITY:
                        # Drop oldest from the in-memory ring.
                        overflow = len(self._bars) - RING_CAPACITY
                        del self._bars[:overflow]
                self._write_snapshot()
            finally:
                self._q.task_done()

    def _write_snapshot(self) -> None:
        with self._lock:
            payload: dict[str, Any] = {
                "symbol": self.symbol,
                "writer_pid": self._pid,
                "session_id": self.session_id,
                "updated_utc": datetime.now(UTC).isoformat(),
                "invalid_rejected_count": self._invalid_rejected,
                "bars": list(self._bars),
            }
        path = _ring_path(self.symbol)
        tmp = path.with_suffix(".json.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(str(tmp), str(path))
            with self._lock:
                self._consecutive_write_failures = 0
        except OSError:
            with self._lock:
                self._consecutive_write_failures += 1
                fails = self._consecutive_write_failures
            if fails >= _WRITE_FAIL_CRITICAL_THRESHOLD:
                log.critical(
                    "bar_ring(%s): %d consecutive write failures — dashboard chart will be stale",
                    self.symbol,
                    fails,
                    exc_info=True,
                )
            else:
                log.warning(
                    "bar_ring(%s): write failed (failures=%d)",
                    self.symbol,
                    fails,
                    exc_info=True,
                )
            tmp.unlink(missing_ok=True)


# Module-level writer registry. One _RingWriter per symbol per process.
# Session-singleton (instance_lock) already enforces one process per symbol.
_writers: dict[str, _RingWriter] = {}
_writers_lock = threading.Lock()


def _get_or_create_writer(symbol: str, session_id: str | None = None) -> _RingWriter:
    with _writers_lock:
        w = _writers.get(symbol)
        if w is None:
            w = _RingWriter(symbol=symbol, session_id=session_id)
            _writers[symbol] = w
        return w


def enqueue_bar(symbol: str, bar: Bar, session_id: str | None = None) -> WriteResult:
    """Enqueue ``bar`` for the symbol's ring file. Called from BarPersister.append.

    Mock-contamination guard raises TypeError on mock objects (mirrors
    ``bot_state.write_state``). Real bars failing ``is_valid()`` are
    rejected; counter is exposed via WriteResult.invalid_rejected.

    Returns WriteResult; never raises on disk error (fail-open contract —
    trading must not block on dashboard IPC).
    """
    writer = _get_or_create_writer(symbol, session_id=session_id)
    return writer.enqueue(bar)


def drain_and_stop_writer(symbol: str, timeout: float = 5.0) -> None:
    """Block until queued bars are written, then stop the writer thread.

    Idempotent — calling on an already-stopped or missing writer is a no-op.
    Used by session shutdown ordering: drain queue → flush_to_db → clear_ring.
    """
    with _writers_lock:
        writer = _writers.pop(symbol, None)
    if writer is None:
        return
    writer.drain_and_stop(timeout=timeout)


def clear_ring(symbol: str) -> None:
    """Stop the writer and remove the ring file.

    Called from session shutdown AFTER flush_to_db succeeds. Belt-and-braces
    against stale rings: also called on session startup to clear any prior
    crash residue.
    """
    drain_and_stop_writer(symbol)
    _ring_path(symbol).unlink(missing_ok=True)


def read_bar_ring(symbol: str) -> RingSnapshot:
    """Read the current ring snapshot for ``symbol``.

    Fail-open: missing/corrupt files return an empty RingSnapshot. Never
    raises to the dashboard reader.
    """
    path = _ring_path(symbol)
    if not path.exists():
        return RingSnapshot(symbol=symbol)
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (json.JSONDecodeError, OSError):
        log.warning("bar_ring read failed for %s — returning empty", symbol, exc_info=True)
        return RingSnapshot(symbol=symbol)
    bars_raw = payload.get("bars")
    bars = bars_raw if isinstance(bars_raw, list) else []
    updated_utc: datetime | None = None
    ts_raw = payload.get("updated_utc")
    if isinstance(ts_raw, str):
        try:
            updated_utc = datetime.fromisoformat(ts_raw)
            if updated_utc.tzinfo is None:
                updated_utc = updated_utc.replace(tzinfo=UTC)
        except ValueError:
            updated_utc = None
    return RingSnapshot(
        symbol=str(payload.get("symbol", symbol)),
        bars=bars,
        writer_pid=payload.get("writer_pid") if isinstance(payload.get("writer_pid"), int) else None,
        session_id=payload.get("session_id") if isinstance(payload.get("session_id"), str) else None,
        updated_utc=updated_utc,
        invalid_rejected_count=int(payload.get("invalid_rejected_count", 0) or 0),
    )


def is_stale(snapshot: RingSnapshot, max_age_s: float = DEFAULT_STALE_AFTER_SECS) -> bool:
    """True if the snapshot is missing or older than ``max_age_s``."""
    if snapshot.updated_utc is None:
        return True
    age = (datetime.now(UTC) - snapshot.updated_utc).total_seconds()
    return age > max_age_s
