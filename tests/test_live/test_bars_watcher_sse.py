"""Characterization tests for the ``_bars_watcher`` SSE producer.

The App-Cohesion report *thought* the live candle stream was broken; ground-truth
inspection REFUTED that — the producer chain is wired and API-correct. The durable
cohesion win is therefore not a fix but a regression guard: lock the already-working
producer so a future refactor cannot silently stop the chart updating.

These are producer-side unit tests. They mock the ring (`bar_ring.read_bar_ring` /
`is_stale`), the bot state (`bot_state.read_state`), and the in-process SSE broker,
then drive the watcher's infinite poll loop for a bounded number of iterations by
patching the module's ``asyncio.sleep`` to raise a sentinel — the
``condition-based-waiting`` rule (poll the condition, never wall-clock sleep).

Covers two invariants:
1. Ring-live + heartbeat-fresh → publishes a ``bar`` SSE event with OHLCV for each
   strictly-newer ring row.
2. live → stale heartbeat transition → emits ``bars_source_changed`` exactly once,
   then suppresses further ring pushes.

Companion: ``tests/test_live/test_dashboard_account_selection.py`` (selector backend).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import trading_app.live.bar_ring as bar_ring
import trading_app.live.bot_dashboard as bd
import trading_app.live.bot_state as bot_state


class _StopWatcher(Exception):
    """Sentinel raised from the patched ``asyncio.sleep`` to end the loop."""


class _RecordingBroker:
    """Stand-in for ``_sse_broker`` that records every published event."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def publish(self, event_type: str, data: dict) -> None:
        self.events.append((event_type, data))


def _fresh_heartbeat_state(instrument: str = "MNQ") -> dict:
    """Bot-state dict with a heartbeat well inside the 120s freshness window."""
    return {
        "heartbeat_utc": datetime.now(UTC).isoformat(),
        "lanes": {"L1": {"instrument": instrument}},
    }


def _stale_heartbeat_state(instrument: str = "MNQ") -> dict:
    """Bot-state dict whose heartbeat is older than HEARTBEAT_STALE_AFTER_S."""
    old = datetime.now(UTC) - timedelta(seconds=bd.HEARTBEAT_STALE_AFTER_S + 30)
    return {
        "heartbeat_utc": old.isoformat(),
        "lanes": {"L1": {"instrument": instrument}},
    }


def _ring_snapshot(instrument: str, bars: list[dict]) -> bar_ring.RingSnapshot:
    return bar_ring.RingSnapshot(symbol=instrument, bars=bars)


def _bar_row(ts: datetime, *, close: float = 100.0) -> dict:
    return {
        "ts_utc": ts.isoformat(),
        "open": close - 1.0,
        "high": close + 1.0,
        "low": close - 2.0,
        "close": close,
        "volume": 42,
    }


def _patch_loop(monkeypatch, *, max_iterations: int) -> None:
    """Patch the watcher's ``asyncio.sleep`` to bound the infinite loop.

    Each loop tick ends in ``await asyncio.sleep(2.0)``. We let ``max_iterations``
    ticks run, then raise ``_StopWatcher`` on the next sleep so the coroutine
    terminates deterministically — no real time elapses (condition-based-waiting).
    """
    state = {"calls": 0}

    async def fake_sleep(_seconds: float) -> None:
        state["calls"] += 1
        if state["calls"] >= max_iterations:
            raise _StopWatcher

    monkeypatch.setattr(bd.asyncio, "sleep", fake_sleep)


async def _run_bounded(coro_fn, monkeypatch, *, max_iterations: int) -> None:
    _patch_loop(monkeypatch, max_iterations=max_iterations)
    try:
        await coro_fn()
    except _StopWatcher:
        pass


async def test_ring_live_fresh_heartbeat_publishes_bar_event(monkeypatch):
    """Ring fresh + heartbeat fresh → a ``bar`` SSE event with OHLCV.

    Two ticks: tick 1 bootstraps ``last_seen`` (records latest ts, no publish);
    tick 2 sees a strictly-newer bar and publishes it. This mirrors the real
    bootstrap-then-stream contract — the chart bulk-loads via /api/bars-recent on
    connect, then streams only newer rows.
    """
    broker = _RecordingBroker()
    monkeypatch.setattr(bd, "_sse_broker", broker)

    base = datetime.now(UTC).replace(microsecond=0)
    older = _bar_row(base, close=100.0)
    newer = _bar_row(base + timedelta(minutes=1), close=101.0)

    # Tick 1 sees [older]; tick 2 sees [older, newer]. last_seen bootstraps to
    # `older` on tick 1, so only `newer` publishes on tick 2.
    snapshots = iter(
        [
            _ring_snapshot("MNQ", [older]),
            _ring_snapshot("MNQ", [older, newer]),
        ]
    )
    monkeypatch.setattr(bar_ring, "read_bar_ring", lambda _inst: next(snapshots))
    monkeypatch.setattr(bar_ring, "is_stale", lambda _snap: False)
    monkeypatch.setattr(bot_state, "read_state", _fresh_heartbeat_state)

    await _run_bounded(bd._bars_watcher, monkeypatch, max_iterations=2)

    bar_events = [d for ev, d in broker.events if ev == "bar"]
    assert len(bar_events) == 1, broker.events
    published = bar_events[0]
    assert published["instrument"] == "MNQ"
    assert published["time"] == int((base + timedelta(minutes=1)).timestamp())
    assert published["open"] == 100.0
    assert published["high"] == 102.0
    assert published["low"] == 99.0
    assert published["close"] == 101.0
    assert published["volume"] == 42
    # No fallback event while the ring is live.
    assert not any(ev == "bars_source_changed" for ev, _ in broker.events)


async def test_live_to_stale_transition_emits_source_changed_once(monkeypatch):
    """Ring live → heartbeat stale → one ``bars_source_changed``, then suppressed.

    Tick 1: ring live (records ring_source_live=True). Tick 2: heartbeat stale →
    transition emit. Tick 3: still stale → no second event (idempotent).
    """
    broker = _RecordingBroker()
    monkeypatch.setattr(bd, "_sse_broker", broker)

    base = datetime.now(UTC).replace(microsecond=0)
    live_snap = _ring_snapshot("MNQ", [_bar_row(base)])
    # After the bot dies the ring file is unchanged but heartbeat goes stale.
    stale_snaps = iter([live_snap, live_snap, live_snap])
    states = iter(
        [
            _fresh_heartbeat_state(),  # tick 1: live
            _stale_heartbeat_state(),  # tick 2: transition
            _stale_heartbeat_state(),  # tick 3: still stale, no re-emit
        ]
    )
    monkeypatch.setattr(bar_ring, "read_bar_ring", lambda _inst: next(stale_snaps))
    monkeypatch.setattr(bar_ring, "is_stale", lambda _snap: False)
    monkeypatch.setattr(bot_state, "read_state", lambda: next(states))

    await _run_bounded(bd._bars_watcher, monkeypatch, max_iterations=3)

    source_changed = [d for ev, d in broker.events if ev == "bars_source_changed"]
    assert len(source_changed) == 1, broker.events
    assert source_changed[0]["instrument"] == "MNQ"
    assert source_changed[0]["source"] == "gold_db"


async def test_both_stale_suppresses_ring_push(monkeypatch):
    """Stale heartbeat from the first tick → no ``bar`` event ever published.

    Guards the crash-detection gate: a post-crash ring that is still inside its
    90s freshness window must NOT stream as live when the heartbeat is dead.
    Because the ring was never live, no ``bars_source_changed`` fires either
    (only a real live→stale transition emits the fallback).
    """
    broker = _RecordingBroker()
    monkeypatch.setattr(bd, "_sse_broker", broker)

    base = datetime.now(UTC).replace(microsecond=0)
    snap = _ring_snapshot("MNQ", [_bar_row(base)])
    monkeypatch.setattr(bar_ring, "read_bar_ring", lambda _inst: snap)
    monkeypatch.setattr(bar_ring, "is_stale", lambda _snap: False)
    monkeypatch.setattr(bot_state, "read_state", _stale_heartbeat_state)

    await _run_bounded(bd._bars_watcher, monkeypatch, max_iterations=2)

    assert not any(ev == "bar" for ev, _ in broker.events), broker.events
    assert not any(ev == "bars_source_changed" for ev, _ in broker.events), broker.events
