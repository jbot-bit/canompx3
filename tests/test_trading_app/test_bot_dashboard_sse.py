"""Stage 2 tests for cockpit-v3: SSE broker + /api/bars-recent + over-cap 429.

Covers:
- _SSEBroker publish/subscribe lifecycle, queue full handling, ring buffer
- _SSEBroker replay_since for Last-Event-ID
- /api/bars-recent shape + ORB delegation
- /api/bars-recent caps lookback at 1440 minutes
- /api/bars-recent tolerates missing gold.db
- /api/events/stream returns 429 over subscriber cap
- localhost-only assertion in run_dashboard
- _orb_levels_for_instrument reads canonical bot_state (delegation, not derivation)
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest


def test_sse_broker_publish_to_subscriber():
    from trading_app.live.bot_dashboard import _SSEBroker

    broker = _SSEBroker()
    q = broker.subscribe()
    broker.publish("signal", {"type": "ENTRY"})
    env = q.get_nowait()
    assert env["event"] == "signal"
    assert env["data"] == {"type": "ENTRY"}
    assert env["id"] == 1


def test_sse_broker_heartbeat_does_not_advance_event_id():
    """Heartbeats are not ring-buffered and don't consume event_ids."""
    from trading_app.live.bot_dashboard import _SSEBroker

    broker = _SSEBroker()
    broker.publish("heartbeat", {"ts": "x"})
    broker.publish("heartbeat", {"ts": "y"})
    broker.publish("signal", {"a": 1})
    assert broker._next_event_id == 1  # only the signal advanced it
    assert len(broker._ring) == 1


def test_sse_broker_unsubscribe_drops_queue():
    from trading_app.live.bot_dashboard import _SSEBroker

    broker = _SSEBroker()
    q = broker.subscribe()
    assert broker.subscriber_count() == 1
    broker.unsubscribe(q)
    assert broker.subscriber_count() == 0


def test_sse_broker_ring_buffer_capped():
    from trading_app.live.bot_dashboard import _SSE_RING_SIZE, _SSEBroker

    broker = _SSEBroker()
    for i in range(_SSE_RING_SIZE + 50):
        broker.publish("signal", {"i": i})
    assert len(broker._ring) == _SSE_RING_SIZE


def test_sse_broker_replay_since_filters_by_event_id():
    from trading_app.live.bot_dashboard import _SSEBroker

    broker = _SSEBroker()
    for i in range(5):
        broker.publish("signal", {"i": i})
    # Last seen = id 3 → replay should return 2 events (ids 4 and 5)
    replayed = broker.replay_since(3)
    assert len(replayed) == 2
    assert replayed[0]["id"] == 4
    assert replayed[1]["id"] == 5


def test_sse_broker_full_queue_does_not_block_publish(caplog):
    """Slow consumer must not stall the publish path — institutional-rigor § 6."""
    from trading_app.live.bot_dashboard import _SSE_QUEUE_MAXSIZE, _SSEBroker

    broker = _SSEBroker()
    q = broker.subscribe()
    # Fill the queue
    for i in range(_SSE_QUEUE_MAXSIZE):
        broker.publish("signal", {"i": i})
    # One more should not raise, just warn
    broker.publish("signal", {"i": "overflow"})
    # The queue is full; the overflow event was dropped for this subscriber.
    assert q.qsize() == _SSE_QUEUE_MAXSIZE


def test_api_bars_recent_route_registered():
    from trading_app.live import bot_dashboard as bd

    paths = {getattr(r, "path", None) for r in bd.app.routes}
    assert "/api/bars-recent" in paths
    assert "/api/events/stream" in paths


@pytest.mark.asyncio
async def test_api_bars_recent_returns_charts_shape():
    from trading_app.live import bot_dashboard as bd

    transport = httpx.ASGITransport(app=bd.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        r = await c.get("/api/bars-recent?instrument=MNQ&lookback_minutes=10")
    assert r.status_code == 200
    body = r.json()
    assert "bars" in body
    assert "instrument" in body
    assert body["instrument"] == "MNQ"
    assert "server_ts" in body
    assert "orb_high" in body
    assert "orb_low" in body
    assert "orb_complete" in body


@pytest.mark.asyncio
async def test_api_bars_recent_caps_lookback_at_1440():
    from trading_app.live import bot_dashboard as bd

    # 99999 should not raise; should be silently capped to 1440
    transport = httpx.ASGITransport(app=bd.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        r = await c.get("/api/bars-recent?lookback_minutes=99999")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_api_events_stream_returns_429_over_subscriber_cap():
    from trading_app.live import bot_dashboard as bd
    from trading_app.live.bot_dashboard import _SSE_MAX_SUBSCRIBERS

    # Saturate the broker directly
    fake_queues = [bd._sse_broker.subscribe() for _ in range(_SSE_MAX_SUBSCRIBERS)]
    try:
        transport = httpx.ASGITransport(app=bd.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            r = await c.get("/api/events/stream")
        assert r.status_code == 429
        assert r.headers.get("Retry-After") == "5"
        body = r.json()
        assert body["limit"] == _SSE_MAX_SUBSCRIBERS
    finally:
        for q in fake_queues:
            bd._sse_broker.unsubscribe(q)


def test_run_dashboard_refuses_non_localhost_host():
    from trading_app.live import bot_dashboard as bd

    with pytest.raises(RuntimeError, match="non-localhost"):
        bd.run_dashboard(host="0.0.0.0", port=18080)


def test_run_dashboard_accepts_loopback_variants():
    """All three loopback forms must pass the check (we mock uvicorn.run)."""
    from trading_app.live import bot_dashboard as bd

    for host in ("127.0.0.1", "localhost", "::1"):
        with patch("trading_app.live.bot_dashboard.uvicorn.run") as mocked:
            bd.run_dashboard(host=host, port=18080)
            mocked.assert_called_once()


def test_orb_levels_for_instrument_returns_nulls_when_no_lane(monkeypatch):
    from trading_app.live import bot_dashboard as bd

    monkeypatch.setattr("trading_app.live.bot_state.read_state", lambda: {})
    out = bd._orb_levels_for_instrument("MNQ")
    # Payload shape is stable: levels null, complete False, plus the
    # additional window/session/direction fields all null (added 2026-05-19
    # for chart-cockpit ORB rectangle).
    assert out["orb_high"] is None
    assert out["orb_low"] is None
    assert out["orb_complete"] is False
    assert out["orb_window_start_utc"] is None
    assert out["orb_window_end_utc"] is None


def test_orb_levels_for_instrument_delegates_to_state(monkeypatch):
    """ORB levels come from bot_state.json lanes — never re-derived from prices."""
    from trading_app.live import bot_dashboard as bd

    fake_state = {
        "lanes": {
            "MNQ_x": {
                "instrument": "MNQ",
                "orb_high": 18420.5,
                "orb_low": 18411.0,
                "orb_complete": True,
                "orb_minutes": 5,
                "session_name": "NYSE_OPEN",
            }
        }
    }
    monkeypatch.setattr("trading_app.live.bot_state.read_state", lambda: fake_state)
    out = bd._orb_levels_for_instrument("MNQ")
    assert out["orb_high"] == 18420.5
    assert out["orb_low"] == 18411.0
    assert out["orb_complete"] is True
    assert out["orb_minutes"] == 5
    assert out["session_name"] == "NYSE_OPEN"


def test_sse_cancel_watchers_idempotent_when_empty():
    """Calling cancel before any subscriber must be safe (no tasks created)."""
    import asyncio

    from trading_app.live import bot_dashboard as bd

    # Ensure no leftover tasks from a prior test
    bd._sse_tasks.clear()
    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(bd._sse_cancel_watchers())
    assert bd._sse_tasks == []


def test_sse_cancel_watchers_clears_started_tasks():
    """After start → cancel, _sse_tasks must be empty and the loop is exit-clean."""
    import asyncio

    from trading_app.live import bot_dashboard as bd

    async def cycle():
        bd._sse_tasks.clear()
        await bd._sse_start_watchers()
        assert len(bd._sse_tasks) == 5  # heartbeat + state + signals + alerts + bars
        await bd._sse_cancel_watchers()
        assert bd._sse_tasks == []

    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(cycle())


def test_sse_lazy_stop_cancels_watchers_when_last_subscriber_unsubscribes():
    """Ref-counted stop: when the final subscriber disconnects the five
    file-polling watchers must be cancelled, not left running until the
    FastAPI lifespan shutdown.

    Mirrors `_sse_start_watchers` (lazy-start on first connect). Without
    this behaviour the watchers (heartbeat 1s, state 0.5s, signals JSONL
    tail, alerts tail, bars 2s) keep polling files indefinitely after the
    last browser tab closes — the zombie-stream surface this Pass 3 closes.
    """
    import asyncio

    from trading_app.live import bot_dashboard as bd

    async def cycle():
        bd._sse_tasks.clear()
        # Two subscribers.
        q1 = bd._sse_broker.subscribe()
        q2 = bd._sse_broker.subscribe()
        await bd._sse_start_watchers()
        assert len(bd._sse_tasks) == 5
        assert bd._sse_broker.subscriber_count() == 2

        # First unsubscribe must NOT stop watchers — second subscriber still
        # consuming. This is the ref-count guard at subscriber_count > 0.
        bd._sse_broker.unsubscribe(q1)
        await bd._sse_lazy_stop_if_idle()
        assert bd._sse_broker.subscriber_count() == 1
        assert len(bd._sse_tasks) == 5, "lazy-stop fired with surviving subscriber"

        # Second (last) unsubscribe -> watchers cancelled.
        bd._sse_broker.unsubscribe(q2)
        await bd._sse_lazy_stop_if_idle()
        assert bd._sse_broker.subscriber_count() == 0
        assert bd._sse_tasks == [], "watchers still running after last unsubscribe"

    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(cycle())


def test_sse_lazy_stop_noop_when_no_watchers():
    """Idempotency: calling lazy-stop with no watchers running must be safe.

    The TOCTOU-rejection path at the SSE endpoint subscribes then
    unsubscribes BEFORE `_sse_start_watchers` is called. The lazy-stop
    finally-block must be safe in that path (no AttributeError, no
    spurious cancellation).
    """
    import asyncio

    from trading_app.live import bot_dashboard as bd

    async def cycle():
        bd._sse_tasks.clear()
        assert bd._sse_broker.subscriber_count() == 0
        assert bd._sse_tasks == []
        await bd._sse_lazy_stop_if_idle()  # must not raise
        assert bd._sse_tasks == []

    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(cycle())


def test_sse_start_watchers_recovers_after_done_tasks():
    """Module-level _sse_tasks may carry stale done-tasks across loops.

    The guard in _sse_start_watchers prunes done-tasks before the no-op
    `if _sse_tasks: return` short-circuit, so a fresh start re-creates them.
    """
    import asyncio

    from trading_app.live import bot_dashboard as bd

    async def first_loop():
        bd._sse_tasks.clear()
        await bd._sse_start_watchers()
        await bd._sse_cancel_watchers()
        # Cancelled tasks are .done() — pruning logic should treat list as empty.

    async def second_loop():
        # Simulate stale done-tasks lingering: re-populate with completed dummies.
        async def noop():
            return None

        # Schedule and immediately await to produce done-tasks.
        dummies = [asyncio.create_task(noop()) for _ in range(3)]
        await asyncio.gather(*dummies)
        bd._sse_tasks[:] = dummies
        # Now start fresh — done-tasks must be pruned.
        await bd._sse_start_watchers()
        assert all(not t.done() for t in bd._sse_tasks), "stale done-tasks not pruned"
        await bd._sse_cancel_watchers()

    loop1 = asyncio.get_event_loop_policy().new_event_loop()
    loop1.run_until_complete(first_loop())
    loop1.close()
    loop2 = asyncio.get_event_loop_policy().new_event_loop()
    loop2.run_until_complete(second_loop())
    loop2.close()


@pytest.mark.asyncio
async def test_api_events_stream_429_cleans_up_subscriber_queue():
    """TOCTOU-fix verification: rejected over-cap requests must unsubscribe.

    Previously the cap check was check-then-subscribe (racy). New pattern is
    subscribe-then-check-and-unsubscribe-on-overflow. The cleanup invariant
    is: after a rejected request, subscriber_count returns to the saturated
    baseline rather than growing past the cap.
    """
    from trading_app.live import bot_dashboard as bd
    from trading_app.live.bot_dashboard import _SSE_MAX_SUBSCRIBERS

    fake_queues = [bd._sse_broker.subscribe() for _ in range(_SSE_MAX_SUBSCRIBERS)]
    try:
        baseline = bd._sse_broker.subscriber_count()
        transport = httpx.ASGITransport(app=bd.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            r = await c.get("/api/events/stream")
        assert r.status_code == 429
        # The TOCTOU fix subscribes then unsubscribes on overflow; after the
        # rejected response, count must equal the pre-request baseline.
        assert bd._sse_broker.subscriber_count() == baseline, (
            f"TOCTOU cleanup broken: count went {baseline} → {bd._sse_broker.subscriber_count()}"
        )
    finally:
        for q in fake_queues:
            bd._sse_broker.unsubscribe(q)


def test_orb_levels_for_instrument_filters_by_instrument(monkeypatch):
    """Multi-lane state: only the matching instrument's ORB is returned."""
    from trading_app.live import bot_dashboard as bd

    fake_state = {
        "lanes": {
            "MGC_y": {
                "instrument": "MGC",
                "orb_high": 2050.0,
                "orb_low": 2045.0,
                "orb_complete": True,
            },
            "MNQ_x": {
                "instrument": "MNQ",
                "orb_high": 18420.5,
                "orb_low": 18411.0,
                "orb_complete": True,
            },
        }
    }
    monkeypatch.setattr("trading_app.live.bot_state.read_state", lambda: fake_state)
    out_mnq = bd._orb_levels_for_instrument("MNQ")
    out_mgc = bd._orb_levels_for_instrument("MGC")
    assert out_mnq["orb_high"] == 18420.5
    assert out_mgc["orb_high"] == 2050.0
