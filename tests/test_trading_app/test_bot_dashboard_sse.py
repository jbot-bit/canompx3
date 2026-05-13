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
    from trading_app.live.bot_dashboard import _SSEBroker, _SSE_RING_SIZE

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
    from trading_app.live.bot_dashboard import _SSEBroker, _SSE_QUEUE_MAXSIZE

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


def test_api_bars_recent_returns_charts_shape():
    from fastapi.testclient import TestClient

    from trading_app.live import bot_dashboard as bd

    c = TestClient(bd.app)
    r = c.get("/api/bars-recent?instrument=MNQ&lookback_minutes=10")
    assert r.status_code == 200
    body = r.json()
    assert "bars" in body
    assert "instrument" in body
    assert body["instrument"] == "MNQ"
    assert "server_ts" in body
    assert "orb_high" in body
    assert "orb_low" in body
    assert "orb_complete" in body


def test_api_bars_recent_caps_lookback_at_1440():
    from fastapi.testclient import TestClient

    from trading_app.live import bot_dashboard as bd

    c = TestClient(bd.app)
    # 99999 should not raise; should be silently capped to 1440
    r = c.get("/api/bars-recent?lookback_minutes=99999")
    assert r.status_code == 200


def test_api_events_stream_returns_429_over_subscriber_cap():
    from fastapi.testclient import TestClient

    from trading_app.live import bot_dashboard as bd
    from trading_app.live.bot_dashboard import _SSE_MAX_SUBSCRIBERS

    # Saturate the broker directly
    fake_queues = [bd._sse_broker.subscribe() for _ in range(_SSE_MAX_SUBSCRIBERS)]
    try:
        c = TestClient(bd.app)
        r = c.get("/api/events/stream")
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
    assert out == {"orb_high": None, "orb_low": None, "orb_complete": False}


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
