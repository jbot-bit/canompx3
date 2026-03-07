"""Tests for ui_v2.sse_manager — SSE broadcast infrastructure."""

from __future__ import annotations

import asyncio
import json

import pytest

from ui_v2.sse_manager import SSEManager


@pytest.fixture
def manager():
    return SSEManager()


# ── Connection lifecycle ─────────────────────────────────────────────────────


def test_connect_returns_id(manager: SSEManager):
    cid = manager.connect()
    assert isinstance(cid, str)
    assert len(cid) == 12
    assert manager.connection_count == 1


def test_disconnect_removes_client(manager: SSEManager):
    cid = manager.connect()
    assert manager.connection_count == 1
    manager.disconnect(cid)
    assert manager.connection_count == 0


def test_disconnect_unknown_is_noop(manager: SSEManager):
    manager.disconnect("nonexistent")
    assert manager.connection_count == 0


def test_multiple_clients(manager: SSEManager):
    ids = [manager.connect() for _ in range(5)]
    assert manager.connection_count == 5
    assert len(set(ids)) == 5  # all unique


# ── Broadcast ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_broadcast_to_all_clients(manager: SSEManager):
    """All connected clients receive the same broadcast."""
    cid1 = manager.connect()
    cid2 = manager.connect()

    manager.broadcast("test_event", {"msg": "hello"})

    # Collect from both subscribers
    results = []
    for cid in [cid1, cid2]:
        async for event in manager.subscribe(cid):
            results.append(event)
            break  # just get one event

    assert len(results) == 2
    for r in results:
        assert r["event"] == "test_event"
        assert json.loads(r["data"])["msg"] == "hello"


@pytest.mark.asyncio
async def test_broadcast_no_clients_is_noop(manager: SSEManager):
    """Broadcasting with no clients does not raise."""
    manager.broadcast("test_event", {"msg": "nobody home"})


# ── send_to ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_to_single_client(manager: SSEManager):
    cid1 = manager.connect()
    cid2 = manager.connect()

    manager.send_to(cid1, "targeted", {"for": "cid1"})

    # cid1 should have an event
    q1 = manager._clients[cid1]
    assert not q1.empty()

    # cid2 should NOT have an event
    q2 = manager._clients[cid2]
    assert q2.empty()


def test_send_to_unknown_client_is_noop(manager: SSEManager):
    manager.send_to("nonexistent", "test", {"x": 1})


# ── Subscribe generator ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_subscribe_terminates_on_disconnect(manager: SSEManager):
    """Subscribe generator stops when client is disconnected."""
    cid = manager.connect()

    collected = []

    async def consume():
        async for event in manager.subscribe(cid):
            collected.append(event)

    task = asyncio.create_task(consume())

    # Send one event, then disconnect
    manager.broadcast("ev1", {"n": 1})
    await asyncio.sleep(0.05)
    manager.disconnect(cid)
    await asyncio.sleep(0.05)

    # Task should complete
    await asyncio.wait_for(task, timeout=1.0)
    assert len(collected) == 1
    assert collected[0]["event"] == "ev1"


@pytest.mark.asyncio
async def test_subscribe_unknown_client_returns_immediately(manager: SSEManager):
    """Subscribing to a non-existent client yields nothing."""
    collected = []
    async for event in manager.subscribe("nonexistent"):
        collected.append(event)
    assert collected == []


# ── Heartbeat ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_heartbeat_start_and_shutdown(manager: SSEManager):
    """Start creates a heartbeat task, shutdown cancels it."""
    await manager.start()
    assert manager._heartbeat_task is not None
    assert not manager._heartbeat_task.done()

    await manager.shutdown()
    assert manager._heartbeat_task is None


@pytest.mark.asyncio
async def test_shutdown_disconnects_all_clients(manager: SSEManager):
    manager.connect()
    manager.connect()
    assert manager.connection_count == 2

    await manager.start()
    await manager.shutdown()
    assert manager.connection_count == 0
