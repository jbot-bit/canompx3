"""Tests for ui_v2.session_monitor — file watcher for live signals."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from ui_v2.session_monitor import SessionMonitor
from ui_v2.sse_manager import SSEManager


@pytest.fixture
def sse_manager():
    return SSEManager()


@pytest.fixture
def signals_path(tmp_path: Path) -> Path:
    return tmp_path / "live_signals.jsonl"


@pytest.fixture
def monitor():
    m = SessionMonitor()
    yield m
    m.stop()


# ── File watching ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_detects_new_lines(
    monitor: SessionMonitor,
    sse_manager: SSEManager,
    signals_path: Path,
):
    """New lines appended after start() are detected and broadcast."""
    # Create empty file so monitor starts at offset 0
    signals_path.write_text("", encoding="utf-8")

    cid = sse_manager.connect()
    monitor.start(sse_manager, signals_path)

    # Append a signal line
    signal = {"type": "SIGNAL_ENTRY", "instrument": "MGC", "direction": "long", "price": 2950.0}
    with open(signals_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(signal) + "\n")

    # Wait for poll cycle to pick it up
    await asyncio.sleep(1.0)

    # Check that SSE manager got the broadcast
    q = sse_manager._clients.get(cid)
    assert q is not None
    assert not q.empty()

    event = q.get_nowait()
    assert event["event"] == "signal"
    data = json.loads(event["data"])
    assert data["type"] == "SIGNAL_ENTRY"
    assert data["instrument"] == "MGC"


@pytest.mark.asyncio
async def test_exit_signal_triggers_debrief_required(
    monitor: SessionMonitor,
    sse_manager: SSEManager,
    signals_path: Path,
):
    """Exit signals produce both a 'signal' and a 'debrief_required' event."""
    signals_path.write_text("", encoding="utf-8")

    cid = sse_manager.connect()
    monitor.start(sse_manager, signals_path)

    signal = {"type": "SIGNAL_EXIT", "strategy_id": "MGC_CME_REOPEN_E2", "ts": "2026-03-07T10:00:00Z"}
    with open(signals_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(signal) + "\n")

    await asyncio.sleep(1.0)

    q = sse_manager._clients.get(cid)
    events = []
    while not q.empty():
        events.append(q.get_nowait())

    event_types = [e["event"] for e in events]
    assert "signal" in event_types
    assert "debrief_required" in event_types

    debrief_event = next(e for e in events if e["event"] == "debrief_required")
    debrief_data = json.loads(debrief_event["data"])
    assert debrief_data["strategy_id"] == "MGC_CME_REOPEN_E2"


@pytest.mark.asyncio
async def test_manual_entry_broadcast(
    monitor: SessionMonitor,
    sse_manager: SSEManager,
    signals_path: Path,
):
    """Manual entry/exit signals are broadcast as 'signal' events."""
    signals_path.write_text("", encoding="utf-8")

    cid = sse_manager.connect()
    monitor.start(sse_manager, signals_path)

    signal = {"type": "MANUAL_ENTRY", "instrument": "MNQ", "direction": "short", "price": 21500.0}
    with open(signals_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(signal) + "\n")

    await asyncio.sleep(1.0)

    q = sse_manager._clients.get(cid)
    assert not q.empty()
    event = q.get_nowait()
    assert event["event"] == "signal"
    data = json.loads(event["data"])
    assert data["type"] == "MANUAL_ENTRY"


@pytest.mark.asyncio
async def test_ignores_preexisting_content(
    monitor: SessionMonitor,
    sse_manager: SSEManager,
    signals_path: Path,
):
    """Lines present before start() are NOT broadcast."""
    # Write a line BEFORE starting monitor
    old_signal = {"type": "SIGNAL_ENTRY", "instrument": "MES", "price": 5000.0}
    signals_path.write_text(json.dumps(old_signal) + "\n", encoding="utf-8")

    cid = sse_manager.connect()
    monitor.start(sse_manager, signals_path)

    await asyncio.sleep(1.0)

    q = sse_manager._clients.get(cid)
    assert q.empty(), "Pre-existing lines should NOT be broadcast"


# ── Start / Stop ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_start_stop_idempotent(
    monitor: SessionMonitor,
    sse_manager: SSEManager,
    signals_path: Path,
):
    """start() and stop() are idempotent."""
    signals_path.write_text("", encoding="utf-8")

    monitor.start(sse_manager, signals_path)
    monitor.start(sse_manager, signals_path)  # second call is no-op
    assert monitor._task is not None

    monitor.stop()
    monitor.stop()  # second call is no-op
    assert monitor._task is None


@pytest.mark.asyncio
async def test_handles_missing_file(
    monitor: SessionMonitor,
    sse_manager: SSEManager,
    signals_path: Path,
):
    """Monitor handles a missing file gracefully (file created later)."""
    # Don't create the file
    cid = sse_manager.connect()
    monitor.start(sse_manager, signals_path)

    await asyncio.sleep(0.6)

    # Now create file and write
    signal = {"type": "SIGNAL_ENTRY", "instrument": "M2K", "price": 2100.0}
    with open(signals_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(signal) + "\n")

    await asyncio.sleep(1.0)

    q = sse_manager._clients.get(cid)
    assert not q.empty()
    event = q.get_nowait()
    assert event["event"] == "signal"
