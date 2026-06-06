"""Tests for ``_reconcile_positions_on_reconnect`` (Row 08 MED #2, capital path).

The mid-session reconnect loop must re-sync local position state against broker
truth before resuming. The dangerous case: a fill arrives during the disconnect
window, so the broker holds a position the local tracker is blind to. On resume,
the engine could act as if flat while a REAL position is open → naked exposure.
The fix fails closed (kill switch + emergency flatten), mirroring the F4-3 /
feed-dead path.

Per institutional-rigor.md §11 (verify by known-violation injection), the
load-bearing test seeds the exact desync (broker open, local flat) and asserts the
kill switch + flatten fire. The other tests pin the non-firing cases so the guard
cannot become a hair-trigger that flattens a legitimately-held position.

Reuses the FakeBrokerComponents / build_orchestrator harness from
test_session_orchestrator.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from tests.test_trading_app.test_session_orchestrator import (
    STRATEGY_ID,
    FakeBrokerComponents,
    FakePositions,
    build_orchestrator,
)


def _orch_with_broker_open(broker_positions: list[dict]):
    """Orchestrator whose broker (query_open) reports the given open positions."""
    components = FakeBrokerComponents(orphans=broker_positions)
    orch = build_orchestrator(components)
    orch._emergency_flatten = AsyncMock()
    orch._notify = lambda *a, **k: None
    return orch


async def test_broker_flat_resumes_silently() -> None:
    """Broker reports no open positions → in sync, no kill switch, no flatten."""
    orch = _orch_with_broker_open([])
    await orch._reconcile_positions_on_reconnect(reconnect_count=1)
    assert orch._kill_switch_fired is False
    orch._emergency_flatten.assert_not_awaited()


async def test_untracked_broker_position_kills_and_flattens() -> None:
    """THE desync: broker open + local tracker flat → kill switch + flatten fire.

    This is the naked-exposure case the fix exists to catch — a fill that landed
    during the disconnect window leaves the broker holding a position we never
    tracked. Resuming blind would be capital-unsafe; we close it.
    """
    orch = _orch_with_broker_open([{"contract_id": "MGCJ6", "side": "long", "size": 1, "avg_price": 2350.0}])
    assert orch._positions.active_positions() == []  # local tracker is flat
    await orch._reconcile_positions_on_reconnect(reconnect_count=2)
    assert orch._kill_switch_fired is True, "untracked broker position must fire kill switch"
    orch._emergency_flatten.assert_awaited_once()


async def test_held_position_across_reconnect_resumes_without_flatten() -> None:
    """Broker open AND local tracker active → genuinely-held position, resume.

    Not a desync — a position is open and we know about it. The live engine keeps
    managing it; flattening here would wrongly close a legitimate position.
    """
    orch = _orch_with_broker_open([{"contract_id": "MGCJ6", "side": "long", "size": 1, "avg_price": 2350.0}])
    orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.0, order_id=1)
    orch._positions.on_entry_filled(STRATEGY_ID, 2350.0)
    assert orch._positions.active_positions(), "precondition: local tracker is active"
    await orch._reconcile_positions_on_reconnect(reconnect_count=1)
    assert orch._kill_switch_fired is False, "held position must NOT trigger kill switch"
    orch._emergency_flatten.assert_not_awaited()


async def test_signal_only_skips_reconcile() -> None:
    """Signal-only sessions have no live position → nothing to reconcile."""
    orch = build_orchestrator(FakeBrokerComponents(signal_only=True))
    orch._emergency_flatten = AsyncMock()
    orch._notify = lambda *a, **k: None
    await orch._reconcile_positions_on_reconnect(reconnect_count=1)
    assert orch._kill_switch_fired is False
    orch._emergency_flatten.assert_not_awaited()


async def test_query_open_not_implemented_falls_back_silently() -> None:
    """Adapter without query_open() → warn + fall back to local tracker, no halt.

    Cannot prove a desync, so do not block resume (matches _confirmed_flat_at_broker).
    """
    orch = _orch_with_broker_open([])

    def _raise_not_implemented(account_id):
        raise NotImplementedError

    orch.positions.query_open = _raise_not_implemented
    await orch._reconcile_positions_on_reconnect(reconnect_count=1)
    assert orch._kill_switch_fired is False
    orch._emergency_flatten.assert_not_awaited()


async def test_query_open_failure_fails_closed_with_alert() -> None:
    """A generic query failure must NOT silently resume — alert the operator.

    Fail-closed: we cannot verify broker state, so we surface it. We do NOT fire
    the kill switch on a transient query error (that would be a hair-trigger on
    every flaky reconnect), but the operator is told to verify manually.
    """
    orch = _orch_with_broker_open([])
    notifications: list[str] = []
    orch._notify = lambda msg, *a, **k: notifications.append(msg)

    def _raise(account_id):
        raise RuntimeError("broker unreachable")

    orch.positions.query_open = _raise
    await orch._reconcile_positions_on_reconnect(reconnect_count=3)
    assert orch._kill_switch_fired is False
    orch._emergency_flatten.assert_not_awaited()
    assert any("RECONNECT RESYNC FAILED" in m for m in notifications), "operator must be alerted"


async def test_flatten_failure_alerts_manual_close() -> None:
    """If the emergency flatten itself fails, escalate to MANUAL CLOSE REQUIRED."""
    orch = _orch_with_broker_open([{"contract_id": "MGCJ6", "side": "long", "size": 1, "avg_price": 2350.0}])
    notifications: list[str] = []
    orch._notify = lambda msg, *a, **k: notifications.append(msg)
    orch._emergency_flatten = AsyncMock(side_effect=RuntimeError("broker dead"))
    await orch._reconcile_positions_on_reconnect(reconnect_count=1)
    assert orch._kill_switch_fired is True
    assert any("MANUAL CLOSE REQUIRED" in m for m in notifications)
