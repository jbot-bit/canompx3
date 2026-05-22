"""Tests for Stage 3 broker-state-unknown kill-switch SLA.

The orchestrator watchdog must fire the kill switch when the broker stops
acknowledging equity reads with open positions on the book. This is the
2026-05-18 TopStepX incident class — connection accepted, reads stop.

Covers:
  - fresh equity (source="live") → no kill switch
  - stale equity past EQUITY_AGE_SLA_SECS with active position → kill switch
  - stale equity with NO active position → no kill switch (don't flatten nothing)
  - BrokerHTTPError raised on read → kill switch fires
  - signal-only mode → SLA gate disabled
  - broker adapter without query_equity_with_age (Rithmic) → fail-open, no kill switch
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from tests.test_trading_app.test_session_orchestrator import (
    STRATEGY_ID,
    FakeBrokerComponents,
    build_orchestrator,
)
from trading_app.live.http_client import BrokerHTTPError, EquityReading


def _orch_with_equity(positions: _EquityFakePositions, *, signal_only: bool = False):
    """Build an orchestrator whose `positions` adapter is the supplied stub.

    FakeBrokerComponents.__post_init__ overwrites positions unconditionally,
    so we patch after construction.
    """
    components = FakeBrokerComponents(fill_price=2350.0, signal_only=signal_only)
    components.positions = positions  # type: ignore[assignment]
    orch = build_orchestrator(components)
    orch.positions = positions  # type: ignore[assignment]
    return orch


class _EquityFakePositions:
    """FakePositions extended with a controllable query_equity_with_age."""

    def __init__(self, *, reading: EquityReading | None = None, raise_exc: Exception | None = None):
        self._reading = reading
        self._raise = raise_exc
        self.calls = 0

    def query_open(self, account_id: int) -> list[dict]:
        return []

    def query_equity_with_age(self, account_id: int) -> EquityReading:
        self.calls += 1
        if self._raise is not None:
            raise self._raise
        assert self._reading is not None, "configure either reading or raise_exc"
        return self._reading


# ---------------------------------------------------------------------------
# Direct tests of _broker_equity_stale (pure-function path).
# ---------------------------------------------------------------------------


class TestBrokerEquityStale:
    def test_fresh_live_equity_is_not_stale(self):
        positions = _EquityFakePositions(reading=EquityReading(value=51000.0, age_s=0.0, source="live"))
        orch = _orch_with_equity(positions)
        orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.0, order_id=1)
        orch._positions.on_entry_filled(STRATEGY_ID, 2350.0)

        stale, reason = orch._broker_equity_stale()

        assert stale is False
        assert reason == ""
        assert positions.calls == 1

    def test_stale_cache_past_sla_with_open_position_is_stale(self):
        # age beyond SLA, source="cache" — last-good served by client
        positions = _EquityFakePositions(
            reading=EquityReading(value=51000.0, age_s=120.0, source="cache"),
        )
        orch = _orch_with_equity(positions)
        orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.0, order_id=1)
        orch._positions.on_entry_filled(STRATEGY_ID, 2350.0)

        stale, reason = orch._broker_equity_stale()

        assert stale is True
        assert "equity_age=120s" in reason
        assert f"SLA {orch.EQUITY_AGE_SLA_SECS:.0f}s" in reason

    def test_stale_equity_with_no_active_position_is_not_stale(self):
        # Even if the broker has stopped reading, no exposure to defend.
        positions = _EquityFakePositions(
            reading=EquityReading(value=51000.0, age_s=999.0, source="cache"),
        )
        orch = _orch_with_equity(positions)
        # No on_entry_filled — book is flat.

        stale, reason = orch._broker_equity_stale()

        assert stale is False
        assert reason == ""
        assert positions.calls == 0, "must short-circuit before broker call when flat"

    def test_broker_http_error_marks_stale(self):
        positions = _EquityFakePositions(
            raise_exc=BrokerHTTPError("read timeout", error_class="B"),
        )
        orch = _orch_with_equity(positions)
        orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.0, order_id=1)
        orch._positions.on_entry_filled(STRATEGY_ID, 2350.0)

        stale, reason = orch._broker_equity_stale()

        assert stale is True
        assert "broker_unreachable" in reason
        assert "B" in reason

    def test_signal_only_disables_sla(self):
        positions = _EquityFakePositions(
            reading=EquityReading(value=51000.0, age_s=9999.0, source="cache"),
        )
        orch = _orch_with_equity(positions, signal_only=True)
        # Signal-only orchestrators have no order_router; SLA gate must short-circuit.

        stale, reason = orch._broker_equity_stale()

        assert stale is False
        assert reason == ""
        assert positions.calls == 0

    def test_adapter_without_query_equity_with_age_fails_open(self):
        # Stage 5 typed contract: stock FakePositions returns
        # EquityReading(value=None, age_s=0.0, source="missing"), simulating
        # Rithmic / Tradovate-today (no real query_equity_with_age impl). The
        # orchestrator's _broker_equity_stale recognises source="missing" and
        # fails OPEN — institutionally equivalent to the pre-Stage-5 ducktype
        # branch ("hasattr fell through, return False").
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.0, order_id=1)
        orch._positions.on_entry_filled(STRATEGY_ID, 2350.0)

        stale, reason = orch._broker_equity_stale()

        assert stale is False
        assert reason == ""


# ---------------------------------------------------------------------------
# Asyncio watchdog integration — the SLA branch must fire kill switch + flatten.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_watchdog_fires_on_stale_equity_with_open_position():
    positions = _EquityFakePositions(
        reading=EquityReading(value=51000.0, age_s=120.0, source="cache"),
    )
    orch = _orch_with_equity(positions)
    orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.0, order_id=1)
    orch._positions.on_entry_filled(STRATEGY_ID, 2350.0)
    # Feed-dead branch must NOT fire — bar arrived recently.
    orch._last_bar_at = datetime.now(UTC) - timedelta(seconds=10)
    orch.KILL_SWITCH_CHECK_INTERVAL = 0.01
    orch.EQUITY_AGE_SLA_SECS = 60.0  # below the 120s reading

    task = asyncio.create_task(orch._watchdog())
    await asyncio.sleep(0.1)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert orch._kill_switch_fired is True
    # Emergency flatten ran — book is now empty + router saw exit submit.
    assert orch._positions.active_positions() == []
    assert len(orch.order_router.submitted) == 1


@pytest.mark.asyncio
async def test_watchdog_does_not_fire_when_equity_is_fresh():
    positions = _EquityFakePositions(
        reading=EquityReading(value=51000.0, age_s=0.0, source="live"),
    )
    orch = _orch_with_equity(positions)
    orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.0, order_id=1)
    orch._positions.on_entry_filled(STRATEGY_ID, 2350.0)
    orch._last_bar_at = datetime.now(UTC) - timedelta(seconds=10)
    orch.KILL_SWITCH_CHECK_INTERVAL = 0.01

    task = asyncio.create_task(orch._watchdog())
    await asyncio.sleep(0.1)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert orch._kill_switch_fired is False
    assert orch._positions.active_positions() != []
