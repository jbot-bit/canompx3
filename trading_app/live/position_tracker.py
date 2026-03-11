"""Position lifecycle state machine for live trading.

Tracks each strategy's position through explicit states:
  FLAT → PENDING_ENTRY → ENTERED → PENDING_EXIT → FLAT

Replaces the ad-hoc _entry_prices dict with auditable state transitions,
timeout detection for stuck orders, and order→strategy mapping.
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

log = logging.getLogger(__name__)


class PositionState(Enum):
    FLAT = "FLAT"
    PENDING_ENTRY = "PENDING_ENTRY"
    ENTERED = "ENTERED"
    PENDING_EXIT = "PENDING_EXIT"


@dataclass
class PositionRecord:
    strategy_id: str
    state: PositionState = PositionState.FLAT
    direction: str | None = None
    engine_entry_price: float | None = None
    fill_entry_price: float | None = None
    entry_order_id: int | None = None
    entry_slippage: float | None = None
    contracts: int = 1
    bracket_order_ids: list[int] = field(default_factory=list)
    exit_order_id: int | None = None
    fill_exit_price: float | None = None
    entered_at: datetime | None = None
    state_changed_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class PositionTracker:
    """Manages position lifecycle for all strategies in a session."""

    def __init__(self):
        self._positions: dict[str, PositionRecord] = {}

    def on_entry_sent(
        self,
        strategy_id: str,
        direction: str,
        engine_price: float,
        order_id: int | None = None,
        contracts: int = 1,
    ) -> PositionRecord | None:
        """Record that an entry order has been submitted to the broker.

        Returns None (rejects) if strategy already has an active position
        (PENDING_ENTRY or ENTERED). Allows overwrite from PENDING_EXIT
        (stale from failed rollover exit).
        """
        existing = self._positions.get(strategy_id)
        if existing is not None and existing.state != PositionState.FLAT:
            if existing.state in (PositionState.PENDING_ENTRY, PositionState.ENTERED):
                log.warning(
                    "Entry REJECTED for %s — already in state %s (active position)",
                    strategy_id,
                    existing.state.value,
                )
                return None
            # PENDING_EXIT: stale from failed rollover — allow overwrite
            log.warning(
                "Entry for %s overwriting stale PENDING_EXIT (failed rollover cleanup)",
                strategy_id,
            )
        now = datetime.now(UTC)
        record = PositionRecord(
            strategy_id=strategy_id,
            state=PositionState.PENDING_ENTRY,
            direction=direction,
            engine_entry_price=engine_price,
            entry_order_id=order_id,
            contracts=contracts,
            state_changed_at=now,
        )
        self._positions[strategy_id] = record
        log.debug("Position %s -> PENDING_ENTRY (order=%s)", strategy_id, order_id)
        return record

    def on_signal_entry(
        self, strategy_id: str, engine_price: float, direction: str, contracts: int = 1
    ) -> PositionRecord | None:
        """Record a signal-only entry (no broker interaction).

        Returns None (rejects) if strategy already has an active position.
        """
        existing = self._positions.get(strategy_id)
        if existing is not None and existing.state != PositionState.FLAT:
            if existing.state in (PositionState.PENDING_ENTRY, PositionState.ENTERED):
                log.warning(
                    "Signal entry REJECTED for %s — already in state %s",
                    strategy_id,
                    existing.state.value,
                )
                return None
            log.warning(
                "Signal entry for %s overwriting stale PENDING_EXIT",
                strategy_id,
            )
        now = datetime.now(UTC)
        record = PositionRecord(
            strategy_id=strategy_id,
            state=PositionState.ENTERED,
            direction=direction,
            engine_entry_price=engine_price,
            contracts=contracts,
            entered_at=now,
            state_changed_at=now,
        )
        self._positions[strategy_id] = record
        log.debug("Position %s -> ENTERED (signal-only)", strategy_id)
        return record

    def on_entry_filled(self, strategy_id: str, fill_price: float) -> PositionRecord | None:
        """Record that an entry order has been filled by the broker."""
        record = self._positions.get(strategy_id)
        if record is None:
            log.warning("Entry fill for unknown strategy %s", strategy_id)
            return None
        if record.state == PositionState.ENTERED:
            log.warning(
                "Duplicate entry fill for %s IGNORED (already ENTERED @ %.2f, duplicate=%.2f)",
                strategy_id,
                record.fill_entry_price or 0.0,
                fill_price,
            )
            return record
        now = datetime.now(UTC)
        record.state = PositionState.ENTERED
        record.fill_entry_price = fill_price
        record.entered_at = now
        record.state_changed_at = now
        if record.engine_entry_price is not None:
            record.entry_slippage = fill_price - record.engine_entry_price
        log.debug(
            "Position %s -> ENTERED (fill=%.2f, slip=%s)",
            strategy_id,
            fill_price,
            f"{record.entry_slippage:+.4f}" if record.entry_slippage is not None else "n/a",
        )
        return record

    def on_exit_sent(self, strategy_id: str, exit_order_id: int | None = None) -> PositionRecord | None:
        """Record that an exit order has been submitted."""
        record = self._positions.get(strategy_id)
        if record is None:
            log.warning("Exit sent for unknown strategy %s", strategy_id)
            return None
        record.state = PositionState.PENDING_EXIT
        record.exit_order_id = exit_order_id
        record.state_changed_at = datetime.now(UTC)
        log.debug("Position %s -> PENDING_EXIT (order=%s)", strategy_id, exit_order_id)
        return record

    def on_exit_filled(self, strategy_id: str, fill_price: float | None = None) -> PositionRecord | None:
        """Record that an exit has been filled. Returns the completed record, then removes it."""
        record = self._positions.get(strategy_id)
        if record is None:
            log.warning("Exit fill for unknown strategy %s", strategy_id)
            return None
        if fill_price is not None:
            record.fill_exit_price = fill_price
        record.state = PositionState.FLAT
        record.state_changed_at = datetime.now(UTC)
        # Remove from active tracking — return the final record for P&L computation
        del self._positions[strategy_id]
        log.debug("Position %s -> FLAT (exit_fill=%s)", strategy_id, fill_price)
        return record

    def get(self, strategy_id: str) -> PositionRecord | None:
        """Get the current position record for a strategy."""
        return self._positions.get(strategy_id)

    def best_entry_price(self, strategy_id: str, fallback: float) -> float:
        """Return fill_entry_price if available, else engine_entry_price, else fallback."""
        record = self._positions.get(strategy_id)
        if record is None:
            return fallback
        if record.fill_entry_price is not None:
            return record.fill_entry_price
        if record.engine_entry_price is not None:
            return record.engine_entry_price
        return fallback

    def active_positions(self) -> list[PositionRecord]:
        """Return all non-FLAT positions."""
        return [r for r in self._positions.values() if r.state != PositionState.FLAT]

    def stale_positions(self, timeout_seconds: float = 300.0) -> list[PositionRecord]:
        """Return positions stuck in PENDING states longer than timeout."""
        now = datetime.now(UTC)
        stale = []
        for record in self._positions.values():
            if record.state in (
                PositionState.PENDING_ENTRY,
                PositionState.PENDING_EXIT,
            ):
                elapsed = (now - record.state_changed_at).total_seconds()
                if elapsed > timeout_seconds:
                    stale.append(record)
        return stale

    def pop(self, strategy_id: str) -> PositionRecord | None:
        """Remove and return a position record (for EOD cleanup)."""
        return self._positions.pop(strategy_id, None)

    def clear(self) -> None:
        """Clear all position records (for testing or session reset)."""
        self._positions.clear()
