---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Position State Machine + Bar Heartbeat — 4TP Design

## Goal
Replace `_entry_prices: dict[str, dict]` with an explicit position state machine.
Add bar heartbeat monitoring. Bring live trading from B+ to A-.

## Architecture
- New module: `trading_app/live/position_tracker.py`
- PositionState enum: FLAT → PENDING_ENTRY → ENTERED → PENDING_EXIT → FLAT
- PositionRecord dataclass: stores all prices, order IDs, timestamps
- PositionTracker class: manages state transitions, timeout detection
- Bar heartbeat: simple watchdog in session_orchestrator._on_bar
- Replaces `_entry_prices` dict and `_best_price` static method

## Data Model

```python
class PositionState(Enum):
    FLAT = "FLAT"
    PENDING_ENTRY = "PENDING_ENTRY"
    ENTERED = "ENTERED"
    PENDING_EXIT = "PENDING_EXIT"

@dataclass
class PositionRecord:
    strategy_id: str
    state: PositionState
    direction: str | None = None
    engine_entry_price: float | None = None
    fill_entry_price: float | None = None
    entry_order_id: int | None = None
    entry_slippage: float | None = None
    exit_order_id: int | None = None
    fill_exit_price: float | None = None
    entered_at: datetime | None = None
    state_changed_at: datetime
```

## Key Methods
- on_entry_sent(strategy_id, direction, engine_price, order_id)
- on_entry_filled(strategy_id, fill_price)
- on_exit_sent(strategy_id, exit_order_id)
- on_exit_filled(strategy_id, fill_price)
- on_signal_entry(strategy_id, engine_price) — signal-only mode
- best_entry_price(strategy_id, fallback) — fill > engine > fallback
- stale_positions(timeout_seconds) — detect stuck PENDING states
- active_positions() — all non-FLAT positions

## Not In Scope
- Full feed degradation (polling fallback) — existing reconnect + post_session handles critical case
- WebSocket fill subscription (Layer 3) — future work
- Wiring circuit breaker into orchestrator — separate task
