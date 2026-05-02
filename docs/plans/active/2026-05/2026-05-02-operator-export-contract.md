---
status: active
owner: codex
last_reviewed: 2026-05-02
superseded_by: ""
---

# Operator Export Contract

## Purpose

Define the canonical export surface from `canompx3`'s live brain to any
external operator/charting platform.

This document exists to prevent ad hoc integration work. The external surface
must consume exported truth from the existing live stack; it must not re-derive
signal logic, ORB state, or risk geometry from broker/platform-specific code.

## Scope

In-scope:

- `trading_app/live/bot_state.py`
- `trading_app/live/session_orchestrator.py`
- export of feed/router/broker health needed by external operator surfaces

Out of scope:

- Quantower/MotiveWave/Sierra-specific implementation
- moving core logic into any external platform scripting layer
- changing live entry/exit semantics in `projectx/order_router.py`

## Canonical Sources

- `ExecutionEngine` owns trade lifecycle, ORB completion, entry/stop/target
  geometry, and entry timestamps.
- `ProjectXDataFeed` owns stale/reconnect semantics.
- `ProjectXOrderRouter` owns live order semantics and price-collar checks.
- `CopyOrderRouter` owns multi-account divergence state.
- `SessionOrchestrator` is the only place that should aggregate those surfaces
  into operator-facing state.

## Design Invariants

1. External platforms consume exported truth. They do not infer ORB, stop,
   target, or signal timestamps from price bars.
2. The live brain remains external and intact. No platform-specific rewrite is
   allowed as a first step.
3. Multi-account divergence must be visible in exported state before any manual
   or semi-manual external operator flow is considered safe.
4. Feed staleness must come from canonical feed liveness logic, not a weaker
   UI heuristic.
5. Every new exported field must be traceable to an existing runtime object or
   a documented new state variable in the orchestrator.

## Minimum Export Contract

### Lane-level truth

- `strategy_id`
- `instrument`
- `session_name`
- `orb_minutes`
- `entry_model`
- `status`
- `status_detail`
- `direction`
- `entry_price`
- `stop_price`
- `target_price`
- `risk_points`
- `signal_time_utc`
- `entry_time_utc`
- `exit_time_utc`
- `current_pnl_r`

### ORB truth

- `orb_high`
- `orb_low`
- `orb_size`
- `orb_complete`
- `orb_break_direction`
- `orb_break_time_utc`
- `orb_complete_time_utc`

### Feed health

- `status` (`idle|connecting|reconnecting|healthy|stale|dead`)
- `last_bar_utc`
- `last_stale_utc`
- `gap_seconds`
- `stale_count`
- `bars_received`
- `reconnect_attempts`
- `last_connected_at`

### Router health

- `degraded`
- `degraded_accounts`
- `supports_native_brackets`
- `has_queryable_bracket_legs`

### Broker/session context

- `broker_name`
- `contract_symbol`
- `signal_only`
- `demo`
- `account_id`
- `account_name`

## Sequencing

### Phase 1 — Contract only

- Extend `bot_state.build_state_snapshot()` to support richer lane/ORB/runtime
  fields.
- Keep all reads best-effort and projection-only.

### Phase 2 — Orchestrator wiring

- Pass ORB map, feed health, router health, and broker context into
  `build_state_snapshot()`.
- Add only the minimal new orchestrator state needed to surface feed health.

### Phase 3 — Tests

- Add focused regression coverage for the richer snapshot contract.
- Do not start platform-specific work until the contract is stable and tested.

## Kill Criteria

- Kill any change that re-encodes signal logic outside the engine/orchestrator.
- Kill any integration that requires external platforms to become the source of
  order truth.
- Kill any operator surface that cannot display canonical stale/degraded state.
- Kill any design that hides copy-router divergence behind broker/platform UI.

## Current Workstream Status

- A low-risk projection extension is in progress in `bot_state.py`.
- The next allowed work item is orchestrator wiring against this contract.
- Platform-specific integration remains blocked until this contract is wired and
  tested.
