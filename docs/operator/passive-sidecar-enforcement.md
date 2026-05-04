# Passive Sidecar Enforcement

## Purpose

The passive sidecar is allowed to prepare only one architecture:

- local-device only
- passive only
- non-executing only
- read-only state consumption only

It is not a second trading engine, not a shadow router, and not a place to
hide "disabled" order logic.

## Hard gate

`LIVE_PASSIVE_SIDECAR_ALLOWED` defaults to blocked.

The sidecar must refuse startup before any auth, HTTP, or SignalR work unless
that flag is explicitly set to a truthy value. The error message must state
that written Topstep confirmation is required first.

## Structural non-execution rule

The passive-sidecar package must remain structurally incapable of execution.

Blocked imports and symbols:

- `ProjectXOrderRouter`
- `CopyOrderRouter`
- `BrokerRouter`
- `webhook_server`
- `build_order_spec`
- `build_exit_spec`
- `submit(...)`
- `cancel(...)`
- `cancel_bracket_orders(...)`
- `/api/Order/place`
- `/api/Order/cancel`
- `/api/Order/modify`

## Enforcement mechanism

`scripts/check_passive_sidecar_non_exec.py` scans
`trading_app/live/passive_sidecar/` and fails closed when any forbidden marker
appears.

This check is wired into:

- `.githooks/pre-commit`
- targeted unit tests

## Important boundary

Policy clearance does **not** permit execution logic. It only permits passive,
non-executing tooling if Topstep confirms that explicitly.
