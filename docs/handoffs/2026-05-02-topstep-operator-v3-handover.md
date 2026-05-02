# 2026-05-02 Topstep Operator V3 Handover

## Scope

Fresh-context handover for the isolated worktree:

- Worktree: `/tmp/canompx3-topstep-operator-v3`
- Branch: `codex/topstep-operator-v3`
- Purpose: Topstep/canompx3 external operator architecture spike

This handover is intentionally narrow. It is for starting the operator-architecture
thread cold without dragging in unrelated allocator/chordia work.

## Read First

1. `docs/plans/active/2026-05/2026-05-02-topstep-operator-architecture-v2.md`
2. `trading_app/live/bot_state.py`
3. `trading_app/live/broker_connections.py`
4. `trading_app/live/projectx/data_feed.py`
5. `trading_app/live/projectx/order_router.py`
6. `trading_app/live/copy_order_router.py`

## Verified Starting Point

- `bot_state.py` is still too thin for serious external operator integration.
  Current `lane_cards` do not export `orb_high`, `orb_low`, `stop_price`,
  `target_price`, or `signal_time_utc`.
- `projectx/data_feed.py` already has canonical stale/reconnect semantics.
  Any external surface should consume those signals, not invent weaker ones.
- `projectx/order_router.py` already owns live-entry semantics:
  - `E1`/`E2` only
  - price collar
  - native brackets
  - 429 retry behavior
- `copy_order_router.py` already owns multi-account divergence semantics.
  Any operator surface that hides or bypasses `is_degraded()` is unsafe.

## Correct Next Step

Do not pick a platform first.

The next work item is to define and, if approved, implement the **canonical
operator export contract** from the live brain:

- lane identity
- ORB high/low
- signal time
- entry/stop/target
- feed stale/liveness state
- router degraded state
- connection/account status
- stage-aware warnings if available

Only after that contract exists should any external shell be tested.

## Candidate Order

1. TopstepX-core + read-only/assist sidecar
2. TopstepX + Quantower shell as challenger
3. MotiveWave/Sierra only if Quantower fails and only with a thin overlay path

## Kill Rules

- Kill any path that requires re-encoding ORB/session/filter/risk logic in a
  platform scripting language.
- Kill any path that creates a second truth surface for order/account state.
- Kill any path that does not expose canonical stale/degraded signals to the
  operator.
- Kill any path that depends on a policy route blocked by Topstep in the
  relevant account stage.

## Worktree Notes

- Preflight was run successfully in this worktree after linking the shared venv
  at `.venv-wsl`.
- This worktree uses the shared canonical DB at
  `/mnt/c/users/joshd/canompx3/gold.db`.
- `.venv-wsl` is a local symlink in this worktree and shows as an untracked
  item; treat that as local setup noise, not repo content to commit.
