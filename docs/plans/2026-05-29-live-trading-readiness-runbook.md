# Live Trading Readiness Runbook - Topstep MNQ Auto

Status: NO-GO until strict live readiness is green.

## Scope Freeze

- Profile: `topstep_50k_mnq_auto`.
- Instrument: `MNQ` only.
- Allocator book: 3-lane single-copy pilot book from `docs/runtime/lane_allocation.json`.
- Active pilot lanes: `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`, `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08`, `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15`.
- Parked pilot exclusion: `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` is paused for SR `ALARM`; re-entry requires a separate prereg/re-audit decision.
- First live pilot: single protected primary account only. Use `--copies 1` until per-shadow software loss belts exist.
- No strategy research, lane expansion, parameter changes, or launch-mode edits are part of this runbook.

## Startup Gate

Run these before any live start:

```bash
python scripts/tools/live_readiness_report.py --profile topstep_50k_mnq_auto --copies 1 --strict-zero-warn
python scripts/tools/project_pulse.py --fast --format json
python scripts/audits/run_all.py --phase 7
python scripts/audits/run_all.py --quick
python pipeline/check_drift.py
python scripts/run_live_session.py --profile topstep_50k_mnq_auto --instrument MNQ --live --copies 1 --preflight
```

Live remains blocked if any command fails, or if the operator cannot confirm broker account, mode, and account count from the broker UI.

## Hard Blockers

- Dirty or behind repo state for live mode.
- Active lane count does not match the intended allocator book.
- Any active stale, paused, blocked, or SR `ALARM` lane.
- Telemetry maturity below the profile-scoped floor for real-capital/self-funded live.
- For explicit Express/Funded prop profiles (`is_express_funded=True`, including Topstep XFA), below-floor telemetry is an advisory warning, not a launch blocker.
- Live stage acceptance not green.
- Journal DB unavailable.
- Broker auth/session check unavailable.
- `copies>1` without per-shadow software daily-loss/HWM protection.
- Router degraded from primary/shadow divergence.

## Degraded Router Definition

The router is degraded when any primary/shadow state can differ:

- shadow submit failure after primary accepted/submitted
- shadow cancel failure
- unresolved partial-fill mismatch
- reconnect without reconciliation
- missing journal write after an accepted order

In degraded state, no further entry may be submitted until manual reconciliation clears account positions, open orders, journal state, and router state.

## Kill And Restart

- Kill switch first: stop new entries, flatten/close broker exposure, then verify broker truth.
- Restart only after `live_journal.db` has no unresolved incomplete trade for the target trading day.
- Ring recovery remains dry-run-only unless the operator explicitly approves a real recovery mutation.

## Daily Postmortem

Record after every demo/live session:

- profile, mode, account id, copies, and git head
- active lanes and skipped/blocked lanes
- expected R vs actual R
- slippage and fill quality
- exit reason
- CUSUM/SR alarms
- router degraded events
- stale/paused/alarmed lane leakage
- journal persistence check after restart

## Scale Rule

Scale only after at least five unchanged trading days with:

- no gate bypass
- no unresolved divergence
- execution quality within tolerance
- live behavior matching allocator state
- no journal/ring/restart incident

Any failure keeps size unchanged or rolls back to signal-only/demo.
