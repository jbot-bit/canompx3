# Live Trading Readiness Runbook - Topstep MNQ Auto

Status: NO-GO until live preflight is green with no blocking strict readiness warnings.

## Scope Freeze

- Profile: `topstep_50k_mnq_auto`.
- Instrument: `MNQ` only.
- Allocator book: 3-lane single-copy pilot book from `docs/runtime/lane_allocation.json`.
- Active pilot lanes: `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`, `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08`, `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15`.
- Parked pilot exclusion: `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` is paused for SR `ALARM`; re-entry requires a separate prereg/re-audit decision.
- First live pilot: single protected primary account only. Use `--copies 1` until per-shadow software loss belts exist.
- No strategy research, lane expansion, parameter changes, or launch-mode edits are part of this runbook.

## Startup Gate

Primary operator entrypoint on Windows:

```bat
START_BOT.bat
```

That launcher starts the dashboard/control-room in signal-only mode. Real-money launch is only from the dashboard's hold-to-confirm `HOLD TO GO LIVE` control. The dashboard server pins the effective pilot to `topstep_50k_mnq_auto`, `MNQ`, and `--copies 1`; runs control refresh and live-session preflight before spawning the canonical live runner; and uses the hold gesture as the final operator confirmation.

Underlying gates, for audit/debug:

```bash
git status --short --branch --ahead-behind
python scripts/tools/refresh_control_state.py --profile topstep_50k_mnq_auto
python scripts/tools/live_readiness_report.py --profile topstep_50k_mnq_auto --copies 1 --strict-zero-warn
python scripts/tools/project_pulse.py --fast --format json
python scripts/audits/run_all.py --phase 7
python scripts/audits/run_all.py --quick
python pipeline/check_drift.py
python scripts/run_live_session.py --profile topstep_50k_mnq_auto --instrument MNQ --live --copies 1 --preflight
```

Live remains blocked if any command fails, if `project_pulse` reports a broken/high live-control item or blocked capital recommendation, if `live_readiness_report` has blockers or non-advisory warnings, or if the operator cannot confirm broker account, mode, and account count from the broker UI.

## Dashboard Open vs Live-Launch Allowed (Three-Tier Contract)

"The dashboard opens" and "live launch is allowed" are **distinct states** and must not be conflated. Borrowing the Kubernetes probe model (startup / readiness / liveness are separate concerns), this runbook recognizes three independent tiers:

- **`STARTUP_OK`** — the control-room web server can start and serve status. This is an *observability* guarantee. A dirty development tree, a stale evidence snapshot, or a red live gate must **not** prevent the dashboard from opening in degraded/read-only mode. Failing to open is a startup failure; being unable to trade is not.
- **`SIGNAL_OK`** — read-only / cached state (allocator book, lane status, C11/C12, readiness report) is usable enough for signal monitoring. The operator can see *exactly what is blocked* without debugging git, leases, DB locks, or preflight scripts at the trade window.
- **`LIVE_OK`** — every strict live-launch gate (the Hard Blockers below, plus `live_readiness_report` strict-zero-warn `green`) passes. This is the only tier that permits a real-money `HOLD TO GO LIVE`.

The tiers are **monotonic for safety, not for availability**: `LIVE_OK` implies the lower tiers are fine, but a failure at `LIVE_OK` says nothing about `STARTUP_OK`/`SIGNAL_OK` — the dashboard should still open and still render the blockers. The doctrine is **dashboard fail-open, execution fail-closed**: degraded observability is acceptable and desirable; degraded *execution* is never permitted. A `live_status` of `BLOCKED` must leave the operator *more* informed, not blind.

These tier labels are surfaced as additive, non-binding metadata on the `live_readiness_report` output (`startup_status`, `signal_status`, `live_status`, `launch_impact`); they describe the report's own verdict and do not change any gate. The binding launch decision remains the Hard Blockers and the strict-zero-warn `green` flag below — the tier metadata is an operator-readable projection of that existing truth, never a substitute for it.

## Hard Blockers

- Dirty, ahead, or behind repo state for live mode.
- Active lane count does not match the intended allocator book.
- Any active stale, paused, blocked, or SR `ALARM` lane.
- Telemetry maturity below the profile-scoped floor is an advisory warning for explicit Express/Funded prop profiles (`is_express_funded=True`, including Topstep XFA), not a live-pilot blocker.
- Live stage acceptance not green.
- Journal DB unavailable.
- `project_pulse` live-control or capital recommendation blockers.
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
