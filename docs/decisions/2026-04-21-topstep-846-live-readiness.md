# 2026-04-21 — Topstep account `...846` live readiness

## Scope

Determine the minimum honest path to run the bot against the Topstep funded
account ending `846` without silently routing to the wrong broker account or
the wrong Topstep program type.

## Canonical findings

1. `scripts/run_live_session.py` previously defaulted to the first active broker
   account when `--account-id` was omitted. That is unsafe for a funded account
   when multiple Topstep accounts are active.
2. `trading_app/prop_profiles.py` still treats the active automation profile
   `topstep_50k_mnq_auto` as an Express Funded Account / TopstepX shape:
   `is_express_funded=True`, `is_live_funded=False`.
3. TopstepX Daily Loss Limit is optional for Trading Combine / Express Funded
   accounts, but automatic for Live Funded Accounts.
   Canonical sources:
   - `docs/research-input/topstep/topstep_dll_article.md`
   - `docs/research-input/topstep/topstep_live_funded_parameters.md`
4. ProjectX automation is prohibited on Topstep Live Funded Accounts. Canonical
   repo capture:
   - `resources/prop-firm-official-rules.md`
   - `docs/decisions/2026-04-21-b2-routing-config.md`
5. The repo's own Topstep audit still marks LFA DLL / Dynamic Live Risk
   Expansion as deferred, not runtime-ready:
   - `docs/audit/2026-04-08-topstep-canonical-audit.md`

## What changed in code

### Safe account binding

`trading_app/live/session_orchestrator.py` now fails closed when multiple active
broker accounts exist and no explicit binding is provided. It supports:

- exact `--account-id`
- trailing `--account-suffix` such as `846`

This closes the silent "first account wins" behavior for live/demo routing.

### Safer single-account intent

`scripts/run_live_session.py` now treats explicit `--account-id` or
`--account-suffix` as single-account intent when `--copies` is not explicitly
set. This avoids accidentally inheriting `profile.copies=2` from
`topstep_50k_mnq_auto` and starting copy-trading when the operator meant to
trade one funded account.

### Account discovery utility

`scripts/run_live_session.py --list-accounts` now prints active broker account
ids and names after auth so the operator can confirm the exact binding before a
live launch.

### Honest LFA stop condition

If a Topstep profile is marked `is_live_funded=True`, auto-trading now fails
closed at orchestrator startup. Reason: LFA Daily Loss Limit / Dynamic Live
Risk Expansion is still not wired into `AccountHWMTracker`, so an LFA is not an
honest live-ready path yet.

## Operational interpretation for account `...846`

## Verified runtime state

Verified live against the broker on 2026-04-21 from this worktree using
read-only account discovery and preflight only:

- ProjectX returned three active accounts:
  - `50KTC-V2-451890-20372221 #20859313`
  - `50KTC-V2-451890-67605663 #21390438`
  - `EXPRESS-V2-451890-53179846 #21944866`
- Account suffix `846` resolves uniquely to
  `EXPRESS-V2-451890-53179846 #21944866`
- Broker metadata for `#21944866` reported:
  - `canTrade=True`
  - `simulated=True`
  - `isVisible=True`
- Read-only orphan check returned `open_positions=0`
- Exact preflight rerun with account binding:
  - `--profile topstep_50k_mnq_auto`
  - `--broker projectx`
  - `--account-suffix 846`
  - `--copies 1`
  - `--live --preflight`
  - result: `6/6 passed`

Non-blocking runtime caveats from that same verified preflight:

- `daily_features` warned `atr_vel_regime = None`
- the warning is non-blocking for this profile because the six active lanes do
  not depend on `ATR_VEL`
- regime gating paused 2 strategies at current state; that is expected runtime
  gating, not a launch blocker

### If `...846` is an Express Funded / TopstepX account

This is the likely case if you were able to choose "no daily loss limit" in the
platform. In that case:

- `projectx` routing is allowed
- the active profile `topstep_50k_mnq_auto` is the correct account type shape
- the bot still enforces its own internal session-level risk limits and HWM/DD
  protection even if TopstepX personal DLL is unset

### If `...846` is a true Live Funded Account

This path is still blocked for honest auto-trading:

- ProjectX automation is prohibited on LFA
- repo runtime does not yet model LFA DLL / Dynamic Live Risk Expansion
- current code now refuses to auto-trade an LFA profile rather than pretending
  it is safe

## Minimum honest launch path for `...846`

### Step 1 — identify the active broker account exactly

```bash
python scripts/run_live_session.py --profile topstep_50k_mnq_auto --broker projectx --list-accounts
```

Confirm one account id or account name ends with `846`.

### Step 2 — preflight the exact account binding

```bash
python scripts/run_live_session.py --profile topstep_50k_mnq_auto --broker projectx --account-suffix 846 --copies 1 --preflight
```

### Step 3 — only if preflight is clean, live-launch the exact account

```bash
python scripts/run_live_session.py --profile topstep_50k_mnq_auto --broker projectx --account-suffix 846 --copies 1 --live
```

## Verdict

- **Verified current runtime status for `EXPRESS ...846`: READY_FOR_SUPERVISED_LIVE_LAUNCH**
- **Remaining caveat:** preflight still carries a non-blocking `daily_features`
  warning (`atr_vel_regime=None`), so this is operationally ready but not
  "warning-free"
- **LFA ending `846`: BLOCKED_PENDING_LFA_RUNTIME_SUPPORT**

The critical missing piece was not generic live infra. It was exact account
binding and fail-closed account-type hygiene, and that gap is now closed and
verified against the actual broker account.
