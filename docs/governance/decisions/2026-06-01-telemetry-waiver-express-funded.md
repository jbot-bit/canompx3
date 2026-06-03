# Decision: Telemetry-Maturity Gate Waived for Express-Funded Profiles

- **Date:** 2026-06-01
- **Decider:** operator (joshd)
- **Class:** capital-path readiness gate (Tier B)
- **Status:** ACTIVE — permanent invariant

## Decision

The telemetry-maturity preflight gate (the 30-distinct-trading-day signal-log
floor) is **WAIVED to a clean OK** for any account profile with
`is_express_funded=True` (Topstep XFA, Tradeify, Bulenox — all funded-account
wrappers). It must never block, nor emit WARN noise, on a funded live launch.

Operator wording: *"REMOVE telemetry from Topstep accounts."*

## Rationale

A funded-account wrapper insulates **real personal capital** — the prop firm's
funded/sim account absorbs drawdown, not the operator's own money. The 30-day
telemetry floor exists to stop a *real-capital* account from launching live on
an unvalidated signal stream. That risk does not apply to a funded wrapper, so
the floor is pure friction on the funded path. The operator has decided the
funded live launch should be gated by the substantive checks (broker auth,
portfolio load, daily-features freshness, contract resolution, bracket/fill
probes, journal health) — not by a signal-day count.

## Scope / what it does NOT change

- **Real-capital self-funded profiles keep the gate.** `is_express_funded=False`
  (e.g. `self_funded_tradovate`) still FAILs below the floor. Consistent with
  `.claude/rules/self-funded-sizing-doctrine.md` — real personal capital keeps
  the conservative guardrails.
- **Demo** stays advisory WARN (no capital, but not a deliberate waiver).
- All other preflight gates untouched.

## Implementation

- `scripts/run_live_session.py` § `_check_telemetry_maturity` — Express-Funded
  live branch now returns OK (waived) instead of WARN.
- Keyed on canonical `AccountProfile.is_express_funded`
  (`trading_app/prop_profiles.py`) — no hardcoded profile-id list, so new funded
  profiles inherit the waiver and new real-capital profiles inherit the FAIL.
- `scripts/tools/live_readiness_report.py` already treats express/funded
  telemetry as advisory (`ADVISORY_WARNING_MARKER`, `is_launch_blocking_strict_warning`)
  so gate [11] does not independently re-block — verified 2026-06-01.

## Tests

- `tests/test_scripts/test_run_live_session_telemetry_maturity.py`
  - `test_below_floor_live_xfa_profile_returns_ok_waived` — pins the OK waiver.
  - `test_below_floor_demo_returns_warn` — demo still WARN (no leak).
  - `test_below_floor_live_unknown_profile_returns_failed` — unknown still FAIL.
  - `test_below_floor_live_real_capital_profile_returns_failed` — real-capital
    still FAIL (waiver did NOT leak into the real-capital path).

## Doctrine

- `.claude/rules/telemetry-maturity-waiver.md` — the load-bearing rule.
