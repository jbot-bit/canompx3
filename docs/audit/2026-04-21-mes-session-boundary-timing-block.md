# MES session-boundary v1 — timing-validity block

**Date:** 2026-04-21
**Status:** ABORTED BEFORE DATA CONTACT
**Hypothesis:** `docs/audit/hypotheses/2026-04-21-mes-session-boundary-v1.yaml`

## Why the family is blocked

The locked family `mes-session-boundary-v1` is not safe to execute as written.
Its second predicate, `ASIA_RANGE_ATR_Q67_HIGH`, depends on:

- `session_asia_high`
- `session_asia_low`

The canonical feature-timing table says these are only valid for ORB sessions
starting at or after `17:00` Brisbane:

- `.claude/rules/backtesting-methodology.md:48`
  - `session_asia_high/low | 09:00-17:00 Brisbane | VALID ONLY for ORB starting ≥ 17:00`

The canonical build code matches that timing model:

- `pipeline/build_daily_features.py:427-438`
  - `session_asia_high/low` are computed from the full `asia` session window
    via `_session_utc_window(trading_day, "asia")`
- `pipeline/build_daily_features.py:445-454`
  - the same 09:00-17:00 window is explicitly documented as look-ahead for early
    sessions in the `overnight_*` warning block

The locked family scope is:

- `BRISBANE_1025`
- `TOKYO_OPEN`
- `SINGAPORE_OPEN`

All three start before `17:00` Brisbane, so `ASIA_RANGE_ATR_Q67_HIGH` is a
look-ahead predicate for the entire scoped family.

## Canonical evidence

Current scoped `orb_outcomes` availability for the exact family tuple
(`MES`, `O5`, `E2`, `CB1`, `RR1.0`) is:

- `TOKYO_OPEN`: `1793` rows (`2019-05-06` → `2026-04-16`)
- `SINGAPORE_OPEN`: `1793` rows (`2019-05-06` → `2026-04-16`)
- `BRISBANE_1025`: `0` rows

That row availability does **not** rescue the family. The timing block fires
before any extraction because one of the two locked predicates is invalid for
all intended sessions.

## Decision

Do **not** run Phase 3 extraction for `mes-session-boundary-v1`.

Do **not** silently trim the family down to `GAP_ABS_Q67_HIGH` only.

Do **not** reinterpret `session_asia_high/low` as prior-session values; the
canonical build code does not do that.

The honest next step, if this direction is still wanted later, is a new
pre-registration with timing-valid early-session conviction features only.
