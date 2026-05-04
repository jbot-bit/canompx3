---
task: f6-execution-engine-pd-timestamp-coercion
mode: DESIGN
agent: claude
updated: 2026-05-04
---

# F6 — execution_engine.py raw pd.Timestamp assignment to trade.entry_ts

**Status:** DESIGN (not yet IMPLEMENTATION — needs proper blast-radius + adversarial-audit-gate scoping before code change)
**Date:** 2026-05-04
**Surfaced during:** F3 grounding pass on `_iso_utc` defensive patch.

## What this is

`trading_app/live/bot_state._iso_utc()` was silently returning None for non-datetime inputs (F3 fixed this with logger warnings). Investigating WHY non-datetime values reach `_iso_utc` in production exposed the upstream class-bug:

`trading_app/execution_engine.py` has 3 sites that assign raw `pd.Timestamp` to `trade.entry_ts` without coercion:

- `execution_engine.py:978` — `trade.entry_ts = confirm_bar["ts_utc"]`
- `execution_engine.py:1099` — `entry_ts = bar["ts_utc"]` (later assigned to `trade.entry_ts` at :1198)
- `execution_engine.py:1374` — `trade.entry_ts = bar["ts_utc"]`

Compare to `trading_app/entry_rules.py:288, 360` which DO coerce: `entry_ts = entry_bar["ts_utc"].to_pydatetime()`. So entry_rules paths are fine; execution_engine paths produce raw pd.Timestamp on `trade.entry_ts`.

## Downstream impact

`bot_state.build_state_snapshot._iso_utc(getattr(t, "entry_ts", None))` was silently None'ing `entry_time_utc`, `signal_time_utc`, `exit_time_utc` for every live trade routed through execution_engine.py. F3's logger.warning now makes this visible — the warning will fire on every live trade until F6 lands.

## Why DESIGN not IMPLEMENTATION yet

`trading_app/execution_engine.py` is on the live-trading exposure path. Per `adversarial-audit-gate.md`:
- Touches `trading_app/`
- Modifies trade-state assignment (judgment-classified)
- Severity needs assessment: probably MEDIUM (operator-visibility bug, not capital-exposure bug — the trade still fires; only the timestamp display is wrong) but trace-the-execution-path before classifying.

**Required before IMPLEMENTATION:**
1. Trace each of `execution_engine.py:978/1099/1374` to confirm `bar["ts_utc"]` is always a `pd.Timestamp` and never another type.
2. Decide between two fix shapes:
   - **(a)** Inline `.to_pydatetime()` at each assignment site (3 surgical edits, zero refactor)
   - **(b)** Add a helper `_coerce_to_datetime()` in `entry_rules.py` or a shared utils module, used at both `entry_rules.py:288/360` AND `execution_engine.py:978/1099/1374` (canonical-source rule per integrity-guardian.md sec 2)
3. Audit `paper_trader.py:365/381` (`entry_ts=event.timestamp`) — type depends on Event abstraction; may also need coercion.
4. Verify F3's logger.warning fires in test that simulates the broken path, then DOESN'T fire after F6 lands (proves F6 actually fixed it).

## Out of scope for this stage

- Don't pre-judge fix shape (a) vs (b); decide during proper blast-radius analysis.
- Don't touch this until F1/F2/F3 are stable on origin (they are now: 9ba25af4 pushed 2026-05-04).

## Acceptance — when promoted to IMPLEMENTATION

- All 4-5 assignment sites coerce to datetime at the source
- F3's `_iso_utc` warning does NOT fire in any test or live run after F6 lands
- New test asserts `trade.entry_ts` is `datetime` (not `pd.Timestamp`) post-execution
- Per adversarial-audit-gate.md: dispatch evidence-auditor IF severity classified CRIT/HIGH
- pipeline/check_drift.py passes
