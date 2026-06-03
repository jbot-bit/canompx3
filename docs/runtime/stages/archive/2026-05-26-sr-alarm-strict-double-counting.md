---
task: "Stage 2 (live go-live plan) — resolve SR-alarm strict-report double-counting in live_readiness_report.py; lock the watch-adjudication doctrine without softening the gate"
mode: IMPLEMENTATION
scope_lock:
  - scripts/tools/live_readiness_report.py
  - tests/test_tools/test_live_readiness_report.py
blast_radius: "live_readiness_report.py — read-only aggregation surface for the --live go decision; consumed by run_live_session preflight and the strict-zero-warn gate. No write paths, no broker calls, no schema. Reads gold.db read-only via lifecycle_state. Companion test test_live_readiness_report.py (already covers the watch/unreviewed branches at lines 559-608). Change is to the blocker-list construction in _build_strict_zero_warn_summary only; the green boolean (any-blocker -> not-green) is preserved exactly."
---

## Context

Stage 2 of `~/.claude/plans/get-live-trading-working-safe-secure.md`. Plan framed blockers
1-4 as a "C12 / SR-alarm strict double-counting" doctrine decision.

## Truth state at stage open (2026-05-26)

- Live report run: `Criterion 12 valid=False (db identity mismatch) alarms=0`; all 4 active
  MNQ lanes `sr=None`. The watch-reviewed alarms recorded in memory (2026-05-17) have
  expired/self-healed. **Blockers 1-4 are NOT firing today.** The defect is latent, surfacing
  only once C12 is valid AND a lane is in active ALARM.
- Current 5 strict blockers are C11/C12 envelope identity-mismatch (environmental, Stage 4)
  + telemetry 1/30 (Stage 4) + two live stages not green (Stage 3). None are in Stage 2 scope.

## Doctrine decision (LOCKED — grounded, not changed)

`lifecycle_state.py:237` already encodes the correct lifecycle doctrine: a `watch`-adjudicated
SR alarm sets `blocked=False` (lane MAY trade). The strict-zero-warn gate intentionally asks a
*stricter* question — "is the profile clean enough for a fresh `--live` capital flip?" — and
treats ANY active SR alarm (watch-reviewed or not) as a blocker. Existing test
`test_strict_zero_warn_blocks_any_active_sr_alarm_even_when_watch_reviewed` (line 585) encodes
this deliberately.

Grounding: `docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md` — the SR
monitor is a multi-cyclic quickest-change-point detector calibrated to ARL-to-false-alarm
≈ 60 trading days (~1 per 3 months). An active alarm is a rare, material structural-break
signal. Acknowledging it (watch) permits continued trading under multi-cyclic monitoring, but
does not zero the warning for a NEW capital commitment. **No gate softening** (institutional-rigor).

## The real defect (in scope)

A single SR alarm currently emits 2-3 separate blocker-list entries: the C12 aggregate
(`Criterion 12 alarm count > 0`), the per-lane `lifecycle_blocked` entry, and the per-lane
`SR alarm` entry. The `green` boolean is correct, but the blocker *list/count* is inflated —
one structural-break event reads as multiple distinct problems. Fix: one active alarm
contributes exactly one per-lane blocker entry carrying the watch/unreviewed qualifier;
drop the redundant standalone C12-aggregate entry when per-lane alarm entries already cover it.

## Acceptance

- [ ] One active SR alarm -> exactly one blocker entry naming that lane (qualifier: watch/unreviewed).
- [ ] `green` boolean unchanged for every existing test scenario.
- [ ] Doctrine test (watch-reviewed still blocks) still passes.
- [ ] Full test_live_readiness_report.py passes; drift passes.
