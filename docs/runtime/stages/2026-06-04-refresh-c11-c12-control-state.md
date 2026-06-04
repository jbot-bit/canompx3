---
slug: refresh-c11-c12-control-state
mode: TRIVIAL
status: DONE
created: 2026-06-04
capital_path: false  # writes machine-derived state surfaces only; apply_pauses=False
---

# Refresh stale C11/C12 control-state (DailyRefresh DB-rebuild envelope invalidation)

## Task
Both Criterion 11 (account-survival) and Criterion 12 (SR monitor) lifecycle
state surfaces for `topstep_50k_mnq_auto` read `valid=False reason="db identity
mismatch"`. This is the known DailyRefresh-DB-rebuild artifact (gold.db identity
hash changed; the envelope guard correctly rejected the now-stale state) — a
REGEN, not a debug. See memory
`feedback_daily_refresh_db_rebuild_invalidates_c11_c12_state_envelope_2026_06_01`.

## Live state BEFORE (verified 2026-06-04)
- C11 valid=False gate_ok=False reason="db identity mismatch"
- C12 valid=False reason="db identity mismatch"

## Fix
`python scripts/tools/refresh_control_state.py --profile topstep_50k_mnq_auto`
(idempotent; `_needs_refresh` gate; C12 runs `run_monitor(apply_pauses=False)`
so no live lane is auto-paused — capital-safe).

## Acceptance
- Tool exits 0.
- Re-read lifecycle state: C11 valid=True AND C12 valid=True (no "db identity
  mismatch").
- NOT a capital decision: C11 NO-GO verdict is UNCHANGED by this (regen only
  re-validates the envelope against the current DB; it does not alter the
  survival verdict). gate_ok may remain False — that is the separate C11 NO-GO,
  not staleness.

## scope_lock
- (run-only) scripts/tools/refresh_control_state.py
- writes: C11 account-survival state surface + C12 SR state surface (gold.db /
  state files), no source edits
