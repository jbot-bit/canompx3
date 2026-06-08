---
task: "Capital-review Phase 1 — two mechanical hardening fixes from the 2026-06-07 live-paths review: (1) drift check pinning the C11 6-file fingerprint list so a new live-risk file can't silently escape staleness detection; (2) instance-lock empty-orphan branch must prove no live OS-lock holder before unlinking (anti double-instance race)."
mode: IMPLEMENTATION
scope_lock:
  - pipeline/check_drift.py
  - trading_app/live/instance_lock.py
  - tests/test_hooks/test_instance_lock.py
  - tests/test_check_drift_c11_fingerprint_list.py
---

## Blast Radius

- pipeline/check_drift.py — ADD one GENERIC drift check
  `check_code_fingerprint_registries_pinned` covering the CLASS OF FOUR sibling
  fingerprint registries (blast scan 2026-06-07): `_criterion11_code_paths`
  (account_survival.py, CAPITAL), `_lane_code_paths` (opportunity_awareness.py),
  `_sr_code_paths` (sr_monitor.py), `_overlay_code_paths` (conditional_overlays.py).
  Reads (does not mutate) those four functions. Asserts each returns a non-empty
  list, every listed file exists, and the count cannot SHRINK below a recorded
  floor (defined in the check) without a reviewed bump. Registered in the CHECKS
  4-tuple list → count self-reports via `len()`, no hardcoded-count edits.
  Fail-closed: missing file, empty list, or shrink → FAIL (blocking).
- trading_app/live/instance_lock.py — MODIFY the empty-orphan branch (~lines 104–109).
  Currently unlinks an empty lock file unconditionally. New behavior: attempt a
  NON-DESTRUCTIVE OS-lock acquisition first; only unlink if the lock is acquirable
  (proves no live holder). If a live process holds the msvcrt/flock handle, refuse
  and sys.exit(1) like the live-PID branch. Reuses existing acquire path — does NOT
  add a second lock primitive.
- tests/ — RED-first regression tests for both. instance_lock test must exercise the
  REAL acquire path (a live-holder simulation), not a mocked-out unlink.
- Capital impact: instance_lock change is on the live-arm path (double-instance =
  two bots one account = capital risk). drift check is deploy-readiness (guards the
  C11 survival fingerprint). Both fail-closed.
- Reads: none from gold.db. Writes: none to DB. No schema change.
