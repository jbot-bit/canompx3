---
task: Commit Chordia false-exclusion audit hygiene tail (dead-code sweep + canonical CSV refresh) and close 3 stale stage files whose work has shipped.
mode: TRIVIAL
slug: closeout-chordia-audit-hygiene-2026-05-16
created: 2026-05-16
scope_lock:
  - research/chordia_queue_recompute.py
  - docs/audit/results/2026-05-12-chordia-audit-queue-candidates.csv
  - docs/runtime/stages/chordia-audit-queue-v2-2026-05-12.md
  - docs/runtime/stages/chordia-queue-false-exclusion-audit-2026-05-16.md
  - docs/runtime/stages/chordia-threshold-drift-check.md
---

## Blast Radius
- research/chordia_queue_recompute.py — dead-code removal (`_NEW_GAP_CODES` set never read) + clarifying comments on `_apply_gates` two-path c8_not_passed predicate. Behaviour-preserving.
- docs/audit/results/2026-05-12-chordia-audit-queue-candidates.csv — canonical Mode-A recompute output (last_trade_day refresh 2026-04-23→2026-05-12, n_oos +1 per row). Read-only artifact.
- 3 stage files — delete (work shipped: drift Check 60 registered, audit commit `9c0b1459` FUNNEL_VALIDATED, all audit MDs+CSVs exist on disk).
- Reads: none. Writes: only the scope_lock files.

## Acceptance
1. `python pipeline/check_drift.py` passes.
2. Three stage files deleted.
3. Single commit lands hygiene + CSV refresh.
