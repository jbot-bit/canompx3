## Iteration: 213
## Target: trading_app/live/session_orchestrator.py:1132
## Cluster: 1 finding, types=[fail_open], severity=[HIGH]
## Classification: [judgment]
## Blast Radius: 1 caller (session_orchestrator.__init__ L1062), 1 test file (3 existing test methods + 1 new)
## Invariants:
##   1. Session MUST still start even if lifecycle load fails (don't raise — positions need management)
##   2. Operator MUST be notified when lifecycle blocks cannot be loaded (blocked lanes may trade unblocked)
##   3. No behavior change when lifecycle load succeeds
## Diff estimate: ~8 lines production + ~15 lines test = ~23 lines total
## Doctrine cited: integrity-guardian.md § 3 (fail-closed mindset), institutional-rigor.md § 6 (no silent failures)
## Findings deferred: all other handlers are ACCEPTABLE (documented, justified, or non-capital-path)

Finding SO-213-01 [HIGH] — S2 Fail-open: _load_paused_lane_blocks silently swallows all
exceptions, leaving SR-ALARMed / Criterion-11-failed lanes unblocked. Operator sees only
log.warning; no notification dispatched. Fix: log.critical + _notify so operator knows
lifecycle safety guard failed.
