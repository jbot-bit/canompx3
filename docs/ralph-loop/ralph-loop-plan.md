## Iteration: 224
## Target: trading_app/conditional_overlays.py:82
## Cluster: 1 finding, types=[canonical_violation], severity=[LOW]
## Classification: [mechanical]
## Blast Radius: 1 file edited, 2 importers (lifecycle_state.py, session_orchestrator.py) unaffected — holdout_frozen_from not read outside conditional_overlays.py
## Invariants: spec.holdout_frozen_from value must remain "2026-01-01" ISO string; no behavior change; field is metadata-only (not a logic gate)
## Diff estimate: 3 lines (import + constant substitution)
## Doctrine cited: integrity-guardian.md § 2 / institutional-rigor.md § 10 — never inline date(2026,1,1)
## Findings deferred: F4 (date.today() in RoleResolver) — ACCEPTABLE, repo-wide pattern
