## Iteration: 211
## Target: trading_app/eligibility/builder.py:58
## Finding: VALIDATION_FRESHNESS_DAYS = 180 has no @research-source annotation — unannotated research-derived threshold violates integrity-guardian.md § 8
## Classification: [mechanical]
## Blast Radius: 1 production file (builder.py), annotation-only comment change
## Invariants: value 180 must NOT change; constant name must NOT change; ConditionStatus.STALE_VALIDATION docstring already references 180 days — annotation must be consistent
## Diff estimate: 3 lines (add comment block)
## Doctrine cited: integrity-guardian.md § 8 (Research Finding Staleness — never inline research stats without @research-source annotation)
