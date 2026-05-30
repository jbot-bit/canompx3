## Iteration: 212
## Target: trading_app/opportunity_awareness.py:38 + trading_app/allocation_promotion.py:18
## Finding: Both files define local copies of PASSING/PASS_CHORDIA_VERDICTS instead of calling canonical chordia.chordia_verdict_allows_deploy()
## Classification: [mechanical]
## Blast Radius: 2 production files, 5 importers (opportunity_awareness), 0 production importers (allocation_promotion), 2 test files
## Invariants: PRIME_SHADOW tier logic unchanged; chordia_verdict_allows_deploy() returns identical truth-table; no behavior change
## Diff estimate: 6-8 lines
## Doctrine cited: institutional-rigor.md § 10 (canonical sources — never re-encode); integrity-guardian.md § 2 (never hardcode canonical lists)
## Invariants: value 180 must NOT change; constant name must NOT change; ConditionStatus.STALE_VALIDATION docstring already references 180 days — annotation must be consistent
## Diff estimate: 3 lines (add comment block)
## Doctrine cited: integrity-guardian.md § 8 (Research Finding Staleness — never inline research stats without @research-source annotation)
