## Iteration: 208
## Target: pipeline/system_context.py:882-925
## Finding: evaluate_system_policy read-only/orientation branch duplicates _parallel_claim_issues inline without the mutating-peer escalation logic, producing divergent warning text and missing blocker-vs-warning distinction for mutating peers
## Classification: [mechanical]
## Blast Radius: 10 importers all via session_preflight._evaluate_preflight_policy; no capital-class callers
## Invariants: [1. read-only/orientation still only emits warnings (not blockers) for peer claims; 2. session_start_mutating path unchanged; 3. warning message text uses _parallel_claim_issues canonical text]
## Diff estimate: -21 lines production (deletion) + 3 new lines = ~18 lines net change
## Doctrine cited: integrity-guardian.md § 5 (contract drift — inline re-implementation diverges from extracted helper); institutional-rigor.md § 4 (delegate to canonical sources, never re-encode)
