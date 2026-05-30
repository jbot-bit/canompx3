## Iteration: 221
## Target: trading_app/outcome_builder.py:1098
## Cluster: 1 finding, types=[canonical_violation], severity=[LOW]
## Classification: [mechanical]
## Blast Radius: 1 file (CLI main() only), test_outcome_builder.py (test_help uses --help, unaffected)
## Invariants: build_outcomes() API unchanged; no behavior change for programmatic callers; CLI with explicit --instrument works identically
## Diff estimate: 2 lines (remove default="MGC", add required=True)
## Doctrine cited: integrity-guardian.md § 2 (instrument literals are canonical violations); outcome_builder.py:724-727 self-documents the prior violation
## Findings deferred: TODO(E3-retired) line 964 — ACCEPTABLE pattern 2 (dormant + TODO annotation)
