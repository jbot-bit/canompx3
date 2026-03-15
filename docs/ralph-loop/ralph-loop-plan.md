## Iteration: 93
## Target: scripts/tools/sensitivity_analysis.py:40
## Finding: RR_STEPS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0] duplicates canonical RR_TARGETS from outcome_builder.py — canonical violation
## Classification: [mechanical]
## Blast Radius: 1 file (standalone script, no importers, no callers)
## Invariants: RR sweep ladder stays identical to outcome_builder grid; no logic change; CB_STEPS and G_LADDER unchanged
## Diff estimate: 3 lines (1 import added, 1 assignment replaced, 1 comment updated)
