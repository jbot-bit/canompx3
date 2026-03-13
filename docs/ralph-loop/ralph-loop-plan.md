## Iteration: 36
## Target: pipeline/build_daily_features.py:884,1143
## Finding: BDF1 — Hardcoded ["CME_REOPEN","TOKYO_OPEN","LONDON_METALS"] duplicated at lines 884 and 1143 — canonical violation, no single source of truth for which sessions have compression columns
## Blast Radius: 0 external callers (constant is new; no external import changes), 1 file, 2 substitutions + 1 new constant line
## Invariants:
##   1. The list contents must remain exactly ["CME_REOPEN","TOKYO_OPEN","LONDON_METALS"] — schema columns are tied to these three sessions (init_db.py:257-262)
##   2. Both for-loops (init at line 884, compute at line 1143) must iterate the same sessions
##   3. No changes to logic — only define a constant and reference it
## Diff estimate: 3 lines (1 added constant block, 2 list references replaced)
