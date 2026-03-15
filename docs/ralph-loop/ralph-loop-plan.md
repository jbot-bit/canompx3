## Iteration: 91
## Target: scripts/tools/build_edge_families.py:218
## Finding: Unused loop variable `orb_min` in first strategies loop — ruff B007 introduced by iter 90 fix
## Classification: [mechanical]
## Blast Radius: 1 file, 0 callers, 0 importers
## Invariants:
##   1. Second loop (line 240) still uses `orb_min` for family_key — MUST NOT touch
##   2. No behavior change — rename `orb_min` -> `_orb_min` only on line 218
## Diff estimate: 1 line changed
