## Iteration: 90
## Target: scripts/tools/build_edge_families.py:217-220
## Finding: Dead variable `orb_minutes_map` — built but never read; orphan code left from refactor
## Classification: [mechanical]
## Blast Radius: 1 file, 0 callers, 0 importers, companion test: tests/test_trading_app/test_edge_families.py
## Invariants:
##   1. family_key computation using `orb_min or 5` at line 243 MUST NOT change
##   2. hash_map and fallback_count logic MUST NOT change
##   3. No behavior change — removing an unused dict only
## Diff estimate: 2 lines removed
