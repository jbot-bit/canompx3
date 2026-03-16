## Iteration: 109
## Target: research/research_edge_structure.py + research/research_1015_vs_1000.py
## Finding: Pyright errors — utcoffset() None guard (lines 59/66/44), numpy ufunc tuple|float ambiguity (lines 720-721), unused imports/variables
## Classification: [mechanical]
## Blast Radius: 0 callers, 0 importers, no companion tests (standalone research scripts)
## Invariants:
##   1. is_us_dst / is_uk_dst return same bool with identical logic — only type guard added
##   2. orb_r / orb_p comparison with np.isnan still works after float() cast
##   3. No behavioral change — pure type guard / unused-symbol fixes
## Diff estimate: ~12 lines

### Fixes:
# PE-01: research_edge_structure.py:59 — assert utcoffset() not None before .total_seconds()
# PE-02: research_edge_structure.py:66 — same assert for UK DST check
# PE-03: research_edge_structure.py:720-721 — cast float(orb_r) / float(orb_p) to resolve tuple|float ambiguity
# PE-04: research_edge_structure.py:28 — remove unused `csv` import
# PE-05: research_edge_structure.py:400 — rename `all_days` -> `_all_days` (already-prefixed `_opens` is fine)
# PE-06: research_1015_vs_1000.py:44 — assert utcoffset() not None before .total_seconds()
# PE-07: research_1015_vs_1000.py:253,304,392,476 — rename `all_days` -> `_all_days` at all unpack sites
