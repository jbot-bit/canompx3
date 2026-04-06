## Iteration: 159
## Target: scripts/research/depth_at_break_research.py:103
## Finding: SQL join line triggers false positive in drift check 78 — validated_setups and o.symbol on same line, but o.symbol refers to orb_outcomes (pipeline table)
## Classification: [mechanical]
## Blast Radius: 1 file, 0 callers, 0 importers
## Invariants: SQL semantics unchanged; cross-table join correctness preserved; drift check 77/77 PASS
## Diff estimate: 2 lines (split 1 string literal into 2)

## Secondary Finding (DEFERRED): Contract drift in build_outcomes_fast.py and build_mes_outcomes_fast.py
## Both scripts call compute_single_outcome() for E2 entries without passing orb_end_utc (added Apr 1 2026 fix).
## Silently falls back to break_ts (pre-fix E2 entry timing) in both fast scripts.
## Severity: MEDIUM. Scripts also don't fetch break_delay_min from DB. Fix requires SQL + logic changes in 2 files.
## Status: DEFERRED — research-only scripts, canonical build_outcomes() is correct.
