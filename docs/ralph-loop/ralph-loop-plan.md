## Iteration: 105
## Target: research/research_atr_velocity_gate.py + research/research_mgc_regime_shift.py
## Finding: Batch — AV-01 (aperture mixing in Part 0 COUNT query), AV-02 (fetchone() None guard), RS-01 (aperture mixing in Parts 4+5 of regime_shift)
## Classification: [judgment]
## Blast Radius: 2 files, 0 external callers (standalone research scripts)
## Invariants: research output logic unchanged; only filtering to correct single-aperture scope; COUNT behavior identical when orb_minutes=5 rows are the intended scope
## Diff estimate: ~5 lines production code (well within 20-line cap)

### AV-01 (MEDIUM): research_atr_velocity_gate.py line 87-98 Part 0 query
- Missing `AND o.orb_minutes = 5` in WHERE clause
- COUNT(*) mixes 5m+15m+30m apertures, inflating removal rate stats by ~3x
- Fix: add `AND o.orb_minutes = 5` to WHERE clause

### AV-02 (LOW): research_atr_velocity_gate.py line 101
- `total, skipped, contracting = row` — row from fetchone() is Optional[tuple]
- Pyright flags "None is not iterable" — not a runtime bug (COUNT always returns)
- Fix: add `if row is None: continue` guard before destructure

### RS-01 (MEDIUM): research_mgc_regime_shift.py lines 170-178 + 200-210
- Part 4 query: no orb_minutes filter on orb_outcomes — mixes apertures
- Part 5 query: same issue
- Part 3 and Part 6 are CLEAN (join constrains orb_minutes=5 via d.orb_minutes=5)
- Fix: add `AND o.orb_minutes = 5` to each affected WHERE clause
