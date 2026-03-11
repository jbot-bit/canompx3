# London Adjacent Hypothesis Test — Design

**Date:** 2026-03-11
**Script:** `research/research_london_adjacent.py`

## Hypothesis

The ORB at the hour adjacent to LONDON_METALS (17:00 AEST winter / 18:00 AEST summer) has a tradeable edge with E1/E2 entries. This is the DST audit's "wrong time" — the strongest untested signal from the 9-test adversarial audit.

## Why This Wasn't Tested Before

The DST audit dismissed LONDON_METALS adjacent on structural arguments:
- 81% double-break rate at LONDON_METALS correct time
- E3-only / G8+ filter dependency
- "Maps to no known session"

None of these are statistical tests of the adjacent slot itself. The adjacent slot has different microstructure — those problems may not apply.

## What It Tests

- **Adjacent time:** When LONDON_METALS is at 18:00 (winter), test 17:00. When at 17:00 (summer), test 18:00.
- **Grid:** 4 instruments × 3 apertures × 3 entries (E1 CB1, E1 CB3, E2 CB1) × 6 RR × 5 filters = ~1,080 cells
- **Statistical:** BH FDR at q=0.10, dollar P&L (standing rule), seasonal decomposition

## Success Criteria

- ≥1 BH FDR survivor with N≥30
- Double-break rate < 60%
- Edge at G4/G5 (not just G8+)

## Output

- `research/output/london_adjacent_grid.csv` — grid metrics
- `research/output/london_adjacent_trades.csv` — raw trades
- Console: honest summary (SURVIVED / DID NOT SURVIVE / CAVEATS / NEXT STEPS)
