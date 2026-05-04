---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# MGC Proxy Hypothesis Design — Phase 2

**Date:** 2026-04-10
**Branch:** research/gc-proxy-validity
**Authority:** Amendment 3.1 (pre_registered_criteria.md)

## Context

GC proxy data validated (4-gate research, all PASS). GC pipeline artifacts built:
- daily_features: 4,605 rows (2010-2026), all integrity checks passed
- orb_outcomes: 1,295,064 rows (4,502 trading days)
- prev_day_range 100% populated, gap 100%, overnight_range 88%, atr_20 100%

## Data-Driven Session Selection

| Session | 16yr ExpR (E2 unfiltered) | Era stability | Verdict |
|---|---|---|---|
| NYSE_OPEN | +0.087 | Positive ALL 4 eras | **PRIMARY** |
| US_DATA_1000 | +0.037 | Positive ALL 4 eras | **PRIMARY** |
| LONDON_METALS | +0.074 | Era-dependent (+0.18→+0.01→+0.07) | **SECONDARY** |
| EUROPE_FLOW | +0.022 | Negative pre-2020, +0.10 post | **SECONDARY** |
| US_DATA_830 | +0.015 | Weak but mostly positive | **SECONDARY** |
| SINGAPORE_OPEN | +0.084 | Stable but EXCLUDED (74% dbl-break) | EXCLUDED |
| TOKYO_OPEN | +0.008 | Mixed | LOW PRIORITY |
| COMEX_SETTLE | -0.105 | Structural negative | AVOID |
| CME_REOPEN | -0.173 | Structural negative 3/4 eras | AVOID |

## Filter Universe (all price-safe, well-populated on GC)

- NO_FILTER: unfiltered baseline — maximum N, purest test
- PDR_R080: prior day range >= 80th percentile (~20% pass rate, stable across eras)
- OVNRNG_50: overnight range >= 50th percentile (~50% pass, 88% coverage)
- GAP_R005: |gap| >= 0.5% (~0.6-1.8% pass — sparse but genuine if real)
- ORB_G5/G6: absolute ORB size threshold — regime-dependent pass rates
- COST_LT10: cost ratio < 10% — ORB-size-dependent, scales correctly

## Hypothesis Files

### File 1: Stable sessions (Pathway B individual, theory-driven)
~8 hypotheses covering NYSE_OPEN + US_DATA_1000:
- NYSE_OPEN × NO_FILTER × RR1.0 — Crabel commitment, unfiltered
- NYSE_OPEN × NO_FILTER × RR2.0 — Crabel commitment, continuation
- NYSE_OPEN × PDR_R080 × RR1.0 — Crabel + vol persistence
- NYSE_OPEN × PDR_R080 × RR2.0 — Crabel + vol persistence
- US_DATA_1000 × NO_FILTER × RR1.0 — macro info incorporation
- US_DATA_1000 × NO_FILTER × RR2.0 — macro info continuation
- US_DATA_1000 × PDR_R080 × RR1.0 — macro + vol persistence
- US_DATA_1000 × PDR_R080 × RR2.0 — macro + vol persistence

### File 2: Era-dependent sessions (Pathway B individual)
~6 hypotheses covering LONDON_METALS + EUROPE_FLOW:
- LONDON_METALS × NO_FILTER × RR2.0 — London gold pricing
- LONDON_METALS × PDR_R080 × RR2.0 — London gold + vol persistence
- LONDON_METALS × OVNRNG_50 × RR2.0 — London gold + overnight info
- EUROPE_FLOW × NO_FILTER × RR2.0 — cross-border flow
- EUROPE_FLOW × PDR_R080 × RR2.0 — cross-border + vol persistence
- EUROPE_FLOW × OVNRNG_50 × RR2.0 — cross-border + overnight info

### File 3: Broader sweep (Pathway A family)
~10 hypotheses testing additional dimensions:
- NYSE_OPEN × G5 × RR1.0 and RR2.0
- NYSE_OPEN × G6 × RR2.0
- US_DATA_1000 × G5 × RR1.0
- US_DATA_830 × NO_FILTER × RR1.0
- US_DATA_830 × PDR_R080 × RR1.0
- LONDON_METALS × G5 × RR2.0
- NYSE_OPEN × COST_LT10 × RR1.0
- NYSE_OPEN × GAP_R005 × RR2.0
- EUROPE_FLOW × G5 × RR2.0
- TOKYO_OPEN × NO_FILTER × RR1.0

Total: ~24 hypotheses. Bailey MinBTL at N=24 on 16yr: 2*ln(24)/1.0^2 = 6.36yr << 16yr. PASS.

## Execution Order
1. Write 3 hypothesis files, commit
2. Run discovery for GC (sequential, ~5 min each)
3. Run validation with --testing-mode individual (files 1+2) and family (file 3)
4. Report results with era-split
5. Cross-validate survivors against MGC micro overlap (2022-2025)
