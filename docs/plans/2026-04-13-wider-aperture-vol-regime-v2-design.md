# Wider Aperture Vol-Regime Discovery v2 -- Design Doc

Date: 2026-04-13
Branch: discovery-wave4-lit-grounded
Status: EXECUTING

## Problem

v1 hypothesis (K=24, ORB_G5, all 12 sessions) produced 0/12 BH FDR survivors.
Root causes: (1) MinBTL miscalculated (5.99yr not 6.65yr), (2) ORB_G5 is vacuous
at wider apertures (96% pass), (3) COST_LT is also vacuous (99-100%), (4) dead
sessions wasted K budget.

## Solution

K=8 hypothesis with aperture-invariant vol-regime filters on data-justified sessions.

### Session selection (3 sessions, 4 aperture-pairs)

| Session | Aperture | Claim | Evidence |
|---------|----------|-------|----------|
| SINGAPORE_OPEN | 15m, 30m | NEW ACCESS (sign flip) | 5m=-0.0003, 15m=+0.054, 30m=+0.086 |
| TOKYO_OPEN | 15m | ADDITIVE | TRADING_RULES validates +0.208R premium |
| US_DATA_1000 | 15m | ADDITIVE | 15m matches 5m (+0.094 vs +0.095) |

### Filter selection (aperture-invariant)

| Filter | Pass rate | Invariant? | Selected? |
|--------|-----------|------------|-----------|
| ORB_G5 | 96% | Yes but vacuous | NO |
| COST_LT12 | 99-100% | NO (becomes vacuous) | NO |
| ATR_P50 | 55% | YES (0 mismatches) | YES |
| ATR_P70 | 38% | YES | YES |
| X_MES_ATR60 | ~40% | YES (runtime) | YES (US sessions) |

### MinBTL

K=8, MinBTL=4.16yr, available=5.99yr, headroom=30.6%.

### Adversarial review findings

1. LONDON_METALS dropped (correlated duplicate -- 5m already positive)
2. Cumulative K documented (v1 K=12 + v2 K=8 = separate families)
3. NO_FILTER control available from unfiltered baseline data (no K cost)

## SHA gate fix

`check_single_use` now accepts `orb_minutes` keyword to scope check.
14/14 tests pass. Committed at 27370c35.

## Hypothesis file

`docs/audit/hypotheses/2026-04-13-mnq-wider-aperture-vol-regime-v2.yaml`
Committed at 469731da.

## Execution

- 15m run: 6 hypotheses (IDs 1,2,5,6,7,8)
- 30m run: 2 hypotheses (IDs 3,4)
- BH FDR at K=8 across full family
