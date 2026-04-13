# Wider Aperture Vol-Regime Discovery v2 -- Design Doc

Date: 2026-04-13
Branch: discovery-wave4-lit-grounded
Status: COMPLETE

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

### MNQ Results (committed 77c6d785)

4/12 BH FDR survivors (honest K=12). ALL 4 are SINGAPORE_OPEN (sign-flip session):
- 30m ATR_P50: N=856, ExpR=+0.125, p=0.0019, Sh=1.28
- 30m ATR_P70: N=588, ExpR=+0.146, p=0.0028, Sh=1.23
- 15m ATR_P70: N=594, ExpR=+0.136, p=0.0043, Sh=1.18
- 15m ATR_P50: N=863, ExpR=+0.109, p=0.0057, Sh=1.14

---

## MES Targeted Expansion (hypothesis SHA 87f39f607115)

### Design

28 trials across 4 sessions × 4 filter mechanisms × multiple RR targets.
Motivation: decorrelation play — MES on different sessions gives genuine daily PnL
decorrelation with deployed MNQ lanes (r=0.331 at NYSE_OPEN, best of any combo).

Groups:
- A: OVNRNG_50 (MES-calibrated overnight range, 9 trials)
- B: GARCH_VOL_PCT_LT20 (low-vol regime, 9 trials)
- C: COST_LT10 (wider cost gate at NYSE_OPEN, 3 trials)
- D: CME_PRECLOSE RR extensions (G8+COST_LT08 at RR1.5/2.0, 4 trials)
- Additional: COMEX_SETTLE GARCH×2 + OVNRNG_50×1 (3 trials)

MinBTL: K=28, data=6.65yr, MinBTL=0.151yr → PASS (headroom 4303%).

### Results: 0/28 positive BH FDR survivors. DEAD.

4 FDR-significant results all have NEGATIVE ExpR (anti-edge, two-tailed p):

| Strategy | N | WR | ExpR | p | FDR |
|----------|---|---|----|---|-----|
| CME_PRECLOSE GARCH_VOL_PCT_LT20 RR2.0 | 292 | 30.8% | -0.243 | 0.0003 | Y (negative) |
| COMEX_SETTLE GARCH_VOL_PCT_LT20 RR1.0 | 428 | 54.2% | -0.127 | 0.0012 | Y (negative) |
| CME_PRECLOSE GARCH_VOL_PCT_LT20 RR1.5 | 329 | 39.8% | -0.176 | 0.0017 | Y (negative) |
| COMEX_SETTLE GARCH_VOL_PCT_LT20 RR1.5 | 426 | 42.5% | -0.144 | 0.0032 | Y (negative) |

Best positive result: NYSE_OPEN COST_LT10 RR2.0 (N=988, ExpR=+0.087, p=0.044) — fails BH FDR.

### Group verdicts

**GARCH_VOL_PCT_LT20 (9 trials): ANTI-EDGE.** Low-vol regime actively HURTS MES.
8/9 sessions show negative ExpR. The mechanism that works for MNQ (calm regime =
clean breakouts) does NOT transfer to S&P 500. Structural finding: MES breakouts
in calm regimes are MORE likely to be fakeouts, not less.

**OVNRNG_50 (9 trials): DEAD.** Marginal positive (ExpR +0.005 to +0.117) but tiny
samples (65-107 trades) and no significance. Overnight range signal too weak on MES.

**COST_LT10 at NYSE_OPEN (3 trials): DEAD.** Best shot was RR2.0 (p=0.044), fails
BH FDR at K=28. Wider cost gate does not unlock meaningful edge.

**CME_PRECLOSE RR extensions (4 trials): DEAD.** G8 and COST_LT08 do not extend to
RR1.5/2.0. The proven RR1.0 edge does not have directional follow-through.

### Verification

4/4 spot-checked strategies independently verified against raw orb_outcomes × daily_features.
BH FDR computation verified (0 mismatches across all 28 adjusted p-values).
Initial COST_LT10 discrepancy traced to incorrect friction constant in verification
query ($3.74 vs canonical $3.92); corrected, matches exactly.

### Conclusion

MES remains structurally hard for ORB breakout. The 2 existing validated CME_PRECLOSE
strategies (G8 RR1.0, COST_LT08 RR1.0) appear to be the ceiling. The decorrelation
play via new sessions/filters is exhausted for the current filter universe.
