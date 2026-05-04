---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Wave 4 Discovery Design — Lit-Grounded Hypothesis Search

**Date:** 2026-04-11
**Branch:** `discovery-wave4-lit-grounded`
**Status:** Hypothesis files committed, discovery runs PENDING

## Motivation

5 validated MNQ strategies, 0 MES, 0 MGC. Portfolio is thin and single-instrument.
Criterion 11 survival gate FAILS (26.2%). Need more strategies across instruments.

## Approach

14 pre-registered hypothesis files, 5 mechanisms, 3 instruments.
Each file = independent BH FDR family. 369 total hypotheses.
All sessions tested per instrument (no cherry-picking).

## Audit Findings Applied

12 findings from adversarial self-review:

| # | Finding | Fix |
|---|---|---|
| F1 | COST_LT08 on MNQ = G15 (95% pass) | MNQ uses COST_LT12 (~G10) |
| F2 | OVNRNG_25 passes 85% MNQ (no-op) | MNQ=OVNRNG_100, MES=OVNRNG_25, MGC=OVNRNG_10 |
| F3 | Session selection bias from data peeking | ALL sessions tested, BH FDR sorts |
| F4 | K=27 has 15-day MinBTL margin | Capped at K<=24 for MNQ/MES |
| F5 | COST_LT10 on MES ~ G5 (redundant) | Uses COST_LT08 (~G7, genuinely different) |
| F6 | FAST composites violate break-speed NO-GO | Removed |
| F7 | PIT_MIN has no data | Dropped |
| F8-F12 | Various overlap/missing issues | Separate families, added DOW/GAP/cross-asset |

## Files

### MNQ (6 files, K=24 each)
- `mnq-cost-gate` — COST_LT12, RR 1.0/1.5
- `mnq-overnight` — OVNRNG_100, RR 1.0/1.5
- `mnq-o15-expansion` — ORB_G5 O15, RR 1.5/2.0
- `mnq-gap` — GAP_R015, RR 1.0/1.5
- `mnq-dow` — G5_NOFRI + G5_NOMON, RR 1.0
- `mnq-cross-asset` — X_MES_ATR60, RR 1.0/1.5

### MES (6 files, K=24 each)
- `mes-cost-gate` — COST_LT08, RR 1.0/1.5
- `mes-overnight` — OVNRNG_25, RR 1.0/1.5
- `mes-vol-persist` — ATR_P70, RR 1.0/1.5
- `mes-size-recal` — ORB_G8, RR 1.0/1.5
- `mes-gap` — GAP_R015, RR 1.0/1.5
- `mes-dow` — G8_NOFRI + G8_NOMON, RR 1.0

### MGC via GC proxy (2 files)
- `mgc-proxy-vol` — ATR_P50 + ATR_P70, K=54, RR 1.0/1.5/2.0
- `mgc-proxy-overnight` — OVNRNG_10, K=27, RR 1.0/1.5/2.0

## Execution Order

1. Commit hypothesis files (this commit)
2. Run discovery per instrument: `strategy_discovery.py --instrument X --hypothesis-file Y --holdout-date 2026-01-01`
3. Run validator per instrument: `strategy_validator.py --instrument X`
4. Post-discovery overlap audit (edge_families rebuild)
5. Portfolio construction from survivors

## Post-Discovery Checks Required

- Filter overlap: if COST_LT and G-filter survivors trade same days, count as 1 bet
- Cross-instrument correlation: MES+MNQ at r=0.83, don't stack
- Era stability: any survivor failing pre-2022 eras = flag
