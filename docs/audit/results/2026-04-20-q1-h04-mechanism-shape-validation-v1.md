# Q1 — H04 mechanism-shape validation v1

**Pre-reg:** `docs/audit/hypotheses/2026-04-20-q1-h04-mechanism-shape-validation-v1.yaml` (LOCKED, commit_sha=`c93ba90d`)
**Script:** `research/q1_h04_mechanism_shape_validation_v1.py`
**Lane:** `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` short
**IS window:** trading_day < 2026-01-01
**OOS window:** trading_day >= 2026-01-01

## Verdict: **KILL**

> IS shape does not hold: rho_IS=0.600 surplus_IS=-0.0951

## Parity vs parent H04 result

| metric | parent | this run | tolerance | pass |
|---|---:|---:|---:|:---:|
| H04 cell N   | 128 | 128 | ±3 | YES |
| H04 cell ExpR| +0.3855 | +0.3855 | ±0.015 | YES |
| H04 Δ (on−off)| +0.3579 | +0.3579 | ±0.02 | YES |
| P67 IS threshold (parent frozen 2.012172) | 2.012172 | 2.012172 | — | — |

## Quintile breakpoints (IS-only, frozen)

| edge | rel_vol_COMEX_SETTLE |
|---|---:|
| P20 | 0.894500 |
| P40 | 1.302680 |
| P60 | 1.793560 |
| P80 | 2.558460 |

## Cell matrices

### IS 5×2 matrix (quintile × F6)

| quintile | F6_TRUE N | F6_TRUE ExpR | F6_TRUE WR | F6_FALSE N | F6_FALSE ExpR | F6_FALSE WR |
|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 60 | -0.0332 | 0.433 | 85 | -0.2831 | 0.318 |
| Q2 | 73 | -0.0313 | 0.425 | 72 | +0.0108 | 0.444 |
| Q3 | 76 | -0.0500 | 0.408 | 68 | +0.1157 | 0.485 |
| Q4 | 71 | +0.4287 | 0.620 | 74 | +0.4029 | 0.608 |
| Q5 | 75 | +0.2892 | 0.560 | 70 | +0.1344 | 0.486 |

### 2026 OOS 5×2 matrix (quintile × F6)

| quintile | F6_TRUE N | F6_TRUE ExpR | F6_TRUE WR | F6_FALSE N | F6_FALSE ExpR | F6_FALSE WR |
|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 1 | +1.3508 | 1.000 | 4 | -0.4428 | 0.250 |
| Q2 | 3 | -0.2098 | 0.333 | 7 | +0.3671 | 0.571 |
| Q3 | 3 | +0.5915 | 0.667 | 2 | +0.2092 | 0.500 |
| Q4 | 3 | +0.5639 | 0.667 | 4 | +0.2013 | 0.500 |
| Q5 | 2 | +0.2037 | 0.500 | 5 | -0.5261 | 0.200 |

## Shape metrics

| metric | value | p (10k perm) |
|---|---:|---:|
| Spearman ρ within F6=TRUE, IS  | +0.6000 | 0.3556 |
| Spearman ρ within F6=TRUE, OOS | -0.4000 | 0.5095 |
| Interaction surplus, IS        | -0.0951 | 0.7196 |
| Interaction surplus, OOS       | -1.0638 | 0.5473 |

## OOS power — top cell (Q5 × F6_TRUE)

- IS top-cell ExpR: +0.2892 (N=75)
- OOS top-cell ExpR: +0.2037 (N=2)
- Cohen's d (top_IS vs 0): +0.1923
- OOS power vs IS effect: 5.8%
- Power report: `  OOS power:
    Cohen's d (IS effect): 0.192
    Expected OOS SE:       0.8389
    Expected 95% CI half-width: 1.6442
    Power at alpha=0.05 two-sided: 5.8%
    N per group for 80% power: 426
    RULE 3.3 tier: STATISTICALLY_USELESS`

### IS year-by-year — top-F6-Q5 cell vs rest

| year | N_top | ExpR_top | N_rest | ExpR_rest | Δ |
|---|---:|---:|---:|---:|---:|
| 2019 | 5 | +1.1771 | 34 | -0.5637 | +1.7408 |
| 2020 | 18 | +0.0099 | 86 | +0.2723 | -0.2625 |
| 2021 | 8 | +0.4605 | 108 | +0.0989 | +0.3616 |
| 2022 | 7 | +0.0317 | 113 | -0.1073 | +0.1390 |
| 2023 | 18 | +0.4129 | 103 | +0.0305 | +0.3823 |
| 2024 | 11 | +0.0542 | 101 | +0.0833 | -0.0291 |
| 2025 | 8 | +0.4616 | 106 | +0.2810 | +0.1806 |

### OOS per-month — top-F6-Q5 cell (descriptive, power-qualified)

| month | N_top | ExpR_top | N_rest | ExpR_rest |
|---|---:|---:|---:|---:|
| 2026-01 | 2 | +0.2037 | 9 | -0.2346 |
| 2026-02 | 0 | +nan | 5 | -0.0453 |
| 2026-03 | 0 | +nan | 13 | +0.4726 |
| 2026-04 | 0 | +nan | 5 | -0.0452 |

## Not done by this result

- No capital action, allocator change, or sizing modification.
- Does not bypass the merged H04 shadow prereg's gate_1 (N>=30 on OOS fires of the binary confluence).
- Does not re-fit quintile breakpoints; the IS-only breakpoints are now frozen and citable by any follow-on.
