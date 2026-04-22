# MNQ binary geometry P1 — v1

**Scope:** `MNQ / {US_DATA_1000, COMEX_SETTLE} / O5 / E2 / RR1.0 / CB1 / long`
**Truth layers:** `orb_outcomes` + `daily_features` only
**Active plan:** `docs/plans/2026-04-22-p1-mnq-binary-geometry-only-lock.md`

| hypothesis | N_IS | N_on_IS | ExpR_on_IS | ExpR_off_IS | delta_IS | WR_on_IS | WR_off_IS | N_OOS | N_on_OOS | ExpR_on_OOS | ExpR_off_OOS | delta_OOS | t_IS | p_IS | BH_p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H1_US_DATA_1000_F5_BELOW_PDL | 882 | 136 | +0.3258 | -0.0112 | +0.3370 | 69.1% | 52.3% | 35 | 8 | -0.0243 | -0.0617 | +0.0375 | +4.018 | 0.0001 | 0.0002 |
| H2_COMEX_SETTLE_F6_INSIDE_PDR | 876 | 433 | -0.0296 | +0.1651 | -0.1947 | 53.6% | 64.6% | 34 | 19 | -0.1015 | +0.1321 | -0.2336 | -3.238 | 0.0013 | 0.0013 |

## Decision read

- `H1_US_DATA_1000_F5_BELOW_PDL` is the stronger active candidate: large positive IS delta and same-sign but small OOS delta.
- `H2_COMEX_SETTLE_F6_INSIDE_PDR` is the stronger negative avoid state: large negative IS delta and same-sign OOS delta.
- `H1_US_DATA_1000_F5_BELOW_PDL` clears the strict `t>=3.79` bar on this pass.
- `H2_COMEX_SETTLE_F6_INSIDE_PDR` does not clear the strict `t>=3.79` bar on this pass.
- The family as a whole is still a bounded research result, not a promotion event.

SURVIVED SCRUTINY:
- canonical layers only
- fixed holdout split
- no threshold tuning

DID NOT SURVIVE:
- the two-cell family does not clear a full promotion bar from this pass alone

CAVEATS:
- OOS remains thin
- this is a two-cell family, not a broad geometry claim

NEXT STEPS:
- keep P1 locked to these two cells
- do not widen to clearance bins or side tracks until this pair is fully adjudicated
