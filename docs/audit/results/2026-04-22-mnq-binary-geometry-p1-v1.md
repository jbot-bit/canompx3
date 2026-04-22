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

## Canonical bridge outcome

The formal bridge from `experimental_strategies` to `validated_setups` was run
after this read-only pass. Result: **no promotions**.

| hypothesis | validator outcome | blocking gate | deployable? |
|---|---|---|---|
| H1_US_DATA_1000_F5_BELOW_PDL | REJECTED | `Phase 4b: Insufficient valid windows: 1 < 3` | No |
| H2_COMEX_SETTLE_F6_INSIDE_PDR | REJECTED | `criterion_9: era 2020-2022 ExpR=-0.1296 < -0.05 (N=195)` | No |

`validated_setups` contains no rows for either exact strategy ID after the
validator pass. That means there is no honest runtime promotion, shadow
profile, or live-lane bridge to build from this family today.

SURVIVED SCRUTINY:
- canonical layers only
- fixed holdout split
- no threshold tuning
- discovery write into `experimental_strategies`
- exact validator bridge run on the two pending MNQ rows only

DID NOT SURVIVE:
- the two-cell family does not clear a full promotion bar from this pass alone
- H1 does not clear walkforward sufficiency
- H2 does not clear era stability

CAVEATS:
- OOS remains thin
- this is a two-cell family, not a broad geometry claim
- the read-only pass and the validator answer different questions; the latter
  governs promotion

NEXT STEPS:
- keep P1 locked as adjudicated and closed for promotion purposes
- do not widen to clearance bins, MES, or ML as a rescue path
- if reopened later, it must be a new prereg with a new mechanism question, not
  a relabel of this failed promotion attempt
