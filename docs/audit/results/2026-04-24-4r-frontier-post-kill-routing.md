# 0.4R Frontier Post-Kill Routing Audit

**Date:** 2026-04-24
**Mode:** post-validation routing audit
**Purpose:** decide what to do after the exact highest-ROI `.4R` prereg was
validator-rejected, without forcing adjacent cells through the same known
failure mode.
**Scope:** post-validation routing decision; not a new discovery scan.

## Outputs

[MEASURED] Active `validated_setups` rows with `expectancy_r >= 0.4`: none.

[MEASURED] Non-rejected `experimental_strategies` rows with `sample_size >= 100`
and `expectancy_r >= 0.4`: none.

[MEASURED] The executed exact frontier row is rejected:

| Strategy | N | ExpR | p | Status | Reason |
|---|---:|---:|---:|---|---|
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_F5_BELOW_PDL` | 125 | +0.4033 | 0.000152 | `REJECTED` | `criterion_8: N_oos=8 < 30` |

[MEASURED] The adjacent RR1.0 long row is already rejected as well:

| Strategy | N | ExpR | p | Status | Reason |
|---|---:|---:|---:|---|---|
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_F5_BELOW_PDL` | 126 | +0.3292 | 0.000036 | `REJECTED` | `Phase 4b: Insufficient valid windows: 1 < 3` |

[MEASURED] The currently active prior-day geometry shelf is below `.4R`, but
it is real validated shelf evidence:

| Strategy | N | ExpR | OOS ExpR | WFE |
|---|---:|---:|---:|---:|
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_DISPLACE_LONG` | 192 | +0.2396 | +0.1922 | 0.7552 |
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_CLEAR_LONG` | 211 | +0.2270 | +0.2926 | 1.4639 |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_PD_GO_LONG` | 321 | +0.2222 | +0.2553 | 1.3983 |
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_GO_LONG` | 324 | +0.1934 | +0.2176 | 1.2609 |
| `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_PD_CLEAR_LONG` | 303 | +0.1841 | +0.2212 | 1.9858 |

## Decision

[INFERRED] The proper next move is not another exact `.4R+` `F5_BELOW_PDL`
Pathway-B prereg tonight. The visible `.4R` frontier cells have tiny held-out
samples in the same region as the killed candidate, and the exact long-side
candidate already demonstrates the immediate gate: the repo will not validate
Pathway-B insufficient-OOS rows.

[INFERRED] The highest-ROI path for actual trade improvement is now the
validated prior-day geometry shelf / deployment-readiness branch, not another
standalone `.4R` hunt. That branch is not a `.4R+` claim; it is a practical
validated expectancy branch with positive OOS and WFE already on the shelf.

## Limitations

- Do not implement short-side `F5_BELOW_PDL` just to rerun the NYSE_OPEN short
  cells. That is likely plumbing toward the same OOS-count wall, not a
  higher-EV action.
- Do not relabel the rejected RR1.5 row as a conditional role without a new
  prereg and comparator.
- Do not use `paper_trades` or live routing as validation proof.
- Do not broaden K to rescue the exact cell.

## Next Highest-EV Work Package

Run a deployment-readiness / routing audit for the active prior-day geometry
shelf:

- `PD_DISPLACE_LONG`
- `PD_CLEAR_LONG`
- `PD_GO_LONG`

Questions:

- Which active shelf rows are actually deployable under current profiles and
  lane caps?
- Are any shelf rows validated but not represented in live/deployable routing?
- Does adding the cleanest shelf row improve expected trade set quality without
  violating account/session/instrument concentration controls?
- What blocks live or shadow use: deployment scope, profile gating, account
  capacity, duplicate parent exposure, or monitoring gaps?

This is the next branch with practical EV. It should be labeled as
`deployment_readiness`, not discovery, and must not claim `.4R+`.
