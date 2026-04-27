# Prior-Day Geometry Shelf Deploy-Readiness Refresh

**Date:** 2026-04-24
**Route:** `deployment_readiness`
**Purpose:** follow the practical EV branch after the exact `.4R+`
`F5_BELOW_PDL` frontier was validator-rejected.
**Scope:** prior-day-geometry shelf deployment-readiness; not discovery, not a live-routing change.

This is not discovery and not a live-routing change.

## Outputs

[MEASURED] Active profile lanes resolved from `ACCOUNT_PROFILES` /
`effective_daily_lanes()`:

- `topstep_50k_mnq_auto`
  - `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5`
  - `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15`
  - `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5`
  - `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12`
  - `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12`
  - `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15`

[MEASURED] No active profile currently routes `PD_DISPLACE_LONG`,
`PD_CLEAR_LONG`, or `PD_GO_LONG`.

[MEASURED] The prior-day shelf rows remain active and marked deployable in
`validated_setups`:

| Strategy | Scope | N | ExpR | OOS ExpR | WFE |
|---|---|---:|---:|---:|---:|
| `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_PD_CLEAR_LONG` | deployable | 303 | +0.1841 | +0.2212 | 1.9858 |
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_CLEAR_LONG` | deployable | 211 | +0.2270 | +0.2926 | 1.4639 |
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_DISPLACE_LONG` | deployable | 192 | +0.2396 | +0.1922 | 0.7552 |
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_GO_LONG` | deployable | 324 | +0.1934 | +0.2176 | 1.2609 |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_PD_GO_LONG` | deployable | 321 | +0.2222 | +0.2553 | 1.3983 |

## Existing Blocking Evidence

[MEASURED] The April 23 routing audit already tested free-slot add and
same-session replacement:

- Result doc: `docs/audit/results/2026-04-23-prior-day-geometry-routing-audit.md`
- Outcome: `KEEP_ON_SHELF` for all five rows.
- Reason: additive math was positive, but every candidate collides with an
  existing same-session live lane. Replacement math was negative.

[MEASURED] The April 23 execution-translation audit then tested the only
remaining current-stack path for `US_DATA_1000 O5` coexistence with the live
`US_DATA_1000 O15` lane:

- Result doc: `docs/audit/results/2026-04-23-prior-day-geometry-execution-translation-audit.md`
- `US_DATA_1000` candidates: `ARCHITECTURE_CHANGE_REQUIRED`
- `COMEX_SETTLE PD_CLEAR_LONG`: remains blocked as same-aperture replacement /
  coexistence, with negative replacement math.

Key runtime blocker:

- same-session overlap asks for `0.5x` size-down;
- current MNQ profile rows are 1-contract lanes;
- the live engine now fails closed when the reduction cannot be expressed.

## Verdict

`BLOCKED_PENDING_ARCHITECTURE`

[INFERRED] The validated prior-day shelf is the practical EV branch, but it is
not a direct live-routing branch today. The work needed is not another
discovery run. It is a bounded architecture/policy decision for expressing
same-session coexistence, shadowing it, or explicitly choosing not to use it.

## Limitations / Next Honest Move

Write or execute a narrow same-session translation design only if this branch
is prioritized:

- object: `US_DATA_1000 O5 prior-day geometry` alongside current
  `US_DATA_1000 O15 ORB_G5`
- decision: shadow-only, fractional/lot-expression policy, or no-route
- controls: no full-size accidental coexistence, no same-session replacement
  unless replacement math is positive, profile/account-cap aware,
  monitor-visible

Do not:

- add these rows to `lane_allocation.json` directly;
- treat `deployment_scope='deployable'` as live-book authority by itself;
- reopen prior-day geometry discovery;
- claim `.4R+` from this branch.
