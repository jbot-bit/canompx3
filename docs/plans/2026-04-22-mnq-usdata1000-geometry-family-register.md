# MNQ US_DATA_1000 geometry family register

Date: 2026-04-22  
Branch: `wt-codex-mnq-hiroi-scan`

Purpose:

- durable register of the MNQ `US_DATA_1000` prior-day geometry families found
  and their current truth-state
- prevent rediscovering / re-arguing already-closed local findings
- keep the next research step anchored to what is already proven vs merely
  observed

## Canonical surviving family rows

### 1. `PD_DISPLACE_LONG`

- strategy id: `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_DISPLACE_LONG`
- role: `TAKE`
- exact meaning:
  - ORB midpoint below prior-day low
  - OR within `0.15 ATR-20` of prior-day low
- current state:
  - discovery + validator passed
  - promoted to `validated_setups`
- key metrics:
  - `WFE=0.7552`
  - `oos_exp_r=+0.1922`

### 2. `PD_CLEAR_LONG`

- strategy id: `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_CLEAR_LONG`
- role: `TAKE`
- exact meaning:
  - ORB midpoint outside prior-day congestion regime
  - not strictly inside prior-day range
  - not within `0.50 ATR-20` of prior-day pivot
- current state:
  - discovery + validator passed
  - promoted to `validated_setups`
- key metrics:
  - `WFE=1.4639`
  - `oos_exp_r=+0.2926`

### 3. `PD_GO_LONG` on `RR1.0`

- strategy id: `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_GO_LONG`
- role: `TAKE`
- exact meaning:
  - bounded union of `PD_DISPLACE_LONG` and `PD_CLEAR_LONG`
- current state:
  - discovery + validator passed
  - promoted to `validated_setups`
- key metrics:
  - `WFE=1.2609`
  - `oos_exp_r=+0.2176`

### 4. `PD_GO_LONG` on `RR1.5`

- strategy id: `MNQ_US_DATA_1000_E2_RR1.5_CB1_PD_GO_LONG`
- role: `TAKE`
- exact meaning:
  - same bounded union transferred to adjacent `RR1.5` parent lane
- current state:
  - discovery + validator passed
  - promoted to `validated_setups`
- key metrics:
  - `WFE=1.3983`
  - `oos_exp_r=+0.2553`

## Closed / rejected local candidates

### 1. `F5_BELOW_PDL`

- exact cell:
  - `MNQ US_DATA_1000 O5 E2 RR1.0 long F5_BELOW_PDL`
- read-only result looked strong
- failed promotion bridge on valid-window sufficiency
- status: `CLOSED as non-promotable exact-cell`

### 2. `F3_NEAR_PIVOT_50`

- exact cell:
  - `MNQ US_DATA_1000 O5 E2 RR1.0 long F3_NEAR_PIVOT_50`
- read-only result looked strong as an avoid-state
- failed validator on era stability
- status: `CLOSED as non-promotable exact-cell`

## Current lane-level conclusion

For `MNQ US_DATA_1000 O5 E2 long`, the prior-day geometry surface is now
strong enough that more local slicing is low-EV:

- two broader positive families survive individually
- their bounded union survives at both `RR1.0` and `RR1.5`
- the excluded residual state is negative in both IS and OOS

That means this lane should be treated as:

- `LOCALLY SOLVED ENOUGH` for this mechanism family
- not a target for further exact-cell mining

## Do not forget

Before opening another MNQ geometry branch on this lane, re-read:

- `docs/audit/results/2026-04-22-mnq-usdata1000-downside-displacement-take-v1.md`
- `docs/audit/results/2026-04-22-mnq-usdata1000-clear-of-congestion-take-v1.md`
- `docs/audit/results/2026-04-22-mnq-usdata1000-positive-context-union-v1.md`
- `docs/audit/results/2026-04-22-mnq-usdata1000-rr15-positive-context-union-v1.md`

Next honest move is another MNQ parent lane, not more cuts on this one.
