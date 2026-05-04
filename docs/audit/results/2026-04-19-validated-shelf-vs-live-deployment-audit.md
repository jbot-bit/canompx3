# Validated Shelf vs Live Deployment Audit

Date: 2026-04-19

## Scope

Audit the current profit gap between:

- the deployable validated shelf
- the currently active live deployment surface
- inactive profile inventory

This is an implementation / blocker / portfolio-translation audit, not a new
signal-discovery pass.

## Source-of-Truth Chain

Canonical / published read surfaces used:

1. `trading_app/prop_profiles.py`
   - active profiles
   - allowed instruments / sessions
   - effective lane resolution
2. `docs/runtime/lane_allocation.json`
   - current allocator output for the active dynamic MNQ profile
3. `pipeline/db_contracts.py`
   - `deployable_validated_setups` is the canonical published shelf contract
4. `gold.db`
   - `deployable_validated_setups`
   - `validated_setups`
5. `trading_app/lane_allocator.py`
   - current zero-lookahead trailing allocation logic
6. `data/state/sr_state.json`
   - current SR liveness state used by the allocator

Orientation only, not proof:

- prior handoffs
- prior summaries
- prior result notes

## Executive Verdict

The main profit bottleneck is **not lack of validated edge**.

The real bottleneck is **deployment translation quality**:

- only one profile is active today
- the active profile is internally consistent with the allocator
- most inactive profiles are **not activation-ready** because their hardcoded
  lane lists point to missing strategy IDs rather than the current deployable
  shelf

So the highest-EV next move is **not** another broad edge hunt.
It is a **profile inventory rebuild / deployment translation cleanup** against
the current deployable shelf, followed by a focused activation decision.

## What Is Real

### 1. The deployable shelf is materially larger than the active live set

Direct DB facts from `gold.db`:

- `deployable_validated_setups`: `38` rows
- `validated_setups` retired-but-deployable-history rows: `23`

Current active execution profile facts from `trading_app/prop_profiles.py`:

- only `topstep_50k_mnq_auto` is `active=True`
- its effective lane count is `6`

### 2. The active MNQ live set matches the allocator

The effective live lanes for `topstep_50k_mnq_auto` are:

- `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5`
- `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15`
- `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5`
- `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12`
- `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12`
- `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15`

Re-running `build_allocation(...)` for `2026-04-18` reproduced the same set.

This means there is **no stale JSON / stale profile mismatch** inside the
currently active profile. The current live set is aligned with the allocator.

### 3. Many stronger-looking shelf variants are intentionally suppressed by live liveness logic

Two obvious examples:

- `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`
- `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15`

Why they are not live despite stronger shelf metrics:

- both are in fresh `SR` alarm state in `data/state/sr_state.json`
- allocator discounts `ALARM` lanes by `0.50x` annual-R before ranking
- both are also perfectly correlated (`rho = 1.0`) with the live same-session
  substitutes they would replace

So this is **not** a free extra lane. It is a live-control choice between
same-session substitutes, and the current allocator is doing that consistently.

### 4. The inactive profile book is mostly stale

Profile inventory check against `validated_setups`:

- `tradeify_50k`: `5` lanes total, only `1` still deployable, `4` missing
- `topstep_50k`: `1` lane total, `1` missing
- `topstep_50k_mes_auto`: `1` lane total, `1` deployable
- `topstep_50k_type_a`: `8` lanes total, only `1` deployable, `7` missing
- `topstep_100k_type_a`: `8` lanes total, only `1` deployable, `7` missing
- `tradeify_50k_type_b`: `9` lanes total, only `1` deployable, `8` missing
- `tradeify_100k_type_b`: `9` lanes total, only `1` deployable, `8` missing
- `bulenox_50k`: `5` lanes total, only `1` deployable, `4` missing
- `self_funded_tradovate`: `10` lanes total, only `1` deployable, `9` missing

This is the load-bearing implementation finding.

The dormant profile surface looks broad in config, but most of that breadth is
not a real current shelf. It is stale profile inventory.

## What Is Overstated

### Overstated claim: "38 deployable vs 6 live means a huge immediately deployable untapped book"

That is too loose.

Why:

- the active profile is only allowed to trade a specific MNQ session set
- many shelf rows are same-session substitutes, not additive lanes
- correlation gating and SR alarm discount remove some "obvious" swaps
- most inactive profiles contain missing strategies, so they are not
  activation-ready as written

The true dormant distinct deployable set outside the active profile is much
smaller than the raw `38 vs 6` headline suggests.

## Blocker Audit

### Blocker 1

- owner layer: `trading_app/prop_profiles.py`
- blocker: inactive profiles hardcode stale strategy IDs
- evidence: most inactive profile lanes do not exist in current
  `validated_setups`
- still supported?: yes
- cost of keeping it:
  - fake sense of deployment readiness
  - slows activation of additional accounts
  - obscures the true dormant EV surface
- risk of weakening/removing it:
  - low, if replacement is allocator-driven and validated-shelf-driven
- verdict: `REMOVE`

### Blocker 2

- owner layer: `trading_app/lane_allocator.py` + `data/state/sr_state.json`
- blocker: stronger-looking session variants are SR-alarmed or same-session substitutes
- evidence:
  - fresh SR state as of `2026-04-18`
  - `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` = `ALARM`
  - `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` = `ALARM`
  - both have `rho = 1.0` with the chosen same-session substitute
- still supported?: yes
- cost of keeping it:
  - may leave some per-session upside unrealized
- risk of weakening/removing it:
  - high; this would override current live-drift control
- verdict: `KEEP`

### Blocker 3

- owner layer: `docs/runtime/lane_allocation.json` / session regime gate
- blocker: `MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8` is deployable on shelf but paused by cold session regime
- evidence:
  - deployable shelf row exists
  - allocator output marks MES CME_PRECLOSE paused
- still supported?: yes
- cost of keeping it:
  - delays diversification into MES
- risk of weakening/removing it:
  - medium to high; would ignore the current regime gate
- verdict: `KEEP`

## Missed Opportunities

### 1. Profile-inventory rebuild against the current deployable shelf

- type: blocker removal / implementation
- why it matters:
  - this is the shortest path to more real deployable capacity
  - the current inactive profile book is mostly stale
- why it was missed:
  - old profile literals survived while the validated shelf changed
- honest next test:
  - rebuild inactive profiles from the current deployable shelf and current
    allocator outputs, then re-run account-by-account readiness

### 2. Separate activation-readiness from edge-readiness

- type: portfolio / implementation
- why it matters:
  - today’s repo mixes "good signal exists" with "this profile can go live now"
- why it was missed:
  - dormant profiles created a false impression of optionality
- honest next test:
  - for each inactive profile, prove:
    - every lane exists on the deployable shelf
    - current liveness state is acceptable
    - profile constraints still match the lane set

### 3. Narrow watchlist, not broad reopen

- type: research correction / live-control watch
- candidates:
  - `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`
  - `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15`
- why they matter:
  - both still look strong on shelf and trailing metrics
- why they are not immediate promotions:
  - both are fresh SR alarms
  - both are same-session substitutes, not additive lanes
- honest next test:
  - targeted post-result sanity pass on the SR alarm state and forward stream,
    not blind allocator override

## False Positives / False Negatives

### What may have been called good too fast

- the idea that many inactive profiles are near-ready scaling paths
- current config breadth overstates actual deployable readiness

### What may have been killed too fast

- not a full kill, but the raw shelf optics could tempt someone to dismiss the
  active allocator as too conservative
- that would be wrong here; the active profile is aligned with current
  liveness-aware allocation logic

## Bottom Line

The biggest honest profit gap is **stale deployment inventory**, not missing
signal.

The repo already has enough validated edge to support more than the current
single active profile, but the inactive profile surface is mostly not
activation-ready because it points at missing strategies.

The next correct move is to rebuild the dormant profile book from the current
deployable shelf and allocator truth, then make a fresh activation decision.
