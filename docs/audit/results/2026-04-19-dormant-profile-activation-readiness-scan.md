# Dormant Profile Activation Readiness Scan

Date: 2026-04-19

## Scope

Run the hardened dormant-profile rebuild tool across all inactive profiles and
assess what is actually activation-ready versus merely stale.

This is not a live-activation change. It is a readiness scan against current
allocator truth.

## Source-of-Truth Chain

Canonical surfaces used:

1. `trading_app/prop_profiles.py`
   - inactive profile definitions
   - allowed instruments / sessions
   - effective current lane resolution
2. `gold.db`
   - `deployable_validated_setups`
3. `trading_app/lane_allocator.py`
   - trailing lane scores
   - current liveness enrichment
   - correlation-aware allocation logic
   - session-specific ORB-size stats
4. `data/state/sr_state.json`
   - persisted SR state consumed by allocator liveness logic
5. `scripts/tools/generate_profile_lanes.py`
   - current rebuild support surface

Orientation only:

- prior summaries
- prior handoffs
- informal terminal notes

## Executive Verdict

The dormant profile surface is stale, but the real rebuild opportunity is
smaller and narrower than the raw profile count suggests.

Load-bearing facts:

- `topstep_50k` has no current deployable rebuild at all.
- `topstep_50k_mes_auto` has one valid current lane, but it remains blocked by
  a cold session regime and still has no current rebuild.
- Every inactive profile that *does* rebuild today collapses primarily to the
  same MNQ cluster.
- Most of those rebuilds explicitly displace the same valid incumbent:
  `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`, which remains deployable but is
  `SR=ALARM` and loses to the current allocator-backed substitute.

So the next honest move is **not bulk activation**. It is:

1. rewrite stale inactive profiles from current allocator truth,
2. keep them inactive until account-by-account readiness is explicitly cleared,
3. treat the displaced COMEX incumbent as a watch item, not a surprise.

## Profile Outcomes

### No current rebuild

- `topstep_50k`
  - current state: `0 valid / 1 ghost`
  - rebuild result: `0` lanes
  - blocker: no current deployable `MGC` / `TOKYO_OPEN` candidate
  - verdict: park

- `topstep_50k_mes_auto`
  - current state: `1 valid / 0 ghosts`
  - rebuild result: `0` lanes
  - current valid lane omitted:
    - `MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8`
  - blocker: `PAUSE` via `Session regime COLD (-0.0917)`
  - verdict: keep inactive

### Rebuild available, but not auto-ready

- `tradeify_50k`
  - current state: `1 valid / 4 ghosts`
  - rebuild result: `3` lanes
  - displaced valid incumbent:
    - `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`
    - allocator context: `DEPLOY`, `annual_r=40.1`, `sr=ALARM`
  - recommended sessions:
    - `COMEX_SETTLE`
    - `TOKYO_OPEN`
    - `US_DATA_1000`
  - verdict: rewrite candidate, then explicit activation decision

- `topstep_50k_type_a`
- `topstep_100k_type_a`
  - current state: `1 valid / 7 ghosts`
  - rebuild result: `4` lanes each
  - rebuilt set is entirely `MNQ`, not mixed `MES/MGC/MNQ`
  - displaced valid incumbent:
    - `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`
  - verdict: rewrite candidate, but the original mixed-instrument thesis is no
    longer real on current shelf

- `tradeify_50k_type_b`
- `tradeify_100k_type_b`
  - current state: `1 valid / 8 ghosts`
  - rebuild result: `5` lanes each
  - rebuilt sessions:
    - `EUROPE_FLOW`
    - `SINGAPORE_OPEN`
    - `COMEX_SETTLE`
    - `NYSE_OPEN`
    - `US_DATA_1000`
  - displaced valid incumbent:
    - `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`
  - verdict: strongest rewrite candidates, but still not auto-activate

- `bulenox_50k`
  - current state: `1 valid / 4 ghosts`
  - rebuild result: `4` lanes
  - rebuilt set:
    - `EUROPE_FLOW`
    - `SINGAPORE_OPEN`
    - `COMEX_SETTLE`
    - `TOKYO_OPEN`
  - displaced valid incumbent:
    - `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`
  - verdict: rewrite candidate

- `self_funded_tradovate`
  - current state: `1 valid / 9 ghosts`
  - rebuild result: `6` lanes
  - rebuilt set exactly matches the current active MNQ auto profile lane set
  - displaced valid incumbent:
    - `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`
  - verdict: rewrite candidate, but not a distinct edge source

## What Is Real

- The rebuild tool now makes current truth explicit:
  - ghost cleanup
  - valid-incumbent displacement
  - rebuilt session/instrument footprint
- There are current rebuilds for most inactive profiles.
- Those rebuilds are narrower than the legacy profile configs suggest.

## What Is Overstated

- "We have many dormant profiles ready to switch on."
  - overstated
  - most dormant profiles are stale as configured
  - several rebuilds converge to the same MNQ session cluster

- "The valid COMEX incumbent should just stay because it is still deployable."
  - overstated
  - it is still deployable, but current allocator logic downgrades it via
    `SR=ALARM` and selects a substitute instead

## Operational Interpretation

The high-EV path is now:

1. rewrite stale inactive profiles from current generated `DailyLaneSpec`
   suggestions,
2. preserve inactive status until account-level readiness is reviewed,
3. document that the new dormant book is mostly an MNQ deployment surface,
   not broad cross-instrument optionality,
4. keep the displaced COMEX incumbent under watch rather than silently dropping
   that fact.

## Next Action

Do a **profile rewrite pass only** for the inactive profiles that have current
allocator-backed rebuilds, but do not activate them in the same change.
