# Next Opportunity After Sweep Reclaim V2

**Date:** 2026-04-21  
**Status:** DECISION NOTE

## Immediate ranking

### 1. Highest EV for real PnL

Continue exact-lane ORB conditioner work.

Why:

- already closer to deployment truth
- lower implementation drag
- evidence chain already exists elsewhere in the repo
- does not require a new standalone execution engine

### 2. Highest EV for new-strategy discovery

Next standalone family should be:

- `opening-drive pullback continuation`

Why:

- distinct from the dead sweep-reclaim reversal geometry
- distinct from the dead ORB boundary retest execution variant
- still compatible with the external liquidity/displacement ideas after
  translation into objective rules

## Why not another sweep family immediately

`sweep-reclaim-v2` did not fail only at the cell threshold level.

The pooled reads were negative by:

- instrument
- session
- level side

That means another tiny geometric rescue inside the same family is low EV and
high risk of post-hoc optimization.

## Requirements before the next standalone pre-reg

Before locking `opening-drive pullback continuation`, define:

1. exact displacement condition
2. exact pullback trigger
3. exact invalidation point
4. fixed target mode
5. bounded K with no more than one entry mode and one target mode

Do not lock it until those are stated in geometry-only language.
