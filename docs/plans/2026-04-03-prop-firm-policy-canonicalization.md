# Prop Firm Policy Canonicalization

**Date:** 2026-04-03
**Status:** IMPLEMENTED

## Why

`prop_profiles.py` already modeled execution constraints well enough for:
- automation legality
- DD type and account tiers
- contract caps
- banned instruments
- close times

But it was too flat for payout modeling. Topstep alone now has materially different
paths:
- `Express Standard`
- `Express Consistency`
- `Live Funded`

These should not be collapsed into one generic firm rule because they change:
- payout eligibility math
- whether consistency applies at all
- how early-stage bots should optimize for smooth qualifying days vs raw edge
- how we compare firms for manual vs bot deployment

## Canonical System

New canonical payout-policy layer:
- [`trading_app/prop_firm_policies.py`](/mnt/c/Users/joshd/canompx3/trading_app/prop_firm_policies.py)

Execution-layer canonical source remains:
- [`trading_app/prop_profiles.py`](/mnt/c/Users/joshd/canompx3/trading_app/prop_profiles.py)

Rule split:
- `prop_profiles.py` = execution and firm mechanics
- `prop_firm_policies.py` = payout path mechanics

Profiles now carry `payout_policy_id` so an account can be scored against a
specific path instead of a vague firm average.

## Topstep Paths Modeled

### `topstep_express_standard`
- 5 winning days
- winning day threshold `$150`
- payout cap `50%` of balance
- per-request dollar cap `$5,000`
- 5 additional qualifying days after payout

### `topstep_express_consistency`
- 3 trading days minimum
- largest winning day must be `<= 40%` of positive profit in the payout window
- payout cap `50%` of balance
- per-request dollar cap `$6,000`
- 3 additional trading days after payout

### `topstep_live_funded`
- 5 winning days at `$150`
- payout cap `50%` of balance before daily-unlock regime
- daily payouts unlock after `30` live winning days
- after unlock, up to `100%` of balance is requestable

## Current Modeling Implication

When deciding whether to use a firm for a bot, the ranking order should be:

1. Hard legality and execution viability
2. DD / scaling / banned-instrument fit
3. Payout-path fit for the actual lane mix
4. Raw edge after split

That means:
- a lumpy, high-edge lane can still be bad for `Topstep Express Consistency`
- a smoother lane mix can be worth more on Topstep than a slightly higher-ExpR but windfall-heavy mix
- `Tradeify` and `Apex` should not be compared to Topstep using only a flat split multiplier

## Scope of This Change

Implemented:
- canonical payout-policy module
- profile-level `payout_policy_id`
- payout eligibility tracker now uses policy-aware thresholds
- profile-aware helper functions for consistency and payout checks

Not yet implemented:
- payout-adjusted lane scoring
- payout-window state tracking after actual payouts
- policy-aware weekly review grouped by active profile instead of generic firm
