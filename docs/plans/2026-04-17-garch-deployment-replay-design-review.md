# Garch Deployment Replay Design Review

**Date:** 2026-04-17  
**Status:** DESIGN REVIEW ONLY  
**Purpose:** audit the current deployment-replay concept against project authority, identify gaps, and lock the corrected design before more execution.

**Authority chain:**
- `docs/governance/document_authority.md`
- `docs/governance/system_authority_map.md`
- `RESEARCH_RULES.md`
- `TRADING_RULES.md`
- `docs/institutional/pre_registered_criteria.md`
- `docs/institutional/regime-and-rr-handling-framework.md`
- `docs/plans/2026-04-16-garch-institutional-utilization-plan.md`

## 1. Scope

This review covers the proposed `profile-specific deployment replay` stage for:
- `topstep_50k_mnq_auto`
- `self_funded_tradovate`

It does **not** revisit whether `garch` exists as a regime-family effect. That work lives in the earlier family / additive / sizing audits.

This review asks a narrower question:

**How should a deployment-style replay be designed so it is operationally useful without becoming another source of research bias or account-model fiction?**

## 2. Evidence posture

The replay stage is an **operational stress test**, not a clean validation layer.

Reason:
- `RESEARCH_RULES.md` currently treats the active validated shelf as research-provisional until the holdout-clean Mode-A rebuild is restored.
- Therefore any replay on the current live book may inform implementation and pilot design, but it cannot be promoted as fresh institutional validation evidence.

That distinction must be explicit in both the code and the report.

## 3. Findings

### F1. The replay tier was initially under-specified

The original draft moved too quickly from:
- family evidence
- normalized research-weight utility

to:
- active-profile deployment replay

without first writing the rules that separate:
- common-scaffold vs native-scaffold proxy comparisons
- research-weight arithmetic vs discrete live-contract translation
- validated-family utility vs current-live-book operational replay

That sequencing was wrong.

### F2. The first replay draft mixed canonical and non-canonical assumptions

The draft correctly used:
- canonical outcomes
- canonical profile surfaces
- canonical survival logic

But it also relied on assumptions that were not yet design-justified:
- immediate `0/1/2` contract translation without a formal translation rule section
- profile-specific execution using the current live lane set before writing down what makes that replay scientifically fair

### F3. A real stop-policy bug appeared immediately

The first replay draft defaulted to the strategy snapshot stop multiplier rather than the active profile's actual stop policy.

That is exactly the kind of bug a deployment replay must catch before execution.

### F4. A real dependency bug appeared on self-funded replay

The replay path depended on `validated_setups` membership via `_load_lane_trade_paths()`.

That is not a safe assumption for profile replay, because:
- current profiles may contain lanes not present on the current validated shelf
- a replay should key off the **actual lane definition + canonical outcomes/filter semantics**, not shelf membership

This is a design issue, not just a coding issue.

### F5. The current script is still not clean enough to trust

Current issues visible in `research/garch_profile_production_replay.py`:
- dynamic construction of `TradePath` through `__import__` is sloppy and unnecessary
- use of `native.replace(...)` is incorrect in spirit and obscures what object is being built
- report text still hard-codes Topstep language in the reading section even though the tool is being generalized
- no formal reporting of skipped / unreplayable lanes
- no explicit statement of lane coverage loss if some profile lanes cannot be replayed
- no explicit contract that the replay is allowed to run on inactive profiles for scenario analysis
- parameterized script, but hypothesis/report framing still partly tied to one profile

So the tool is **not yet in a professional state**.

## 4. Correct design

### 4.1 Replay objective

The replay stage should answer:

**Given a fixed regime map that already survived upstream family/additivity work, what happens when we translate it into actual live actions on a specific profile?**

Not:
- "discover better regime maps"
- "rescue weak research results"
- "prove the signal is validated"

### 4.2 Required inputs

For each profile replay:

1. Profile definition from `trading_app/prop_profiles.py`
2. Exact lane set from:
   - `profile.daily_lanes`, or
   - `docs/runtime/lane_allocation.json` through the canonical profile path
3. Canonical trade outcomes from `orb_outcomes + daily_features + filter semantics`
4. Profile stop policy / lane overrides
5. Account-rule surface:
   - prop-firm rules where applicable
   - self-funded DD/risk caps where applicable

### 4.3 Required translation contract

This translation must be explicit and pre-registered:

- hostile low state -> `0` contracts
- neutral state -> `1` contract
- favorable high state -> `2` contracts

And then subject to:
- profile stop policy
- account/scaling limits
- copied-account multiplication where applicable

No fractional weights.
No hidden leverage.

### 4.4 Required output contract

Each replay report must include:
- profile identity and whether it is active or scenario-only
- whether the underlying book is research-provisional
- lane coverage:
  - total lanes requested
  - total lanes replayed
  - any skipped lanes and why
- per-account total $
- copied-account total $
- Sharpe
- max DD $
- worst day $
- worst 5-day $
- max open lots
- 90-day survival probability
- operational pass probability

### 4.5 Cross-profile comparison rule

Comparing Topstep and self-funded is valid only if:
- the regime map is held fixed
- the action translation is held fixed
- only the account-constraint surface changes

If the ranking changes, that is not noise. It means:
- part of the edge is implementation/account-shape dependent

That must be reported as such.

## 5. Mandatory pre-execution gates

Before the replay tool is run again, all of the following must be true:

1. `TradePath` construction is explicit and typed, not dynamic/import-hacky
2. Lane loading is based on canonical profile lane surfaces, not `validated_setups` membership
3. Report language is profile-agnostic except where the profile itself is the subject
4. Unreplayable lanes are listed explicitly
5. Coverage loss is surfaced in the report and treated as a caveat
6. The replay is labelled as operational stress-test evidence only
7. Profile-specific hypothesis files exist for every profile run

## 6. Implementation changes required

The next implementation pass should do exactly this:

1. Refactor `research/garch_profile_production_replay.py`
   - import and use `TradePath` directly
   - remove dynamic object construction
   - remove any stale Topstep-only wording from generic paths
   - add lane coverage reporting
   - add skipped-lane reporting

2. Keep lane loading canonical
   - build lane definitions directly from `AccountProfile` + `DailyLaneSpec` / allocation JSON
   - do not depend on `get_profile_lane_definitions()` where it intentionally excludes some profile classes

3. Build outcomes from lane definitions
   - use `_load_strategy_outcomes()` with exact lane parameters
   - apply profile stop policy after loading
   - then build explicit `TradePath` objects

4. Re-run in this order
   - `topstep_50k_mnq_auto`
   - `self_funded_tradovate`

5. Compare only after both runs complete

## 7. Decision discipline

The replay stage is allowed to influence:
- pilot/shadow prioritization
- implementation order
- profile-specific deployment doctrine

It is not allowed to influence:
- discovery truth
- family existence claims
- holdout-clean validation claims

## 8. Immediate next step

Do **not** run further replay executions until the tool is refactored to the above design.

Next action:
- patch the replay tool to the corrected design
- then rerun Topstep and self-funded cleanly
