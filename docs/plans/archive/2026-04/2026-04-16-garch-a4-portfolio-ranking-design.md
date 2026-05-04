---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Garch A4 Portfolio-Ranking Design

**Date:** 2026-04-16  
**Status:** ACTIVE DESIGN  
**Purpose:** define the first honest `A4` test for `garch` as a
portfolio-ranking / scarce-risk allocator, with the accounting, fairness, and
anti-bias controls that were still missing from the broader attack plan.

**Authority chain:**
- `CLAUDE.md`
- `RESEARCH_RULES.md`
- `TRADING_RULES.md`
- `docs/institutional/pre_registered_criteria.md`
- `docs/institutional/regime-and-rr-handling-framework.md`
- `docs/institutional/mechanism_priors.md`
- `docs/audit/hypotheses/2026-04-16-garch-mechanism-hypotheses.md`
- `docs/plans/2026-04-16-garch-deployment-map-proof-plan-reset.md`
- `docs/plans/2026-04-16-deployment-map-incremental-edge-proof-plan.md`
- `docs/plans/2026-04-17-garch-deployment-allocator-architecture.md`
- `docs/plans/2026-04-16-garch-institutional-attack-plan.md`

---

## 1. Why A4 exists

`A1` through `A3` were useful, but they still tested the wrong economic shape
for a state variable that may be more valuable as a **relative allocator**
than as a per-trade action map.

What those stages established:

1. `A1` discrete replay can create profile-level gains, but only as
   operational evidence.
2. `A2` continuous sizing failed after real contract translation.
3. `A3` simple confluence helped in places, but did not dominate the best solo
   map.

What they did **not** test:

- whether the state is more useful for deciding **which eligible opportunity
  gets scarce risk budget first**
- whether continuous state value survives better in **ranking / routing**
  than in isolated single-lane size changes

So `A4` exists to test the mechanism family that remains live after `A1-A3`:

**M3 - allocator-not-gate**

---

## 2. Gap audit on the current program

The main attack plan was directionally right, but still left several design
gaps that would have created ambiguity if `A4` were run directly from it.

### Gap G1 - scarce-risk unit was not explicit

The plan said "scarce daily risk budget" but did not freeze the first budget
unit. That would allow accidental drift between:

- slot count
- contract count
- dollar risk

### Gap G2 - collision-day scope was not explicit

If the candidate and baseline act the same on non-collision days, the test must
say so clearly. Otherwise the result can look broader than it is.

### Gap G3 - baseline ordering was not explicit

Ranking tests are invalid if the baseline route is not fixed. The candidate
must be compared to a deterministic base ordering under the same slot budget.

### Gap G4 - tie-break policy was not explicit

A score-ranker must define what happens on ties before first look.

### Gap G5 - leverage creep risk

If `A4` changes both **selection** and **size** at once, it becomes impossible
to tell whether routing or leverage caused the gain.

### Gap G6 - discovery / validated / replay boundary needed a sharper split

`A4` must use:

- canonical truth to justify the score family
- validated shelf to define the candidate book
- profile replay to test deployment translation

Those must stay separate.

### Gap G7 - concentration proof was underspecified

Because ranking can accidentally push the same few lanes repeatedly, `A4` needs
explicit concentration outputs before any doctrine claim.

---

## 3. Design resolution

This document closes those gaps for the first `A4` pass.

### 3.1 First-pass budget unit

The first `A4` pass uses **daily slot budget**, not dollar-risk budget and not
fractional sizing.

Definition:
- slot budget = `profile.max_slots`
- each chosen trade consumes `1` slot
- each chosen trade stays at `1x`

Reason:
- this expresses allocator value without hidden leverage
- it avoids repeating `A2` contract-translation collapse
- it matches the live account/profile surface already used by the replay layer

### 3.2 First-pass action surface

The first `A4` pass is **routing only**:

- no `0x/1x/2x` map
- no upsizing
- no continuous sizing
- no contract-count changes

The only allowed action is:
- choose which eligible lanes consume the limited daily slot budget first

### 3.3 Collision-day scope

The candidate differs from base **only** on days when:

- more eligible lanes exist than the slot budget allows

If `eligible_lanes <= slot_budget`, candidate and base must behave identically.

This makes the economic meaning explicit:
- `A4` is a scarce-resource routing test, not a universal gating test

### 3.4 Baseline ordering

The baseline route must be deterministic and pre-declared.

First-pass baseline:
- current profile lane order as loaded from
  `prop_profiles.py` / `lane_allocation.json`

This is not claimed as optimal. It is simply the fixed operational comparator.

### 3.5 Candidate ordering

Candidate ordering must also be deterministic and fixed before first look.

First-pass candidate:
- rank by locked pre-entry score descending
- tie-break by:
  1. baseline lane order
  2. `strategy_id`

### 3.6 First-pass score family

The first `A4` pass should test **one** mechanism family only.

Recommended first family:
- `M3` simple composite state score

Recommended first score:
- mean of
  - `garch_forecast_vol_pct`
  - `overnight_range_pct`
  - `atr_20_pct`

Reason:
- it is the cleanest literature-shaped composite
- it is already used in `A3`
- it avoids introducing another arbitrary feature at the routing stage

### 3.7 Evidence-tier split

`A4` must respect this split:

- canonical truth:
  broad research established that vol-state information is not obviously noise
- validated utility:
  the map is only allowed to operate on the replayable validated shelf
- deployment translation:
  the profile replay tests whether routing under slot scarcity improves utility

---

## 4. Exact A4 question

The first `A4` question is:

> Under a fixed per-profile daily slot budget, does ranking same-day eligible
> lanes by a locked composite pre-entry state score improve profile utility
> versus the baseline lane-order route, without hidden leverage or
> unacceptable concentration?

This is a **deployment allocator** question.

It is not:
- a discovery-truth question
- a standalone signal-edge question
- a session-doctrine question

---

## 5. First-pass test specification

### 5.1 Universe

Use:
- exact profile lane universe from `prop_profiles.py`
- canonical trade paths from `account_survival.py`
- same stop policy as the profile replay baseline
- same period as prior production-style replays

Do not use:
- `live_config` as truth
- full broad research grid as replay universe
- unreconciled session-attribution tables as evidence

### 5.2 Eligibility

For each trading day:

1. determine which profile lanes have canonical eligible trades that day
2. if eligible count is less than or equal to slot budget:
   - candidate = base
3. if eligible count exceeds slot budget:
   - base takes the first `K` lanes by baseline order
   - candidate takes the first `K` lanes by candidate score order

Where:
- `K = profile.max_slots`

### 5.3 Score construction

Primary candidate:
- `TRIPLE_MEAN_RANK`

Definition:
- `score = mean(garch_forecast_vol_pct, overnight_range_pct, atr_20_pct)`

Constraints:
- no post-look weight tuning
- no nonlinear transformation in first pass
- no score family mixing inside the same hypothesis file

### 5.4 Output surfaces

Required primary outputs:
- per-account total dollars
- copied-account total dollars
- annualized Sharpe on daily dollars
- max drawdown dollars
- worst day dollars
- worst 5-day dollars
- 90-day survival probability
- operational pass probability

Required routing-specific outputs:
- collision-day count
- pct of days affected by rerouting
- total delta from collision days only
- budget utilization rate
- candidate-vs-base lane selection table on affected days
- top-lane contribution share of delta
- top-session contribution share of delta

### 5.5 Fairness rules

The candidate and baseline must keep constant:

1. same lane set
2. same period
3. same stop policy
4. same account profile
5. same copies / copier arithmetic
6. same `1x` per chosen lane
7. same daily slot budget

If any of those change, the comparison is invalid.

---

## 6. Bias controls

### 6.1 No lookahead

Only pre-entry state is allowed:
- `garch_forecast_vol_pct`
- `overnight_range_pct`
- `atr_20_pct`

No post-entry outcomes, no later-day information, no future realized range.

### 6.2 No hidden leverage

The first `A4` pass is routing-only at `1x`.

So if the candidate wins, it wins by better **selection under scarcity**, not
by taking more risk.

### 6.3 No threshold fishing

There are no thresholds in first-pass `A4`.

This is a ranking test, not a new `70/30` threshold sweep.

### 6.4 No baseline drift

The baseline route is frozen before first look and must remain fixed.

### 6.5 No session-story inflation

If session-attribution is still unreconciled, no session doctrine may be
derived from the A4 result. Session contribution can be descriptive only if the
underlying accounting surface reconciles exactly.

### 6.6 Concentration guard

No positive result is promotable if the rerouted gain is dominated by:
- one lane
- one session
- one short window

---

## 7. Success and kill conditions

### Survive

The first `A4` candidate survives only if:

1. headline totals recompute from raw trade paths
2. it beats `BASE_1X` on the profile's primary objective
3. it does not materially worsen non-primary risk metrics
4. the gain is not dominated by one lane or one window
5. it remains stable under neighboring-window checks

### Kill or demote

Kill or demote `A4` if:

1. rerouting barely changes any days
2. gains come from one narrow lane cluster only
3. candidate underperforms once concentration or survival is considered
4. the result depends on accounting ambiguity
5. the candidate only looks good because the baseline order is weak in a way
   unrelated to the state score

---

## 8. What A4 will and will not prove

### If A4 survives

It would support:
- portfolio / routing utility of the state score under profile scarcity

It would **not** by itself support:
- standalone signal-edge proof
- session doctrine
- production promotion without shadow

### If A4 fails

It would mean:
- this remaining routing-shaped use is weaker than expected under current
  profile geometry

It would **not** automatically mean:
- `garch` is worthless in all forms

But it would materially strengthen the case to park the allocator program until
new mechanism evidence appears.

---

## 9. Implementation map

Expected new artifacts:

- hypothesis:
  - `docs/audit/hypotheses/2026-04-16-garch-a4-portfolio-ranking-allocator.yaml`
- replay script:
  - `research/garch_profile_portfolio_ranking_replay.py`
- results:
  - `docs/audit/results/2026-04-16-garch-a4-portfolio-ranking-replay-topstep-50k-mnq-auto.md`
  - `docs/audit/results/2026-04-16-garch-a4-portfolio-ranking-replay-self-funded-tradovate.md`

Expected code reuse:
- lane universe and trade paths from `research/garch_profile_production_replay.py`
- feature cache / score extraction from `research/garch_profile_confluence_replay.py`
- survival and profile rules from `trading_app/account_survival.py`
- profile definition from `trading_app/prop_profiles.py`

---

## 10. Recommendation

This is the correct next step.

It is narrower than another broad scan, more honest than more filter stacking,
and more suited to the remaining live hypothesis than either `A2` or another
thresholded `A3` variant.

The next execution step should therefore be:

1. lock the first `A4` hypothesis file
2. implement the replay script
3. run it on both profiles
4. independently recompute the headline result from raw trade paths
5. then decide whether the allocator program is still alive
