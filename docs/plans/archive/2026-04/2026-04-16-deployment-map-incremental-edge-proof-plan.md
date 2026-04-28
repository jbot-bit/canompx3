---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Deployment-Map Incremental Edge Proof Plan

**Date:** 2026-04-16  
**Status:** ACTIVE PLAN  
**Purpose:** define the exact proof standard for testing whether a candidate
deployment map adds **incremental profile-specific utility** over `BASE_1X`,
rather than merely replaying as a profitable artifact.

**Authority chain:**
- `CLAUDE.md`
- `RESEARCH_RULES.md`
- `TRADING_RULES.md`
- `REPO_MAP.md`
- `docs/plans/2026-04-16-garch-deployment-map-proof-plan-reset.md`
- `docs/plans/2026-04-17-garch-deployment-allocator-architecture.md`

**Operating principle:** allocator-accounting validation first, edge proof
second.

---

## 1. Claim Formulation

This plan does **not** ask whether a signal is a universal edge. It asks a
narrower question:

**Strict claim form**

> Relative to `BASE_1X`, a candidate deployment map improves pre-declared
> profile utility on unseen data under the same lanes, same execution
> assumptions, same stop policy, and same account constraints.

### 1.1 Prop-style / Topstep claim

> For the `topstep_50k_mnq_auto` profile, candidate map `M` improves
> deployment utility relative to `BASE_1X` by raising take-home dollars and/or
> improving survival-consistent risk utility, without creating unacceptable
> degradation in drawdown, worst-day clustering, or profile-rule compatibility.

### 1.2 Self-funded claim

> For the `self_funded_tradovate` profile, candidate map `M` improves
> deployment utility relative to `BASE_1X` by improving terminal wealth and/or
> risk-adjusted capital growth under the same lane set and stop policy, without
> materially worsening drawdown pain or capital efficiency.

These claims are profile-specific by design. They must not be restated as one
universal map claim.

---

## 2. Null Hypothesis

The null must be practical and decision-facing:

> Relative to `BASE_1X`, candidate map `M` does not improve pre-declared
> profile utility on unseen data once replay accounting, account rules, and
> execution assumptions are held constant.

Equivalent restatement:

- apparent replay gains are within noise / path dependence / implementation error
- or the gains are not stable enough to justify changing deployment doctrine

---

## 3. What Counts As "Edge"

This layer must distinguish three different concepts:

### 3.1 Signal edge

Question:
- does the underlying variable discover a real market-state effect?

Authority:
- discovery truth
- family-framed canonical tests

Not what this plan proves.

### 3.2 Portfolio allocation edge

Question:
- does a map improve which opportunities receive scarce risk budget?

Authority:
- allocator design
- portfolio comparison under matched conditions

May be partially addressed here.

### 3.3 Profile / risk-constraint edge

Question:
- does a map improve utility **for a specific account geometry**?

Authority:
- profile replay
- survival logic
- profile-specific utility metrics

This is the main object of this plan.

So the operative meaning of "edge" here is:

**incremental deployment utility under a specified account profile**

not:
- signal discovery
- generic alpha
- universal map superiority

---

## 4. Pre-Registered Test Design

Each candidate map needs its own hypothesis file, but the structure must stay fixed.

### 4.1 Hypothesis file spec

Required fields:

- `profile_id`
- `baseline_map`
- `candidate_map`
- `lane_source`
- `stop_policy_source`
- `execution_translation`
- `is_window`
- `oos_window`
- `forward_shadow_window`
- `primary_metric`
- `secondary_metrics`
- `fairness_constraints`
- `promotion_gates`
- `kill_criteria`

### 4.2 Test-period structure

Required ladder:

1. **IS / design**
   - design only, no proof claim
2. **historical OOS / holdout**
   - replay comparison on unseen historical window
3. **forward shadow**
   - locked doctrine logging with no live action change
4. **promotion decision**
   - only after shadow agreement

### 4.3 Baseline

- Baseline is always `BASE_1X`
- no changing lanes
- no changing stop policy
- no hidden leverage changes

---

## 5. Metrics By Profile

The profile ranking is allowed to differ because the constraint surface differs.

### 5.1 Topstep / prop-style metrics

Primary:
- `operational_pass_probability`
- `dd_survival_probability`

Secondary:
- max DD
- worst day
- worst 5-day stretch
- payout / consistency compatibility
- trailing-DD safety
- terminal PnL

Reason:
- prop utility is dominated by survival and rule compatibility before raw dollars

### 5.2 Self-funded metrics

Primary:
- terminal wealth / total $
- Sharpe

Secondary:
- max DD
- worst 5-day stretch
- survival
- capital efficiency
- pain metric such as ulcer-like drawdown burden if implemented

Reason:
- self-funded utility tolerates different risk tradeoffs and is not governed by prop constraints

### 5.3 Why ranking may differ

Legitimate causes:
- copier arithmetic
- trailing-DD geometry
- daily-loss ceilings
- scaling-plan limitations
- tolerance for clustered pain

That is not a bug. It is the point of profile-specific doctrine.

---

## 6. Fairness / Apples-to-Apples Rules

No incremental edge claim is valid unless all of the following are held constant:

1. same lane set
2. same historical period
3. same stop policy
4. same profile/account constraints
5. same execution assumptions
6. explicit lane coverage reporting
7. explicit skipped-lane reporting
8. no hidden shelf assumptions
9. no partial-book comparisons unless explicitly labeled partial-book

If any one of these changes, the comparison is not an incremental edge test.

---

## 7. Robustness Requirements

A candidate cannot be called an incremental deployment edge unless it survives:

1. multiple non-overlapping OOS windows
2. no single-lane dominance
3. no single-regime dominance
4. zero unexplained skipped lanes
5. neighboring-window sensitivity checks
6. bounded degradation versus base on non-primary metrics
7. stable sign under small window shifts

This layer is explicitly more about **stability and utility dominance** than classic signal p-values.

---

## 8. Forward Proof Plan

Replay is necessary but insufficient.

Required proof ladder:

1. historical replay
2. locked doctrine note
3. shadow deployment
4. forward comparison to base
5. promotion decision

### 8.1 Shadow requirement

During shadow:
- log candidate-map action
- log base action
- log realized divergence
- monitor lane drift
- monitor DD drift
- monitor worst-day / worst-5-day drift

No live promotion before shadow.

---

## 9. Statistical / Decision Standard

This layer should be treated primarily as **decision-theoretic validation**, not
pure alpha discovery.

### 9.1 Practical standard

A candidate map survives only if:

1. effect size over base is economically material
2. the effect is stable over time
3. profile-specific utility dominance is clear on the primary metric
4. no major failure occurs on the secondary risk metrics

### 9.2 Multiple-testing treatment

Because this layer compares a small number of deployment candidates against one
fixed baseline, treat it as a **small family of deployment decisions**, not a
massive alpha-discovery search.

Implications:
- no global alpha-discovery BH framing is needed at this layer
- but the candidate family must be declared in advance
- no post-hoc expanding candidate set after first look

### 9.3 Decision rule

Prefer:
- stable profile utility dominance

over:
- the single highest lucky replay PnL

---

## 10. Failure Modes

Ways a candidate can look real while being false:

1. lucky window
2. path dependence
3. one-lane concentration
4. regime-specific windfall
5. hidden leverage effect
6. replay/live mismatch
7. profile-rule mismatch
8. implementation leakage
9. shelf contamination
10. using provisional research book as production truth
11. attribution / accounting mismatch
12. skipped-lane concealment

Any one of these can invalidate the claim.

---

## 11. Recommended Decision

### 11.1 What can be said now

- headline replay totals are usable after independent recomputation
- some candidate maps improve deployment utility relative to base in replay
- the value currently looks more like allocator / sizing behavior than binary gating

### 11.2 What cannot be said now

- that any map is proven as a validated deployment edge
- that replay alone proves promotion readiness
- that any current map is universally best
- that unreconciled session-attribution explains the gain

### 11.3 Next proof step

The next proof step is:

1. close or demote attribution gaps
2. lock candidate family
3. compare against `BASE_1X` on matched replay surfaces
4. move the leading candidate into forward shadow

That is the first point at which a serious incremental-edge claim becomes possible.

---

## 12. Implementation Plan

### 12.1 Hypothesis registration

- `docs/audit/hypotheses/`
- `trading_app/hypothesis_loader.py`

### 12.2 Replay comparison

- `research/garch_profile_production_replay.py`
- `research/garch_profile_policy_surface_replay.py`
- `research/garch_profile_continuous_sizing_replay.py`
- `research/garch_profile_confluence_replay.py`

### 12.3 Profile utility scoring

- `trading_app/account_survival.py`
- `trading_app/prop_firm_policies.py`
- `trading_app/consistency_tracker.py`
- `trading_app/portfolio.py`

### 12.4 Forward shadow tracking

- `trading_app/live/trade_journal.py`
- `trading_app/live/performance_monitor.py`
- `trading_app/live/sr_monitor.py`
- `trading_app/live/session_orchestrator.py`

### 12.5 Promotion reporting

- `docs/audit/results/`
- `HANDOFF.md`
- profile-state surfaces in `data/state/` as needed

---

## 13. Final Recommendation

Use the **first prompt** for skeptical reset and proof-boundary control.  
Use the **second prompt** for the actual deployment-edge proof program.

The second prompt is more useful for current understanding and search because it
forces:
- profile-specific claims
- baseline-relative proof
- metric hierarchy by account geometry
- fairness rules
- robustness and forward-proof ladder
- implementation mapping

Current stance:

- replay gives operational evidence
- it does not yet give validated edge proof
- the right next move is not random filter stacking
- the right next move is a locked, profile-specific incremental-edge program
  against `BASE_1X`, followed by shadow

