# Garch Deployment Allocator Architecture

**Date:** 2026-04-17  
**Status:** ACTIVE DESIGN  
**Purpose:** define the correct architecture for turning `garch_forecast_vol_pct`
and related regime variables into profile-aware deployment policy without
claiming a universal edge, overfitting the allocator, or collapsing a
continuous state variable into an ad hoc one-off map.

**Authority chain:**
- `RESEARCH_RULES.md`
- `TRADING_RULES.md`
- `docs/institutional/pre_registered_criteria.md`
- `docs/institutional/regime-and-rr-handling-framework.md`
- `docs/institutional/mechanism_priors.md`
- `docs/plans/2026-04-16-garch-institutional-utilization-plan.md`
- `docs/plans/2026-04-17-garch-deployment-replay-design-review.md`
- `docs/institutional/literature/chan_2008_ch7_regime_switching.md`
- `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md`
- `docs/institutional/literature/harvey_liu_2015_backtesting.md`
- `docs/institutional/literature/lopez_de_prado_2020_ml_for_asset_managers.md`

---

## 1. Problem framing

The current replay work established something useful but incomplete:

- the ranking of regime maps changes by account profile
- this means the problem is partly deployment engineering, not one universal
  best map
- the current discrete `0/1/2` replay is a valid operational stress test
- it is **not** the full economically correct use of a regime variable

`garch_forecast_vol_pct` is a continuous, session-conditional state variable.
Per Chan 2008, the correct use is current-state classification, not a Markov
transition-probability story. Per Carver 2015, the correct economic end-state
is a bounded forecast / sizing / allocation policy, not only binary gates.

So the allocator problem is:

**Given a selected account profile, a lane set, and a pre-entry regime state,
what action should the system take so that take-home $ improves without hidden
leverage, survival degradation, or regime-overfit?**

---

## 2. What this architecture is and is not

### It is

- a policy-layer architecture for deployment
- a way to translate validated regime information into profile-aware actions
- a structure for comparing Topstep, self-funded, and future profile classes
- a way to shadow and promote allocator behavior honestly

### It is not

- proof that `garch` is a universal deployable edge
- permission to weaken BH or holdout discipline
- a black-box classifier program
- a justification for locking the current `70/30` discrete maps as final truth

---

## 3. Design principles

1. **Allocator, not edge-claim**
   - deployment logic may vary by profile without changing the underlying
     research truth

2. **Current observable state only**
   - no transition-forecast or post-entry features
   - no hidden Markov story

3. **Continuous object, bounded actions**
   - regime variables may be continuous even if live actions are discrete

4. **Lane-aware, not just portfolio-aware**
   - some sessions/lane families react differently to the same regime state

5. **Profile-aware, not universal**
   - prop-style DD and self-funded DD are different economic surfaces

6. **Operational realism beats pretty expectancy**
   - prioritize take-home $, survival, drawdown clustering, and slot efficiency

7. **Shadow before promotion**
   - no live action change before replay-to-shadow agreement is checked

---

## 4. Correct object model

The allocator must treat these as separate layers.

### 4.1 Research truth layer

Answers:
- is the regime effect real?
- where is it real?
- what sign and shape does it have?

Valid inputs:
- canonical-layer regime findings
- family-framed BH results
- additive-value results
- monotonicity / tail-shape results

### 4.2 Policy design layer

Answers:
- for a given lane/session family, how should the state be used?
- skip
- neutral size
- upsize
- blended confluence score

Valid inputs:
- research truth layer only
- no profile replay results may alter research truth

### 4.3 Deployment allocator layer

Answers:
- under a specific profile, what bounded action is feasible and attractive?

Valid inputs:
- policy design layer
- profile/account rules
- live lane set
- contract rounding
- copier/scaling rules

### 4.4 Shadow/promotion layer

Answers:
- does live shadow agree with replay enough to promote?

Valid inputs:
- logged hypothetical actions
- live realized PnL/DD/drift

---

## 5. Allocator stack

The full allocator should be built as five layers.

### Layer A — State inputs

Allowed first-class inputs:
- `garch_forecast_vol_pct`
- `overnight_range_pct`
- `atr_20_pct`
- `atr_vel_ratio` or equivalent velocity regime
- lane/session identity
- instrument identity
- profile identity

Rules:
- all must be pre-entry and lookahead-clean
- no post-entry outcomes
- no selection on forward window

### Layer B — Regime representation

Represent the same state in two forms:

1. **continuous representation**
   - normalized percentile or bounded transformed score

2. **discrete state representation**
   - hostile
   - neutral
   - favorable

Reason:
- continuous representation is needed for honest information-content testing
- discrete representation is needed for operational translation and UI clarity

### Layer C — Lane policy

Each lane/session family gets a bounded policy:

- `AVOID_LOW`
- `UPSIZE_HIGH`
- `AVOID_LOW_AND_UPSIZE_HIGH`
- `NEUTRAL`

This policy is determined from upstream research and not from profile replay.

### Layer D — Profile translation

Translate lane policy into profile-feasible actions.

Examples:
- Topstep may prefer bounded survival-preserving upsizing
- self-funded may accept more raw-dollar aggression in some families

Allowed actions:
- `0x`
- `1x`
- `2x`
- bounded continuous size if and only if the continuous allocator audit passes

### Layer E — Portfolio governance

Final action is clipped by:
- DD budget
- max open lots
- profile/account limits
- concentration guard
- copier/scaling rules

This is the layer that turns local lane attractiveness into actual take-home $.

---

## 6. Action ladder

The allocator program should progress in this order:

### A1 — Discrete shadow maps

This is what the current replay does.

Purpose:
- low-risk first operational slice
- easy to audit
- easy to explain

Valid claim:
- operational evidence only

### A2 — Continuous bounded size allocator

Purpose:
- stop wasting information by forcing a continuous state into a coarse map
- test whether Carver-style bounded scaling beats hard gating

Rules:
- risk-normalized against base
- bounded to a small action range
- no hidden leverage

### A3 — Multi-signal confluence allocator

Purpose:
- allow `garch`, `overnight`, and ATR-state to complement rather than compete

Form:
- simple scorecard first
- regularized linear combiner second
- no tree model until simpler layers fail

### A4 — Profile doctrine layer

Purpose:
- define defaults by profile only after A1-A3 are compared on the same
  objective hierarchy

---

## 7. Objective hierarchy

Every allocator candidate is ranked by:

1. take-home $
2. survival / breach probability
3. max DD and clustered-loss behavior
4. Sharpe / profit-to-DD
5. interpretability / stability

This prevents false ranking from:
- better kept-trade expectancy but worse total $
- better raw $ caused by unacceptable breach risk
- more activity in the wrong profile shape

---

## 8. Testing program

The allocator program is not complete unless it covers all four questions.

### 8.1 Information-content question

Test:
- continuous slope
- monotonic buckets
- within-family sign retention
- additive value beyond other vol proxies

Goal:
- prove the variable carries usable state information

### 8.2 Policy question

Test:
- `SKIP_LOW_ONLY`
- `TAKE_HIGH_ONLY`
- `UPSIZE_HIGH_ONLY`
- `AVOID_LOW_AND_UPSIZE_HIGH`

Goal:
- determine the economically correct lane policy

### 8.3 Translation question

Test:
- discrete contract replay
- bounded continuous-size replay
- common-scaffold vs native-scaffold comparison
- row-level vs profile-level accounting reconciliation
- session-attribution reconciliation

Goal:
- determine whether the policy survives real execution constraints

### 8.4 Promotion question

Test:
- replay vs shadow tolerance
- lane drift
- DD / survival drift
- worst-day / worst-5d drift

Goal:
- stop replay overfitting from leaking into production

---

## 9. Initial doctrine from current evidence

This is provisional doctrine, not final doctrine.

1. `garch` belongs in the allocator stack.
2. `garch` is not yet proven as the single best standalone map.
3. `overnight_range_pct` may be the stronger raw-dollar partner or competitor.
4. Current discrete replays are a valid first slice, not the final shape.
5. The economically correct endpoint is likely:
   - lane-aware policy
   - profile-aware translation
   - bounded continuous sizing or simple confluence

---

## 10. Immediate build order

1. repair and verify discrete replay accounting and attribution
2. keep only the reconciled discrete replay results as **A1**
3. design and run **A2 bounded continuous-size replay**
4. design and run **A3 simple confluence allocator**
5. compare A1 vs A2 vs A3 on Topstep and self-funded
6. define profile doctrine only after that comparison
7. shadow the selected doctrine before promotion

---

## 11. What not to do

- do not freeze current profile defaults as final truth yet
- do not let profile replay rewrite research truth
- do not skip continuous-value testing because discrete maps look good
- do not jump to tree-based allocators before scorecard and linear forms
- do not claim live readiness from replay alone
- do not mix “best universal map” and “best profile-specific policy” into one claim

---

## 12. Decision standard

The allocator architecture is ready to move from design to implementation only
if:

1. research truth remains separated from deployment replay
2. discrete replay is treated as first-slice operational evidence only
3. continuous and confluence stages are explicitly planned next
4. promotion is gated by shadow agreement, not replay aesthetics

If any of those fail, the architecture is not yet institutional-grade.
