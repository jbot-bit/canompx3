---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Garch Institutional Utilization Plan

**Date:** 2026-04-16
**Status:** ACTIVE PLAN
**Purpose:** turn `garch_forecast_vol_pct` from an audited regime-family observation into an institutionally valid profit-improvement program without bias, look-ahead, or ad hoc deployment.

**Authority chain:**
- `RESEARCH_RULES.md`
- `docs/institutional/pre_registered_criteria.md`
- `docs/institutional/regime-and-rr-handling-framework.md`
- `docs/institutional/mechanism_priors.md`
- `docs/institutional/literature/chan_2008_ch7_regime_switching.md`
- `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md`
- `docs/institutional/literature/harvey_liu_2015_backtesting.md`
- `docs/institutional/literature/lopez_de_prado_2020_ml_for_asset_managers.md`

---

## 1. Executive view

### 1.1 What has already been established

From the 2026-04-16 audit series:

- `garch_forecast_vol_pct` is **look-ahead clean**.
- Global BH rejects the claim that `garch` is a **universal production overlay**.
- Family-framed fixed-tail audit supports a **real regime-family effect**:
  - high-side positive cells: `304/431`
  - low-side negative cells: `290/431`
  - shuffle-null p-values: `0.0099` / `0.0099`
  - directional BH survivors: `14`
  - tail-bias BH survivors: `6`
- Structural decomposition shows `garch` is primarily a **volatility-regime variable**, strongly related to `atr_20_pct`, moderately related to `overnight_range_pct`, and weakly related to gap/calendar flags.
- Utilization audit shows:
  - `TAKE_HIGH_ONLY` often improves **trade quality** but reduces **total R**
  - `SKIP_LOW_ONLY` is sometimes better but not universally
  - the most likely correct exploitation path is **R3/R7**, not broad `R1`

### 1.2 Core implication

`garch` should be treated as a **market-state input**, not as a magic standalone filter.

The right institutional objective is:

1. verify where the regime effect is real,
2. verify whether it is additive beyond existing vol proxies,
3. determine the economically correct use:
   - `R1` binary skip/take
   - `R3` size tilt
   - `R7` confluence / classifier
4. deploy only after holdout-clean forward-shadow evidence

### 1.3 Why this object needs a different audit rubric

`garch_forecast_vol_pct` is not the same research object as a normal binary setup filter.

It is a:
- continuous percentile-ranked state variable
- mostly session-conditional regime signal
- tail-sensitive variable where economics may show up in sizing, avoidance, or classifier states rather than pure trade/no-trade gating

That means the plan must explicitly test four distinct questions:

1. **Existence:** is there a real regime-family effect?
2. **Distinctness:** is that effect additive beyond existing vol proxies?
3. **Utilization:** is the right use gate, size, or classifier?
4. **Portfolio value:** does any use materially improve take-home $ after realistic risk constraints?

If we judge `garch` only as a binary filter, we will likely under-measure its valid uses.

### 1.4 Evidence tiers — must not be blended

This program now explicitly separates three evidence tiers. They answer
different questions and must never be collapsed into one headline.

1. **Discovery truth**
   - Canonical-layer, family-framed, mechanism-aware work
   - Answers: "Is there a real regime effect at all?" and "where?"
   - Valid surfaces: `orb_outcomes`, `daily_features`, fixed family tests
   - Invalid use: claiming deployment readiness

2. **Validated-family utility**
   - Exact validated strategy populations only
   - Answers: "Does this variable improve the actual validated shelf?"
   - Valid use: deciding whether the signal deserves a role in the deployment program
   - Invalid use: claiming live profitability under account rules
   - Current caveat: until the Mode-A clean rerun is restored, the existing
     validated shelf is research-provisional per `RESEARCH_RULES.md`

3. **Profile-specific deployment replay**
   - Active profile lane set + active firm/account rules + discrete action translation
   - Answers: "Does this survive contract rounding, scaling rules, DD rules, and take-home-$ accounting?"
   - Valid use: pilot / shadow decision support
   - Invalid use: discovering new regime families or rescuing weak research claims
   - Current caveat: replay on the current live book is an operational stress
     test, not a clean validation layer, until the holdout-clean shelf is rebuilt

Each later stage of this plan must declare which tier it belongs to.

### 1.5 Fairness doctrine — common scaffold vs native scaffold

There are two legitimate but different proxy-comparison questions:

1. **Common-scaffold question**
   - Hold the session map fixed and swap the score
   - Answers: "inside the same operational scaffold, which score is most useful?"

2. **Native-scaffold question**
   - Let each score earn its own session-family map under the same family method
   - Answers: "if each score gets a fair chance, which one is best?"

Both are allowed. Neither is allowed to impersonate the other.

Rules:
- common-scaffold results are useful for implementation choice inside a fixed operating design
- native-scaffold results are the fair test of comparative edge across scores
- no proxy-comparison claim is valid unless it states which scaffold question it answers
- if the two answers diverge, do not smooth that over; treat the divergence as information about role and substitutability

---

## 2. Success condition

The program succeeds only if it delivers **one or more** of the following on a holdout-clean / forward-judged basis:

1. higher **total R** without materially worse drawdown
2. higher **Sharpe / WFE / OOS retention**
3. lower **loss capture** via hostile-regime avoidance
4. higher **take-home $** at the portfolio / account level after realistic risk normalization

It does **not** succeed if it only raises `ExpR` on kept trades while destroying too much total R or adding hidden leverage.

### 2.1 Objective hierarchy

All candidate uses of `garch` should be ranked by this hierarchy:

1. **Take-home $ under real constraints**
2. **Survivability** under portfolio and account DD limits
3. **Risk-adjusted quality** (`Sharpe`, `profit / DD`, OOS retention)
4. **Interpretability / robustness**

This explicitly prevents optimizing for:
- local p-values without portfolio benefit
- kept-trade `ExpR` without total-R benefit
- raw PnL that comes only from leverage
- black-box classifier fit with no stable mechanism story

---

## 3. Risk inventory — what can go wrong

This plan explicitly covers these failure modes:

### 3.1 Statistical / discovery risk

- wrong BH family
- post-hoc threshold rescue
- over-reading local winners
- family survivorship from one hot session
- rare-tail underpower
- confusing informational monotonicity with deployment evidence

### 3.2 Bias / contamination risk

- feature look-ahead
- implicit holdout leakage
- discovery on contaminated derived layers
- mixing filtered and unfiltered populations
- using post-entry variables in classifier logic

### 3.3 Data integrity / plumbing risk

- stale `daily_features` or `orb_outcomes` rows by symbol/session
- join mismatches between `orb_outcomes` and `daily_features`
- incorrect filter semantics when recreating validated populations
- DST / session-boundary contamination
- silently missing symbols, sessions, or feature columns

### 3.4 Model-governance risk

- forcing a continuous variable into arbitrary bins without monotonicity evidence
- interaction explosion from `garch × atr × overnight × session × direction`
- overfitting a classifier because the model class is too flexible
- letting state labels drift until they merely rename PnL clusters
- selecting a model on the sacred forward window

### 3.5 Economic / implementation risk

- improving kept-trade quality while reducing total portfolio R
- treating a state variable as a hard gate when size tilt is correct
- double-counting what `atr_20_pct` or `overnight_range_pct` already capture
- hidden leverage in size overlays
- portfolio correlation shift after regime activation
- prop-firm DD breaches from concentrated favorable-regime sizing

### 3.6 Structural interpretation risk

- claiming `garch` is unique when it is mostly an ATR proxy
- assuming all sessions share one mechanism
- treating `NYSE_OPEN` behavior as equivalent to `COMEX_SETTLE`
- assuming favorable regime means “take only then” rather than “size more there”
- confusing common-scaffold proxy wins with native-scaffold proxy wins
- confusing research-weight gains with discrete live-contract gains

---

## 4. Decision tree — how `garch` can be used

`garch` has four plausible roles. The plan must discriminate among them.

### Role A — `R0` null / descriptive only

Use if:
- additive value collapses after controls
- no forward retention
- normalized size audit shows no portfolio improvement

### Role B — `R1` hostile-regime gate

Use if:
- `SKIP_LOW_ONLY` improves total R and/or drawdown materially
- effect survives within natural session families
- forward shadow confirms loss reduction

### Role C — `R3` size tilt

Use if:
- high-regime trades are consistently better quality
- hard gating loses too much total R
- normalized weighting improves portfolio PnL or Sharpe without hidden leverage

### Role D — `R7` regime classifier component

Use if:
- `garch` is partially redundant but still additive with ATR / overnight / gap state
- best value comes from combined state labels rather than any single threshold

**Default prior from current evidence:** `R3/R7` is more likely than `R1`.

---

## 5. Program architecture

The work should be executed in seven stages.

### Stage G0 — Research-object typing and data-integrity preflight

**Goal:** lock the object definition before any more exploitation work.

Mandatory checks:
1. environment / interpreter preflight
2. canonical-layer freshness by symbol
3. feature-construction audit for `garch_forecast_vol` and `garch_forecast_vol_pct`
4. exact validated-population rebuild audit
5. holdout boundary confirmation (`2026-01-01`)
6. destruction controls:
   - shuffled `garch_pct`
   - shifted `garch_pct`
   - placebo percentile feature

Pass condition:
- no unresolved data-plumbing or temporal-contamination issue remains open

Failure condition:
- any discrepancy in join logic, feature timing, or filtered-population recreation invalidates downstream economics until fixed

### Stage G1 — Lock the audited truth set

**Goal:** freeze what we already know so future runs are not ad hoc.

Inputs already created:
- `docs/audit/hypotheses/2026-04-16-garch-regime-family-audit.yaml`
- `docs/audit/results/2026-04-16-garch-regime-family-audit.md`
- `docs/audit/results/2026-04-16-garch-structural-decomposition.md`
- `docs/audit/results/2026-04-16-garch-regime-utilization-audit.md`

Required outputs:
- no code changes to thresholds or families without a new hypothesis file
- no promotion claims from exploratory runs

Pass condition:
- all later work references this audit set as the starting state

---

### Stage G2 — Additive value audit

**Goal:** determine whether `garch` adds value beyond existing regime proxies.

Questions:
1. Is `garch` mostly subsumed by `atr_20_pct`?
2. Does it add conditional information after controlling for:
   - `atr_20_pct`
   - `overnight_range_pct`
   - `atr_vel_ratio`
   - gap flags
   - calendar flags
3. Which sessions still show persistence after those controls?

Required tests:
1. pooled family correlations
2. within-stratum sign persistence
3. family-level 2x2 conditional tables:
   - `high_garch × high_atr`
   - `high_garch × low_atr`
   - `low_garch × high_atr`
   - `low_garch × low_atr`
4. dominance check:
   - `garch_only`
   - `atr_only`
   - `overnight_only`
   - `garch + atr`
   - `garch + overnight`
5. leave-one-feature-out ablation:
   - full state set
   - drop `garch`
   - drop `atr`
   - drop `overnight`
6. continuous-value check:
   - monotone deciles or quintiles
   - slope test on percentile rank
   - family-level incremental-value readout

Kill conditions:
- no distinct persistence beyond ATR / overnight families
- effect disappears once another proxy is fixed

Pass conditions:
- at least one family keeps the expected sign under the control strata
- combined-state version outperforms simpler proxies honestly
- incremental value survives on the continuous formulation, not only one arbitrary threshold pair

---

### Stage G3 — Utilization-mode audit

**Goal:** decide *how* to use `garch`, not just whether it exists.

Modes to test:

1. `TAKE_HIGH_ONLY`
2. `SKIP_LOW_ONLY`
3. `SIZE_TILT_LINEAR`
4. `SIZE_TILT_CLIPPED`
5. `COMPOSITE_STATE_GATE`

#### 5.3.1 Binary-gate economics

Already started in:
- `docs/audit/results/2026-04-16-garch-regime-utilization-audit.md`

Need to extend:
- include drawdown impact
- include per-session and portfolio-level totals
- include forward window only
- include trade-frequency / opportunity-cost effect

#### 5.3.2 Normalized sizing audit

This is mandatory and still missing.

For each trade:

```text
base weight   = 1.0
tilted weight = f(garch_pct)
```

But normalize so:

```text
mean(tilted weight) = 1.0
```

and compare:

```text
R_base = Σ pnl_r
R_tilt = Σ tilted_weight * pnl_r
```

Required metrics:
- total R
- annualized Sharpe
- max DD in R
- OOS retention
- tail risk of weighted losses
- account-rule consumption:
  - daily loss usage
  - trailing DD usage
  - worst 5-day weighted loss

Candidate weight maps:
- linear from percentile
- clipped piecewise:
  - `<=30` → `0.5x`
  - `30-70` → `1.0x`
  - `>=70` → `1.5x`
- session-specific maps only if pre-registered
- monotone scorecard map from classifier state only if the classifier survives on its own

Kill conditions:
- gain comes only from leverage, not selection
- DD rises too much for the improvement
- result depends on one arbitrary map

Pass conditions:
- normalized tilt beats base on total R and/or Sharpe with tolerable DD

#### 5.3.3 Composite classifier audit

Build a read-only classifier first, not an execution change.

Allowed feature menu:
- `garch_forecast_vol_pct`
- `atr_20_pct`
- `overnight_range_pct`
- `atr_vel_ratio`
- gap flags
- calendar flags

Do **not** use:
- `day_type`
- post-entry features
- any look-ahead columns
- unbounded feature engineering beyond the pre-registered menu
- flexible black-box models as the starting point

Model ladder:
1. theory-first scorecard / state machine
2. regularized linear score on the same locked features
3. depth-limited tree only if the simpler forms fail and only under a new hypothesis file

Classifier output:
- discrete regime labels or a bounded score

Candidate classes:
- `quiet`
- `trend_vol`
- `event_vol`
- `noisy_vol`
- `hostile_reversion`
- `unknown`

Test:
- lane-family compatibility matrix
- session-family total R under classifier states
- walk-forward or era-split stability of the learned states
- leave-one-session-out stability check for the state definitions
- ablation vs simpler gates:
  - classifier vs `garch_only`
  - classifier vs `atr_only`
  - classifier vs `garch + atr` rule table

Pass condition:
- classifier beats single-feature garch gates on either total R, DD, or stability
- classifier state definitions remain interpretable and stable across eras

#### 5.3.4 Proxy-comparison fairness audit

This stage is now mandatory because the first additive pass exposed a real design risk:
proxy scores can look stronger or weaker depending on whether they are forced
through another score's discovered session map.

Required paired tests:
1. **common scaffold**
   - run `garch`, `ATR`, `overnight`, and combined-state scores inside the same locked session map
2. **native scaffold**
   - let each score earn its own session-family support via the same fixed-threshold family audit
3. compare ranking stability between the two views

Questions:
- does `garch` still rank well when proxies get their own fair session maps?
- is `garch` best as a standalone score, a stabilizer, or a combo ingredient?
- does a proxy win only because it fits the other score's map better?

Pass conditions:
- a map's role is the same or at least interpretable across common/native views
- ranking shifts are explainable and documented, not ignored

Failure condition:
- the comparison answer changes materially with no design explanation, meaning the operating scaffold is still under-specified

---

### Stage G4 — Holdout-clean and forward-shadow validation

**Goal:** separate “interesting backtest regime behavior” from deployable improvement.

Requirements:
1. holdout-clean discovery populations only
2. 2026 forward window used for judgment, not selection
3. shadow logs for:
   - would-trade / would-skip
   - weighted size
   - realized counterfactual R
4. no threshold, weight-map, or class-definition changes after first forward look without a fresh hypothesis file

For each candidate role:
- IS metrics
- OOS metrics
- forward-shadow metrics

Minimum gates before any live use:
- OOS direction match
- OOS effect retention `>= 0.40 × IS`
- no severe DD blowout under normalized size
- no major drift alarm under SR monitor once shadow begins
- if the underlying shelf is research-provisional, treat all replay results as
  provisional operational evidence only

---

### Stage G5 — Portfolio and take-home-$ audit

**Goal:** answer the only question that matters at the PM level: does this raise net take-home dollars after real constraints?

Must evaluate:
- single-lane effect
- full portfolio effect
- prop-style DD constraints
- self-funded DD constraints
- turnover / implementation drag
- correlation concentration in favorable regimes
- capital-efficiency per account slot / lane slot

Required comparisons:
1. baseline portfolio
2. binary regime-gated portfolio
3. normalized size-tilted portfolio
4. classifier-controlled portfolio

Outputs:
- annualized total R
- total $
- max DD $
- worst day $
- profit/DD ratio
- risk-adjusted take-home per account type
- annualized turnover
- % days at elevated DD utilization
- account-breach probability

Decision rule:
- prefer the configuration with highest net take-home $ subject to DD / survivability
- do not prefer higher `ExpR` if it reduces realistic take-home $

### Stage G5b — Profile-specific deployment replay

**Tier:** profile-specific deployment replay

**Goal:** translate the surviving regime maps into the actual discrete choices
the live system can make.

This stage is the **first operational slice** of the allocator problem.
It is intentionally conservative and should not be confused with the final
allocator architecture. See:
- `docs/plans/2026-04-17-garch-deployment-replay-design-review.md`
- `docs/plans/2026-04-17-garch-deployment-allocator-architecture.md`

This stage exists because research weights are not live contracts.

Required translation:
- hostile low state -> `0` contracts or reduced size
- neutral state -> base live size
- favorable high state -> increased size only if the active profile permits it

Required constraints:
- active profile only (or explicitly named candidate profile)
- exact live lane set from `prop_profiles.py` / `lane_allocation.json`
- actual stop policy
- contract rounding
- Topstep / prop scaling plan or self-funded equivalent
- daily DD and trailing DD realism
- copied-account arithmetic if the profile actually trades via copier

Required outputs:
- per-account total $
- copied-account total $
- actual-path max DD $
- worst day $
- worst 5-day run $
- max open lots
- 90-day survival probability
- operational pass probability

Rules:
- no fractional-weight result can be promoted as a deployment claim
- no profile replay may discover new families; it only translates already-supported regime maps
- no deployment claim is valid if it improves take-home while materially degrading survival or scaling feasibility
- if the replay uses the current contaminated/research-provisional live book, the
  result is useful for operational stress-testing and pilot design only, not as
  clean validation evidence

### Stage G5c — Deployment allocator architecture

**Tier:** design / deployment-policy architecture

**Goal:** prevent the program from freezing on one-off discrete maps when the
research object is actually a continuous regime allocator problem.

This stage formalizes:
- the separation between research truth, policy design, deployment replay, and shadow promotion
- lane-aware policy vs profile-aware translation
- the progression from discrete maps to bounded continuous sizing and simple confluence allocation

Required questions:
1. is the discrete map the right permanent form, or only a low-risk first slice?
2. does bounded continuous sizing beat hard gating under the same profile constraints?
3. does simple confluence (`garch + overnight + ATR-state`) beat single-score maps?
4. does the preferred policy differ by profile because of real constraint geometry?

Required outputs:
- allocator architecture doc
- action ladder (`A1` discrete, `A2` continuous bounded size, `A3` confluence)
- locked objective hierarchy for comparing allocator candidates
- shadow/promotion contract

Rules:
- no profile doctrine is final until `A1` vs `A2` vs `A3` is compared
- no allocator claim is valid unless it is explicit whether it is lane policy, profile translation, or portfolio governance
- no new allocator stage may use the forward window for model selection

### Stage G5d — Allocator accounting verification

**Tier:** verification / deployment replay integrity

**Goal:** prevent false allocator conclusions caused by translation or reporting
bugs between raw row-level policy logic and profile-level replay outputs.

This stage is mandatory before any discrete, continuous, or confluence
allocator result is treated as interpretable.

Required reconciliations:
1. raw row-level policy totals reproduce the unconstrained policy-surface audit
2. profile replay baseline reproduces the canonical production replay baseline
3. per-policy profile replay totals reconcile to row-level trade-path arithmetic
4. per-session attribution reconciles to the same signed dollar deltas as the replay totals
5. copied-account totals equal per-account totals times copier count
6. skipped-lane accounting is explicit and zero-checked before any full-book claim

Required checks:
- one global policy and one session-aware policy per profile must be recomputed independently from raw rows before results are called verified
- any disagreement between headline totals and attribution tables blocks interpretation until fixed
- reporting tables are evidence-bearing outputs and must be verified, not treated as cosmetic summaries

Failure conditions:
- skipped-trade deltas do not reconcile
- session attribution omits or sign-flips part of the policy delta
- profile replay totals depend on unstated rounding or copy arithmetic

Pass condition:
- row-level policy logic, profile replay totals, and attribution tables all reconcile within rounding tolerance

---

### Stage G6 — Production migration path

**Goal:** implement only what survives the above.

Deployment order:

1. visibility only
   - dashboard regime readout
   - no execution change
2. shadow mode
   - log hypothetical skip / size decisions
3. one-family pilot
   - smallest clean candidate family
4. broader regime integration

Production states should mirror the regime framework:
- `LIVE`
- `LIVE_REGIME_ONLY`
- `MONITOR_ONLY_OUT_OF_REGIME`

No direct production move from exploratory family audit.

---

## 6. Testing matrix — mandatory coverage

This is the minimum complete matrix. Anything less leaves gaps.

### 6.1 Data integrity / validity

- environment preflight
- canonical freshness audit
- look-ahead audit
- holdout audit
- canonical-layer audit
- filter-semantics audit
- shuffle-null destruction test
- shifted-placebo destruction test

### 6.2 Statistical framing

- global BH
- family BH
- directional sign tests
- monotonicity / tail-bias tests
- OOS / forward sign retention
- continuous-rank slope / bucket tests

### 6.3 Regime-object correctness

- continuous-vs-threshold comparison
- rare-tail power check
- family-level rather than universal interpretation
- role-by-role evaluation (`R1`, `R3`, `R7`)

### 6.4 Redundancy / additivity

- correlation vs ATR / overnight / ATR-velocity
- within-stratum persistence
- pairwise overlay tests
- combined-state tests
- leave-one-feature-out ablation

### 6.5 Economic exploitation

- `TAKE_HIGH_ONLY`
- `SKIP_LOW_ONLY`
- normalized size maps
- classifier-gated activation
- total R, not just kept-trade ExpR
- drawdown and opportunity-cost decomposition
- common-scaffold and native-scaffold proxy comparison
- research-weight vs discrete-contract translation

### 6.6 Classifier governance

- locked feature menu
- locked model ladder
- state-definition stability
- interpretability check
- no forward-window model selection

### 6.7 Deployment realism

- portfolio DD
- account survival / prop-firm realism
- SR monitor integration
- regime-only shadow period
- turnover / implementation drag
- correlation concentration under favorable regimes
- active-profile replay with contract rounding
- copied-account take-home arithmetic
- scaling-plan feasibility under max open lots

---

## 7. What “institutional grade” means here

This plan counts as institutional only if it satisfies all of the following:

1. **No ad hoc thresholds** after seeing results
2. **No universal claim** from family-level evidence
3. **No deployment** from exploratory evidence alone
4. **No profit claim** without DD-normalized portfolio arithmetic
5. **No uniqueness claim** unless additive value beyond ATR / overnight survives
6. **No take-high-only claim** unless total R and DD both justify it
7. **No classifier claim** unless its state labels improve over simpler gates
8. **No regime-variable claim** unless continuous and family-framed tests agree on the sign story
9. **No portfolio-improvement claim** unless account-breach risk and implementation drag are included
10. **No proxy-comparison claim** unless common-vs-native scaffold framing is explicit
11. **No deployment-replay claim** unless discrete contract translation replaces research-weight arithmetic

---

## 8. Recommended immediate next actions

Order matters.

### Next 1

Implement the **G0 preflight pack** and freeze the exploitation input set.

Why:
- it closes the remaining data-integrity / object-definition loopholes
- it ensures later economics are grounded in the right populations and timestamps

### Next 2

Run the **normalized size audit** on the current audited family set.

Why:
- this is the largest current economic gap
- it is the most likely true utilization path
- it directly answers the PM question of take-home $

### Next 3

Run the **additive value audit** against:
- `atr_20_pct`
- `overnight_range_pct`
- `atr_vel_ratio`

Why:
- determines whether `garch` deserves its own role or just belongs inside a combined state variable

### Next 4

Build the **read-only regime classifier prototype** and test it in backtest/shadow mode.

Why:
- likely the best institutional way to use the signal family
- avoids premature hard-gate decisions

### Next 5

Do the **portfolio take-home-$ audit** after the normalized size and classifier passes exist.

Why:
- this is where “overall profile/profits/take home $$” gets answered honestly

### Next 6

Do the **profile-specific deployment replay** only after:
- common/native proxy fairness is resolved
- the best candidate maps are clear
- the discrete translation rules are pre-registered

Why:
- this is the first point where a live-operational answer is scientifically fair
- doing it earlier risks baking scaffold bias into deployment logic

### Next 7

Extend the replay program into the **deployment allocator architecture**:
- keep current discrete replay as `A1`
- run bounded continuous-size replay as `A2`
- run simple confluence allocator replay as `A3`

Why:
- the current discrete maps are probably useful, but they are unlikely to be the final economically correct use of a continuous regime variable
- this is the honest way to avoid tunnel vision and leaving allocator value on the table

---

## 9. Explicit non-goals

This plan is **not**:

- a justification to deploy `garch` immediately
- a reason to weaken BH or criteria
- a claim that `garch` is universally useful
- a claim that favorable sessions should always be traded high-only

---

## 10. Final doctrine

The correct posture is:

- keep `garch` alive
- treat it as a volatility-regime variable
- test it as a **state input**, not a folklore edge
- maximize **portfolio take-home $**, not local `ExpR` aesthetics
- promote only what survives holdout-clean, forward-judged, DD-normalized scrutiny
