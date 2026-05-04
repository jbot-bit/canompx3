---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Garch Institutional Attack Plan

**Date:** 2026-04-16  
**Status:** ACTIVE PROGRAM  
**Purpose:** turn the current `garch` work into one coherent institutional
program with explicit proof boundaries, mechanism families, test order,
stop/promote gates, and raw-number verification rules.

**Authority chain:**
- `CLAUDE.md`
- `RESEARCH_RULES.md`
- `TRADING_RULES.md`
- `docs/institutional/pre_registered_criteria.md`
- `docs/institutional/regime-and-rr-handling-framework.md`
- `docs/institutional/mechanism_priors.md`
- `docs/plans/2026-04-16-garch-deployment-map-proof-plan-reset.md`
- `docs/plans/2026-04-16-deployment-map-incremental-edge-proof-plan.md`
- `docs/plans/2026-04-17-garch-deployment-allocator-architecture.md`
- `docs/audit/hypotheses/2026-04-16-garch-mechanism-hypotheses.md`
- `docs/plans/2026-04-16-garch-program-audit.md`

---

## 1. Program objective

The objective is **not** to prove "`garch` is the edge."

The objective is to answer four narrower questions, in order:

1. Is `garch_forecast_vol_pct` a real market-state input in canonical truth?
2. Does that state have utility on the validated / live-like shelf?
3. Can that utility survive honest allocator translation under real account
   geometry?
4. If yes, what is the correct role:
   - `R1` binary gate
   - `R3` size modifier
   - `R7` confluence input
   - `R8` portfolio / routing allocator

This program is complete only when one of the following is true:

- a use survives and is promoted to shadow / live doctrine, or
- the remaining hypotheses are falsified and the work is parked.

---

## 2. Current proof state

### 2.1 Verified now

These statements are currently supported:

1. `A1` discrete allocator headline replay totals recompute independently from
   raw trade paths.
2. `A1` behaves more like allocator / sizing value than simple
   `TAKE_HIGH_ONLY` or `SKIP_LOW_ONLY` gating.
3. `A2` bounded continuous sizing is negative under current account geometry
   after real integer-contract translation.
4. `A3` simple confluence can improve on `BASE_1X` in some cases, but did not
   beat the best solo map on either profile in the first clean pass.
5. The current evidence justifies continuing allocator research.

### 2.2 Not verified now

These statements are currently **not** supported:

1. `garch` as a standalone deployable signal edge.
2. any session-level doctrine from replay attribution tables
3. any hard app default for a candidate map
4. any claim that a deployment map is a validated incremental edge over
   `BASE_1X`
5. any claim that replay alone is enough for promotion

### 2.3 Immediate boundary

- **Authoritative now:** headline replay totals that were raw-row recomputed
- **Non-authoritative now:** per-session attribution from the current replay
  layer until exact reconciliation closes

---

## 3. Anti-bias operating rules

These rules are mandatory for every further step.

1. **Hypothesis-first**
   - no new run without a committed hypothesis / plan artifact
2. **Canonical truth first**
   - discovery uses canonical layers, not `live_config`, notes, or shelf state
3. **Validated shelf for utility**
   - deployment-utility questions must also be tested on the validated /
     live-like population
4. **Replay is operational evidence only**
   - replay can justify shadow, not proof of edge
5. **Raw numbers before doctrine**
   - every promoted statement must be backed by directly recomputable trade-path
     arithmetic
6. **Demote weak evidence**
   - if a surface does not reconcile or hold up, demote it rather than rescue it
7. **No filter-menu browsing**
   - only mechanism-shaped combinations are allowed
8. **No profile confusion**
   - Topstep and self-funded are different objective surfaces and may rank maps
     differently
9. **No non-binding allocator surfaces**
   - if the proposed budget or action cannot actually change the path, the test
     is invalid and must be killed before interpretation
10. **No invented neutral comparator**
   - every allocator baseline must come from a named canonical repo surface and
     be audited before the candidate run

---

## 3A. Mandatory allocator preflight

Every future allocator stage must pass this preflight before execution.

### P1 - Binding check

Prove that the proposed scarce-resource surface actually binds.

Examples:
- routing test:
  - collision-day count must be non-zero
- sizing test:
  - translated contracts must change on a non-zero share of trades
- budget test:
  - candidate budget must exclude some otherwise-eligible opportunities

If the action surface cannot change the path, kill the stage immediately.

### P2 - Comparator audit

Name and justify the neutral comparator from an existing canonical repo surface.

Allowed:
- `trading_app/prop_portfolio.py`
- `trading_app/lane_allocator.py`
- other explicitly audited canonical allocator surfaces

Not allowed:
- ad hoc invented baseline order
- undocumented operational default

### P3 - Path-change proof

Show exactly how the candidate can differ from base:

- on which days / states
- through which action channel
- under which pre-entry information

### P4 - Raw recompute contract

Before any doctrine statement:
- headline result must recompute from raw trade paths
- if attribution is reported, it must reconcile exactly or be explicitly demoted

### P5 - Early kill rule

If preflight fails, do not rescue the stage with narrative or extra variants.
Demote it and redesign the surface.

---

## 4. Correct data tiers

The main source of confusion so far has been mixing the wrong universe for the
wrong question. This program fixes that.

### Tier T1 — Discovery truth

Question:
- what does the state variable mean structurally?
- where is the sign / shape real?

Source:
- canonical layers only:
  - `bars_1m`
  - `daily_features`
  - `orb_outcomes`

Use:
- broad family scans
- monotonicity / tail-shape checks
- additive / redundancy checks
- mechanism falsification

### Tier T2 — Validated utility

Question:
- does the state help the actual validated / live-like book?

Source:
- exact validated populations and locked family definitions

Use:
- fixed-family tests
- candidate policy surfaces
- profile-agnostic utility checks

### Tier T3 — Deployment translation

Question:
- does the candidate policy survive real profile/account constraints?

Source:
- replay infrastructure
- profile lane sets
- account rules
- stop policy
- contract translation

Use:
- `A1`, `A2`, `A3`, `A4`

### Tier T4 — Forward proof

Question:
- does shadow agree with replay enough to justify promotion?

Source:
- locked forward shadow logs
- live-like drift / survival / DD comparisons

Use:
- promotion or kill

### Rule

No tier is allowed to rewrite a different tier’s truth:

- T3 cannot prove T1 discovery truth
- T1 cannot by itself prove T3 deployment utility
- T4 is required for promotion

---

## 5. Mechanism families in scope

Only the following mechanism families are in scope now.

### M1 — Latent expansion

Definition:
- high `garch_forecast_vol_pct`
- low to moderate `overnight_range_pct`

Economic question:
- does latent conditional volatility, not yet expressed overnight, improve
  event-session expansion quality?

Expected use:
- lane ranking / allocator preference, not pure binary gating

### M2 — Active trend-vol

Definition:
- high `garch_forecast_vol_pct`
- high `atr_20_pct`

Economic question:
- does agreement between conditional and realized vol identify an active
  continuation regime?

Expected use:
- bounded upweighting on already-eligible trades

### M3 — Allocator-not-gate

Definition:
- composite vol-state score across `garch`, overnight range, ATR state

Economic question:
- is the value mainly in cross-opportunity ranking under scarce daily risk
  budget, rather than in yes/no trade decisions?

Expected use:
- `R8` portfolio / routing allocator

### M4 — Profile-specific translation

Definition:
- same upstream state, different feasible actions by profile

Economic question:
- can the same research truth map to different deployment doctrine under prop
  versus self-funded constraints?

Expected use:
- profile-specific policy layer after proof

### Out of scope until new theory appears

- random additional filters with no mechanism note
- threshold fishing after first look
- black-box tree / nonlinear model search
- session doctrine from non-reconciled attribution

---

## 6. Workstreams and required order

This is the official execution order. No skipping.

### W0 — Governance and accounting hygiene

Goal:
- make sure no attractive result is being carried by a broken surface

Tasks:
1. maintain the proof reset and incremental-edge plan
2. keep `HANDOFF.md` current
3. close or formally demote the current session-attribution gap

Deliverable:
- either exact attribution reconciliation or explicit non-authoritative label

### W1 — Discovery truth refresh

Goal:
- keep the structural picture honest and prevent deployment replay from
  becoming the theory

Tasks:
1. confirm the sign / shape of `garch` on canonical families
2. keep additive / redundancy checks vs:
   - `overnight_range_pct`
   - `atr_20_pct`
   - `atr_vel_ratio`
3. use only pre-2026 discovery / holdout-clean discipline for new truth claims

Deliverable:
- refreshed mechanism-aware truth note if a mechanism is falsified or sharpened

### W1b — State distinctness / incremental-value audit

Goal:
- determine whether `garch` adds usable information beyond adjacent vol-state
  proxies, or whether the real object is a broader vol-state family

Tasks:
1. audit overlap versus:
   - `atr_20_pct`
   - `overnight_range_pct`
   - `atr_vel_ratio`
2. test persistence of the `garch` sign inside proxy strata
3. test incremental value, not just correlation
4. explicitly record whether `garch` is:
   - distinct
   - complementary
   - or largely subsumed

Deliverable:
- state-family distinctness note

Status:
- design locked and executed on 2026-04-16
- output:
  - `docs/audit/results/2026-04-16-garch-state-distinctness-audit.md`

Current read:
- `garch` is not globally dominant
- it still looks locally distinct or complementary in parts of the locked
  family set, especially against `overnight_range_pct` and `atr_vel_ratio`
- `atr_20_pct` overlap remains the strongest subsumption risk and did not earn
  a clean distinctness verdict in this stage
- verdicts are local to the locked family set and do not promote deployment
  doctrine

### W2 — Mechanism pairing on the validated shelf

Goal:
- test whether one realistic state + setup pairing improves the validated /
  live-like shelf without drifting into random filter stacking

Tasks:
1. keep candidate mechanism family fixed
2. test exact validated populations only
3. pair `garch` only in the role of:
   - conditioner
   - confluence score
   - allocator state
   not standalone direction signal
4. prevent broad-grid drift from being mistaken for deployable utility

Deliverable:
- validated utility note for each promoted mechanism family

Status:
- executed on 2026-04-16
- design:
  - `docs/plans/2026-04-16-garch-w2-mechanism-pairing-design.md`
- hypothesis:
  - `docs/audit/hypotheses/2026-04-16-garch-w2-mechanism-pairing.yaml`
- result:
  - `docs/audit/results/2026-04-16-garch-w2-mechanism-pairing-audit.md`

Current read:
- `M2` active transition (`high garch + atr_vel Expanding`) is the stronger
  survivor
- `M1` latent expansion (`high garch + overnight not high`) is mostly
  `garch_distinct` or `unclear`, not a broad complementary winner
- W2 remains validated-shelf utility only; it does **not** promote deployment
  doctrine or allocator translation yet

### W2b — Partner-state provenance audit

Goal:
- verify that the current W2 partner encodings are principled local mechanism
  representations rather than arbitrary convenience cutoffs

Tasks:
1. hold `garch_high` fixed
2. test only a small locked set of neighboring or alternate partner
   representations
3. keep the scope on the validated shelf only
4. allow only representation conclusions:
   - `supported_current`
   - `neighbor_stable`
   - `alternate_better`
   - `weak_mechanism`
   - `unclear`
5. keep prior-day levels and prior-session carry explicitly **out** of this
   stage

Deliverable:
- partner-state provenance note tied back to W2, with no deployment promotion

Status:
- executed on 2026-04-16
- design:
  - `docs/plans/2026-04-16-garch-partner-state-provenance-design.md`
- hypothesis:
  - `docs/audit/hypotheses/2026-04-16-garch-partner-state-provenance.yaml`
- result:
  - `docs/audit/results/2026-04-16-garch-partner-state-provenance-audit.md`

Current read:
- `atr_vel_regime == Expanding` remains a defensible M2 representation, but it
  is not uniquely privileged:
  - `COMEX_SETTLE_high`: `ATR_PCT_GE_70` slightly edges it and the family reads
    `neighbor_stable`
  - `EUROPE_FLOW_high`: current `atr_vel Expanding` remains viable and
    neighboring velocity cutoffs stay supportive
  - `TOKYO_OPEN_high`: current `atr_vel Expanding` and `atr_vel_ratio >= 1.05`
    are effectively the same local read
  - `SINGAPORE_OPEN_high`: stricter `atr_vel_ratio >= 1.10` beats the current
    representation
- `overnight_range_pct < 80` is **not** a generally strong M1 representation:
  - `COMEX_SETTLE_high`: current read is only `neighbor_stable`; tighter
    `OVN_NOT_HIGH_60` improves conjunction quality
  - `EUROPE_FLOW_high`: current representation is demoted in favor of
    `OVN_MID_ONLY`
  - `TOKYO_OPEN_high`: M1 is weak regardless of representation
  - `SINGAPORE_OPEN_high`: current representation is demoted in favor of
    `OVN_MID_ONLY`
- the honest implication is:
  - M2 is the main carry-forward mechanism
  - M1 should not be treated as one broad "overnight not high" doctrine
  - partner-state representation matters and must stay local / family-aware

Queue kept explicit:
- prior-day levels
- prior-day realized range
- prior-session carry / cascade

These remain valid future mechanism families, but they do not belong inside
W2b.

### W2c — Conservative M2 validated utility carry check

Goal:
- test whether the surviving local M2 partner representation actually improves
  validated-shelf utility beyond base and `garch_high` alone, without opening a
  new representation search

Tasks:
1. freeze family-local M2 representation from W2b:
   - `neighbor_stable` or `supported_current` keeps current representation
   - `alternate_better` may switch only to the named locked alternate
2. run exact validated populations only
3. compare:
   - base family expectancy
   - `garch_high` expectancy
   - conjunction expectancy
4. keep OOS / 2026 descriptive only
5. make per-cell detail unambiguous by carrying ORB aperture where the
   validated shelf contains multiple apertures for the same filter / RR family

Deliverable:
- validated-shelf M2 carry note with raw-checked family summaries

Status:
- executed on 2026-04-16
- design:
  - `docs/plans/2026-04-16-garch-w2c-m2-validated-utility-design.md`
- hypothesis:
  - `docs/audit/hypotheses/2026-04-16-garch-w2c-m2-validated-utility.yaml`
- result:
  - `docs/audit/results/2026-04-16-garch-w2c-m2-validated-utility-audit.md`

Current read:
- stage verdict: `M2_carry`
- all carried local families improved conjunction expectancy over both base and
  `garch_high` alone:
  - `COMEX_SETTLE_high`:
    - base `+0.107822`
    - `garch_high` `+0.238249`
    - conjunction `+0.277584`
    - `Δ conj-garch +0.039335`
  - `EUROPE_FLOW_high`:
    - base `+0.096774`
    - `garch_high` `+0.148081`
    - conjunction `+0.261281`
    - `Δ conj-garch +0.113200`
  - `TOKYO_OPEN_high`:
    - base `+0.082299`
    - `garch_high` `+0.140209`
    - conjunction `+0.282653`
    - `Δ conj-garch +0.142445`
  - `SINGAPORE_OPEN_high`:
    - base `+0.120994`
    - `garch_high` `+0.144872`
    - conjunction `+0.406274`
    - `Δ conj-garch +0.261402`
- `SINGAPORE_OPEN_high` initially looked duplicated in detail output; review
  confirmed those are real validated rows split by ORB aperture (`O15` and
  `O30`), and the report now carries ORB minutes explicitly
- this is still validated utility only:
  - no deployment doctrine
  - no allocator translation
  - no promotion from descriptive 2026 OOS

Immediate implication:
- carry `M2` forward as a live mechanism family candidate
- keep `M1` demoted from generic doctrine
- next mechanism-family stage can reopen a **different** queue item
  (`prior-day levels` or `prior-session carry`) without collapsing it into W2c

### W3 — Allocator accounting verification

Goal:
- make sure deployment arithmetic is exact before interpreting anything

Tasks:
1. raw row-level arithmetic
2. rounded scaled trade-path arithmetic
3. daily totals
4. profile headline totals
5. attribution roll-up

Acceptance:
- exact reconciliation or formal demotion

Deliverable:
- accounting-verification note tied to each replay surface

### W4 — A-series allocator tests

Goal:
- exhaust the valid deployment uses in the right order

Sequence:
1. `A1` discrete maps
2. `A2` bounded continuous sizing
3. `A3` simple confluence allocator
4. `A4` portfolio-ranking / scarce-risk allocation

Rule:
- each stage must be judged against `BASE_1X` and against the best surviving
  prior stage

### W5 — Forward proof ladder

Goal:
- prove or kill the surviving candidate under forward evidence

Sequence:
1. locked doctrine note
2. shadow deployment
3. forward comparison to base
4. promotion / kill decision

---

## 7. A-series verdict and next target

### A1 — Discrete maps

Status:
- operationally verified at headline level

Current verdict:
- worth continuing
- not enough for edge proof

### A2 — Bounded continuous sizing

Status:
- tested

Current verdict:
- negative under current profile/account translation
- do not keep trying to rescue it unless the translation model itself changes

### A3 — Simple confluence

Status:
- tested

Current verdict:
- additive in some cases
- not a dominant winner yet
- keep as a contender, not doctrine

### A4 — Portfolio-ranking / scarce-risk allocation

Status:
- next stage, not yet run

Why this is next:
- it is the cleanest remaining literature-shaped use
- it tests `M3` directly
- it is the best way to express continuous state value without pretending
  single-lane fractional sizing is always implementable

---

## 8. Exact next test: A4 portfolio-ranking allocator

This is the next execution target.

### 8.1 Claim

Relative to `BASE_1X`, a pre-registered state score improves profile utility by
allocating scarce daily risk budget toward the best eligible same-day
opportunities without hidden leverage or profile-rule violations.

### 8.2 Baseline

- same lane universe
- same profile
- same stop policy
- same period
- same daily risk budget
- baseline allocation rule = existing `BASE_1X` handling

### 8.3 Candidate score families

Only one mechanism family is allowed per initial A4 hypothesis file.

Allowed first candidates:
1. `M3` simple composite:
   - standardized / discretized blend of `garch`, `overnight`, `ATR`
2. `M1` latent expansion score:
   - high `garch`, low/mod overnight
3. `M2` active trend-vol score:
   - high `garch`, high ATR

### 8.4 Action form

No fancy model at first pass.

Allowed first-pass action:
- when multiple lanes are eligible on the same day, rank them by the locked
  score and allocate finite risk budget to the top-ranked opportunities first

Required first-pass constraints:
- fixed 1x contract handling only
- no upsizing, downsizing, or fractional translation in the first A4 pass
- action may differ from base only on same-day collision sets
- budget unit must be pre-declared and profile-native
- tie-break order must be deterministic and fixed before first look

### 8.5 Required outputs

1. headline profile totals
2. raw recomputable trade-path totals
3. budget utilization stats
4. concentration stats
5. no-lane-left-behind coverage report
6. degradation versus base on:
   - drawdown
   - worst day
   - worst 5-day
   - survival
7. collision-day only delta decomposition
8. top-lane and top-session contribution shares for the rerouted delta

### 8.6 Acceptance standard

To survive to shadow, A4 must:

1. beat `BASE_1X` on the profile’s primary objective
2. not be dominated by one lane or one isolated window
3. not rely on hidden leverage
4. remain stable under neighboring-window sensitivity
5. use only pre-entry data
6. preserve exact lane universe, stop policy, and profile rules

---

## 9. Decision gates

### Continue gates

Continue a line of work only if:

1. accounting is exact or explicitly demoted
2. headline gains recompute from raw trade paths
3. the mechanism is still plausible and unfalsified
4. utility survives on the right tier for the question being asked

### Kill gates

Kill or park a line of work if:

1. it requires narrative rescue after a negative result
2. it only works through non-implementable fractional sizing
3. it is dominated by one narrow window or one lane
4. it fails neighboring-window sensitivity
5. it degrades the profile’s primary objective beyond tolerance
6. a better simpler surviving map dominates it

### Promotion gates

Promote only if:

1. historical replay survives
2. doctrine is locked
3. forward shadow matches replay within tolerance
4. profile utility still beats base on the chosen objective hierarchy

---

## 10. Review and verification protocol

Nothing becomes “verified” until all of the following happen:

1. the hypothesis / plan file exists first
2. the script runs cleanly
3. the script output is read
4. at least one independent raw-row recompute confirms the headline claim
5. the conclusion is checked against:
   - sample size
   - period
   - IS/OOS/holdout status
   - multiple-testing context
   - mechanism plausibility
   - implementability

Every result note should end in this format:

```text
SURVIVED SCRUTINY:
DID NOT SURVIVE:
CAVEATS:
NEXT STEPS:
```

---

## 11. Current recommendation

### Do not do

- do not park `garch`
- do not claim `garch` edge proved
- do not hard-wire production defaults yet
- do not keep stacking random filters
- do not use session attribution as evidence until it reconciles

### Do now

1. keep `garch` alive as an allocator / state-variable program
2. demote or close the attribution gap explicitly
3. freeze `A2` as currently negative
4. keep `A3` as a contender, not a winner
5. run `A4` portfolio-ranking / scarce-risk allocation as the next real test

### Working one-line doctrine

`garch_forecast_vol_pct` is currently best treated as a **candidate allocator
state input** whose standalone signal edge is unproved, whose simple continuous
sizing translation currently fails, and whose next honest test is
portfolio-ranking under scarce risk budget.
