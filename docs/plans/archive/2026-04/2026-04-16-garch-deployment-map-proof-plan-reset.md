---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Garch Deployment-Map Proof Plan Reset

**Date:** 2026-04-16  
**Status:** ACTIVE RESET  
**Purpose:** redesign the proof plan for deployment-map / allocator evidence after
audit findings showed the prior cycle was too close to overclaiming.

**Authority chain:**
- `CLAUDE.md`
- `RESEARCH_RULES.md`
- `TRADING_RULES.md`
- `REPO_MAP.md`
- `docs/plans/2026-04-16-garch-institutional-utilization-plan.md`
- `docs/plans/2026-04-17-garch-deployment-allocator-architecture.md`

---

## 1. What Is Actually Proved Now

Only the following claims are supported at this point:

1. The replay layer can produce headline profile totals that independently
   recompute from raw trade paths.
2. The current value signal is more consistent with allocator / sizing behavior
   than with simple `TAKE_HIGH_ONLY` or `SKIP_LOW_ONLY` gating.
3. Two concrete replay defects were real and were fixed:
   - bad import / object reference for `replace`
   - skipped-trade deltas missing from attribution
4. The replay work is operationally informative enough to justify a stricter
   allocator program.

These are the strongest currently supportable Topstep and self-funded A1
headline facts:

- `GLOBAL_HIGH_2X_ONLY` and `SESSION_CLIPPED_0_1_2` both improve headline
  total dollars versus `BASE_1X` in the current profile replays.
- `TAKE_HIGH_ONLY` and `SKIP_LOW_ONLY` do not emerge as the dominant economic
  use in the current replay layer.

What is proved here is **translation-layer usefulness**, not edge proof.

---

## 2. What Is Not Proved

The following are **not** proved:

1. That `garch` is a validated standalone edge.
2. That any session-level doctrine is proven from the current replay outputs.
3. That the per-session attribution tables are authoritative.
4. That the current best A1 discrete policy is the final deployment doctrine.
5. That the current replay winners survive a cleaner A2 continuous or A3
   confluence comparison.
6. That replay evidence alone is enough for live promotion.
7. That the current operational gains are independent of the current
   research-provisional live shelf.

Current proof boundary:

- **Headline totals:** usable after independent recomputation.
- **Session attribution:** not yet fully reconciled, therefore not evidence-grade.

---

## 3. Evidence-Tier Split

This program must explicitly separate three evidence tiers.

### 3.1 Discovery truth

Question:
- Is there a real regime-family effect at all?

Allowed sources:
- canonical trade rows
- family-framed audits
- structural decomposition
- additive-value work

Not allowed:
- profile replay results rewriting discovery truth

### 3.2 Validated-family utility

Question:
- Does the variable improve the actual validated / live-like shelf?

Allowed sources:
- exact validated populations
- fixed family tests
- raw row-level policy surfaces

Not allowed:
- profile/account geometry being mistaken for proof of edge

### 3.3 Deployment replay / allocator translation

Question:
- Does a candidate policy survive real account geometry, stop policy, contract
  rounding, and survival rules?

Allowed sources:
- replay infrastructure
- profile lane sets
- account-survival logic
- copied-account arithmetic

Not allowed:
- session-edge discovery
- promotion claims from replay alone

---

## 4. Revised Null / Alternative Hypotheses

The prior framing was too close to:
"there is a deployment-map edge and we just need to prove it."

That is not the correct null.

### Null hypotheses

`H0a`
- The current replay gains are accounting / translation artifacts, not robust
  allocator value.

`H0b`
- Any apparent gains come from coarse leverage application rather than correct
  state usage.

`H0c`
- The current A1 winners do not survive stricter allocator comparison once A2
  continuous sizing and A3 confluence are tested.

`H0d`
- Session-level conclusions are unsafe until attribution is fully reconciled.

### Alternative hypotheses

`H1a`
- There is real allocator value in the variable, but it shows up as bounded
  sizing / translation behavior rather than simple gating.

`H1b`
- The correct object is profile-aware policy translation, not one universal map.

`H1c`
- A2 or A3 may dominate A1 once information is used more efficiently.

The active working question is therefore:

**Can allocator value be demonstrated under verified accounting and realistic
profile constraints, without claiming a standalone edge has been proved?**

---

## 5. Exact Verification Plan To Close Or Demote The Attribution Gap

This is the immediate blocker.

### 5.1 Objective

Either:
- close the attribution gap exactly, or
- formally demote session-attribution tables to non-authoritative explanatory output

### 5.2 Required reconciliation chain

For each selected policy / profile pair:

1. raw row-level policy arithmetic
2. rounded scaled trade-path arithmetic
3. daily scenario totals
4. profile replay headline total
5. session attribution roll-up

These must reconcile on the same arithmetic surface.

### 5.3 Minimum mandatory checks

Per profile:
- one global policy
- one session-aware policy

Profiles:
- `topstep_50k_mnq_auto`
- `self_funded_tradovate`

Policies:
- `GLOBAL_HIGH_2X_ONLY`
- `SESSION_CLIPPED_0_1_2`

### 5.4 Acceptance standard

- Headline replay deltas must reconcile exactly or within stated rounding tolerance
- Session attribution must reconcile exactly to the same signed headline delta
- No hidden residual line is allowed in the final doctrine output

### 5.5 If the gap closes

- Session tables may be treated as descriptive allocator evidence
- They still do not prove edge by themselves

### 5.6 If the gap does not close

- Keep headline totals
- Demote session tables explicitly
- Ban session-level deployment-map claims until repaired

This is the default skeptical outcome if reconciliation remains imperfect.

---

## 5A. Allocator preflight hardening

Any future allocator test must clear these gates **before** execution:

1. **Binding surface**
   - prove the proposed budget / routing / sizing surface can actually change
     the path
2. **Named canonical comparator**
   - baseline must come from a declared repo allocator surface, not an ad hoc
     invented order
3. **Path-change map**
   - state exactly when and how candidate differs from base
4. **Early null kill**
   - if the binding check fails, demote the stage immediately instead of
     interpreting a no-op replay

This was added after `A4a` active-profile slot routing turned out to be null by
construction on the current books.

---

## 6. Safe Claims Allowed Now

These claims are allowed now:

1. Headline A1 replay totals are independently recomputed and usable.
2. The current replay surface does not support simple `TAKE_HIGH_ONLY` as the
   main use case.
3. The current replay surface is more consistent with allocator / sizing behavior.
4. Replay evidence is sufficient to justify A2 and A3 testing.
5. Session-attribution tables are not yet authoritative evidence.

---

## 7. Unsafe Claims Prohibited Now

These claims are prohibited now:

1. "`garch` edge is proved."
2. "`SESSION_X` is definitely the source of the gain."
3. "`GLOBAL_HIGH_2X_ONLY` is the best final map."
4. "The replay proves live readiness."
5. "The current session attribution establishes profile doctrine."
6. "A1 discrete replay is enough to skip A2 / A3."

---

## 8. A2 Bounded Continuous Sizing Test Spec

**Goal:** test whether the information in the regime score is better used
through bounded continuous sizing than through coarse A1 maps.

### 8.1 Inputs

- `garch_forecast_vol_pct`
- same replayable lane sets as A1
- same profile/account rules as A1

### 8.2 Allowed forms

Start with simple bounded monotone functions only:

1. clipped linear:
   - `<=30 -> 0.5x`
   - `30-70 -> 1.0x`
   - `>=70 -> 1.5x`
2. profile-bounded linear ramp:
   - percentile mapped into a narrow profile-safe range
3. session-aware bounded form only if it uses already-supported lane policy

### 8.3 Hard constraints

- average risk must be normalized against base
- no hidden leverage
- no use of forward window for curve shaping
- no flexible non-monotone functions

### 8.4 Required outputs

- per-account total $
- copied-account total $
- Sharpe
- max DD $
- worst day $
- worst 5-day $
- survival
- operational pass
- concentration / open-lots stress

### 8.5 Success condition

A2 is worth keeping only if it beats A1 on the objective hierarchy:

1. take-home $
2. survival
3. clustered-loss behavior
4. Sharpe / profit-to-DD

### 8.6 Kill condition

Kill A2 if:
- it only wins by stealth leverage
- it worsens survival materially
- it is too sensitive to small shape changes
- it collapses back to A1 after realistic contract rounding

---

## 9. A3 Simple Confluence Allocator Test Spec

**Goal:** test whether `garch` is more valuable as one component in a simple
multi-signal allocator than as a solo score.

### 9.1 Allowed signals

- `garch_forecast_vol_pct`
- `overnight_range_pct`
- `atr_20_pct`
- optionally `atr_vel_ratio` if the signal is already clean and pre-entry

### 9.2 Allowed model class

Only simple, auditable forms:

1. scorecard / rule table
2. regularized linear combiner

Not allowed at this stage:
- tree models
- interaction sprawl
- opaque non-linear fitting

### 9.3 Comparison framework

Run both:

1. common scaffold
2. native scaffold

Questions:
- does `garch` still matter once combined with other vol-state signals?
- is it the primary driver, a stabilizer, or redundant?

### 9.4 Required outputs

- same economic outputs as A2
- plus ablation reads:
  - full confluence
  - drop `garch`
  - drop `overnight`
  - drop `atr`

### 9.5 Success condition

A3 only survives if it improves the objective hierarchy versus the best A1 and A2
candidate without becoming opaque or overfit.

### 9.6 Kill condition

Kill A3 if:
- the gain disappears under ablation
- one proxy fully subsumes the others
- the improvement is not stable across profiles

---

## 10. Promotion Gates

Nothing gets promoted from replay to doctrine unless all of the following hold:

1. allocator-accounting verification passed
2. A1 vs A2 vs A3 comparison completed on the same objective hierarchy
3. no session-level claim relies on unreconciled attribution
4. the preferred candidate is profile-scoped, not falsely universal
5. replay result is converted into shadow logging before any live action change
6. replay-to-shadow agreement is inside tolerance

---

## 11. Kill / Demotion Gates

Demote or kill a candidate if any of the following occur:

1. attribution remains unreconciled
2. improvement disappears after proper risk normalization
3. winner changes wildly under minor policy-shape changes
4. survival deterioration outweighs dollar improvement
5. the signal is fully subsumed in A3 by another proxy
6. the result depends on the current research-provisional shelf in a way that
   prevents clean interpretation

---

## 12. Repo File / Module Map

Core modules for this proof program:

- [research/garch_profile_policy_surface_replay.py](/mnt/c/Users/joshd/canompx3/research/garch_profile_policy_surface_replay.py:1)
  First-slice A1 discrete profile replay for raw verified policies.
- [research/garch_discrete_policy_surface_audit.py](/mnt/c/Users/joshd/canompx3/research/garch_discrete_policy_surface_audit.py:1)
  Raw row-level policy-surface audit.
- [research/garch_profile_production_replay.py](/mnt/c/Users/joshd/canompx3/research/garch_profile_production_replay.py:1)
  Canonical profile replay infrastructure.
- [trading_app/account_survival.py](/mnt/c/Users/joshd/canompx3/trading_app/account_survival.py:377)
  Daily scenario construction and survival logic.
- [pipeline/build_daily_features.py](/mnt/c/Users/joshd/canompx3/pipeline/build_daily_features.py:1)
  Feature construction authority.
- [trading_app/holdout_policy.py](/mnt/c/Users/joshd/canompx3/trading_app/holdout_policy.py:1)
  Holdout boundary authority.
- [docs/plans/2026-04-16-garch-institutional-utilization-plan.md](/mnt/c/Users/joshd/canompx3/docs/plans/2026-04-16-garch-institutional-utilization-plan.md:1)
  Main staged utilization program.
- [docs/plans/2026-04-17-garch-deployment-allocator-architecture.md](/mnt/c/Users/joshd/canompx3/docs/plans/2026-04-17-garch-deployment-allocator-architecture.md:1)
  Allocator architecture and objective hierarchy.

---

## 13. Final Recommendation

Treat this as an allocator-accounting validation program first.

Immediate order:

1. close or formally demote the session-attribution layer
2. freeze only the verified A1 headline totals
3. run A2 bounded continuous sizing
4. run A3 simple confluence allocator
5. rank A1 / A2 / A3 on the same profile-aware objective hierarchy
6. only then discuss profile doctrine

Current honest recommendation:

- continue the work
- do not claim edge proved
- do not use session-attribution tables as evidence yet
- treat the verified replay headlines as sufficient reason to keep testing
  allocator behavior, not as sufficient reason to declare deployment-map truth
