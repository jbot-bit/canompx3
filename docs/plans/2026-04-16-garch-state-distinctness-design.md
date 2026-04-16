# Garch State Distinctness Design

**Date:** 2026-04-16  
**Status:** ACTIVE DESIGN  
**Purpose:** define the next institutional audit for `garch_forecast_vol_pct`
as a state-family input, answering whether it is distinct, complementary, or
mostly subsumed by adjacent volatility-state proxies before any further pairing
or deployment work.

**Authority chain:**
- `CLAUDE.md`
- `RESEARCH_RULES.md`
- `TRADING_RULES.md`
- `docs/institutional/pre_registered_criteria.md`
- `docs/institutional/mechanism_priors.md`
- `docs/institutional/regime-and-rr-handling-framework.md`
- `docs/audit/results/2026-04-16-garch-regime-audit-synthesis.md`
- `docs/audit/results/2026-04-16-garch-structural-decomposition.md`
- `docs/audit/results/2026-04-16-garch-additive-sizing-audit.md`
- `docs/audit/results/2026-04-16-garch-proxy-native-sizing-audit.md`
- `docs/plans/2026-04-16-garch-program-audit.md`
- `docs/plans/2026-04-16-garch-institutional-attack-plan.md`

---

## 1. Why this stage exists

The program has already established three things:

1. `garch_forecast_vol_pct` is look-ahead clean and structurally non-random in
   several natural regime families.
2. It does **not** currently justify a broad production-ready overlay claim on
   the validated shelf.
3. It overlaps materially with nearby volatility-state proxies, especially
   `atr_20_pct`, but that overlap has not yet been turned into a clean
   incremental-value verdict.

That means the next correct question is not:

- "what deployment map should we use?"
- "what random extra filter should we bolt on?"

It is:

> What exactly does `garch` add beyond the adjacent vol-state family, and in
> what role can that added information be used honestly?

---

## 2. Core question

The audit should classify `garch` into one of three buckets:

1. **Distinct**
   - adds information that survives conditioning on nearby vol proxies
2. **Complementary**
   - not fully distinct, but useful in combination with one or more nearby
     proxies for a mechanism-shaped state family
3. **Subsumed**
   - most apparent value is already captured by another simpler proxy

This classification must be made separately for:

- **state truth**
- **validated utility**

Those are not the same question.

### Scope boundary

This stage is a **local distinctness audit on pre-selected garch-anchored
families**, not a global winner-take-all proxy ranking.

That means:

- a `distinct` verdict means "distinct inside these already-audited families"
- a `subsumed` verdict means "mostly subsumed inside these already-audited
  families"
- it does **not** mean:
  - `garch` is globally superior across the whole book
  - `ATR`, `overnight`, or `atr_vel` have been globally defeated
  - a deployment doctrine is proved

Global proxy monetization remains a separate question and stays in the existing
native-utility / deployment workstreams.

---

## 3. What this stage must answer

### Q1 - State truth

On canonical populations, does `garch` continue to show the expected sign after
conditioning on:

- `atr_20_pct`
- `overnight_range_pct`
- `atr_vel_ratio`

### Q2 - Utility relevance

On the validated / live-like shelf, does `garch` improve utility after nearby
proxies are given a fair shot with their own native scaffolds?

### Q3 - Pairing role

If `garch` is not fully distinct, does it still belong in:

- `R3` size modulation
- `R7` confluence scoring
- `R8` allocator ranking

and **not** in:

- standalone directional signal discovery
- universal binary gate doctrine

---

## 4. Scope

### Included proxies

These are the only adjacent state proxies in scope for this stage:

- `atr_20_pct`
- `overnight_range_pct`
- `atr_vel_ratio`

Reason:
- all are already in the repo
- all are pre-entry and look-ahead clean
- all are already implicated by the existing structural work

Working prior on their roles:

- `garch_forecast_vol_pct`
  - conditional / latent volatility expectation
- `overnight_range_pct`
  - realized pre-session expansion already expressed before the trade window
- `atr_vel_ratio`
  - transition / acceleration state, i.e. whether volatility is actively
    changing into the session

So the audit must explicitly allow for the possibility that:

- `garch` is partly subsumed by `overnight_range_pct`
- `garch` is partly subsumed by `atr_vel_ratio`
- or the real value sits in one interaction:
  - latent-but-not-yet-realized
  - active acceleration
  - already-expanded and therefore lower-quality

### Excluded for this stage

- calendar-only flags as primary competitors
- gap flags as primary competitors
- any new exotic features
- any black-box latent factors

Those may appear as descriptive controls only, not as new competing state
families.

---

## 5. Evidence tiers

### Tier D1 - Canonical distinctness

Universe:
- canonical trade populations from `orb_outcomes` + `daily_features`

Question:
- what is the structural relation between `garch` and nearby proxies?

### Tier D2 - Validated utility distinctness

Universe:
- exact `validated_setups` populations only

Status note:
- this tier is still **research-provisional**, not production truth, because
  the current validated shelf was discovered during the temporary Mode B
  contamination window documented in `RESEARCH_RULES.md`

Question:
- does `garch` still help after nearby proxies are compared fairly?

### Rule

Tier D1 may classify the state relation.
Tier D2 may classify utility relevance.
Neither tier may by itself promote a deployment doctrine.

### Sacred-window rule

This stage may report pre-existing 2026 forward/OOS readouts only as
**descriptive drift context**. It may not use 2026 forward performance to choose
families, choose partners, rescue a weak verdict, or decide the classification.

---

## 6. Required test blocks

This stage is only complete if it covers all six blocks.

### Block B1 - Overlap table

For each target family:

- pooled trade count
- mean proxy levels
- pairwise correlations:
  - `garch` vs ATR
  - `garch` vs overnight
  - `garch` vs ATR velocity

Purpose:
- descriptive overlap only, not final classification

### Block B2 - Conditional sign persistence

For each target family and each proxy:

- does the `garch` sign persist inside:
  - proxy-high stratum
  - proxy-low stratum

Purpose:
- tell whether `garch` survives conditioning on the adjacent proxy state

Required conditioning conventions:

- `atr_20_pct`
  - use fixed high / low tails (`>=80`, `<=20`) as already used in the prior
    structural pass
- `overnight_range_pct`
  - use fixed high / low tails (`>=80`, `<=20`)
- `atr_vel_ratio`
  - do **not** force fake `70/30` percentiles
  - use canonical `atr_vel_regime` (`Expanding`, `Stable`, `Contracting`) as
    the primary conditioning surface
  - optional descriptive backup: within-family IS terciles (`P33`, `P67`) only
    if needed for continuity with the four-cell block

Thin-stratum rule:

- if a conditioned stratum does not have enough support to evaluate honestly,
  mark it `thin` / `n/a`
- do not treat missing support as survival or failure

### Block B3 - Four-cell mechanism decomposition

For each paired mechanism family:

- low proxy / low garch
- low proxy / high garch
- high proxy / low garch
- high proxy / high garch

Required readout:
- `N`
- `ExpR`
- `sr`
- dollar totals where relevant

Purpose:
- identify whether the value is:
  - marginal
  - shared
  - or interaction-like

Required proxy pairings:
- `garch` x `overnight_range_pct`
- `garch` x `atr_vel_ratio`
- `garch` x `atr_20_pct`

Pair-construction rule:

- `garch` / `overnight` / `atr_20` use fixed tails from the existing audited
  convention
- `garch` x `atr_vel_ratio` must use either:
  - `Contracting` / `not Contracting` as the primary binary split, or
  - pre-declared IS terciles derived within the family
- no ad hoc threshold search is allowed

Minimum-support rule:

- if any four-cell decomposition has fewer than `20` observations in a cell, it
  is descriptive only and cannot carry the final verdict on its own

### Block B4 - Fair utility comparison

Use the existing fair native-scaffold logic:

- `garch`
- `ATR`
- `overnight`

Purpose:
- determine whether `garch` contributes utility after each proxy earns its own
  natural map

Utility-comparison boundary:

- this block is for **solo-proxy utility comparison on the locked family set**
  plus reference to already-completed native utility audits
- it is **not** a new confluence search
- new pairwise / composite utility candidates belong to the later
  mechanism-pairing stage, not this stage

### Block B5 - Distinctness verdict

For each proxy and each main family, classify:

- `distinct`
- `complementary`
- `subsumed`
- `unclear`

This classification must be earned from:
- persistence
- four-cell decomposition
- utility comparison

### Block B6 - Role implication

Translate the distinctness verdict into the allowed role:

- `R3`
- `R7`
- `R8`
- or `no standalone role`

---

## 7. Main families to audit

The stage must stay anchored to the strongest already-audited families, not a
fresh broad search.

Initial family set:

- `COMEX_SETTLE high`
- `EUROPE_FLOW high`
- `TOKYO_OPEN high`
- `SINGAPORE_OPEN high`
- `LONDON_METALS high`
- `NYSE_OPEN low`

Reason:
- these were already identified in the family-framed audit and structural
  decomposition
- `NYSE_OPEN low` is kept partly as a hostile / counterexample family, not as a
  favorable deployment candidate

No expansion beyond this family set without a new hypothesis file.

---

## 8. What this stage must not do

1. It must not create a new deployment map.
2. It must not use profile/account geometry as evidence.
3. It must not discover new sessions.
4. It must not use 2026 forward data to choose mechanisms.
5. It must not turn pairwise correlation into a storytelling shortcut.

---

## 9. Decision rules

### Distinct

Classify `garch` as **distinct** relative to a proxy if:

1. sign persistence survives in both overall and at least one conditioned stratum
2. four-cell decomposition shows non-trivial lift unique to a `garch` state
3. fair utility comparison shows `garch` or a `garch`-containing family still
   matters economically

### Complementary

Classify `garch` as **complementary** if:

1. overlap is material
2. `garch` alone is not clearly dominant
3. but interaction-like or combination utility remains plausible and survives

### Subsumed

Classify `garch` as **subsumed** if:

1. conditioned sign mostly disappears
2. four-cell decomposition shows no meaningful marginal contribution
3. fair utility comparison is dominated by a simpler proxy

### Unclear

Use **unclear** if the signs conflict across the blocks or sample support is
too weak to decide honestly.

### Minimum evidence for a non-unclear verdict

Do not issue `distinct`, `complementary`, or `subsumed` unless:

1. at least two of the three core evidence blocks are non-thin:
   - sign persistence
   - four-cell decomposition
   - utility comparison
2. none of the supporting blocks rely on 2026 forward data for the decision
3. the verdict remains local to the locked family set

---

## 10. Expected outcomes and next actions

### If mostly distinct

Next stage:
- one mechanism-shaped pairing audit on validated populations

### If mostly complementary

Next stage:
- one locked confluence / conditioner pairing audit

Expected first pairing priority:
1. `garch` x `overnight_range_pct`
2. `garch` x `atr_vel_ratio`
3. `garch` x `atr_20_pct`

Reason:
- `overnight` and `atr_vel` are the most plausible places where trapped or
  constrained profit may be hiding because they encode:
  - realized overnight expression
  - active transition into the session

### If mostly subsumed

Next stage:
- demote `garch` from center stage
- shift the program focus to the stronger proxy family

### If mixed by family

Next stage:
- allow family-specific role mapping
- do not force one universal doctrine

---

## 11. Implementation map

Expected new artifacts:

- plan / design:
  - `docs/plans/2026-04-16-garch-state-distinctness-design.md`
- hypothesis:
  - `docs/audit/hypotheses/2026-04-16-garch-state-distinctness-audit.yaml`
- research script:
  - `research/garch_state_distinctness_audit.py`
- result:
  - `docs/audit/results/2026-04-16-garch-state-distinctness-audit.md`

Expected code reuse:

- `research/garch_structural_decomposition.py`
- `research/garch_additive_sizing_audit.py`
- `research/garch_proxy_native_sizing_audit.py`
- `research/garch_regime_family_audit.py`

---

## 12. Recommendation

This is the correct next research stage.

It is broader than deployment translation but narrower than another full search.
It directly answers the live methodological risk:

> are we over-focusing on `garch` itself, or are we correctly identifying a
> usable state-family input?

Only after this stage is complete should the program move into:
- mechanism-shaped pairing
- or revised allocator translation
