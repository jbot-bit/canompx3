# Garch W2 Mechanism Pairing Design

**Date:** 2026-04-16  
**Status:** ACTIVE DESIGN  
**Purpose:** define the next validated-shelf-only stage after W1 state
distinctness: test whether one locked mechanism pairing actually improves
validated setup quality without drifting into random filter stacking or
deployment overclaiming.

**Authority chain:**
- `CLAUDE.md`
- `RESEARCH_RULES.md`
- `TRADING_RULES.md`
- `docs/institutional/pre_registered_criteria.md`
- `docs/institutional/mechanism_priors.md`
- `docs/institutional/regime-and-rr-handling-framework.md`
- `docs/plans/2026-04-16-garch-institutional-attack-plan.md`
- `docs/plans/2026-04-16-garch-program-audit.md`
- `docs/audit/results/2026-04-16-garch-state-distinctness-audit.md`

---

## 1. Why this stage exists

W1 answered the structural question:

- `garch` is not globally dominant
- `garch` is not dead
- `atr_20_pct` remains the strongest overlap / subsumption risk
- `overnight_range_pct` and `atr_vel_ratio` are the most plausible mechanism
  partners in the locked family set

That means the next correct question is not:

- "what deployment map should we use?"
- "what other filter should we bolt on?"

It is:

> On the validated shelf only, does one realistic `garch`-anchored mechanism
> pairing improve setup quality in a way that survives explicit marginal and
> conditional comparison?

---

## 2. Scope boundary

This stage is deliberately narrow.

It is:

- a **validated-shelf mechanism test**
- a **local family study**
- a **pairing utility** question

It is not:

- a global feature search
- a deployment / allocator / profile translation test
- a new session discovery pass
- a production doctrine decision

### Data boundary

- Universe: `validated_setups` only
- Status: still research-provisional, not production truth, per
  `RESEARCH_RULES.md`
- Exact filter semantics only; no `live_config`, no docs-as-truth shortcuts

### Sacred-window boundary

- IS / discovery logic must be based on pre-2026 data only
- 2026 forward/OOS may be reported only as descriptive context
- 2026 OOS may not be used to choose mechanism, family, or verdict

---

## 3. Local family scope

W2 stays on the already-audited high-garch family surface from W1.

Locked family set:

- `COMEX_SETTLE high`
- `EUROPE_FLOW high`
- `TOKYO_OPEN high`
- `SINGAPORE_OPEN high`
- `LONDON_METALS high`

Excluded:

- `NYSE_OPEN low`

Reason:

- W2 is testing favorable high-garch mechanism pairings
- `NYSE_OPEN low` remains hostile / mixed and is not the correct surface for
  these two high-side mechanisms

### Non-pigeonholing rule

The stage must remain open to four outcomes:

1. mechanism survives cleanly
2. `garch` survives but the partner adds little
3. partner dominates and `garch` is not needed
4. both fail and the family should be demoted

This stage is not allowed to assume the pairing works.

---

## 4. Mechanisms in scope

Only two mechanisms are allowed in this stage.

### M1 — Latent expansion

**Hypothesis:** when `garch` is high but overnight realized expansion is not
already high, event-session breakout quality may be better because latent
conditional volatility has not yet been fully expressed.

Canonical feature definitions:

- `garch_high`:
  - `garch_forecast_vol_pct >= 70`
- `overnight_not_high`:
  - `overnight_range_pct < 80`
- descriptive overnight buckets:
  - `overnight_low`: `overnight_range_pct <= 20`
  - `overnight_mid`: `20 < overnight_range_pct < 80`
  - `overnight_high`: `overnight_range_pct >= 80`

Primary pairing state:

- `garch_high AND overnight_not_high`

Primary falsification comparator:

- `garch_high AND overnight_high`

### M2 — Active transition

**Hypothesis:** when `garch` is high and volatility is actively expanding into
the session, breakout quality may improve because the state is not merely
latent; it is already accelerating.

Canonical feature definitions:

- `garch_high`:
  - `garch_forecast_vol_pct >= 70`
- `atr_vel_favorable`:
  - `atr_vel_regime = 'Expanding'`
- descriptive comparison states:
  - `Stable`
  - `Contracting`

Primary pairing state:

- `garch_high AND atr_vel_regime = 'Expanding'`

Primary falsification comparator:

- `garch_high AND atr_vel_regime != 'Expanding'`

### Open-outcome rule

These two mechanisms are the only **candidate explanations** being tested in
W2. They are not privileged conclusions.

The stage must remain open to these mutually exclusive results:

1. one mechanism survives
2. both mechanisms survive in different local families
3. neither survives and `garch` remains mostly standalone / local-distinct
4. neither survives and the partner state dominates, implying the broader
   vol-state family is the real object
5. all readings stay too thin / mixed and W2 ends `unclear`

W2 fails if the implementation behaves like it is trying to rescue one favored
mechanism.

---

## 5. Exact questions to answer

For each mechanism and each local family:

### Q1 — Baseline quality

How good is the validated family with no extra mechanism pairing?

### Q2 — Marginal garch effect

Does `garch_high` improve the family relative to all other days?

### Q3 — Marginal partner effect

Does the partner state improve the family relative to all other days?

### Q4 — Conditional partner effect inside garch

Inside `garch_high` days, does the partner improve quality?

Examples:

- M1:
  - compare `garch_high & overnight_not_high`
    vs `garch_high & overnight_high`
- M2:
  - compare `garch_high & atr_vel_expanding`
    vs `garch_high & not_expanding`

### Q5 — Conditional garch effect inside partner

Inside the partner-favorable days, does `garch_high` improve quality?

This prevents the stage from mistaking a strong partner for a true
`garch`-anchored mechanism.

### Q6 — Conjunction quality

What is the actual trade quality of the conjunction subset?

### Q7 — Distinct vs complementary

Is the result better described as:

- `garch_distinct`
- `complementary_pair`
- `partner_dominant`
- `unclear`

---

## 6. Metrics

### Required per family / mechanism

- `N_total`
- `ExpR_base`
- `N_garch_high`, `ExpR_garch_high`
- `N_partner_favorable`, `ExpR_partner_favorable`
- `N_conjunction`, `ExpR_conjunction`
- `SR_conjunction`
- `conjunction_fire_pct`

### Required pair-comparison metrics

- marginal garch lift:
  - `ExpR(garch_high) - ExpR(not garch_high)`
- marginal partner lift:
  - `ExpR(partner_favorable) - ExpR(not partner_favorable)`
- conditional partner lift inside garch:
  - `ExpR(garch_high & partner_favorable) - ExpR(garch_high & partner_unfavorable)`
- conditional garch lift inside partner:
  - `ExpR(garch_high & partner_favorable) - ExpR(not garch_high & partner_favorable)`

### Support metrics

- support cell count by family
- support fraction by family
- year sign support where enough data exists

### OOS / 2026 metrics

- descriptive only:
  - OOS conjunction ExpR
  - OOS conditional sign match if support exists

These cannot decide the verdict.

---

## 7. Sample and support rules

### Cell inclusion

- exact validated cell only
- `N_total >= 50`

### Thin-state rules

- any primary group with `< 10` observations is invalid for that comparison
- any conjunction bucket with `< 30` observations is descriptive only

### Non-unclear verdict requirement

To issue `garch_distinct`, `complementary_pair`, or `partner_dominant` for a
family-mechanism result, require:

1. conjunction `N >= 30`
2. at least one valid conditional comparison
3. at least one valid marginal comparison
4. no contradiction strong enough to reverse the claimed interpretation

Otherwise verdict = `unclear`.

---

## 8. Baseline comparison rules

Baseline must be the exact validated family as currently defined.

Not allowed:

- comparing against a deployment replay map
- comparing against a profile-specific route
- comparing against a newly invented neutral surface

This is a setup-quality question, not a portfolio allocation question.

---

## 9. Distinct vs complementary decision rules

### `garch_distinct`

Use only if:

1. `garch_high` retains positive marginal effect
2. partner conditional effect is weak / mixed / unnecessary
3. conjunction does not materially improve over `garch_high` alone

Interpretation:

- keep `garch` central
- do not promote the partner as necessary

### `complementary_pair`

Use only if:

1. conjunction quality is positive
2. partner conditional effect inside `garch_high` is positive
3. conjunction improves meaningfully over at least one marginal component

Interpretation:

- pair is economically useful
- candidate for next-stage validated utility note

### `partner_dominant`

Use only if:

1. partner marginal effect is stronger than `garch`
2. conditional garch effect inside partner is weak / mixed
3. conjunction does not justify keeping `garch` central

Interpretation:

- do not center the program on `garch` for this family/mechanism

### `unclear`

Use if:

- sample is too thin
- conditional and marginal reads conflict
- or the conjunction does not improve the story cleanly

---

## 10. Failure modes to defend against

1. **Subset illusion**
   - conjunction looks good only because it isolates a tiny subset
2. **Partner dominance misread**
   - partner does the work, `garch` gets the credit
3. **One-cell concentration**
   - pooled family result is driven by one validated cell
4. **OOS leakage by narration**
   - descriptive 2026 read is allowed to choose the winner
5. **Profile creep**
   - mechanism result gets reinterpreted as deployment doctrine
6. **Proxy-threshold abuse**
   - `atr_vel` forced into fake percentile logic

---

## 11. Promotion and kill criteria

### Promote to next mechanism utility note only if

1. at least one local family returns `complementary_pair` or a strong
   `garch_distinct` result
2. conjunction support is not thin
3. result is not carried by a single validated cell
4. no strong contradiction appears in the required falsification comparator

### Kill or demote if

1. both mechanisms are mostly `unclear` or `partner_dominant`
2. conjunctions are thin across the board
3. conjunction does not improve over marginals in a meaningful way
4. the stage drifts into new search instead of testing the locked mechanisms

---

## 12. Expected artifacts

- plan:
  - `docs/plans/2026-04-16-garch-w2-mechanism-pairing-design.md`
- hypothesis:
  - `docs/audit/hypotheses/2026-04-16-garch-w2-mechanism-pairing.yaml`
- script:
  - `research/garch_w2_mechanism_pairing_audit.py`
- result:
  - `docs/audit/results/2026-04-16-garch-w2-mechanism-pairing-audit.md`

---

## 13. Recommendation

This is the correct next stage.

It keeps the program open-minded without reverting to menu-browsing:

- it tests two realistic mechanism families
- it keeps `garch` honest against partner dominance
- it stays on the validated shelf
- and it blocks premature deployment conclusions
