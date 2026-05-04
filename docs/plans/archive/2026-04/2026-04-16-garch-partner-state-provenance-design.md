---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Garch Partner-State Provenance Design

**Date:** 2026-04-16  
**Status:** ACTIVE DESIGN  
**Purpose:** verify whether the current W2 partner-state definitions are
principled representations of the mechanism, or merely convenient cutoffs that
happen to look good once.

**Authority chain:**
- `CLAUDE.md`
- `RESEARCH_RULES.md`
- `TRADING_RULES.md`
- `docs/institutional/pre_registered_criteria.md`
- `docs/institutional/mechanism_priors.md`
- `docs/plans/2026-04-16-garch-program-audit.md`
- `docs/plans/2026-04-16-garch-state-distinctness-design.md`
- `docs/plans/2026-04-16-garch-w2-mechanism-pairing-design.md`
- `docs/audit/results/2026-04-16-garch-state-distinctness-audit.md`
- `docs/audit/results/2026-04-16-garch-w2-mechanism-pairing-audit.md`
- `pipeline/build_daily_features.py`

---

## 1. Why this stage exists

W2 established an honest local read:

- `M2` active transition, `high garch + atr_vel Expanding`, survived better than
  `M1`
- `M1` latent expansion, `high garch + overnight not high`, did not emerge as a
  broad complementary winner

That is **not** enough to conclude that:

- `atr_vel_regime == Expanding` is the uniquely right partner state
- `overnight_range_pct < 80` is the right realized-overnight representation
- static ATR level is irrelevant

This stage exists to answer a narrower institutional question:

> Are the current partner states canonically grounded and locally robust, or
> are they just convenient cutoffs that need to be tightened, widened, or
> replaced before further mechanism or deployment work?

---

## 2. Institutional framing

This is a **representation / provenance** audit, not a new search.

It is grounded in three ideas that match both the repo doctrine and the
literature already gathered:

1. volatility-state variables should be tested as **state representations**, not
   treated as magical exact thresholds
2. neighboring cutoffs matter; if the result disappears with a tiny threshold
   move, the state is fragile
3. latent-vol and realized-vol measures are often best handled as complementary
   objects, not as rivals where one scalar must be "the truth"

This stage therefore tests a **small locked set** of neighboring or alternative
representations for the two already-locked W2 mechanisms.

It does **not**:

- discover new families
- open new sessions
- compare deployment maps
- revive random filter stacking
- use 2026 data to select winners

---

## 3. Scope boundary

### Universe

- `validated_setups` only
- exact canonical re-joins to:
  - `orb_outcomes`
  - `daily_features`

### Family boundary

Same W2 locked family set:

- `COMEX_SETTLE_high`
- `EUROPE_FLOW_high`
- `TOKYO_OPEN_high`
- `SINGAPORE_OPEN_high`
- `LONDON_METALS_high`

### Time boundary

- pre-2026 data drives all interpretation
- 2026 OOS may appear as descriptive context only
- 2026 may not choose the preferred representation

---

## 4. Canonical partner-state definitions

Before testing variants, freeze what is already canonical in the repo:

### Garch anchor

- `garch_high := garch_forecast_vol_pct >= 70`

This stays fixed in this stage. The question is not the garch cutoff; it is the
partner-state representation.

### Overnight realized state

- `overnight_range_pct`
- built as the prior-only percentile rank of `overnight_range` against the
  prior `60` trading days
- min prior support `20`

### ATR level state

- `atr_20_pct`
- built as the prior-only percentile rank of `atr_20` against the prior
  `252` trading days
- min prior support `60`

### ATR transition state

- `atr_vel_ratio := atr_20 / avg(prior 5 atr_20)`
- `atr_vel_regime`
  - `Expanding` if `atr_vel_ratio > 1.05`
  - `Contracting` if `atr_vel_ratio < 0.95`
  - else `Stable`

This matters: `atr_vel_regime` is **not** an ad hoc threshold from the garch
project. It is a canonical state variable already defined in the feature
pipeline.

---

## 5. Representations in scope

Only the following representations are allowed.

### R1 — Overnight representations for M1 latent expansion

The mechanism claim is:

> latent conditional volatility may be useful when realized overnight expansion
> has not already fully expressed it.

Allowed partner-favorable encodings:

1. `OVN_NOT_HIGH_60`
   - `overnight_range_pct < 60`
2. `OVN_NOT_HIGH_70`
   - `overnight_range_pct < 70`
3. `OVN_NOT_HIGH_80`
   - `overnight_range_pct < 80`
   - this is the current W2 representation
4. `OVN_MID_ONLY`
   - `20 < overnight_range_pct < 80`
   - tests whether "not high" only works because we are really excluding both
     extreme quiet and extreme overnight expansion

Why this set is honest:

- it checks small neighboring cutoffs
- it checks the only obvious structural alternative for the same mechanism
- it does not explode into an arbitrary threshold grid

### R2 — ATR representations for M2 active transition

The mechanism claim is:

> high garch is more useful when volatility is actively expanding into the
> session, not merely elevated in static level terms.

Allowed partner-favorable encodings:

1. `ATRVEL_EXPANDING`
   - `atr_vel_regime == 'Expanding'`
   - current W2 representation
2. `ATRVEL_GE_100`
   - `atr_vel_ratio >= 1.00`
3. `ATRVEL_GE_105`
   - `atr_vel_ratio >= 1.05`
   - numeric equivalent neighborhood around the canonical categorical split
4. `ATRVEL_GE_110`
   - `atr_vel_ratio >= 1.10`
5. `ATR_PCT_GE_70`
   - `atr_20_pct >= 70`
6. `ATR_PCT_GE_80`
   - `atr_20_pct >= 80`

Why this set is honest:

- it tests whether the edge is really in **transition** or merely in **high
  static ATR**
- it checks immediate numeric neighbors around the canonical `1.05` expansion
  split
- it avoids a large ATR threshold sweep

---

## 6. Exact questions

For each family and each allowed representation:

1. Does the representation improve quality inside `garch_high` relative to its
   complementary state?
2. Is the conjunction quality better than the base validated family?
3. Does the current W2 representation remain among the best-supported reads?
4. Are nearby thresholds directionally stable?
5. Is there evidence that the mechanism should be represented by:
   - current canonical state
   - a neighboring threshold band
   - a different representation entirely
   - no useful representation

---

## 7. Metrics

### Required per family × representation

- `cells`
- `N_total`
- `N_conj`
- `base_exp`
- `conj_exp`
- `conj_exp_minus_base`
- `support_cells_partner_inside_garch`
- `valid_cells_partner_inside_garch`
- `support_cells_garch_inside_partner`
- `valid_cells_garch_inside_partner`
- `support_share_partner_inside_garch`
- `support_share_garch_inside_partner`

### Required representation comparison outputs

- best representation by `conj_exp`
- best representation by conditional support share
- whether the current W2 representation is:
  - `supported_current`
  - `neighbor_stable`
  - `alternate_better`
  - `weak_mechanism`
  - `unclear`

---

## 8. Support / thin-cell rules

- include cell only if `N_total >= 50`
- side-by-side comparison valid only if both sides have `>= 10` rows
- pooled conjunction interpretation requires `N_conj >= 30`
- non-`unclear` representation verdict requires:
  - at least one valid conditional comparison
  - pooled conjunction support not thin
  - no reliance on 2026 context

---

## 9. Distinctness and complementarity rule

This stage does **not** rediscover W1 or W2. It only asks whether the chosen
representation is credible.

Interpretation hierarchy:

1. `supported_current`
   - current W2 representation remains among the best-supported reads
2. `neighbor_stable`
   - nearby thresholds look similar, so the mechanism seems real but exact
     cutoff is not unique
3. `alternate_better`
   - another locked representation is materially better than the current W2
     state
4. `weak_mechanism`
   - no representation adds enough to justify carrying the mechanism forward
5. `unclear`
   - too mixed or too thin

No representation may be promoted to deployment logic from this stage.

---

## 10. Prior-level / prior-session queue

This stage is **not** the place to re-open every prior-context idea at once.
But the queue must stay explicit so nothing gets lost.

Outstanding mechanism families to revisit later, separately:

1. prior-day level interaction
   - PDH / PDL / pivot type context
2. prior-day realized range interaction
   - `prev_day_range / atr_20`
3. prior-session carry / cascade
   - earlier-session state influencing later-session quality

These remain valid future mechanism candidates, but they must get their own
locked hypothesis files. They are not allowed to creep into this provenance
audit.

---

## 11. Promotion / kill logic

### Promote to next stage only if

- one or more families show `supported_current`, `neighbor_stable`, or
  `alternate_better`
- support is not driven by a single tiny conjunction pocket
- result does not depend on 2026 context

### Kill / demote if

- most families are `weak_mechanism` or `unclear`
- current W2 representation collapses and no alternative representation survives
- result is unstable across immediate neighboring cutoffs

---

## 12. What this stage should answer

After this run, we should be able to state honestly:

- whether `atr_vel Expanding` is the right way to represent the M2 partner, or
  whether static ATR or a neighboring velocity cutoff is better
- whether `overnight_range_pct < 80` was too loose, too broad, or directionally
  wrong for M1
- whether the current W2 mechanism results are grounded enough to carry
  forward, or should be demoted before any further translation work
