# Garch W2c M2 Validated Utility Design

**Date:** 2026-04-16  
**Status:** ACTIVE DESIGN  
**Purpose:** carry the surviving `M2` mechanism into one stricter validated-shelf
utility stage, using only the partner-state representation choices justified by
the completed provenance audit.

**Authority chain:**
- `CLAUDE.md`
- `RESEARCH_RULES.md`
- `TRADING_RULES.md`
- `docs/institutional/pre_registered_criteria.md`
- `docs/plans/2026-04-16-garch-institutional-attack-plan.md`
- `docs/plans/2026-04-16-garch-w2-mechanism-pairing-design.md`
- `docs/plans/2026-04-16-garch-partner-state-provenance-design.md`
- `docs/audit/results/2026-04-16-garch-w2-mechanism-pairing-audit.md`
- `docs/audit/results/2026-04-16-garch-partner-state-provenance-audit.md`

---

## 1. Why this stage exists

W2 showed that `M2` active transition is the stronger `garch`-anchored
mechanism family on the validated shelf.

W2b then sharpened the representation question:

- some families support the current canonical state
- some are only `neighbor_stable`
- one family (`SINGAPORE_OPEN_high`) clearly wants a stricter locked alternate

The next correct question is therefore:

> If we freeze a conservative family-local M2 representation choice rule, does
> the resulting `garch + M2` mechanism still improve validated-family utility
> over both the base family and `garch_high` alone?

This is a utility stage, not a search stage.

---

## 2. Scope boundary

This stage is:

- validated shelf only
- exact canonical re-joins only
- M2 only
- family-local but conservatively chosen

This stage is not:

- a new partner-state search
- a deployment or allocator stage
- a profile translation stage
- a prior-day level or prior-session carry stage

### Universe

- `validated_setups` only
- canonical joins to:
  - `orb_outcomes`
  - `daily_features`

### Time boundary

- pre-2026 data drives interpretation
- 2026 remains descriptive only

---

## 3. Conservative representation-selection rule

This rule is frozen from W2b. No new selection logic is allowed here.

### Rule

1. If W2b verdict for a family was `neighbor_stable`, keep the **current W2
   canonical state**.
2. If W2b verdict was `supported_current`, keep the current state.
3. If W2b verdict was `alternate_better`, switch only to the named locked
   alternate representation.
4. If W2b verdict was `weak_mechanism` or `unclear`, do not carry the family
   into the promoted local-M2 set.

### Resulting family map

- `COMEX_SETTLE_high`:
  - use `ATRVEL_EXPANDING`
- `EUROPE_FLOW_high`:
  - use `ATRVEL_EXPANDING`
- `TOKYO_OPEN_high`:
  - use `ATRVEL_EXPANDING`
- `SINGAPORE_OPEN_high`:
  - use `ATRVEL_GE_110`
- `LONDON_METALS_high`:
  - excluded, no supported validated-family W2 claim

This rule intentionally sacrifices some local optimization to reduce overfit
risk.

---

## 4. Exact question

For each carried family:

1. What is the base validated-family ExpR?
2. What is the `garch_high` utility alone?
3. What is the chosen local-M2 conjunction utility?
4. Does the conjunction beat:
   - base family
   - `garch_high` alone
5. Is the family-level conjunction concentrated in one cell, or distributed
   across the validated family?

---

## 5. Metrics

### Required per family

- `cells`
- `N_total`
- `N_garch`
- `N_conj`
- `base_exp`
- `garch_exp`
- `conj_exp`
- `delta_garch_vs_base`
- `delta_conj_vs_base`
- `delta_conj_vs_garch`
- weighted `partner_inside_garch` support share
- weighted `garch_inside_partner` support share
- `max_conj_cell_share`
- `N_conj_oos`
- `conj_exp_oos`

### Utility interpretation

This stage is about validated utility, so the primary comparison is:

- `conj_exp - garch_exp`

Secondary:

- `conj_exp - base_exp`

---

## 6. Thin-cell and concentration rules

- family pooled conjunction requires `N_conj >= 30`
- valid conditional comparison requires both sides `>= 10`
- if `max_conj_cell_share > 0.50`, mark the family as concentration-risk
- concentration-risk families may not be promoted as clean local utilities

---

## 7. Verdicts

Allowed family verdicts:

- `carry_local_m2`
  - conjunction beats base and `garch_high`
  - partner-inside-garch support is positive
  - no severe one-cell concentration
- `partial_local_m2`
  - conjunction beats base but not clearly `garch_high`, or carries
    concentration risk
- `demote_local_m2`
  - conjunction does not beat `garch_high` or support is weak
- `unclear`
  - thin or mixed

Allowed stage-level outcomes:

- `M2_carry`
- `M2_partial`
- `M2_demote`

---

## 8. Queue discipline

Do not widen this stage into:

- `M1`
- prior-day levels
- prior-day range
- prior-session carry
- deployment allocator translation

Those remain separate queued families.
