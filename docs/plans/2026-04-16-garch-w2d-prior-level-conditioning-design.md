# Garch W2d Prior-Level Conditioning Design

**Date:** 2026-04-16  
**Status:** DEMOTED_AFTER_FEASIBILITY_CHECK  
**Purpose:** choose the next `garch` mechanism-family stage without drifting into
ad hoc queue-hopping. This stage decides whether prior-day level context is the
correct next pairing family for `garch` on the validated shelf.

---

## 1. Design question

After `W2c`, the active carried mechanism is:

- `M2`: `garch_high` + local active-transition partner state

The next open queue items are:

1. prior-day levels / prior-day zone position
2. prior-day realized range
3. prior-session carry / same-day resolved prior-session context

This design answers which of those should be tested next under the same
institutional constraints:

- canonical truth only
- validated shelf only for utility
- no deployment inference
- no brute-force context menu search

---

## 2. Queue decision

### 2.1 Initial choice

The initial choice was to test prior-day level context next.

### 2.2 Why this queue item wins

Because the repo already contains stronger and cleaner evidence for prior-day
level context than for prior-session carry:

- `docs/institutional/mechanism_priors.md` has explicit, pre-existing mechanism
  priors for PDH / PDL / pivot behavior.
- `research/output/pdh_pdl_signal_findings.md` shows real, pre-existing
  directional effects for PDH / PDL gap-taken states.
- `docs/audit/hypotheses/2026-04-15-prior-day-zone-positional-features-orb.md`
  is already a serious pre-registration around prior-day level geometry.
- `docs/audit/hypotheses/2026-04-15-comprehensive-deployed-lane-gap-audit.md`
  records actual deployed-lane relevance for level context.
- Broad session-cascade has already been marked NO-GO in the archive, so
  prior-session carry must remain narrower and more defensive when reopened.

### 2.3 Why prior-session carry is not next

Because the repo’s current state on prior-session carry is:

- broad session-cascade = historically demoted / NO-GO
- only narrow same-day resolved prior-session context remains admissible
- that narrower family is still worth doing later, but it has weaker current
  repo evidence than prior-day levels

So the correct order is:

1. prior-day level conditioning
2. then, if still warranted, narrow prior-session carry

---

## 3. Mechanism prior for W2d

The combined mechanism is:

- `garch_high` = elevated expected volatility state before the target session
- prior-day levels = known liquidity pools and value anchors before the trade

The hypothesis family is **not** "levels work because garch is high."

It is narrower:

- when volatility state is elevated, level-based friction or displacement may
  matter more, so the level context may become a stronger conditioner of
  breakout quality

That gives three admissible roles:

1. **reinforcement**
   - `garch_high` makes an already-favorable level context more favorable
2. **penalty amplification**
   - `garch_high` makes an already-hostile level context more hostile
3. **no added value**
   - level context does not add enough beyond `garch_high` alone

This is still a conditioning test, not a new directional signal discovery.

---

## 4. Admissible feature set

### 4.1 Allowed canonical inputs

Only these pre-trade-safe fields may enter:

- `garch_forecast_vol_pct`
- `prev_day_high`
- `prev_day_low`
- `prev_day_close`
- `atr_20`
- `gap_type`
- session ORB midpoint from `daily_features`:
  - `(orb_{session}_high + orb_{session}_low) / 2`

### 4.2 Allowed prior-level states

Only previously-grounded local level contexts are admissible:

- `NEAR_PDH`
- `NEAR_PIVOT`
- `BELOW_PDL`
- `INSIDE_PDR`

No full 8-feature sweep is allowed in this stage.

### 4.3 Banned inputs for W2d

Do **not** use:

- `break_dir`
- `break_ts`
- `break_delay_min`
- `double_break`
- `took_pdh_before_1000`
- `took_pdl_before_1000`
- `overnight_took_pdh`
- `overnight_took_pdl`
- `overnight_range_pct`
- any prior-session resolved outcome fields

Those belong to separate stages or are explicitly blacklisted here.

---

## 5. Local scope

This stage is **validated shelf only**.

It is also **local-family only**. The first pass is restricted to the already
documented prior-level contexts that have live repo support:

1. `BELOW_PDL` long on `US_DATA_1000`
2. `INSIDE_PDR` on `US_DATA_1000` if validated rows exist
3. `NEAR_PDH` short on `NYSE_CLOSE`
4. `NEAR_PIVOT` long on sessions where validated rows exist and prior evidence
   is already documented

If a candidate family has no validated rows with minimum support, it is logged
as `not_testable_here`, not rescued by broadening scope.

---

## 6. Test structure

For each admissible local family:

1. compute base family expectancy
2. compute level-only expectancy
3. compute `garch_high` expectancy
4. compute conjunction expectancy
5. compare:
   - conjunction vs base
   - conjunction vs `garch_high`
   - conjunction vs level-only

Required verdict classes:

- `complementary_pair`
- `garch_only`
- `level_only`
- `unclear`
- `not_testable_here`

This stage is about distinctness / complementarity, not deployment ranking.

---

## 7. Support and guardrails

- minimum total per evaluated validated row: `N >= 50`
- minimum conjunction support per local family summary: `N_conj >= 30`
- descriptive 2026 OOS only
- raw joined-path recheck required on every carried family summary
- if a report table becomes ambiguous because multiple validated rows share the
  same instrument / filter / RR family, the disambiguating field must be
  carried explicitly into the report

---

## 8. Failure modes

This stage fails if any of these happen:

1. it silently reopens a full prior-day feature sweep
2. it mixes prior-day level context with prior-session carry
3. it uses banned time-dependent fields
4. it treats lack of validated support as permission to widen scope
5. it interprets descriptive OOS as promotion evidence
6. it jumps from validated utility to deployment doctrine

---

## 9. Deliverable

Deliverables for W2d:

- locked hypothesis file
- one research script
- one validated-shelf result note
- one raw-check note inside the result

No config change. No runtime implementation. No allocator translation.

---

## 10. Feasibility correction

The validated-shelf overlap check invalidated this as the **immediate** next
branch.

Observed validated overlap at the time of review:

- `EUROPE_FLOW`: present
- `US_DATA_1000`: absent
- `NYSE_CLOSE`: absent
- `LONDON_METALS`: effectively absent for meaningful validated utility

That means the broad prior-level conditioning program is still a valid **future
queue item**, but not the right next validated-shelf execution branch after
`W2c`.

Corrected implication:

- keep this family in the explicit queue
- do **not** execute it next under a validated-shelf-only boundary
- prefer the narrower prior-session carry branch, which actually binds to the
  current validated shelf
