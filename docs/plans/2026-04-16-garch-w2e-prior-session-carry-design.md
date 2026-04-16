# Garch W2e Prior-Session Carry Design

**Date:** 2026-04-16  
**Status:** LOCKED DESIGN  
**Purpose:** define the next executable `garch` mechanism-family stage after the
validated-shelf feasibility check demoted broad prior-level conditioning as the
immediate next branch.

---

## 1. Why this is next

After `W2c`, the next queue item must satisfy **both**:

1. mechanism grounding already present in the repo
2. actual validated-shelf overlap

The feasibility check showed:

- broad prior-day level conditioning does **not** have enough current
  validated-shelf overlap to be the immediate next branch
- narrow prior-session carry **does** have validated-shelf overlap through
  `EUROPE_FLOW`

That makes the next honest branch:

- **same-day prior-session carry for `LONDON_METALS -> EUROPE_FLOW`**

This is consistent with:

- `docs/audit/hypotheses/2026-04-11-cross-session-context-audit.yaml`
- `docs/plans/2026-04-11-cross-session-state-round3-memo.md`

---

## 2. Critical safety correction

This stage must **not** use static session order.

Earlier ML work found a seasonal chronology problem for
`LONDON_METALS <-> EUROPE_FLOW`. That means this stage is admissible only if it
uses:

1. dynamic target-session start timing
2. actual prior trade `exit_ts`
3. an explicit check that the prior `LONDON_METALS` trade is fully resolved
   before the target `EUROPE_FLOW` session starts

If this timing cannot be enforced per row, the stage is invalid.

---

## 3. Mechanism prior

This is a state-conditioning test, not a new signal search.

The mechanism family is:

- prior session resolves in a meaningful state
- target session inherits or fights that state
- `garch_high` may amplify either continuation or hostility

For `LONDON_METALS -> EUROPE_FLOW`, the repo evidence is narrower than the US
handoffs. So this stage stays narrow:

- primary hostile-state test
- optional continuation-state comparison

---

## 4. Local scope

### 4.1 Universe

- validated shelf only
- target session: `EUROPE_FLOW`
- prior session: `LONDON_METALS`
- same trading day

### 4.2 Instruments

Only instruments with actual validated `EUROPE_FLOW` rows at run time.

Current feasibility snapshot:

- `MNQ`: meaningful overlap
- `GC`: one validated row only, likely too thin for interpretation

No broadening to non-validated rows is allowed.

---

## 5. Admissible prior-session states

Only the following prior-session states are allowed:

1. `PRIOR_WIN_ALIGN`
   - prior `LONDON_METALS` trade outcome = win
   - prior trade direction aligned with target `EUROPE_FLOW` breakout direction

2. `PRIOR_WIN_OPPOSED`
   - prior `LONDON_METALS` trade outcome = win
   - prior trade direction opposed to target `EUROPE_FLOW` breakout direction

The current repo memo suggests `PRIOR_WIN_OPPOSED` is the stronger live watchlist
branch for this handoff, so that is the primary test.

No other sibling states are allowed in this stage.

---

## 6. Garch role

`garch` stays in the same role as W2/W2c:

- local state-family input
- not standalone direction signal

The specific test is whether:

- `garch_high` improves the usefulness of a carry state
- or whether the carry state dominates and `garch` adds little

---

## 7. Exact row validity rules

A target `EUROPE_FLOW` trade row is valid only if:

1. there is a same-symbol `LONDON_METALS` trade on the same trading day
2. the prior trade has non-null:
   - `entry_ts`
   - `exit_ts`
   - `outcome`
3. the prior trade `exit_ts` is strictly earlier than the dynamically-resolved
   `EUROPE_FLOW` session start timestamp for that trading day
4. the target row has non-null `garch_forecast_vol_pct`

If any rule fails, that row is excluded rather than inferred.

---

## 8. Comparisons

For each admissible local family and prior-session state:

1. base family expectancy
2. `garch_high` expectancy
3. carry-state expectancy
4. conjunction expectancy:
   - `garch_high AND carry_state`

Primary comparisons:

- conjunction vs base
- conjunction vs `garch_high`
- conjunction vs carry-state alone

Because one state is expected hostile, the stage must support both:

- positive complementary carry
- negative complementary veto

---

## 9. Verdicts

Allowed verdicts:

- `carry_take_pair`
- `carry_veto_pair`
- `garch_only`
- `carry_only`
- `unclear`
- `not_testable_here`

No deployment or allocator verdict is allowed here.

---

## 10. Guardrails

- minimum total per validated row: `N >= 50`
- minimum conjunction support per family-state summary: `N_conj >= 30`
- descriptive 2026 OOS only
- raw joined-path recheck required for any carried verdict
- if only one instrument supports the branch, report that explicitly

---

## 11. Immediate implication

The next executable research target after `W2c` is:

- **W2e — `LONDON_METALS -> EUROPE_FLOW` prior-session carry conditioned by
  `garch_high`**

Broad prior-day level conditioning stays in the explicit queue, but it is no
longer the immediate next validated-shelf branch.
