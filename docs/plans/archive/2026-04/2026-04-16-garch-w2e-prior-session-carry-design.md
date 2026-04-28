---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Garch W2e Prior-Session Carry Design

**Date:** 2026-04-16  
**Status:** BROADENED_AFTER_USER_FEEDBACK (2026-04-16 post-crash session)  
**Purpose:** define the next executable `garch` mechanism-family stage after the
validated-shelf feasibility check demoted broad prior-level conditioning as the
immediate next branch.

> **2026-04-16 scope broadening note.** The original V1 of this design locked
> scope to `LONDON_METALS -> EUROPE_FLOW` only, on `MNQ` only. Codex hit its
> usage limit before running the audit. User feedback arrived immediately
> before the limit cutoff: *"don't just test 1 session or lane and call it
> done — there's lots of variables."* That feedback was never applied in V1.
>
> V2 (this revision) broadens scope to **all chronologically-admissible same-day
> session handoffs × the full validated shelf**, enforcing the same dynamic
> chronology and validated-shelf rules. Institutional reasoning: a 1-handoff
> × 1-instrument audit is a tunnel-vision artefact — RULE 5 (comprehensive
> scope) of `.claude/rules/backtesting-methodology.md` requires enumeration
> of all declared axes. The original narrow design is preserved below as the
> seed hypothesis; the broadening is an expansion of the test battery, not a
> new hypothesis. See § 12 "V2 scope broadening" for exact changes.

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

- **W2e — prior-session carry conditioned by `garch_high`, across all
  chronologically-admissible same-day session handoffs on the validated shelf**

Broad prior-day level conditioning stays in the explicit queue, but it is no
longer the immediate next validated-shelf branch.

---

## 12. V2 scope broadening (2026-04-16 post-crash revision)

### 12.1 Why

V1 locked `prior = LONDON_METALS`, `target = EUROPE_FLOW`, `instrument = MNQ`.
That is one cell of a much larger admissible space. User feedback at the moment
Codex hit its usage limit: *"there's lots of variables — don't test one lane."*

V1's narrow choice was defensible under "first branch with validated-shelf
overlap that compiles," but it does not meet the comprehensive-scope bar in
`.claude/rules/backtesting-methodology.md` RULE 5. Running V1 as-is would have
produced a single verdict with no family context — no way to tell whether any
lift is universal across handoffs or an artefact of one pair.

### 12.2 What broadened

| Axis | V1 | V2 |
|---|---|---|
| Target session | EUROPE_FLOW only | all 5 validated families (COMEX_SETTLE, EUROPE_FLOW, TOKYO_OPEN, SINGAPORE_OPEN, LONDON_METALS) |
| Prior session | LONDON_METALS only | any session in `pipeline.dst.SESSION_CATALOG` that resolves before the target (per-row dynamic check) |
| Instrument | MNQ only | full validated shelf (MNQ, MES, MGC, GC) |
| Carry states | PRIOR_WIN_ALIGN, PRIOR_WIN_OPPOSED | unchanged — hypothesis unchanged |
| Threshold | garch_forecast_vol_pct ≥ 70 | unchanged |
| Chronology | static | unchanged — per-row `prior.exit_ts < target_start_ts` |
| Validated-shelf rule | validated-only | unchanged |
| Null test | none | **added** — bootstrap shuffle of carry-state membership on the filtered target population, 1000 iters |
| Raw sanity | relied on validated metadata | **added** — raw N per cell pulled from canonical `orb_outcomes`, independent of `validated_setups.sample_size` |

### 12.3 What did NOT change

- The pre-registered hypothesis is identical: *garch_high adds value to
  fully-resolved same-day prior-session carry state on validated shelf cells*.
- The carry-state definitions (PRIOR_WIN_ALIGN, PRIOR_WIN_OPPOSED) are
  identical.
- MIN_TOTAL = 50, MIN_CONJ = 30, 2026 descriptive-only — unchanged.
- No deployment or allocator verdict allowed — unchanged.

### 12.4 K budget

V2 comprehensive scope produces roughly:
- 5 target sessions × 8 candidate priors × 4 instruments × 2 states ≈ **320
  handoff-cells max**.

Real count is smaller because:
- TOKYO_OPEN target has no admissible priors (first in trading day) → 0 cells.
- Many instrument/session/prior combos have no shared trading days.
- Many have N < MIN_TOTAL or N_conj < MIN_CONJ → report `not_testable_here`.

The actual K is computed and reported in the audit output header. BH-FDR is
applied at the HANDOFF level (K = # admissible handoffs) since each handoff is
a distinct hypothesis.

### 12.5 Interpretation rule

V2 report verdicts are per-(prior, target, instrument, state). A handoff is
"supported" only if:
- `N_conj ≥ 30`
- `Δ_conj_vs_base`, `Δ_conj_vs_garch`, `Δ_conj_vs_carry` all match
  expected_role direction
- bootstrap null p ≤ 0.05 (descriptive, not a promotion gate — because this is
  state-conditioning not discovery)
- dir_match across the state's expected role

A handoff **family** (e.g., LONDON_METALS → EUROPE_FLOW across instruments) is
"supported" only if ≥ 2 instruments independently support. Single-instrument
support is flagged `thin_handoff_support`.
