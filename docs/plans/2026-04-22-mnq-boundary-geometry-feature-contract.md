# MNQ Boundary Geometry Feature Contract

**Created:** 2026-04-22  
**Owner:** canompx3  
**Status:** LOCKED — design-only, no production build yet  
**Authority:** `RESEARCH_RULES.md`, `docs/institutional/pre_registered_criteria.md`, `docs/institutional/mechanism_priors.md`, `pipeline/session_guard.py`

---

## Purpose

Define the canonical feature contract for the highest-EV MNQ overlay path:

- prior-day / session-boundary geometry
- expressed in `R`
- fully pre-trade safe
- usable first as `R1` and then `R3`

This is the feature-build gate that must precede any new MNQ prereg in this line.

---

## Why MNQ first

The repo’s strongest live-adjacent exploitation surface is still MNQ, not a side program.

Committed evidence on `main` already shows repeated MNQ level-context effects:

- `docs/audit/results/2026-04-15-prior-day-features-orb-mega-exploration.md`
  - `MNQ US_DATA_1000 O5 RR1.0 long F5_BELOW_PDL`
  - `MNQ COMEX_SETTLE O5 RR1.0 long F6_INSIDE_PDR`
  - `MNQ NYSE_CLOSE O5 RR1.5 long F3_NEAR_PIVOT_15`
- `docs/institutional/mechanism_priors.md` already treats this mechanism class as live prior, not noise

So the missing object is not “another idea.” It is a cleaner canonical encoding of the same geometry class.

---

## Canonical source fields

These fields already exist and are sufficient for the first build:

- `daily_features.prev_day_high`
- `daily_features.prev_day_low`
- `daily_features.prev_day_close`
- `daily_features.overnight_high`
- `daily_features.overnight_low`
- `daily_features.orb_{session}_high`
- `daily_features.orb_{session}_low`
- `daily_features.orb_{session}_break_dir`

Optional second-pass fields:

- `daily_features.atr_20`
- `daily_features.gap_open_points`

No new external data is required for Phase 1.

---

## Feature definitions

All formulas assume:

- `orb_risk = orb_high - orb_low`
- long trades reference overhead resistance
- short trades reference underlying support

If `orb_risk <= 0`, fail closed.

### 1. `clear_to_prevday_overhead_r`

**Long formula:**

`(prev_day_high - orb_high) / orb_risk`

**Short formula:**

`(orb_low - prev_day_low) / orb_risk`

**Meaning:** how much room exists to the nearest prior-day extreme in the direction of the breakout.

### 2. `co_located_prevday_break`

Boolean state:

- long: `orb_high >= prev_day_high`
- short: `orb_low <= prev_day_low`

**Meaning:** the ORB break is also a break of the prior-day extreme.

This should be treated as its own regime, not folded into “open air.”

### 3. `inside_prevday_range`

Boolean state:

- `prev_day_low < orb_mid < prev_day_high`

where:

- `orb_mid = (orb_high + orb_low) / 2`

**Meaning:** the ORB is forming inside yesterday’s structure rather than outside it.

### 4. `clear_to_pivot_r`

`abs(orb_mid - prev_day_pivot) / orb_risk`

with:

- `prev_day_pivot = (prev_day_high + prev_day_low + prev_day_close) / 3`

This is a distance metric, not a near/far binary.

### 5. `clear_to_overnight_extreme_r`

Only valid where session ordering permits.

Long:

`(overnight_high - orb_high) / orb_risk`

Short:

`(orb_low - overnight_low) / orb_risk`

This is a second-phase feature, not mandatory for the first prereg.

---

## Session-safety matrix

Per `pipeline/session_guard.py`:

- `prev_day_high`, `prev_day_low`, `prev_day_close` are safe across all ORB sessions
- `overnight_high` and `overnight_low` are guarded to `LONDON_METALS`

So:

- **Phase 1 safe everywhere:** prior-day geometry
- **Phase 2 safe only at or after `LONDON_METALS`:** overnight geometry

Do not widen overnight features beyond the `session_guard` contract.

---

## Role order

### Phase 1 — `R1` filter / confluence

Allowed uses:

- avoid trades breaking into a nearby wall
- treat co-located breaks as a separate state
- treat inside-range formation as a separate state

### Phase 2 — `R3` size modifier

Allowed only after `R1` survives.

Candidate size map:

- blocked geometry: `0.75x`
- neutral geometry: `1.00x`
- clean geometry / co-located escape: `1.25x`

### Not allowed yet

- target modifiers
- stop-geometry rewrites
- rolling ML on these features

---

## First exact prereg surface after build

The first prereg should stay small and use already-motivated MNQ cells:

1. `MNQ US_DATA_1000 O5 RR1.0 long`
2. `MNQ COMEX_SETTLE O5 RR1.0 long`
3. `MNQ NYSE_CLOSE O5 RR1.5 long`

Those are not yet being pre-registered here, because this doc only locks the feature contract. The next step is to write the prereg once the canonical feature implementation path is agreed.

---

## Shortcut result that changes the priority

The read-only shortcut run on:

- `MNQ / US_DATA_1000 / O5 / E2 / RR1.0 / long`

using on-the-fly `clearance_r` bins produced:

- `co_located_break`: `N_IS=273`, `ExpR=+0.0813`
- `choked`: `N_IS=86`, `ExpR=-0.0691`
- `mid_clearance`: `N_IS=95`, `ExpR=-0.0326`
- `open_air`: `N_IS=427`, `ExpR=+0.0535`

But the bucket-vs-rest Welch checks were weak, and OOS was thin/noisy:

- `co_located_break`: `t=+0.856`, `p=0.392`
- `choked`: `t=-1.121`, `p=0.265`
- `open_air`: `t=+0.386`, `p=0.700`

Implication:

- the **geometry class is still alive**
- the **four-bin clearance framing is not yet the right first deployable encoding**
- immediate research priority should stay with the already-stronger binary states
  - `below_pdl`
  - `inside_prevday_range`
- clearance bins should be treated as a **secondary refinement** after the binary states are locked

---

## Explicit non-goals

- No universal clearance rule
- No direct jump to production `daily_features` changes from this doc alone
- No reinterpretation as market-maker microstructure modeling
- No use of this doc to justify rolling ML or black-box adaptation

This is a canonical feature contract, not a performance claim.
