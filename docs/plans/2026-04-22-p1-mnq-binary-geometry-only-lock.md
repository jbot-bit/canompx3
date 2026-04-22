# P1 Lock — MNQ Binary Geometry Only

**Created:** 2026-04-22  
**Owner:** canompx3  
**Status:** LOCKED  
**Authority:** `RESEARCH_RULES.md`, `docs/institutional/pre_registered_criteria.md`, `docs/institutional/mechanism_priors.md`

---

## Decision

Freeze the active research plan to:

- **P1 = MNQ binary geometry only**

Explicitly excluded from the active plan:

- clearance-bin families as the primary path
- MES participation-family execution
- MGC participation-family execution
- rolling / adaptive ML overlays
- microstructure/L3 upgrade work

Those paths are **deferred**, not active.

---

## Why this is the correct freeze

Canonical evidence says the strongest live-adjacent exploitation surface is still
MNQ conditional structure on positive parent lanes.

Two exact binary states remain the cleanest grounded pair:

1. `MNQ US_DATA_1000 O5 E2 RR1.0 long F5_BELOW_PDL`
   - `N_IS=882`
   - `N_on_IS=136`
   - `ExpR_on_IS=+0.3258`
   - `ExpR_off_IS=-0.0112`
   - `delta_IS=+0.3370`
   - `N_OOS=35`
   - `N_on_OOS=8`
   - `delta_OOS=+0.0375`

2. `MNQ COMEX_SETTLE O5 E2 RR1.0 long F6_INSIDE_PDR`
   - `N_IS=876`
   - `N_on_IS=433`
   - `ExpR_on_IS=-0.0296`
   - `ExpR_off_IS=+0.1651`
   - `delta_IS=-0.1947`
   - `N_OOS=34`
   - `N_on_OOS=19`
   - `delta_OOS=-0.2336`

These are stronger and more directly exploitable than the current secondary
execution-safe participation path.

The read-only clearance shortcut was still useful, but it demoted itself:

- it supports the geometry class
- it does **not** support a clean universal four-bin clearance rule as the active first path

So the right sequence is:

1. lock binary MNQ geometry
2. run it properly
3. only then revisit continuous geometry refinement

---

## Active counted hypotheses

### H1

`MNQ US_DATA_1000 O5 E2 RR1.0 long F5_BELOW_PDL`

Expected role:

- positive binary take / emphasis overlay on a strong lane

### H2

`MNQ COMEX_SETTLE O5 E2 RR1.0 long F6_INSIDE_PDR`

Expected role:

- negative binary avoid overlay on a positive lane

---

## Explicitly deferred

### D1 — Clearance bins

Deferred because the first read-only shortcut showed:

- weak bucket-vs-rest separation
- thin/noisy OOS
- better immediate value in the stronger binary states

### D2 — MES / MGC execution-safe participation

Deferred because it is secondary to the primary-book exploitation path.

### D3 — ML / current-meta adaptive overlay

Deferred because rolling ORB ML remains blocked by repo doctrine until a static
overlay proves incremental value first.

---

## Immediate next move

Write and run the exact P1 prereg / harness on the two locked MNQ binary states.
