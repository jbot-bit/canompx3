# MNQ NYSE_OPEN avoid-congestion queue park v1

Date: 2026-04-22
Branch: `wt-codex-mnq-hiroi-scan`
Candidate: `family::NYSE_OPEN::1.5::long::AVOID_CONGESTION`

## Question

Should the current top queued family candidate advance immediately into an
exact bridge via its positive complement on:

- `MNQ NYSE_OPEN O5 E2 RR1.5 long`

The family surface says prior-day congestion is hostile here, but the next
honest move must still be justified by the bounded exact evidence rather than
by the family rank alone.

## Family-board truth

From `docs/audit/results/2026-04-22-mnq-prior-day-family-board-v1.md`:

- `AVOID_CONGESTION`
  - `N_on_IS=633`
  - `ExpR_on_IS=+0.0520`
  - `ExpR_off_IS=+0.1859`
  - `delta_IS=-0.1339`
  - `t=-1.36`
  - `BH=0.3515`
  - `N_on_OOS=21`
  - `delta_OOS=-0.7023`
  - `same_sign_oos=True`

This is useful route-map evidence, but it is not a promotable family row on its
own:

- multiple-testing support is weak
- the family still needs an exact bridge candidate, not a prose upgrade

## Layered exact-state check

From `docs/audit/results/2026-04-22-mnq-layered-candidate-board-v1.md`, the
congestion-side exact avoid rows are directionally aligned but still weak or
thin as bridge seeds:

- `F6_INSIDE_PDR`
  - `delta_IS=-0.2066`
  - `BH=0.2141`
  - `N_on_OOS=21`
  - `delta_OOS=-0.7023`
- `F3_NEAR_PIVOT_15`
  - `delta_IS=-0.2608`
  - `BH=0.0913`
  - `N_on_OOS=4`
  - `delta_OOS=-0.7158`
- `F3_NEAR_PIVOT_50__AND__F6_INSIDE_PDR`
  - `delta_IS=-0.1820`
  - `BH=0.2851`
  - `N_on_OOS=20`
  - `delta_OOS=-0.4299`

That means the family shape is descriptively interesting, but the exact bridge
surface is not yet clean enough to force a discovery write from this lane.

## Complement transfer check

The natural exact take-side complement of `AVOID_CONGESTION` is
`PD_CLEAR_LONG`. From
`docs/audit/results/2026-04-22-mnq-geometry-transfer-board-v1.md`:

- `NYSE_OPEN RR1.5 long PD_CLEAR_LONG`
  - `N_on_IS=180`
  - `ExpR_on_IS=+0.1700`
  - `ExpR_off_IS=+0.0823`
  - `delta_IS=+0.0877`
  - `t=+0.85`
  - `BH=0.5968`
  - `N_on_OOS=7`
  - `delta_OOS=+0.7023`

And the active workflow register in
`docs/plans/2026-04-22-mnq-geometry-transfer-workflow.md` already tightens the
interpretation:

- this row is a `watchlist` row, not an approved bridge row
- it is weaker and less stable than it first looked
- `3/7` negative years exist
- ATR quartiles flip sign across the surface

## Decision

Park `family::NYSE_OPEN::1.5::long::AVOID_CONGESTION` for the current bounded
queue.

Do **not** prereg `PD_CLEAR_LONG` from this lane on this iteration. The honest
interpretation is:

- the family board captured a real congestion-vs-clear route-map observation
- the exact complement is not strong enough yet to justify a new bridge
- this lane should remain watchlist-only until a stronger exact seed survives
  the bounded stack

## Queue consequence

- frontier decision: `parked`
- queue reason:
  - family rank overstated the readiness of the exact complement
  - weak transfer-board support beats forced bridge momentum
- next focus:
  - keep the lens broad across the remaining queued family and non-geometry
    mechanism candidates instead of forcing this NYSE congestion branch

## Verdict

`family::NYSE_OPEN::1.5::long::AVOID_CONGESTION` is parked.

This is not a kill on the mechanism story. It is a bounded queue decision: the
lane stays as watchlist evidence, but it should not consume the next exact
bridge iteration from the current board stack.
