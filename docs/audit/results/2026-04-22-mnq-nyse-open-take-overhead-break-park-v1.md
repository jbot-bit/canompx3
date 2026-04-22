# MNQ NYSE_OPEN take-overhead-break queue park v1

Date: 2026-04-22
Branch: `wt-codex-mnq-hiroi-scan`
Candidate: `family::NYSE_OPEN::1.5::long::TAKE_OVERHEAD_BREAK`

## Question

Should the current top queued family candidate advance immediately into an
exact bridge on:

- `MNQ NYSE_OPEN O5 E2 RR1.5 long`

The family surface says prior-day overhead-break context is directionally
helpful here, but the next bounded move still needs one honest exact seed
rather than a family-rank story.

## Family-board truth

From `docs/audit/results/2026-04-22-mnq-prior-day-family-board-v1.md`:

- `TAKE_OVERHEAD_BREAK`
  - `N_on_IS=320`
  - `ExpR_on_IS=+0.1705`
  - `ExpR_off_IS=+0.0293`
  - `delta_IS=+0.1412`
  - `t=+1.65`
  - `BH=0.2953`
  - `N_on_OOS=7`
  - `delta_OOS=+0.6962`
  - `same_sign_oos=True`

This is alive route-map evidence, but it is not clean enough by itself to
force a prereg:

- multiple-testing support is still weak
- OOS breadth is thin
- the family still has to survive through one exact bridge candidate

## Layered exact-state check

From `docs/audit/results/2026-04-22-mnq-layered-candidate-board-v1.md`, the
family decomposes into one thin positive seed and one OOS-failing member:

- `F4_ABOVE_PDH`
  - `delta_IS=+0.2198`
  - `BH=0.2141`
  - `N_on_OOS=5`
  - `delta_OOS=+0.8972`
- `F1_NEAR_PDH_15`
  - `delta_IS=+0.0945`
  - `BH=0.7066`
  - `N_on_OOS=8`
  - `delta_OOS=-0.4530`
- `F1_NEAR_PDH_15__AND__F4_ABOVE_PDH`
  - `delta_IS=+0.0569`
  - `BH=0.8903`
  - `N_on_OOS=0`
  - `delta_OOS=nan`

That means the family read is not backed by a clean exact bridge:

- `F4_ABOVE_PDH` is directionally attractive but still thin
- `F1_NEAR_PDH_15` does not confirm the same story out of sample
- the overlap state provides no OOS support at all

So the current family row looks more like a useful route-map than a ready exact
promotion surface.

## Lane-level transfer check

The same lane still has queued prior-day transfer complements on
`docs/audit/results/2026-04-22-mnq-geometry-transfer-board-v1.md`, but they do
not rescue the bridge decision:

- `NYSE_OPEN RR1.5 long PD_CLEAR_LONG`
  - `delta_IS=+0.0877`
  - `BH=0.5968`
  - `N_on_OOS=7`
  - `delta_OOS=+0.7023`
- `NYSE_OPEN RR1.5 long PD_GO_LONG`
  - `delta_IS=+0.0355`
  - `BH=0.7841`
  - `N_on_OOS=10`
  - `delta_OOS=+0.7687`

And `docs/plans/2026-04-22-mnq-geometry-transfer-workflow.md` already says the
`NYSE_OPEN` transfer lane is watchlist-only, not an approved active bridge row.

## Decision

Park `family::NYSE_OPEN::1.5::long::TAKE_OVERHEAD_BREAK` for the current
bounded queue.

Do **not** prereg `F4_ABOVE_PDH` from this lane on this iteration. The honest
read is:

- the family observation is still alive
- the only positive exact seed is too thin to spend the next bridge on
- the companion family member does not confirm the same story out of sample
- the transfer-board complements remain watchlist context, not promotion
  support

## Queue consequence

- frontier decision: `parked`
- queue reason:
  - family rank overstated the readiness of the exact bridge seed
  - exact decomposition is mixed rather than clean
- next focus:
  - move to the remaining queued review-batch candidates instead of letting the
    NYSE prior-day family lane monopolize another iteration

## Verdict

`family::NYSE_OPEN::1.5::long::TAKE_OVERHEAD_BREAK` is parked.

This is not a mechanism kill. It is a bounded queue decision: the lane remains
alive as research context, but it should not consume the next exact bridge from
the current board stack.
