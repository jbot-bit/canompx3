# MNQ NYSE_OPEN RR1.5 short pivot50 avoid queue park v1

Date: 2026-04-22
Branch: `wt-codex-mnq-hiroi-scan`
Candidate: `cell::NYSE_OPEN::1.5::short::F3_NEAR_PIVOT_50::AVOID`

## Question

Should the current diversified-review exact candidate advance immediately into a
new prereg on:

- `MNQ NYSE_OPEN O5 E2 RR1.5 short`
- `F3_NEAR_PIVOT_50`
- role = `AVOID`

The queue explicitly routed this row into review after the US_DATA exact-cell
park so the honest task here is to decide whether this is a clean next bounded
bridge or just another exact-row observation that should not consume the next
iteration.

## Layered-board truth

From `docs/audit/results/2026-04-22-mnq-layered-candidate-board-v1.md`:

- `F3_NEAR_PIVOT_50`
  - `N_on_IS=622`
  - `ExpR_on_IS=+0.0589`
  - `ExpR_off_IS=+0.2516`
  - `Delta_IS=-0.1927`
  - `t=-1.9562`
  - `BH=0.3810`
  - `N_on_OOS=23`
  - `Delta_OOS=-0.0570`

This is directionally consistent, but the bridge quality is still weak:

- in-sample separation is real-looking but not clean after multiple-testing
- out-of-sample direction matches, but the effect is small
- this is an exact-row story, not a broader bounded family story

## Broader-stack check

The current bounded stack does not provide a clean parent bridge behind this
row:

- `docs/audit/results/2026-04-22-mnq-prior-day-family-board-v1.md`
  - no matching `NYSE_OPEN RR1.5 short` family candidate is active on the
    bounded family surface
- `docs/audit/results/2026-04-22-mnq-geometry-transfer-board-v1.md`
  - contains only long-side prior-day transfer rows for the active unsolved
    lanes
  - no short-side transfer bridge supports this cell
- `docs/plans/2026-04-22-mnq-geometry-transfer-workflow.md`
  - explicitly says the current honest move is to re-rank the next MNQ-first
    mechanism class instead of forcing another geometry-family bridge

So this row would be advancing as a standalone exact cell with no family or
transfer support from the current board stack.

## Queue-quality check

This candidate is also not obviously the best use of the next bounded bridge:

- it is weaker than the stronger NYSE long avoid/take exact rows on raw
  evidence
- it is less grounded than the still-queued mechanism-distinct transfer rows
- it does not solve a broad active family lane the way a clean transfer or
  family bridge would

That makes it a poor prereg candidate for the next slot even though the sign is
not contradictory.

## Decision

Park `cell::NYSE_OPEN::1.5::short::F3_NEAR_PIVOT_50::AVOID` for the current
bounded queue.

Do **not** write a prereg for this exact short avoid-state on this iteration.
The honest read is:

- the row is alive as route-map context
- the OOS effect is too modest to justify spending the next exact bridge here
- the bounded family/transfer stack does not support it as the next move

## Queue consequence

- frontier decision: `parked`
- queue reason:
  - exact-cell evidence without broader parent support
  - same-sign but modest OOS effect
  - not the highest-EV next use of a bounded bridge
- next focus:
  - move to the remaining non-geometry review-batch candidates before
    returning to more NYSE exact slicing or watchlist geometry rows

## Verdict

`cell::NYSE_OPEN::1.5::short::F3_NEAR_PIVOT_50::AVOID` is parked.

This is not a mechanism kill. It is a bounded queue decision: the row remains
alive as local context, but it should not consume the next prereg from the
current broad-but-bounded frontier.
