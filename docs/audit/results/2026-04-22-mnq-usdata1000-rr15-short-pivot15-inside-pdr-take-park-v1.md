# MNQ US_DATA_1000 RR1.5 short pivot15-inside-pdr take queue park v1

Date: 2026-04-22
Branch: `wt-codex-mnq-hiroi-scan`
Candidate: `cell::US_DATA_1000::1.5::short::F3_NEAR_PIVOT_15__AND__F6_INSIDE_PDR::TAKE`

## Question

Should the current diversified-review exact candidate advance immediately into a
new prereg on:

- `MNQ US_DATA_1000 O5 E2 RR1.5 short`
- `F3_NEAR_PIVOT_15__AND__F6_INSIDE_PDR`

The capsule explicitly routed this row into review before letting NYSE or
geometry rows reclaim the loop, so the honest task here is to decide whether
this conjunction is actually a bounded bridge candidate or just a thin local
observation.

## Layered-board truth

From `docs/audit/results/2026-04-22-mnq-layered-candidate-board-v1.md`:

- `F3_NEAR_PIVOT_15__AND__F6_INSIDE_PDR`
  - `N_on_IS=202`
  - `ExpR_on_IS=+0.1898`
  - `Delta_IS=+0.0860`
  - `t=+0.8933`
  - `BH=0.7338`
  - `N_on_OOS=6`
  - `Delta_OOS=+0.8287`

Directionally this is alive, but the bridge quality is still weak:

- multiple-testing support is poor
- OOS breadth is still thin
- the row is not clearly stronger than the simpler nearby states on the same
  lane

## Exact-state decomposition check

The same layered board shows the conjunction is not adding a clean incremental
mechanism:

- `F3_NEAR_PIVOT_15`
  - `Delta_IS=+0.0791`
  - `BH=0.7480`
  - `N_on_OOS=6`
  - `Delta_OOS=+0.8287`
- `F4_ABOVE_PDH`
  - `Delta_IS=+0.0455`
  - `BH=0.8903`
  - `N_on_OOS=9`
  - `Delta_OOS=+0.5155`
- `F6_INSIDE_PDR`
  - `Delta_IS=+0.0188`
  - `BH=0.9586`
  - `N_on_OOS=19`
  - `Delta_OOS=-0.1306`

That means:

- the queued conjunction does not improve meaningfully on `F3_NEAR_PIVOT_15`
- the positive OOS read is effectively the same as the simpler pivot-near row
- the `F6_INSIDE_PDR` side of the conjunction does not independently support a
  positive short-side story

So the current exact row looks more like local slicing than a distinct bridge.

## Parent-surface check

This candidate is also unsupported by the broader bounded stack:

- `docs/audit/results/2026-04-22-mnq-prior-day-family-board-v1.md`
  - contains no `US_DATA_1000 RR1.5 short` family row
  - the active prior-day family board is currently about other parent lanes,
    not this short exact cell
- `docs/audit/results/2026-04-22-mnq-geometry-transfer-board-v1.md`
  - contains no matching short-side transfer candidate for this lane
- `docs/audit/results/2026-04-20-mnq-unfiltered-baseline-cross-family-v1.md`
  - still shows `MNQ 15m RR1.5 US_DATA_1000` as broad route-map evidence, but
    that surface is explicitly not direct promotion authority for a new exact
    short prereg

So there is no family or transfer bridge behind this row. Advancing it now
would be exact-cell tunnel vision, which the current workflow is specifically
trying to avoid.

## Decision

Park `cell::US_DATA_1000::1.5::short::F3_NEAR_PIVOT_15__AND__F6_INSIDE_PDR::TAKE`
for the current bounded queue.

Do **not** write a prereg for this conjunction on this iteration. The honest
read is:

- the row is directionally interesting but statistically weak
- the conjunction is not cleaner than the simpler `F3_NEAR_PIVOT_15` state
- the broader stack does not provide a parent-family or transfer bridge

## Queue consequence

- frontier decision: `parked`
- queue reason:
  - thin exact-row evidence without parent support
  - no meaningful incremental edge over the simpler pivot-near state
- next focus:
  - move to the remaining diversified review-batch candidates instead of
    spending another loop on this US_DATA exact slice

## Verdict

`cell::US_DATA_1000::1.5::short::F3_NEAR_PIVOT_15__AND__F6_INSIDE_PDR::TAKE` is
parked.

This is not a mechanism kill. It is a bounded queue decision: the row can stay
alive as route-map context, but it should not consume the next prereg from the
current board stack.
