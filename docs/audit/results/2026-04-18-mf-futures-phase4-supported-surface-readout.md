# Phase 4 Supported-Surface Readout

**Date:** 2026-04-18  
**Scope:** `mf_futures` Phase 4 research harness on the currently supported
surface only:

- `MES`
- `MNQ`
- `MGC`

This is a **research-only readout** from the new harness. It is **not** a
deployability verdict and **not** a tradable-chain realized-PnL simulation.

## Method

Run settings:

- capital: `$100,000`
- return basis: `research_close_to_close`
- position sizing: existing `mf_futures` target-position engine
- annualized vol lookback: `63`
- minimum history before simulation: `260` daily observations
- walk-forward:
  - anchored
  - non-overlapping
  - `252` train days
  - `63` test days

Costs used were the canonical repo round-trip frictions from
`pipeline.cost_model.COST_SPECS`:

- `MES`: `$3.92`
- `MNQ`: `$2.92`
- `MGC`: `$5.74`

The run ended at `2026-04-03`, which is the last date where the currently
assembled stats/carry surface remained honest across the supported symbols.

## Results

### MES

- coverage: `2019-05-06` → `2026-04-03`
- snapshots: `2021`
- simulated rows: `1761`
- carry-available simulation days: `15`
- positive daily net rows: `946`
- negative daily net rows: `808`
- average turnover: `0.0403` contracts/day
- average absolute position: `2.2800` contracts
- aggregate gross PnL: `$21,867.50`
- aggregate net PnL: `$21,624.46`
- total turnover: `62` contracts
- walk-forward windows: `23`
- positive walk-forward windows: `12 / 23`
- positive-window ratio: `52.17%`
- best window net PnL: `$8,212.50`
- worst window net PnL: `-$5,106.59`

### MNQ

- coverage: `2019-05-06` → `2026-04-03`
- snapshots: `2021`
- simulated rows: `1761`
- carry-available simulation days: `14`
- positive daily net rows: `970`
- negative daily net rows: `781`
- average turnover: `0.0085` contracts/day
- average absolute position: `1.0522` contracts
- aggregate gross PnL: `$14,402.00`
- aggregate net PnL: `$14,366.96`
- total turnover: `12` contracts
- walk-forward windows: `23`
- positive walk-forward windows: `14 / 23`
- positive-window ratio: `60.87%`
- best window net PnL: `$6,061.00`
- worst window net PnL: `-$7,636.84`

### MGC

- coverage: `2010-06-07` → `2026-04-02`
- snapshots: `4603`
- simulated rows: `4343`
- carry-available simulation days: `452`
- positive daily net rows: `2175`
- negative daily net rows: `2105`
- average turnover: `0.0962` contracts/day
- average absolute position: `3.2068` contracts
- aggregate gross PnL: `$48,162.00`
- aggregate net PnL: `$45,860.26`
- total turnover: `401` contracts
- walk-forward windows: `64`
- positive walk-forward windows: `32 / 64`
- positive-window ratio: `50.00%`
- best window net PnL: `$16,700.00`
- worst window net PnL: `-$5,454.92`

## Interpretation

### 1. MNQ is the cleanest next research candidate

On this harness readout, `MNQ` is the strongest of the three supported
instruments:

- best positive-window ratio: `60.87%`
- lowest turnover drag
- lowest average absolute positioning
- positive aggregate net PnL after fixed friction

That does **not** make it validated. It does make it the cleanest next
candidate for a narrower formal research pass.

### 2. MES stays viable, but not clean

`MES` remains positive on aggregate, but the window split is much weaker:

- `12 / 23` positive windows
- positive-window ratio only `52.17%`

That is enough to keep `MES` in the supported research surface, but not enough
to call it the lead candidate.

### 3. MGC is positive, but noisier and more turnover-heavy

`MGC` has the largest aggregate net PnL in dollars because it has the deepest
history and larger average position size. That is not the same thing as having
the cleanest research profile.

The more important signals are:

- `32 / 64` positive windows exactly `50%`
- materially higher turnover than `MES` or `MNQ`
- much larger carry-available count than the index micros, but still only a
  partial carry surface relative to the full history

So `MGC` should remain in the supported surface, but it should **not** be
treated as the best immediate candidate from this pass.

### 4. This is still not a full trend+carry verdict

The current harness uses `research_close_to_close`, not a full tradable-chain
PnL across rolls. That matters.

Also, carry availability is currently sparse for the index micros:

- `MES`: `15` carry-available simulation days
- `MNQ`: `14` carry-available simulation days

So the `MES` and `MNQ` outputs in this pass are effectively **trend-dominated**
with only occasional carry contribution. `MGC` has materially more carry days
(`452`), so the three instruments are not yet on an equivalent carry footing.

## Honest verdict

SURVIVED SCRUTINY:

- `MNQ` as the best next supported-surface research candidate
- `MES` as a secondary supported candidate worth retaining
- `MGC` as a still-viable but noisier candidate, not the lead lane

DID NOT SURVIVE:

- any claim that this pass validates a deployable trend+carry strategy
- any claim that index-micro carry is already well-covered in the current data
  surface
- any claim that aggregate dollar PnL alone is the correct ranking criterion

CAVEATS:

- return basis is `research_close_to_close`
- no tradable-chain realized PnL across roll transitions
- no full portfolio-level correlation / sector-cap accounting in this pass
- `MES` / `MNQ` carry availability is currently sparse
- fixed friction only; no richer slippage model in this harness

NEXT STEPS:

1. Treat `MNQ` as the first candidate for a formal narrow research memo built
   on the new harness output.
2. Keep `MES` as a secondary follow-up on the same surface.
3. Keep `MGC` in the supported universe, but do not promote it above `MNQ`
   from this readout alone.
4. Do **not** expand to `ZN` / `ZB` / `M6E` / `6J` until raw stats exist.
5. Before any promotion talk, add a stricter tradable-chain accounting pass so
   the harness is not judged as if it were a live-ready simulator.
