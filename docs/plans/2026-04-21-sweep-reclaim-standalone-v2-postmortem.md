# Sweep Reclaim Standalone V2 — Postmortem

**Date:** 2026-04-21  
**Status:** CLOSED  
**Scope:** exact family locked in
`docs/audit/hypotheses/2026-04-21-sweep-reclaim-standalone-v2.yaml`

## Verdict

This exact standalone family is `DEAD`.

That statement is stronger than "failed to reach `N>=100` on a warm cell."
Canonical results show:

- `0 / 8` cells pass `H1`
- `0 / 8` cells classify `RESEARCH_SURVIVOR`
- `0 / 8` cells classify `CANDIDATE_READY`
- `7 / 8` cells have negative IS `ExpR`
- the only positive IS cell (`MNQ EUROPE_FLOW prev_day_low`) is thin (`N=26`,
  `ExpR=+0.1349`, `t=+0.64`)

## What canonical data actually proved

From `research/output/sweep_reclaim_standalone_v2_cells.csv` and
`research/output/sweep_reclaim_standalone_v2_trades.csv`:

- pooled by instrument:
  - `MES`: `N=181`, `ExpR=-0.1699`
  - `MNQ`: `N=216`, `ExpR=-0.0671`
- pooled by session:
  - `EUROPE_FLOW`: `N=121`, `ExpR=-0.1850`
  - `NYSE_OPEN`: `N=276`, `ExpR=-0.0828`
- pooled by level:
  - `prev_day_high`: `N=228`, `ExpR=-0.1455`
  - `prev_day_low`: `N=169`, `ExpR=-0.0714`

So the family is not failing only because the positive cells are underpowered.
The broader pooled economics are also negative.

## What this does and does not kill

### Killed

- the exact standalone trade geometry:
  - `reclaim_close`
  - stop one tick beyond sweep extreme
  - fixed `1.5R`
  - `MNQ/MES`
  - `EUROPE_FLOW/NYSE_OPEN`
  - `PDH/PDL`

### Not killed

- the broader liquidity / displacement prompt family
- ORB context overlays from other branches
- any future standalone family with a genuinely different mechanism
- any sweep/reclaim use in a different role that is pre-registered separately

## Biggest mistake corrected

The repo previously sat in an ambiguous middle state:

- `sweep-reclaim-v1` was a useful event study
- but it was not a trade strategy
- and the branch risked being over-killed or over-rescued depending on who read
  it

V2 resolves that ambiguity. The exact standalone family was actually tested and
it failed.

## Best opportunity now

Two different answers depending on objective:

### Near-term profit / lowest drag

Stay with exact-lane ORB conditioner work.

That branch has already shown local surviving context overlays and has much
lower implementation drag than inventing another standalone engine.

### New strategy discovery

Do **not** reopen this same sweep/reclaim geometry.

The next honest standalone family should be structurally different:

- `opening-drive pullback continuation`

Reason:

- different mechanism from the dead reversal family
- different failure mode from the dead ORB boundary retest route
- still compatible with the translator's geometry-first approach

## What not to do next

- do not widen this family to more sessions just to raise sample size
- do not add overnight levels as a rescue
- do not change target mode post-hoc to save it
- do not treat the warm `MNQ EUROPE_FLOW prev_day_low` cell as a survivor

## Net closeout

The exact standalone sweep-reclaim branch is closed.

The correct interpretation is:

- `standalone sweep-reclaim v2`: dead
- `broader liquidity/displacement research`: still open, but must move to a
  different family
