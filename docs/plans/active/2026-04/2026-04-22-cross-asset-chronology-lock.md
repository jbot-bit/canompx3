---
status: active
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Cross-Asset Chronology Lock

**Date:** 2026-04-22
**Status:** LOCKED NEXT STEP
**Purpose:** choose the next honest post-action-pack research path without reopening dead families or drifting into chronology leakage.

## Binding facts

- The 2026-04-21 action pack executed three locked paths and exhausted that queue:
  - `COMEX` unfiltered overlay family: `PARK` in PR #78.
  - `L1 EUROPE_FLOW` pre-break context family: `KILL` in PR #79.
  - `MNQ US_DATA_1000 F5_BELOW_PDL` single-cell confirmation: `PARK` in PR #80.
- `HTF freshness / distance / outside-range` remains blocked because the canonical feature layer does not exist yet. No scan before feature build. Source: [2026-04-21-post-stale-lock-action-queue.md](2026-04-21-post-stale-lock-action-queue.md), [2026-04-20-htf-branch-closeout.md](2026-04-20-htf-branch-closeout.md).
- `Cross-asset earlier-session context for later ORB quality` remains live, but the blocker was chronology discipline, not mechanism. Source: [2026-04-21-post-stale-lock-action-queue.md](2026-04-21-post-stale-lock-action-queue.md), [RESEARCH_RULES.md](../../RESEARCH_RULES.md).
- Repo-local mechanism prior is explicit: large `MGC US_DATA_830` ORB should be treated as a volatility transmission state for later equity-index ORBs, not as a direction forecast. Source: [RESEARCH_RULES.md](../../RESEARCH_RULES.md), [mechanism_priors.md](../institutional/mechanism_priors.md).

## Chronology correction

- Older prose in [RESEARCH_RULES.md](../../RESEARCH_RULES.md) describes `MGC US_DATA_830 -> MES/MNQ NYSE_OPEN` as having a "90-minute temporal gap."
- The actual safe condition is stronger and more precise than that prose:
  - source session = `US_DATA_830`, anchored at `8:30 AM ET`
  - target session = `NYSE_OPEN`, anchored at `9:30 AM ET`
  - source feature must be fully determined by **source ORB close**
  - target feature must be evaluated only on the later target lane after its own ORB closes
  - no source field may depend on target-session bars, and no target trade may be conditioned on unresolved source-state
- This plan locks chronology by event ordering, not by a hand-wavy minute count. Session anchors come from [pipeline/dst.py](../../pipeline/dst.py).

## Chosen next path

- **KEEP and prereg now:** `MGC US_DATA_830` source-session ORB-size context as a conditioner for the **unfiltered** `MNQ NYSE_OPEN E2 RR1.0 CB1` lane.
- **Role:** `R1` binary overlay only.
- **What it is allowed to ask:** whether high source-session magnitude changes target-lane quality.
- **What it is not allowed to ask:** target direction prediction, pooled cross-asset discovery, or same-turn selection across both `MNQ` and `MES`.

## Why this exact path

- `RESEARCH_RULES.md` already names this mechanism class as live and not dead.
- [2026-03-22-live-config-redesign.md](2026-03-22-live-config-redesign.md) contains the only repo-local instrument-specific note for this exact family at `NYSE_OPEN`: `X_MGC_ATR70 (MNQ, 0.19R)`.
- Choosing `MNQ` only keeps the next test at `K=1` and avoids pair-shopping across `MES` and `MNQ`.
- Choosing the **unfiltered** lane preserves the discipline established in the 2026-04-21 action pack: do not layer a new context study on top of a prefiltered target lane unless the base lane has already been evaluated honestly.

## Explicit non-choices

- `MES` is not killed. It is deferred to a later sibling prereg if and only if the `MNQ` path survives.
- `HTF` stays parked.
- No broad cross-session lead-lag sweep.
- No reuse of generic `X_MGC_ATR70` daily-vol framing as a substitute for session-specific chronology.

## Next action

1. Lock the paired-session prereg for `MGC US_DATA_830 -> MNQ NYSE_OPEN`.
2. Build the minimum runner only after the prereg is committed.
3. Do not add `MES`, `US_DATA_1000`, or alternative source features to the same trial budget.
