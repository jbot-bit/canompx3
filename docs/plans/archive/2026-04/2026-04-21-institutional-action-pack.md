---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Institutional Action Pack

**Date:** 2026-04-21
**Status:** ACTIVE
**Purpose:** convert the 2026-04-21 institutional audits into a single honest execution pack so the next research step is pre-registered, scoped, and non-duplicative.

**This doc supersedes** [2026-04-21-post-stale-lock-action-queue.md](2026-04-21-post-stale-lock-action-queue.md) for next-step research routing.

## Binding facts

- `GARCH R3` is already past prereg and sits in forward-monitoring only. No new discovery or retuning belongs there. Source: [2026-04-21-garch-r3-session-clipped-shadow.md](../audit/results/2026-04-21-garch-r3-session-clipped-shadow.md), [2026-04-21-post-stale-lock-action-queue.md](2026-04-21-post-stale-lock-action-queue.md).
- `L1 EUROPE_FLOW` does **not** justify an ATR-normalized ORB_G5 replacement prereg. That path is dead. Source: [2026-04-21-l1-europe-flow-filter-diagnostic.md](../audit/results/2026-04-21-l1-europe-flow-filter-diagnostic.md).
- `L3 COMEX_SETTLE ORB_G5` had real historical discrimination, but that does **not** prove a live selector because the no-fire bucket is extinct-era. Source: [2026-04-21-comex-settle-orb-g5-failure-pocket-audit.md](../audit/results/2026-04-21-comex-settle-orb-g5-failure-pocket-audit.md).
- `Prior-day level` work is not dead, but the honest next move is a narrow single-cell confirmation, not another family sweep. Source: [2026-04-15-prior-day-features-orb-mega-exploration.md](../audit/results/2026-04-15-prior-day-features-orb-mega-exploration.md), [2026-04-15-t0-t8-audit-o5-patterns.md](../audit/results/2026-04-15-t0-t8-audit-o5-patterns.md).
- `E2 break_*` features are not admissible predictor inputs for new preregs. The canonical rule file already bans them, and the 2026-04-21 handoff quantified the leak at 41.3% on real E2 trades. Source: [HANDOFF.md](../../HANDOFF.md).

## Decision matrix

| Path | Verdict | Action | Why |
|---|---|---|---|
| GARCH `R3` session-clipped shadow | KEEP AS MONITOR ONLY | No new prereg | Already locked and running; new work here would be drift from the audited role |
| MNQ `COMEX_SETTLE` unfiltered overlay | PREREG NOW | New prereg file | Highest-EV open path after the COMEX audit; tests where current edge may actually live |
| MNQ `EUROPE_FLOW` pre-break context | PREREG NOW | New prereg file | Filter-replacement path is dead, but pre-break context on the unfiltered lane remains open |
| MNQ `US_DATA_1000` `F5_BELOW_PDL` single-cell | PREREG NOW | New prereg file | Cleanest narrow prior-day cell: hot in the mega scan, conditional in T0-T8, predicate already canonical |
| Cross-asset earlier-session context | DEFER | No prereg yet | Still plausible, but chronology discipline is not yet frozen tightly enough |
| HTF freshness / distance / outside-range | DEFER | No prereg yet | Requires canonical feature build first; no honest scan before feature layer exists |

## What this pack does

1. Locks the highest-EV `COMEX` next-step study as an unfiltered lane overlay family.
2. Locks the `L1` next-step study as a pre-break-context family on the unfiltered lane, not an ORB_G5 replacement.
3. Locks one narrow prior-day single-cell confirmation path instead of reopening the whole family.
4. Explicitly records the deferred paths so they are not forgotten or quietly widened later.

## What this pack does not do

- No code changes.
- No rule-file changes.
- No holdout use.
- No rediscovery scans.
- No pre-registration for cross-asset or HTF work before the blockers are removed.

## Execution order after merge

1. Keep `GARCH R3` in forward-monitoring unchanged.
2. Implement the `COMEX` prereg harness first.
3. Implement the `L1` prereg harness second.
4. Implement the narrow prior-day single-cell harness third.
5. Revisit cross-asset chronology spec only after the three locked paths above have verdicts.

## Locked prereg set created by this pack

- [2026-04-21-mnq-comex-unfiltered-overlay-v1.yaml](../audit/hypotheses/2026-04-21-mnq-comex-unfiltered-overlay-v1.yaml)
- [2026-04-21-l1-europe-flow-pre-break-context-prereg.yaml](../audit/hypotheses/2026-04-21-l1-europe-flow-pre-break-context-prereg.yaml)
- [2026-04-21-mnq-us-data-1000-f5-below-pdl-v1.yaml](../audit/hypotheses/2026-04-21-mnq-us-data-1000-f5-below-pdl-v1.yaml)
