---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Cross-Asset RR1.5 Reopen Lock

**Date:** 2026-04-22
**Status:** LOCKED NEXT STEP
**Purpose:** reopen the cross-asset mechanism class only where the prior kill no longer binds.

## Binding facts

- PR #82 killed the exact `R1` binary overlay on:
  - source: `MGC US_DATA_830`
  - target: `MNQ NYSE_OPEN`
  - lane: `O5 / E2 / CB1 / RR1.0`
  - result: `delta_IS=+0.0393R`, `Welch p=0.5922`, `null_floor p=0.2987`
- That kill remains fully binding for `RR1.0`.
- It does **not** automatically bind `RR1.5`, because the target lane geometry is different and the raw canonical overlay economics differ materially:
  - `RR1.0`: `delta_IS=+0.0393R`
  - `RR1.5`: `delta_IS=+0.1447R`
  - same pair, same source feature, same IS Q3 split logic
- This is therefore an admissible reopen under the project rule: different role is not enough, but a materially different target lane is.

## What stays locked

- Same chronology discipline from PR #81.
- Same source feature class:
  - `source_orb_size_norm = orb_US_DATA_830_size / atr_20`
- Same source instrument and session:
  - `MGC / US_DATA_830 / O5`
- Same target instrument and session:
  - `MNQ / NYSE_OPEN / O5 / E2 / CB1`
- Same role:
  - `R1` binary overlay only

## What changes

- Target lane `rr_target` changes from `1.0` to `1.5`.
- Nothing else changes.

## Why this reopen is allowed

- The old kill was methodology-clean, but lane-specific.
- The user explicitly challenged whether the result was just a `1R` artifact.
- Canonical spot-check on the same frozen source logic shows the target RR matters materially.
- That makes `RR1.5` a legitimate new prereg, not a post-hoc excuse for the dead `RR1.0` result.

## Non-choices

- No same-turn `MES` sibling.
- No `R3` sizing follow-up.
- No widening to `US_DATA_1000`.
- No reuse of live-config historical notes as proof; those are only motivation, not evidence.

## Next action

1. Lock the exact `RR1.5` prereg.
2. Do not implement the runner until the prereg is committed.
