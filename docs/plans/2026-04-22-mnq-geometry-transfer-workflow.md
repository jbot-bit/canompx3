# MNQ Geometry Transfer Workflow

Date: 2026-04-22
Branch: `wt-codex-mnq-hiroi-scan`

## Purpose

Use the already-proven MNQ prior-day geometry families to find the next
promotable parent-lane transfer without reopening broad feature discovery or
falling back into exact-cell sniping.

## Active families

- `PD_DISPLACE_LONG`
- `PD_CLEAR_LONG`
- `PD_GO_LONG`

## Scope rules

- Instrument: `MNQ` only until transfer quality weakens
- Canonical truth layers only:
  - `orb_outcomes`
  - `daily_features`
- Structural start: `2020-01-01`
  - matches discovery `WF_START_OVERRIDE`
- Sacred holdout: `2026-01-01`
- Entry model: `E2`
- ORB minutes: `5`
- Confirm bars: `1`
- No new feature invention during transfer work

## Fast path

1. Run `research/mnq_geometry_transfer_board_v1.py`
   - rank unsolved parent lanes for the existing family set
   - keep solved lanes in the board for context only
   - derive masks from canonical `ALL_FILTERS`, not hand-coded feature logic
   - do not pre-filter away invalid feature rows; let canonical filters fail
     closed into the off-side
2. Select one exact candidate
   - same-sign OOS
   - strongest bounded family row among unsolved lanes
   - mechanism distinct from the already solved lane
3. Cheap gate
   - run `research/phase4_candidate_precheck.py`
   - require `accepted_raw_trials=1`
   - require zero unexpected experimental or validated rows for the hypothesis SHA
4. Full bridge only after the cheap gate passes
   - discovery write
   - validator
5. Record the result
   - result note
   - update the active queue / family register

## Kill conditions

- Family row only exists because of thin OOS or solved-lane leakage
- Cheap gate does not resolve to one exact accepted combo
- Discovery row does not match the audited surface
- Validator fails on walk-forward, era stability, or lifecycle illegality

## Current queue

1. `MNQ EUROPE_FLOW O5 E2 RR1.0 long PD_GO_LONG`
2. `MNQ NYSE_OPEN O5 E2 RR1.5 long PD_CLEAR_LONG`
3. Re-check whether a broader non-geometry transfer family is warranted before
   touching a third MNQ session

`CME_PRECLOSE` is not in the active queue for this family because the board is
negative or inconsistent there.
