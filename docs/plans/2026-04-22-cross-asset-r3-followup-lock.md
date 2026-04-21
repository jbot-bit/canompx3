# Cross-Asset R3 Follow-Up Lock

**Date:** 2026-04-22
**Status:** LOCKED NEXT STEP
**Purpose:** decide whether the cross-asset mechanism class deserves one more honest role-specific test after the exact `R1` binary overlay failed.

## Binding facts

- The exact `R1` binary overlay prereg on `MGC US_DATA_830 -> MNQ NYSE_OPEN` is a clean `KILL`. Source chronology is safe, but the frozen overlay economics fail:
  - `N_on_IS=226`
  - `ExpR_on_IS=+0.1603`
  - `ExpR_off_IS=+0.1210`
  - `delta_IS=+0.0393R`
  - `Welch p=0.5922`
  - `null_floor p=0.2987`
  Source: PR #82 result note.
- The target lane itself remains positive on the usable sample:
  - baseline target-lane `IS ExpR=+0.1308`
  - baseline target-lane `OOS ExpR=+0.1327`
  Source: PR #82 result note.
- This means the failure is not “no signal anywhere.” The failure is “not strong enough to justify binary skip/take.”
- The chronology lock in PR #81 remains binding. No widening to `MES`, no direction prediction, no feature swapping, no generic `cross_atr_MGC_pct` substitution.

## Why an R3 follow-up is still allowed

- `mechanism_priors.md` explicitly maps context features to multiple roles and treats `R3` size modulation as a different deployment question from `R1` filtering.
- The dead path here is specifically:
  - `R1`
  - binary overlay
  - same pair
  - same source feature
- The alive question is narrower:
  - same chronology-clean pair
  - same source feature
  - `R3` only
  - frozen, low-amplitude size translation
- This is not a post-hoc claim that the binary overlay “really worked.” It did not. The role changes because off-signal expectancy stayed positive, which makes full suppression the wrong economic action to test first.

## Chosen next path

- **PREREG NOW:** `MGC US_DATA_830` source-session size context as an `R3` size modifier for the unfiltered `MNQ NYSE_OPEN E2 RR1.0 CB1` lane.
- **Not allowed:** binary veto, direction gating, allocator routing, cross-instrument expansion, or threshold shopping.

## Frozen policy shape

- Single source feature: `source_orb_size_norm = orb_US_DATA_830_size / atr_20`
- Single threshold: IS-only Q3 on the source feature
- Single size map:
  - `1.25x` when source state is `HIGH_Q3`
  - `1.00x` otherwise
- No `0.75x`, no multi-bucket map, no optimizer, no continuous fit.

## Why this exact map

- It is the minimum non-trivial `R3` translation.
- It preserves the sign implied by the dead `R1` test without pretending the sign was strong enough for take/skip.
- It caps implementation freedom and avoids turning a role-change into covert retuning.

## Non-choices

- No same-turn `MES` sibling prereg.
- No `R8` allocator version.
- No use of broad cross-asset ATR infrastructure as a surrogate for this session-specific source state.
- No further action on the binary overlay path beyond its published kill.

## Next action

1. Lock the `R3` prereg only.
2. Do not implement the runner until the prereg is committed and reviewed.
