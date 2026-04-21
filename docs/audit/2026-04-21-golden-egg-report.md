# 2026-04-21 golden-egg report

## Session summary

This report closes the orthogonal canonical golden-egg hunt on
`research/orthogonal-golden-egg-hunt-v1`.

Session trial count:

- executed trials: `20`
- blocked before data contact: `1` family (`mes-session-boundary-v1`)
- remaining trial budget: `280`

Primary deliverable status:

- posture-clearing contribution (`PCC-1` through `PCC-6`): complete
- orthogonal hunt top-3 families: complete

## Family verdicts

| Hypothesis | Empirical verdict | Deployment-posture verdict | Evidence |
|---|---|---|---|
| `2026-04-21-mgc-regime-ortho-v1` | `DEAD` | `N/A` | Zero family BH-FDR survivors; strongest raw cell failed family q and both family nulls stayed inside noise. |
| `2026-04-21-mes-session-boundary-v1` | `DEAD` | `N/A` | Blocked before data contact because the locked `ASIA_RANGE_ATR_Q67_HIGH` feature is look-ahead for all scoped sessions. |
| `2026-04-21-mgc-microstructure-v1` | `DEAD` | `N/A` | Timing-valid at ORB close, but zero family BH-FDR survivors and both family nulls kept max-t inside noise. |

There are **no GOLD** candidates and **no SILVER** candidates in this session.
Nothing in this hunt is currently waiting on Terminal 1's ONC / posture
resolution, because nothing survived empirically to reach that posture stage.

## Evidence pack

### `2026-04-21-mgc-regime-ortho-v1`

- result: `docs/audit/results/2026-04-21-mgc-regime-ortho-v1.md`
- current strongest cell:
  - `LDM_PREVWEEK_LONG`
  - `Δ_IS = +0.2079`
  - `t_IS = 2.602`
  - `q_family = 0.0587`
- family nulls:
  - destruction-shuffle `p=0.3466`
  - rng-null `p=0.3586`
- disposition:
  - empirically dead under the locked family gate
  - not a posture-blocked candidate

### `2026-04-21-mes-session-boundary-v1`

- block note: `docs/audit/2026-04-21-mes-session-boundary-timing-block.md`
- failure mode:
  - `session_asia_high/low` is only complete after `17:00` Brisbane
  - scoped sessions are `BRISBANE_1025`, `TOKYO_OPEN`, `SINGAPORE_OPEN`
  - therefore the locked `ASIA_RANGE_ATR_Q67_HIGH` predicate is timing-invalid
- disposition:
  - not an empirical miss
  - the exact locked family belongs in the NO-GO memory as a timing-invalid
    construction
  - this does **not** kill the broader session-boundary POV family forever

### `2026-04-21-mgc-microstructure-v1`

- result: `docs/audit/results/2026-04-21-mgc-microstructure-v1.md`
- timing validity:
  - both features use only the five ORB formation bars in `[orb_start_utc, orb_end_utc)`
  - fully known at ORB close, before `E2`
- coverage reality:
  - only `US_DATA_830` had scoped rows
  - `BRISBANE_1025` stayed zero-coverage
- strongest raw cell:
  - `USD830_RANGECONC_LONG`
  - `Δ_IS = +0.0828`
  - `t_IS = 0.924`
  - `q_family = 1.0000`
- family nulls:
  - destruction-shuffle `p=0.4263`
  - rng-null `p=0.3944`
- disposition:
  - empirically dead under the locked family gate
  - not a posture-blocked candidate

## Questions required by the hunt

### Q1. Are there any GOLD candidates?

No.

### Q2. Are there SILVER candidates waiting on Terminal 1's posture resolution?

No.

The hunt did not produce any empirical survivor that reached the posture stage.
Terminal 1's ONC work remains critical for the live-six baseline, but it does
not currently unlock any orthogonal survivor from this session.

### Q3. Which dead candidates should land in the NO-GO registry?

Drafted NO-GO candidates from this session:

- `2026-04-21-mgc-regime-ortho-v1`
  - reason: empirically dead at family level; no survivor and nulls non-rejecting
- `2026-04-21-mgc-microstructure-v1`
  - reason: empirically dead at family level; timing-valid but no signal
- `2026-04-21-mes-session-boundary-v1` exact locked family
  - reason: timing-invalid design because `ASIA_RANGE_ATR_Q67_HIGH` is
    look-ahead for the scoped sessions

These are drafted dispositions only. No non-trivial amendment to
`docs/STRATEGY_BLUEPRINT.md` has been made.

### Q4. What's the single best critical-path move to a first honest live strategy given current state?

Use the PCC artifacts, not the orthogonal candidates, to tighten the posture
path on the live-six baseline. The clearest contribution from this session is
the posture map:

- `PCC-1` ruled out intrinsic framework-calibration bias in the Phase B gate stack
- `PCC-4` clarified that all six current live lanes are provisional only
- `PCC-6` ranked the nearest clean Mode A rediscovery path:
  - `TOKYO_OPEN`
  - `SINGAPORE_OPEN`
  - `US_DATA_1000`

The hunt itself did **not** produce a faster alternate route to first live.

### Q5. Did any PCC item materially shorten Terminal 1's path?

Yes.

`PCC-6` materially shortened the next-step choice by ranking the six live lanes
for clean Mode A rediscovery, while `PCC-1` removed the need to chase a false
"framework is intrinsically biased" theory.

## Phase 7 fork memo

Scoring scale: `1` low / poor, `5` high / strong.

| Option | S1 posture clearing | S2 time-to-live | S3 independence from Terminal 1 | S4 integrity risk | S5 reversibility | Notes |
|---|---:|---:|---:|---:|---:|---|
| `A*` escalate PCC findings to user for Terminal 1 re-scoping | 5 | 4 | 3 | 5 | 5 | Best use of this session's real output; no surviving alternate candidate exists. |
| `B` advance a posture-blocker-proof GOLD candidate | 0 | 0 | 0 | 5 | 5 | Inapplicable because no GOLD candidate exists. |
| `C` hold SILVER candidates behind Terminal 1 | 0 | 0 | 0 | 5 | 5 | Inapplicable because no SILVER candidate exists. |
| `D` expand POV map next tier | 2 | 2 | 4 | 3 | 4 | Possible, but lower EV than using the PCC outputs already earned. |
| `E` retire deads to NO-GO registry and stop | 3 | 2 | 5 | 5 | 5 | Reasonable cleanup move, but still secondary to feeding the posture path. |
| `F` pause for Terminal 1 ONC mid-session landing | 0 | 0 | 0 | 5 | 5 | Did not occur during this run. |

### Recommendation

**Recommend `A*`: escalate the PCC findings as the main output of this session.**

Rationale:

- no orthogonal candidate survived empirically
- no posture-blocker-proof hedge exists right now
- the highest-EV contribution is the posture-shortening evidence already
  produced in `PCC-1`, `PCC-4`, and `PCC-6`
