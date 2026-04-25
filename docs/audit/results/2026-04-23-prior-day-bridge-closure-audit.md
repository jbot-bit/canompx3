# Prior-Day Bridge Closure Audit

**Date:** 2026-04-23

## Purpose

Close the gap between the 2026-04-22 MNQ prior-day bridge executions and the
current repo story.

This is **not** a fresh discovery pass. It is a closure audit:

- which exact bridge hypotheses were already consumed
- which ones survived to the validated shelf
- whether the branch is still open at the signal layer or only at the
  portfolio / routing layer

## Source-of-Truth Split

### Signal truth

Canonical only:

- `orb_outcomes`
- `daily_features`
- `trading_app.config.ALL_FILTERS` exact `PrevDayGeometryFilter` semantics

### Shelf / routing state

Operational state, not signal proof:

- `experimental_strategies`
- `validated_setups`
- `docs/runtime/lane_allocation.json`
- `trading_app/validated_shelf.py`

Orientation only, not proof:

- archived handoffs
- prior summaries

## MEASURED — hypothesis consumption state

All six bridge hypothesis files were already consumed exactly once through the
discovery/validation path.

| hypothesis file | strategy_id | experimental status | validated shelf |
|---|---|---|---|
| `2026-04-22-mnq-usdata1000-near-pivot-50-avoid-v1.yaml` | `MNQ_US_DATA_1000_E2_RR1.0_CB1_F3_NEAR_PIVOT_50` | `REJECTED` | no |
| `2026-04-22-mnq-usdata1000-downside-displacement-take-v1.yaml` | `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_DISPLACE_LONG` | `PASSED` | yes |
| `2026-04-22-mnq-usdata1000-clear-of-congestion-take-v1.yaml` | `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_CLEAR_LONG` | `PASSED` | yes |
| `2026-04-22-mnq-usdata1000-positive-context-union-v1.yaml` | `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_GO_LONG` | `PASSED` | yes |
| `2026-04-22-mnq-usdata1000-rr15-positive-context-union-v1.yaml` | `MNQ_US_DATA_1000_E2_RR1.5_CB1_PD_GO_LONG` | `PASSED` | yes |
| `2026-04-22-mnq-comex-pd-clear-long-take-v1.yaml` | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_PD_CLEAR_LONG` | `PASSED` | yes |

Rejected exact-cell detail:

- `MNQ_US_DATA_1000_E2_RR1.0_CB1_F3_NEAR_PIVOT_50`
  - rejection reason: `criterion_9: era 2020-2022 ExpR=-0.0645 < -0.05 (N=284)`

## MEASURED — canonical signal recheck

Independent recheck from `orb_outcomes + daily_features`, using the exact live
filter logic in `PrevDayGeometryFilter.matches_df(...)` and the project holdout
boundary `2026-01-01`.

These are descriptive signal-layer facts only. Small 2026 counts are **not**
promotion proof by themselves.

| filter | session | RR | N_IS_on | ExpR_IS | N_OOS_on | ExpR_OOS |
|---|---|---:|---:|---:|---:|---:|
| `F3_NEAR_PIVOT_50` | `US_DATA_1000` | 1.0 | 618 | -0.0344 | 19 | -0.1771 |
| `PD_DISPLACE_LONG` | `US_DATA_1000` | 1.0 | 205 | +0.2320 | 10 | +0.1745 |
| `PD_CLEAR_LONG` | `US_DATA_1000` | 1.0 | 232 | +0.2155 | 13 | +0.1962 |
| `PD_GO_LONG` | `US_DATA_1000` | 1.0 | 350 | +0.1811 | 15 | +0.2993 |
| `PD_GO_LONG` | `US_DATA_1000` | 1.5 | 347 | +0.2078 | 15 | +0.2964 |
| `PD_CLEAR_LONG` | `COMEX_SETTLE` | 1.0 | 338 | +0.1602 | 15 | +0.1321 |

## MEASURED — shelf vs live-routing state

- Five prior-day bridge survivors are on the validated shelf.
- Zero of those five strategy IDs appear in `docs/runtime/lane_allocation.json`.
- The repo's shelf semantics explicitly allow this:
  - `validated_setups` is a deployable shelf, not the live book
  - lane routing is a later policy / allocator decision

So **not being live-routed is not itself a bug**.

## MEASURED — actual closure failure

The failure is narrower:

- the hypothesis files were consumed
- the strategies were either rejected or promoted
- but the branch was not closed into a durable live result / policy decision
  surface in the working tree

Practical effect:

- later sessions could still treat Track A as "execute next"
- the repo looked as if discovery was unfinished
- the real open question (portfolio role / routing) was obscured

## Verdict

### Signal layer

**Closed.**

- `F3_NEAR_PIVOT_50` exact avoid cell is dead on this lane.
- The broader positive prior-day geometry families are alive on:
  - `MNQ US_DATA_1000 O5 E2 RR1.0 long`
  - `MNQ US_DATA_1000 O5 E2 RR1.5 long`
  - `MNQ COMEX_SETTLE O5 E2 RR1.0 long`

### Portfolio / routing layer

**Open.**

The honest next question is no longer:

- "run Track A"

It is:

- are these five shelf survivors additive to the current MNQ live book,
- same-session substitutes,
- or valid shelf rows that should remain unrouted?

## Next Step

Run one bounded **prior-day geometry routing / additivity audit** against the
current MNQ live profile.

Required output:

1. one table for the five promoted bridge survivors
2. overlap / substitution view versus the current live same-session lanes
3. portfolio contribution view versus the current `topstep_50k_mnq_auto` book
4. explicit decision per row:
   - `ROUTE_LIVE`
   - `KEEP_ON_SHELF`
   - `PARK_NON_ADDITIVE`

## Do Not Do

- do not rerun the six consumed hypothesis files
- do not reopen broad prior-day mega exploration
- do not treat shelf presence as automatic deployment
- do not reopen the dead `MES E1 rel_vol` family as a binary confirmation gate

## Bottom Line

Track A did not stall at discovery. It stalled at closure.

The prior-day bridge signal work is already mostly done:

- 1 exact avoid cell dead
- 5 broader geometry shelf survivors alive

The missing work is the institutional last mile:

- durable closure
- portfolio-role decision
- routing or explicit park
