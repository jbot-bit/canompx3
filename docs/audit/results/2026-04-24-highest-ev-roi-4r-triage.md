# Highest-EV ROI Triage For Next 0.4R+ Trade Set

**Date:** 2026-04-24
**Mode:** discovery triage / route selection
**Branch:** `codex/ev-roi-4r-hunt`
**Scope:** route selection for 0.4R+ trade-set discovery; not a live trade or deployment decision.
**Outcome:** route to one-shot validator on the highest-EV exact frontier cell (see § Final Recommendation).

## Goal

Find the highest expected-value next research action for surfacing a selected
trade set around `+0.4R` or better, without reopening broad scans or treating
derived shelves as proof.

This is not a live trade recommendation and not a deployment decision.

## Reproduction

- Repo doctrine used: `RESEARCH_RULES.md`,
  `docs/prompts/INSTITUTIONAL_DISCOVERY_PROTOCOL.md`, and
  `docs/institutional/research_pipeline_contract.md`.
- Truth layer used for measured claims: canonical `gold.db` tables
  `orb_outcomes`, `daily_features`, `validated_setups`, and
  `experimental_strategies`.
- Holdout boundary used for descriptive checks: `2026-01-01`.
- No Databento pull, paid data fetch, live broker query, or DB write was run.

## MEASURED Facts

Canonical data availability:

- `orb_outcomes` spans `2010-06-07` through `2026-04-19`.
- There are `4,565` distinct `orb_outcomes` trading days.

Existing shelf / candidate state:

- `validated_setups` has no active row with `expectancy_r >= 0.4`.
- `experimental_strategies` has no non-rejected row with `sample_size >= 100`
  and `expectancy_r >= 0.4`.
- Therefore the repo does not currently contain a validated or ready candidate
  inventory item that honestly answers "next 0.4R+ trade set."

Prior-day geometry shelf rows:

| Strategy | Status | N | ExpR | OOS ExpR |
|---|---:|---:|---:|---:|
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_DISPLACE_LONG` | active | 192 | +0.2396 | +0.1922 |
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_CLEAR_LONG` | active | 211 | +0.2270 | +0.2926 |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_PD_GO_LONG` | active | 321 | +0.2222 | +0.2553 |
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_GO_LONG` | active | 324 | +0.1934 | +0.2176 |
| `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_PD_CLEAR_LONG` | active | 303 | +0.1841 | +0.2212 |

Those are useful shelf facts, but they are not `0.4R+` selected-trade sets.

Exact `F5_BELOW_PDL` cells that still look like the 0.4R frontier:

| Cell | IS N | IS AvgR | IS p | 2026 N | 2026 AvgR | All N | All AvgR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `MNQ US_DATA_1000 O5 RR1.0 long F5_BELOW_PDL` | 136 | +0.3258 | 0.000037 | 8 | -0.0243 | 144 | +0.3064 |
| `MNQ US_DATA_1000 O5 RR1.5 long F5_BELOW_PDL` | 135 | +0.4022 | 0.000131 | 8 | -0.0878 | 143 | +0.3748 |
| `MNQ US_DATA_1000 O5 RR2.0 long F5_BELOW_PDL` | 133 | +0.3377 | 0.007675 | 8 | -0.2710 | 141 | +0.3032 |
| `MNQ NYSE_OPEN O5 RR1.0 short F5_BELOW_PDL` | 120 | +0.3531 | 0.000027 | 8 | -0.0167 | 128 | +0.3300 |
| `MNQ NYSE_OPEN O5 RR1.5 short F5_BELOW_PDL` | 117 | +0.4688 | 0.000037 | 8 | -0.0755 | 125 | +0.4340 |
| `MNQ NYSE_OPEN O5 RR2.0 short F5_BELOW_PDL` | 115 | +0.4149 | 0.002735 | 8 | +0.1094 | 123 | +0.3950 |

## INFERRED Decision

The highest-EV ROI research action is not a new broad scan. It is a bounded
confirmation design around the exact `F5_BELOW_PDL` frontier.

The cleanest current-stack candidate is:

`MNQ US_DATA_1000 O5 E2 RR1.5 long F5_BELOW_PDL`

Reasons:

- It is the only exact long-side `F5_BELOW_PDL` cell in this triage with
  pre-2026 selected-trade average above `+0.4R`.
- The long-side `PrevDayGeometryFilter` semantics already exist in code as a
  supported mode, although the exact `F5_BELOW_PDL` key is not currently a
  registered hypothesis-scoped filter.
- It sits near already-validated prior-day geometry shelf rows, so the
  mechanism family is not cold-start.

The stronger-looking `MNQ NYSE_OPEN O5 RR1.5 short F5_BELOW_PDL` cell is not
the best immediate ROI path despite higher all-period average:

- it requires a new short-side prior-day geometry predicate, because the
  current canonical `PrevDayGeometryFilter` owns long-direction semantics for
  this family;
- its 2026 descriptive sample is only `N=8` and negative;
- it is farther from the current validated prior-day shelf path.

## Limitations

There is no honest validated `0.4R+` setup ready to route today.

The 0.4R-looking path is an exact-cell research frontier, not a live book
decision. The 2026 descriptive samples are too small to confirm or kill alone,
but the negative 2026 prints on the top long/short `RR1.5` cells are a real
warning against promotion optimism.

## Recommended Next Action

Write one narrow prereg:

- route: `standalone_discovery`
- object: exact standalone candidate
- candidate: `MNQ US_DATA_1000 O5 E2 RR1.5 CB1 F5_BELOW_PDL`
- allowed filter implementation: register exact hypothesis-scoped
  `F5_BELOW_PDL` using the existing `PrevDayGeometryFilter(mode="below_pdl_long")`
- K: `1`
- holdout: `2026-01-01`
- required kill checks:
  - validator rejects lifecycle or phase-4 legality
  - `2026` OOS remains non-positive or below Criterion 8 effect ratio once the
    exact validator path computes it
  - era stability fails
  - standalone candidate loses to the broader `PD_DISPLACE_LONG` /
    `PD_GO_LONG` shelf rows on risk-adjusted or routing-relevant metrics

Do not do:

- do not reopen broad prior-day mega exploration;
- do not rerun consumed `PD_*` hypothesis files;
- do not use the validated prior-day shelf rows as proof of a `0.4R+` set;
- do not implement short-side `F5_BELOW_PDL` until the long-side exact cell is
  confirmed or killed.

## Final Recommendation

**NARROW.**

If the target is specifically "find a `0.4R+` trade set," the best ROI path is
the exact `MNQ US_DATA_1000 O5 RR1.5 long F5_BELOW_PDL` prereg. If the target
is "add deployable expected R soon," the prior-day shelf/routing branch remains
cleaner, but it is not a `0.4R+` path.
