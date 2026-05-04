---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# MZC Stage-1 Findings

**Date:** 2026-04-20
**Parent spec:** `docs/plans/2026-04-20-mzc-stage1-spec.md`
**Status:** Stage 1 complete

---

## Verdict

**`GO_TO_STAGE_2`, but only with `ZC` as the primary research proxy.**

`MZC` is not dead, and the data budget is not the problem. The constraint is
the tape quality of the native micro contract around the exact USDA event
window.

The honest read is:

- `MZC` is suitable as the eventual execution vehicle / translation target
- `MZC` is **not** suitable as the sole Stage-2 research tape for 12:00 PM ET
  USDA event windows
- `ZC` is the correct primary research tape for the Stage-2 event study

---

## Grounded event windows

Official USDA / NASS grounding used:

- WASDE monthly releases: `12:00 PM ET`
  - USDA WASDE page: <https://www.usda.gov/about-usda/general-information/staff-offices/office-chief-economist/commodity-markets/wasde-report>
- Prospective Plantings:
  - 2026 release on `March 31, 2026`
  - USDA/NASS release pages indicate the report-day release cadence at `12:00 PM ET`
- Acreage / Grain Stocks:
  - NASS calendar shows `12:00 PM ET` on `June 30, 2025`
  - page: <https://www.nass.usda.gov/Publications/Calendar/reports_by_date.php?month=06&year=2025>

For the 2025-2026 spot checks below, `12:00 PM ET` corresponds to `16:00 UTC`
because those dates are during U.S. DST.

---

## Gate A. Vendor / data budget

Verified via Databento metadata:

- `MZC.FUT` `ohlcv-1m`, `2025-02-01 -> 2026-04-20`
  - cost: `$0.0`
  - billable size: `3,109,456`
- `ZC.FUT` `ohlcv-1m`, `2010-06-06 -> 2026-04-20`
  - cost: `$0.0`
  - billable size: `1,197,474,432`

Decision:

- Stage 1 is **not** blocked by data budget
- Stage 2 can justify `ZC` onboarding if the structural case holds

---

## Gate B. Event-window tape compatibility

Direct Databento minute-bar spot checks were run on:

- `2025-06-30` (`Acreage` / `Grain Stocks`)
- `2026-03-31` (`Prospective Plantings`)
- `2026-04-09` (`WASDE`)

Window checked:

- broad event window: `15:30-16:59 UTC`
- shock window: `16:00-16:04 UTC`

### Distinct covered minutes in the shock window

| Symbol | 2025-06-30 | 2026-03-31 | 2026-04-09 |
|---|---:|---:|---:|
| `MZC` | 4 / 5 | 2 / 5 | 3 / 5 |
| `ZC`  | 5 / 5 | 5 / 5 | 5 / 5 |

### Distinct covered minutes in the 90-minute event window

| Symbol | 2025-06-30 | 2026-03-31 | 2026-04-09 |
|---|---:|---:|---:|
| `MZC` | 26 / 90 | 69 / 90 | 33 / 90 |
| `ZC`  | 90 / 90 | 90 / 90 | 90 / 90 |

Interpretation:

- `MZC` is tradable and does print around the event
- but native micro coverage is too sparse and inconsistent for a clean
  primary Stage-2 event study
- `ZC` gives full minute coverage and is the honest research tape

---

## Gate C. Sample-path realism

`MZC` launched in 2025, so even with acceptable vendor access:

- native micro history is short
- event count is limited
- event-window sparsity compounds the sample problem

That means `MZC` alone is too weak a base for mechanism discovery.

`ZC` resolves both issues:

- long history
- dense report-window tape

So the right design is:

- discover on `ZC`
- translate carefully to `MZC`
- validate any deployment claim on native `MZC`, not on `ZC`

---

## Stage-1 decision

### `MZC`

- **Do not use as the sole Stage-2 research tape**
- keep as the eventual execution / translation surface

### `ZC`

- **Use as the primary Stage-2 research proxy**
- explicitly declare proxy mode in any hypothesis file

---

## What Stage 2 should be

A narrow `ZC`-based USDA-response study for corn only:

- event family:
  - WASDE
  - Prospective Plantings
  - Acreage / Grain Stocks
- fixed structures only:
  - immediate continuation
  - failed first move / reversal

No broad ORB or generic ag-session sweep should be run.

---

## What this finding does NOT claim

- It does not claim a corn edge exists.
- It does not claim `MZC` is deployable.
- It does not claim `ZC` and `MZC` have interchangeable execution behavior.

It only claims that:

- the ag branch is worth continuing
- and the honest research path is `ZC` proxy first, `MZC` translation second

