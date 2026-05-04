---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# ZT Stage-1 Triage Gate

**Date:** 2026-03-15
**Status:** Ready
**Scope:** Read-only triage for `ZT` data arrival and first-pass research viability

---

## Why This Exists

The repo already has a live rates thread:

- `2YY` is present in [asset_configs.py](/mnt/c/users/joshd/canompx3/pipeline/asset_configs.py)
- there are fresh `2YY` design notes and a narrow rates event-study direction

At the same time, `ZT` matters for a different reason:

- it is the cleaner institutional benchmark for 2-year U.S. rates exposure
- it is likely to have better market quality and broader adoption than `2YY`
- if the goal is a real diversification lane, `ZT` is worth a direct viability check the moment data lands

This document does **not** replace the `2YY` docs.
It defines the read-only gate for assessing `ZT` as:

- a benchmark sanity check
- a possible better first rates expression
- a candidate for a narrow event-window research pass

---

## Decision Question

When `ZT` files finish downloading, answer this question first:

**Does direct `ZT` data fit this repo cleanly enough, and look liquid/structured enough, to justify a narrow rates research pass before any strategy engineering?**

This is not asking whether `ZT` is already a strategy.
It is only asking whether `ZT` deserves Stage-1 research budget.

---

## What This Gate Is Allowed To Do

- inspect metadata and file coverage
- inspect naming / contract pattern compatibility
- assess whether the product is a clean fit for the repo's data path
- define the narrow first-pass event windows
- decide `GO`, `NO-GO`, or `BENCHMARK_ONLY`

## What This Gate Must Not Do

- add `ZT` to production trading logic
- assume session-open ORB is the right model
- kick off a broad Treasury scan
- widen into `ZN`, curve spreads, auctions, Fed speakers, and macro kitchen sink all at once
- call anything an edge

---

## Immediate Arrival Checks

Run these as soon as the `ZT` folder exists.

### 1. Metadata sanity

Inspect:

- `DB/ZT_DB/metadata.json`
- `DB/ZT_DB/condition.json`

Verify:

- dataset is `GLBX.MDP3`
- schema is `ohlcv-1m`
- symbol request is direct `ZT.FUT`
- start/end dates are long enough for a serious event study
- packaging looks identical to other repo instruments

### 2. Coverage sanity

Check:

- number of daily `.dbn.zst` files
- first and last file dates
- obvious missing multi-month holes

Minimum standard:

- enough history to make monthly event families meaningful
- no visible “tiny fragment” issue

### 3. Contract-expression sanity

Decide whether the symbol expression looks standard enough for normal onboarding:

- expected pattern should be `ZT[FGHJKMNQUVXZ]YY`
- no weird special-case symbol bridge
- no evidence that the product requires a custom one-off ingest path

### 4. Repo-fit sanity

Ask:

- does this look like one more normal futures instrument in the current DBN workflow?
- or does it look like a product that will create path-specific complexity immediately?

If it requires special casing on day one, that is a warning sign.

---

## Product-Level Questions

Before any strategy work, answer these structurally:

### A. Why should `ZT` diversify the current book?

Expected answer:

- different macro driver than gold/equity ORB
- front-end U.S. rates repricing around scheduled macro events
- different stress behavior than the current gold plus equity-micro cluster

Bad answer:

- “Treasuries are probably uncorrelated”

### B. Why should `ZT` have tradable intraday structure at all?

Expected answer:

- macro release windows
- FOMC windows
- Treasury auction windows
- cash-session and settlement behavior

Bad answer:

- “we should just try the same ORB sessions and see”

### C. Why `ZT` instead of only `2YY`?

Possible acceptable answers:

- better market-quality benchmark
- cleaner institutional expression of 2-year rates
- better chance of a durable execution lane if the mechanism survives

### D. What would kill it quickly?

Examples:

- data quality is fragmentary
- no clean event response in the narrow first pass
- behavior looks thin, erratic, or entirely threshold-dependent
- it adds narrative but not portfolio value

---

## First-Pass Research Scope

If `ZT` passes arrival checks, the first pass must stay narrow.

### In Scope

Wave 1A only:

1. `CPI` at `8:30 ET`
2. `NFP` at `8:30 ET`
3. `FOMC statement` at `2:00 PM ET`

Optional only if Wave 1A survives:

4. exact Treasury auction windows

### Model Families In Scope

Only two model framings:

1. continuation after a real event shock
2. failed-first-move reversal after an event shock

### Explicitly Out Of Scope

- generic 24-hour session scans
- copied gold/equity ORB session grids
- broad filter soups
- confirm-bar / RR / threshold explosions
- calling `ZT` an ORB candidate just because it is a futures contract

---

## Stage-1 Viability Criteria

`ZT` earns a `GO` only if all of these look credible:

1. **Data fit**
   - direct minute data is present, clean, and structurally normal

2. **Mechanism**
   - the event thesis can be explained in one paragraph without hand-waving

3. **Sample path**
   - the chosen event families can plausibly reach PRELIMINARY / CORE evidence over time

4. **Execution realism**
   - expected move size versus friction is not obviously hopeless

5. **Portfolio role**
   - the lane has a plausible path to being meaningfully different from the current book

---

## Classification

Use one of these outcomes after arrival triage:

### `GO`

Use when:

- data landed cleanly
- repo fit is normal
- the product has a credible mechanism and sample path
- it deserves a narrow event-study implementation spec

### `BENCHMARK_ONLY`

Use when:

- data is good
- product is useful as a market-quality comparison
- but the first actual research lane should still stay with `2YY` or another rates expression

### `NO-GO`

Use when:

- data quality or repo fit is poor
- the event-study path already looks too weak
- the product adds complexity without clear research upside

---

## Exact First Response Template

When `ZT` finishes downloading, respond in this structure:

```text
ZT ARRIVAL CHECK

DATA:
- metadata verdict
- coverage verdict
- symbol / contract-pattern verdict

MECHANISM:
- one-paragraph institutional case

RISKS:
- what could make this a bad rates lane

CLASSIFICATION:
- GO / BENCHMARK_ONLY / NO-GO

NEXT STEP:
- exact narrow research pass, or stop reason
```

---

## Commands To Run Immediately

These are the first local checks to run once the folder exists:

```bash
sed -n '1,220p' DB/ZT_DB/metadata.json
find DB/ZT_DB -maxdepth 1 -name '*.dbn.zst' | wc -l
find DB/ZT_DB -maxdepth 1 -name '*.dbn.zst' | sort | sed -n '1,3p'
find DB/ZT_DB -maxdepth 1 -name '*.dbn.zst' | sort | tail -n 3
```

Optional benchmark comparison versus `2YY`:

```bash
sed -n '1,80p' DB/2YY_DB/metadata.json
sed -n '1,80p' DB/ZT_DB/metadata.json
```

---

## Bottom Line

When `ZT` lands, the first job is not to trade it.

The first job is to decide whether it is:

- the better real rates lane
- only a benchmark cross-check
- or not worth more repo attention

That decision should be made explicitly, fast, and read-only.
