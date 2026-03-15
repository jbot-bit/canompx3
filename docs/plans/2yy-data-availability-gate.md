# 2YY Data Availability Gate

**Date:** 2026-03-15
**Status:** Design doc only
**Scope:** Pre-implementation gate for `2YY` research data viability and proxy policy

---

## Purpose

Before any `2YY` research code is written, answer one question honestly:

**Can this repo support a defensible `2YY` event-study path with direct data, or does the path stop here unless a clean proxy bridge is approved?**

This note exists to prevent a fake start where the research design looks clean on paper but the market-data path is weak, ad hoc, or silently widened.

This is not approval for:

- new ingestion code
- new pipeline tables
- new config entries
- broad rates onboarding

It is only the `GO / STOP / PROXY` gate.

---

## Current Repo Truth

Right now, the repo does **not** have a live `2YY` data path.

What exists:

- the canonical pipeline is Databento `GLBX.MDP3` -> `bars_1m` -> `bars_5m` -> `daily_features`
- the active ORB universe is still built around current configured symbols in `pipeline/asset_configs.py`
- `scripts/tools/refresh_data.py` only maps `MGC`, `MNQ`, `MES`, and `M2K` for automatic Databento backfill

What does **not** exist yet:

- a `2YY` asset config
- a `2YY` DBN path
- a `2YY` download symbol mapping
- a decided proxy policy for rates research

That means the Stage-1 spec is methodologically ready, but the data path is not yet proven.

---

## External Source Truth

The external picture is encouraging, but not enough on its own.

### Databento coverage

Databento currently lists:

- `2YY` on `GLBX.MDP3` with historical coverage shown from `2010-06-06`
- `ohlcv-1m` among the available historical schemas for the instrument

This is a positive sign because it means the current repo's canonical vendor and dataset appear to cover the market needed for the first pass.

But that does **not** prove the path is ready here yet.
The repo still needs a defensible direct-data decision before implementation.

### CME product truth

CME's yield futures material supports the mechanism case:

- `2YY`, `5YY`, `10Y`, and `30Y` are real yield futures products
- the contracts are cash-settled and designed around fixed DV01 exposure
- CME frames them as tools for expressing views directly in yield terms rather than bond-price terms

This supports the strategic rationale for researching the product.
It does **not** remove the need to verify tradability, history depth, and practical research viability in this repo.

---

## GO / STOP / PROXY Decision Framework

The path only gets a `GO` if all of the following are true.

### Condition 1: Direct vendor path is real

You can identify a direct Databento path for `2YY` that matches the repo's research requirements:

- same vendor family already used by the repo
- minute data available
- timestamps suitable for `8:30 ET` event work
- enough history to make monthly event research worth doing

If this fails, stop unless a proxy bridge is written and approved.

### Condition 2: Repo integration path is simple

The repo can onboard `2YY` without inventing a weird side system:

- normal `asset_configs` pattern
- normal Databento parent or instrument mapping
- normal DBN ingest path
- no one-off sidecar dataset

If `2YY` requires a custom special-case ingestion path just to get started, that is a warning sign.

### Condition 3: Proxy, if needed, is structurally clean

If direct `2YY` is not viable for the needed lookback, a proxy is acceptable only if:

- the proxy measures the same economic object closely enough
- the execution product remains `2YY`
- the proxy-to-execution bridge can be stated in one short paragraph
- the bridge has a clear invalidation condition

If the proxy requires a long explanation or hand-wavy equivalence, do not use it.

### Condition 4: Tradability is not obviously fake

Before deeper work, the path must pass a basic reality check:

- the product is actually tradable in a way that could matter later
- the event-study idea is not built on a dead or unusably thin contract
- the expected event move is not obviously too small relative to friction and slippage reasoning

This does not require a full execution model yet.
It does require enough realism to avoid wasting research budget.

---

## Approved Outcomes

Only three outcomes are allowed from this gate.

### 1. `GO_DIRECT`

Use direct `2YY` via the normal Databento -> DBN -> pipeline path.

This is the preferred outcome.

### 2. `GO_PROXY`

Proceed only with a formally declared proxy bridge note that states:

- exact proxy instrument or series
- why it is structurally close enough
- why `2YY` remains the intended execution product
- what evidence would invalidate the bridge

This is acceptable only if direct `2YY` history is genuinely insufficient or operationally unusable for the first pass.

### 3. `STOP`

Stop the rates path here for now.

This is the correct result if:

- direct `2YY` data is weaker than it looked
- the proxy bridge is not clean
- the repo integration path becomes ad hoc
- tradability or history depth looks too weak to justify the research budget

Stopping early is a success if the path is not defensible.

---

## Proxy Policy

The default policy is:

**direct `2YY` first, proxy second, no silent substitution**

Proxy use is only acceptable under these rules:

1. The direct path was checked first and documented.
2. The proxy is declared explicitly before any event-study output is produced.
3. The proxy is used to answer the same narrow Stage-1 question, not to widen the program.
4. The proxy does not create a second research universe with different sessions, different mechanics, or different event behavior.

Bad proxy behavior includes:

- quietly switching to a different rates product because data is easier
- using a broader Treasury complex proxy that changes the mechanism materially
- using a proxy whose relationship to `2YY` is mostly narrative

---

## Minimum Evidence Package Before Implementation

Before any research code starts, write a short evidence note covering:

1. **Direct `2YY` vendor confirmation**
   - dataset
   - schema
   - history start shown

2. **Repo onboarding shape**
   - what would need to be added to support direct `2YY`
   - whether that follows the existing pipeline pattern cleanly

3. **Proxy decision**
   - `none required`, `required and justified`, or `path stopped`

4. **Reality check**
   - short statement on whether the product still looks worth the first research dollar

If this note is not clean, there is no implementation case.

---

## Bottom Line

The right next question is not "how do we code the `2YY` research?"

It is:

**"Does direct `2YY` data fit this repo cleanly enough to deserve any coding at all, and if not, is there a proxy bridge strong enough to defend?"**

That decision should be made explicitly before any rates research implementation begins.

---

## Official Sources

- Databento `2YY` catalog page:
  - https://databento.com/catalog/cme/GLBX.MDP3/futures/2YY
- CME Group yield futures overview:
  - https://www.cmegroup.com/articles/2024/understanding-yield-futures.html
