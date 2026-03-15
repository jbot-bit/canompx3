# 2YY Direct Data Evidence Note

**Date:** 2026-03-15
**Status:** Design doc only
**Decision:** `GO_DIRECT`

---

## Decision

Proceed with the `2YY` research path using **direct `2YY` data**, not a proxy.

This is the correct call based on the current evidence because:

- the canonical repo vendor appears to support `2YY` directly
- the product fits the existing Databento `GLBX.MDP3` pipeline model
- the available historical depth appears long enough for monthly event research
- there is no current evidence that a proxy is needed to answer the Stage-1 question

The first implementation task should therefore be:

**prove the direct `2YY` pull works in this repo's normal data path**

Not:

- build a proxy research branch
- widen to `5YY`
- start coding strategy logic before data viability is proved

---

## Evidence

### 1. Vendor confirmation

Databento currently lists a direct `2YY` product page under the same CME dataset family already used by the repo:

- dataset: `GLBX.MDP3`
- product: `2YY`
- schema availability includes `OHLCV-1m`
- listed historical coverage begins `2010-06-06`

That is enough to justify a direct-path first decision.

This matters because the Stage-1 spec only needs:

- direct minute bars
- exact `8:30 ET` event alignment
- enough history for `CPI` and `NFP` event studies

The listed coverage is comfortably long enough for monthly macro-release research if the pull works cleanly in practice.

### 2. Repo fit

The current repo architecture already expects:

- Databento DBN files
- ingest into `bars_1m`
- deterministic build into `bars_5m`
- research and downstream analysis from the canonical DuckDB path

That means `2YY` is not asking for a different research system.
It is asking for one more instrument onboarding path inside the existing one.

### 3. Onboarding shape looks standard

Nothing currently inspected suggests `2YY` would require an exotic integration.

The visible gaps are ordinary onboarding gaps:

- add `2YY` to `pipeline/asset_configs.py`
- add a normal DBN path
- add a normal download mapping if `refresh_data.py` should support it
- ingest and build bars through the standard pipeline

That is normal repo work, not architectural drift.

### 4. No clean reason for a proxy yet

A proxy should only be used when the direct path is unavailable or too weak.

That is not the current situation.
Right now, the evidence points the other way:

- direct product exists
- direct minute schema exists
- direct history appears long enough

So using a proxy at this stage would add explanation burden without a clear need.

---

## What This Decision Does And Does Not Mean

### What it means

- `2YY` has enough external and architectural support to justify a direct implementation attempt
- the first implementation step should be a live vendor/data-path proof, not more planning
- proxy use is not justified as the default route

### What it does not mean

- `2YY` is now fully onboarded in the repo
- the local Databento account/license is already proven for this product
- the direct pull has been executed successfully in this session
- the strategy idea itself is validated

This is a **path viability decision**, not a research win.

---

## Why Not `GO_PROXY`

`GO_PROXY` would only be better if one of these were true:

- direct `2YY` minute data was unavailable
- direct history was too short
- repo integration required a messy special case
- the direct product was obviously too thin or unusable

None of those are the best current reading of the evidence.

Using a proxy now would increase narrative risk:

- more explanation
- more bridge assumptions
- more ways to accidentally widen the mechanism

That is bad process for the first pass.

---

## Why Not `STOP`

`STOP` would be correct if the path looked weak at the vendor or repo-fit level.

That is not what the evidence says.

The evidence says:

- the market exists
- the vendor path exists
- the repo path is plausible
- the research question is narrow enough to justify one direct test

So stopping now would be overly conservative.

---

## Implementation Opening Task

The next task should be one small, concrete proof:

### Objective

Show that the repo can obtain direct `2YY` minute data in the same canonical shape as existing instruments.

### Success criteria

- direct `2YY` symbol/product request is accepted
- a small test pull succeeds
- timestamps are usable for `8:30 ET` event work
- the onboarding path can follow the normal instrument pattern

### Failure criteria

- product pull fails under the actual account/vendor setup
- schema is not practically usable
- instrument mapping is weirder than expected
- direct path turns out weaker than the catalog evidence suggested

If that happens, revisit the gate and consider `GO_PROXY` or `STOP`.

---

## Bottom Line

The rates path is now past the vague-idea stage.

The correct next call is:

**`GO_DIRECT` on `2YY`, with the first live pull as the opening implementation proof.**

That is the cleanest, most honest way to move from diversification theory toward actually finding a new strategy.

---

## Sources

- Databento `2YY` catalog:
  - https://databento.com/catalog/cme/GLBX.MDP3/futures/2YY
- Databento futures overview:
  - https://databento.com/futures
- CME yield futures overview:
  - https://www.cmegroup.com/articles/2024/understanding-yield-futures.html
