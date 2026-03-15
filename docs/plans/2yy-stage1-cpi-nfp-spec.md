# 2YY Stage-1 CPI / NFP Research Spec

**Date:** 2026-03-15
**Status:** Design doc only
**Scope:** One exact first-pass research spec for `2YY` on `CPI` and `NFP`

---

## Purpose

Define the first implementable diversification research pass without giving permission for a broader rates program.

This spec inherits the guardrails in:

- `RESEARCH_RULES.md`
- `TRADING_RULES.md`
- `docs/plans/diversification-research-program.md`
- `docs/plans/2yy-5yy-stage0-research-memo.md`

This is **not** a production feature spec.
It does **not** approve new tables, config entries, or broad rates research.

---

## Exact Research Question

For `2YY` only:

**On `CPI` and `NFP` release days, does the first post-release price response contain a repeatable continuation or failed-first-move pattern that is economically plausible, small enough to test honestly, and potentially additive to the current ORB book?**

This first pass is intentionally narrower than the Stage-0 memo:

- one product: `2YY`
- two event families: `CPI`, `NFP`
- two model framings: continuation, failed-first-move
- fixed holding structures

If this pass is weak, stop. Do not widen to `5YY`, `FOMC`, auctions, or broader rates work.

---

## Why This Exact Cut

This is the cleanest first pass because:

- `2YY` is the closest tenor to front-end policy repricing
- `CPI` and `NFP` are the most defensible first event families at `8:30 ET`
- the event clock is exact and well known
- monthly event frequency gives a plausible path to a meaningful event sample over a long history

This is an event study first, not a session-open strategy search.

---

## In Scope

### Product

- `2YY` only

### Event Families

- `CPI` release days at `8:30 ET`
- `NFP` release days at `8:30 ET`

### Time Windows

Use these exact windows unless data quality forces a stop:

1. **Pre-event reference**
   - `8:20 ET` to `8:29:59 ET`

2. **Initial shock window**
   - `8:30 ET` to `8:34:59 ET`

3. **Follow-through window A**
   - `8:35 ET` to `8:39:59 ET`

4. **Follow-through window B**
   - `8:35 ET` to `8:44:59 ET`

5. **Follow-through window C**
   - `8:35 ET` to `8:49:59 ET`

These three follow-through windows are the only allowed holding structures in this pass.

---

## Out Of Scope

- `5YY`
- `10Y`
- `FOMC`
- auctions
- PPI, retail sales, ISM, Fed speakers, or any other `8:30 ET` event
- generic ORB cloning
- all-session scans
- threshold grids
- new overlays on the current book
- any production or schema work

---

## Data Requirements

The implementation pass should stop before coding if any of these fail:

1. `2YY` minute data around `8:30 ET` is not available with reliable timestamps.
2. Event-date mapping for `CPI` and `NFP` cannot be assembled defensibly.
3. The available history for direct `2YY` data is too short to make the event count meaningful.
4. A proxy is required but the proxy-to-execution bridge cannot be defended cleanly.

### Proxy Rule

Direct `2YY` is the default.

If direct `2YY` history is too short, the implementer must write a short bridge note before proceeding that states:

- what proxy is being used
- why the proxy is structurally close enough
- what the execution product would be
- what would invalidate that bridge

No silent substitution.

---

## Signal Definitions

The point of Stage 1 is to test a mechanism cleanly, not to optimize thresholds.

Use exactly two model framings.

### Model A: Continuation

Definition:

- measure the signed net move during the initial shock window (`8:30-8:34:59 ET`)
- measure the signed net move over each allowed follow-through window
- continuation exists when the follow-through move keeps the same sign as the initial shock

Primary question:

- after a meaningful first release impulse, does the move continue in the same direction over the next `5`, `10`, or `15` minutes?

### Model B: Failed First Move

Definition:

- measure the signed net move during the initial shock window
- classify a failed-first-move only when the direction of the follow-through window is opposite the initial shock

Primary question:

- after the first release impulse, is there a repeatable reversal over the next `5`, `10`, or `15` minutes?

### Important Constraint

Do **not** add extra signal filters in this pass:

- no volatility thresholds beyond the fixed event framing
- no confirm-bar counts
- no additional direction filters
- no day-type or regime overlays
- no secondary event classification

If the mechanism needs that much help to appear, it is not ready.

---

## Allowed Variation Count

The entire first pass is capped at **12 structured variations**:

- 2 event families: `CPI`, `NFP`
- 2 model framings: continuation, failed-first-move
- 3 follow-through windows: `5`, `10`, `15` minutes

That is the full search space.

No additional branches are allowed inside Stage 1.

If an implementer proposes more than 12 structured variations, the spec is being violated.

---

## Output Shape

Stage 1 should produce two artifacts only.

### 1. Event-Level Research Table

One row per event instance with these minimum fields:

- `event_date`
- `event_family`
- `instrument`
- `pre_event_open`
- `pre_event_close`
- `shock_open`
- `shock_close`
- `shock_direction`
- `shock_magnitude`
- `fw_5m_close`
- `fw_10m_close`
- `fw_15m_close`
- `cont_5m`
- `cont_10m`
- `cont_15m`
- `rev_5m`
- `rev_10m`
- `rev_15m`
- `usable_event_flag`
- `exclusion_reason`

### 2. Research Memo / Summary Report

The summary must report, for each of the 12 structured variations:

- event count
- exact date range
- average signed move
- median signed move
- win rate
- exact p-value where appropriate
- variation count tested
- first-half / second-half split behavior
- notes on parameter fragility or instability

It must end with:

```text
SURVIVED SCRUTINY:
DID NOT SURVIVE:
CAVEATS:
NEXT STEPS:
```

No DB writes are approved by this spec.
Flat files or local research outputs are sufficient.

---

## Screening Standards

The implementation must explicitly answer these questions:

1. Does either event family show stable directional behavior after the first shock?
2. Is the sign of the effect consistent enough across the sample to matter?
3. Does the effect survive simple first-half / second-half splits?
4. Does the effect remain after a basic friction sanity check?
5. Is the event-day behavior different enough from the current ORB book to justify deeper work?

The last point can be lightweight at Stage 1:

- event-day overlap with representative current ORB activity
- whether the promising days are obviously just the same high-vol days the current book already wins on

If the answer is "this is just another way of monetizing the same days," that counts against continuation.

---

## Kill Conditions

Stop and mark the path `NO-GO` if any of these happen:

1. **Data failure**
   - timestamps, event mapping, or direct `2YY` data are not defensible

2. **Weak event study**
   - both `CPI` and `NFP` look directionally unstable across all three holding windows

3. **Variation creep**
   - the implementation starts adding extra thresholds, filters, or event classes

4. **Proxy dependence**
   - the result only exists under a weakly justified proxy

5. **Economically thin effect**
   - the average move is too small or too inconsistent to survive basic friction reasoning

6. **Same-day clone risk**
   - the promising cases look like the same macro-volatility days already captured by the current ORB book, with little independent diversification value

7. **Narrative over evidence**
   - the macro explanation sounds good but the actual event results are unstable, sign-flipping, or sample-fragile

---

## Continuation Gate

Only continue to `5YY` or a broader rates program if at least one `2YY` event family survives with all of the following:

- a clear one-paragraph mechanism
- stable sign across at least one follow-through window
- no variation creep beyond the 12 allowed branches
- no obvious dependence on a weak proxy bridge
- enough event count to justify deeper validation
- at least a plausible case for portfolio additivity versus the current ORB book

If those conditions are not met, stop.

---

## Recommended Implementation Order

1. Prove direct `2YY` data availability and timestamp quality.
2. Build the `CPI` and `NFP` event calendar.
3. Produce the event-level table with no strategy engineering.
4. Compute the 12 structured variation summaries.
5. Write the honest summary memo.
6. Decide `GO / NO-GO` before any `5YY`, `FOMC`, or auction work.

---

## Bottom Line

This Stage-1 spec exists to answer one narrow question:

**Does `2YY` show a real post-`CPI` or post-`NFP` event effect worth taking seriously, or does the rates path die immediately under disciplined scrutiny?**

That is the whole job.
