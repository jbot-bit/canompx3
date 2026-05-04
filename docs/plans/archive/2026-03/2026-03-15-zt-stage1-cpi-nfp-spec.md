---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# ZT Stage-1 CPI / NFP Research Spec

**Date:** 2026-03-15
**Status:** Ready
**Scope:** One exact first-pass research spec for `ZT` on `CPI` and `NFP`

---

## Purpose

Define the first implementable `ZT` research pass without opening a broad Treasury program.

This spec inherits the guardrails in:

- `RESEARCH_RULES.md`
- `TRADING_RULES.md`
- `docs/plans/diversification-research-program.md`
- `docs/plans/2026-03-15-zt-stage1-triage-gate.md`

This is **not** a production feature spec.
It does **not** approve broad rates scanning, live trading, or generic ORB work.

---

## Exact Research Question

For `ZT` only:

**On `CPI` and `NFP` release days, does the first post-release price response contain a repeatable continuation or failed-first-move pattern that is economically plausible, small enough to test honestly, and potentially additive to the current ORB book?**

This pass is intentionally narrow:

- one product: `ZT`
- two event families: `CPI`, `NFP`
- two model framings: continuation, failed-first-move
- fixed holding structures only

If this pass is weak, stop.
Do not widen immediately into `ZN`, `FOMC`, auctions, curve work, or session scans.

---

## Why This Exact Cut

This is the cleanest first `ZT` pass because:

- `ZT` is the institutional benchmark expression for 2-year U.S. rates exposure
- `CPI` and `NFP` are the most defensible first `8:30 ET` event families
- the event clock is exact
- monthly event frequency gives a believable path to a meaningful long-horizon sample

This is an event study first, not a session-open strategy search.

---

## In Scope

### Product

- `ZT` only

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

- `ZN`
- `2YY`
- `5YY`
- `FOMC`
- Treasury auctions
- generic ORB cloning
- all-session scans
- threshold grids
- new overlays on the current live book
- any production or schema work

---

## Data Requirements

The implementation pass should stop before coding if any of these fail:

1. `ZT` minute data around `8:30 ET` is not available with reliable timestamps.
2. Event-date mapping for `CPI` and `NFP` cannot be assembled defensibly.
3. The available direct `ZT` history is too short to make the event count meaningful.
4. The apparent event move is too small relative to realistic friction to matter.

Direct `ZT` is the default and already passed arrival triage.
No proxy is approved by this spec.

---

## Signal Definitions

Stage 1 tests the mechanism cleanly, not an optimization grid.

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
- hit rate
- exact p-value where appropriate
- variation count tested
- first-half / second-half split behavior
- notes on fragility or instability

It must end with:

```text
SURVIVED SCRUTINY:
DID NOT SURVIVE:
CAVEATS:
NEXT STEPS:
```

Flat files or local research outputs are sufficient.

---

## Screening Standards

The implementation must explicitly answer these questions:

1. Does either event family show stable directional behavior after the first shock?
2. Is the sign of the effect consistent enough across the sample to matter?
3. Does the effect survive simple first-half / second-half splits?
4. Does the effect survive a basic friction sanity check?
5. Is the behavior different enough from the current ORB book to justify deeper work?

The last point can stay lightweight at Stage 1:

- event-day overlap with representative current ORB activity
- whether promising days are obviously just the same broad risk-on / risk-off days the current book already monetizes

If the answer is “this is just another way of winning on the same days,” that counts against continuation.

---

## Kill Criteria

Kill the path early if any of these happen:

1. **Weak mechanism**
   - the event story is vague or mostly narrative

2. **Search breadth drift**
   - the pass starts expanding into too many windows, filters, or model classes

3. **Bad sample economics**
   - usable event count is too sparse or too noisy to ever matter

4. **Thin economics**
   - the event effect is too small relative to realistic Treasury futures friction

5. **Fake diversification**
   - standalone metrics look fine but the days are obviously the same concentration days as the current book

6. **Narrative without edge**
   - the macro story sounds right but the measured behavior is unstable

---

## Continuation Criteria

Continue to deeper validation only if all of these are true:

- the mechanism can be stated cleanly in one paragraph
- the tested event family survives the narrow pass
- the result is not obviously parameter-fragile
- the candidate is plausibly additive to the existing portfolio at the daily-PnL level
- the next phase can still be kept narrow

If any of these are weak, stop.

---

## What This Spec Is Not

This spec does **not** approve:

- production wiring
- a generic Treasury research program
- broad rates market expansion
- ORB session exploration for `ZT`

It only defines the first narrow research gate for `ZT`.

---

## Recommended Immediate Next Step

After this spec, the next deliverable should be:

**a read-only research script and event calendar input for `ZT` CPI/NFP continuation vs failed-first-move analysis**

Nothing broader than that.
