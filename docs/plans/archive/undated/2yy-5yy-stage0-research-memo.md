---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# 2YY / 5YY Stage-0 Research Memo

**Date:** 2026-03-15
**Status:** Design doc only
**Purpose:** Define the first narrow diversification research pass on U.S. Treasury Yield futures

---

## Decision

The first diversification path to pursue is:

- `2YY` event-window research
- `5YY` event-window research

This is **not** approval for broad rates research, broad ORB scans, or production wiring.
It is only approval for a narrow Stage-0 / Stage-1 research pass on the strongest first rates candidates.

---

## Why Rates First

This choice follows directly from current repo truth:

- fake diversification inside the current book was already disproven
- broad non-ORB exploration already failed
- the next best shot is a different macro driver, not another variant of the same ORB story

Why rates specifically:

- U.S. Treasury yield futures are driven by a different core mechanism set than gold/equity session-open breakouts:
  - macro release repricing
  - Fed path repricing
  - Treasury auction dynamics
  - duration demand / supply balance
- CME Yield futures are purpose-built, cash-settled contracts priced directly in yield and structured around fixed `$10 DV01`
- this gives a cleaner institutional case than stretching the current book into more same-family variations

---

## Why `2YY` And `5YY` Before `10Y`

Start with `2YY` and `5YY` first.

Reasoning:

- `2YY` is closest to front-end policy repricing
- `5YY` still captures policy-sensitive rates behavior but is less pure front-end noise than `2YY`
- `10Y` mixes more duration / term-premium behavior and is better treated as second-wave only

This ordering keeps the mechanism cleaner:

- front-end / belly yields around key U.S. macro events
- narrower event thesis
- less temptation to turn the first pass into a generic rates exploration project

---

## Research Question

The first-pass question is:

**Do `2YY` and `5YY` show repeatable, economically plausible event-window behavior around major U.S. macro releases that is different enough from the current ORB book to justify deeper validation?**

That question is intentionally narrow.
It does not ask:

- whether rates have any tradable strategy at all
- whether rates ORBs work across every session
- whether `10Y`, curve, or every Treasury product should be explored

This memo is deliberately framed as an **institutional event study first**:

- start from a known macro transmission mechanism
- define the event clock exactly
- measure whether the post-event behavior is stable enough to deserve any strategy engineering at all

If the event study is weak, there is no justification for a broader rates program.

---

## Event Windows In Scope

Wave 1 should test only event windows that have a strong institutional reason to move front-end and belly yields.

### In Scope

Wave 1A should be limited to these exact event families:

1. **CPI at 8:30 ET**
   - direct inflation repricing
   - highest-priority macro event for the first pass

2. **Non-Farm Payrolls at 8:30 ET**
   - labor / policy-path repricing
   - second high-priority event for the first pass

3. **FOMC statement window at 2:00 PM ET**
   - separate family
   - only evaluate as its own event set, never blended into the 8:30 ET releases

Wave 1B is optional only if Wave 1A survives:

4. **Treasury auction-adjacent windows**
   - only if the exact tenor, timestamp, and mechanism are explicitly stated
   - treat as separate event families, not as a blended rates bucket

### Out Of Scope

- generic 24-hour session scans
- fixed session ORBs copied from the current book
- day-of-week / calendar overlays as the primary research question
- large multi-event blended scans
- adding PPI, retail sales, ISM, auctions, Fed speakers, or curve trades into Wave 1A just because they are available

---

## Model Family In Scope

The first pass is allowed **at most two** model framings:

1. **Immediate continuation after a genuine event shock**
2. **Failed first move / reversal after an event shock**

That is the full Stage-0 search family.

Do not add:

- multi-threshold ORB filters
- large confirm-bar grids
- many target variants
- multiple unrelated model classes in the same pass

The purpose is to answer whether the mechanism exists, not to optimize it.

---

## Search-Space Cap

To stay within the repo's anti-overfitting standards, keep the first pass extremely small.

Cap the initial search to:

- 2 products: `2YY`, `5YY`
- 3 Wave 1A event families max: CPI, NFP, FOMC statement
- 2 model framings: continuation, failed-first-move
- 1-2 holding/exit structures max per framing
- **16 total structured variations max** before any follow-up work is proposed

If the planned variation count starts drifting upward, stop and shrink the design before any implementation begins.

Stage-0 is successful if it prevents a bloated Stage-1.

---

## Data Requirements

Minimum data requirements for continuing past Stage-0:

1. **Reliable event timestamps**
   - exact event timing must be available and defensible

2. **Enough market history**
   - enough event count to make the path capable of reaching PRELIMINARY / CORE thresholds eventually

3. **Executable market data**
   - data quality must be good enough to evaluate realistic event-window behavior

4. **Proxy path if needed**
   - if direct history on the chosen implementation contract is short or awkward, a closely aligned research proxy is acceptable only if the proxy/execution relationship is explicitly justified

5. **Sample path credibility**
   - each chosen event family must have a believable path to a useful sample without excessive slicing
   - if the design depends on carving 10 years of history into too many sub-buckets, stop

If any of those fail, the path should stop before research code is written.

---

## Screening Metrics

Before anything is called promising, it must be evaluated on:

- sample size and time span
- event count quality
- exact variation count tested
- exact date range and event-family definition used
- exact p-values when any claim approaches a tradable conclusion
- average trade economics after friction
- parameter fragility
- daily-PnL correlation versus representative existing ORB families
- same-day loss clustering versus the current book
- marginal portfolio impact, not just standalone performance

Low raw correlation alone is not enough.
Portfolio usefulness is the actual test.

---

## Kill Criteria

Kill the path early if any of the following happen:

1. **Weak mechanism**
   - the event story is vague, post-hoc, or not specific enough to institutional behavior

2. **Search breadth drift**
   - the first pass starts expanding into many windows, many thresholds, or many model classes

3. **Bad sample economics**
   - the usable event count is too sparse to ever produce a meaningful result

4. **Pre-cost or parameter-fragile edge**
   - the result only looks attractive before costs or after threshold tuning

5. **Fake diversification**
   - standalone metrics improve but same-day stress still clusters with the current book enough to add little portfolio value

6. **Narrative without edge**
   - the macro story sounds good but the measured event behavior is unstable or inconsistent

7. **Proxy mismatch**
   - the research proxy is materially different from the intended execution product and the bridge cannot be defended cleanly

---

## Continuation Criteria

Continue to a deeper validation phase only if all of the following are true:

- the mechanism can be stated cleanly in one paragraph
- the tested event family shows behavior that survives the first-pass narrow scan
- the result is not obviously parameter-fragile
- the candidate is plausibly additive to the existing portfolio at the daily-PnL level
- the next phase can still be kept narrow

If any of these are weak, stop.

---

## What This Memo Is Not

This memo does **not** approve:

- new production code
- new database tables
- new config entries
- broad rates market expansion
- a generic Treasury research program

It only defines the first narrow research gate for `2YY` and `5YY`.

---

## Recommended Immediate Next Step

The next deliverable after this memo should be:

**a Stage-1 implementation spec for one exact first pass**

Recommended initial ordering:

1. `2YY` around `CPI` and `NFP` only
2. `5YY` under the same exact event framing only if the `2YY` pass is still worth continuing
3. `FOMC` statement window as a separate family only after the 8:30 ET path is judged clean enough to keep going
4. optional auction-window follow-up only if the macro-release path justifies continuing

That spec should lock:

- exact events
- exact holding logic
- exact variation count
- exact output table/report shape
- exact kill condition
- exact proxy/execution mapping if a proxy is used

---

## Source Grounding

Primary market-structure sources:

- CME Group, Yield futures overview:
  - https://www.cmegroup.com/articles/2024/understanding-yield-futures.html
- CME Group, 2025 quarterly ADV release:
  - https://www.cmegroup.com/media-room/press-releases/2025/4/02/cme_group_sets_newall-timequarterlyadvrecordof298millioncontract.html

Internal repo grounding:

- `RESEARCH_RULES.md`
- `ROADMAP.md`
- `docs/RESEARCH_ARCHIVE.md`
- `docs/plans/diversification-research-program.md`
- `docs/plans/diversification-candidate-shortlist.md`

Methodology background in `resources/`:

- `Robert Carver - Systematic Trading.pdf`
- `Lopez_de_Prado_ML_for_Asset_Managers.pdf`
- `Two_Million_Trading_Strategies_FDR.pdf`

---

## Bottom Line

If the next research dollar is spent anywhere, it should be spent here first:

**`2YY` / `5YY`, narrow event-window research, with hard kill criteria and no scan bloat.**

That is the highest-quality next bet from the current project state.
