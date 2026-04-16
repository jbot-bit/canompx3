# 2026-04-11 Cross-Session State Round-3 Memo

## Purpose

Convert the external-context audit into an honest, non-ML research direction.
This memo records what actually survived after read-only testing on canonical
`daily_features + orb_outcomes`, what stays alive as secondary leads, and what
should be killed or deprioritized.

The guiding rule was strict:

- no external/vendor metadata trusted
- no imported thresholds
- no look-ahead
- literature can provide family shape only, not evidence

## Literature Grounding

These sources justify testing continuation, reversal, and session spillover as
*hypothesis families*:

- JFE, *Market intraday momentum*:
  <https://www.sciencedirect.com/science/article/pii/S0304405X18301351>
  - early intraday state can predict later intraday state
- Finance Research Letters, Chinese index futures intraday momentum:
  <https://www.sciencedirect.com/science/article/pii/S1544612319304337>
  - first trading-session return can predict later trading-session return
- Journal of Banking & Finance, VIX futures intraday momentum/spillover:
  <https://www.sciencedirect.com/science/article/pii/S0378426622003260>
  - continuation/reversal can spill across different trading sessions
- Finance Research Letters, intraday momentum and reversal coexistence:
  <https://www.sciencedirect.com/science/article/pii/S1544612318307414>
  - continuation and reversal can coexist depending on state
- commodity futures reversal evidence:
  <https://www.sciencedirect.com/science/article/pii/S0927538X24002865>
  - reversal and cross-predictive effects can coexist in derivatives markets

This repo did **not** import any paper thresholds or claimed returns.

## Translation Into Our Grammar

We tested only prior-session state that is fully resolved before the later
session starts:

- prior outcome: `win` / `loss`
- prior direction relative to later breakout: `aligned` / `opposed`

That yields four symmetric states:

- `TAKE PRIOR_WIN_ALIGN`
- `VETO PRIOR_WIN_OPPOSED`
- `VETO PRIOR_LOSS_ALIGN`
- `TAKE PRIOR_LOSS_OPPOSED`

These four states were tested across adjacent chronologically-safe session
handoffs only.

## Evidence Summary

### Strongest Surviving Branch

`NYSE_OPEN -> US_DATA_1000` on `MES+MNQ`

Why this branch is real enough to promote:

- all four sibling states matter
- pooled result is not hiding a contradiction; `MES` and `MNQ` both contribute
- OOS sign is positive
- family behaves like a coherent state machine, not a one-off filter

Round-2 / matrix survivors:

- `TAKE PRIOR_WIN_ALIGN`
  - IS benefit `+0.1395R`
  - BH `q=0.0038`
  - OOS benefit `+0.4810R`
- `VETO PRIOR_WIN_OPPOSED`
  - IS benefit `+0.1853R`
  - BH `q=0.0038`
  - OOS benefit `+0.3396R`
- `VETO PRIOR_LOSS_ALIGN`
  - IS benefit `+0.1466R`
  - BH `q=0.0087`
  - OOS benefit `+0.4065R`
- `TAKE PRIOR_LOSS_OPPOSED`
  - IS benefit `+0.1285R`
  - BH `q=0.0112`
  - OOS benefit `+0.2847R`

Interpretation:

- continuation after successful aligned prior state is credible
- fading a successful prior state is bad and should likely be vetoed
- repeating a failed prior direction is bad and should likely be vetoed
- reversal after failed prior direction is credible

### Secondary Branch

`US_DATA_830 -> NYSE_OPEN`

This branch is weaker than `NYSE_OPEN -> US_DATA_1000`, but still worth active
follow-up.

Survivors:

- `MES+MNQ VETO PRIOR_LOSS_ALIGN`
  - IS benefit `+0.1503R`
  - BH `q=0.0061`
  - OOS benefit `+0.0430R`
- `MGC TAKE PRIOR_WIN_ALIGN`
  - IS benefit `+0.1845R`
  - BH `q=0.0198`
  - OOS benefit `+0.1376R`
- `MGC VETO PRIOR_LOSS_ALIGN`
  - IS benefit `+0.2413R`
  - BH `q=0.0095`
  - OOS benefit `+0.3609R`
- `MNQ VETO PRIOR_LOSS_ALIGN`
  - IS benefit `+0.1816R`
  - BH `q=0.0253`
  - OOS benefit `+0.1769R`

Interpretation:

- the cleanest use here is veto logic
- the branch is heterogeneous across instruments, so pooled promotion would be
  premature without per-instrument validation

### Watchlist Branches

`CME_REOPEN -> TOKYO_OPEN`

- `MES+MNQ VETO PRIOR_LOSS_ALIGN`
  - IS benefit `+0.1698R`
  - BH `q=0.0070`
  - OOS benefit `+0.1602R`
- `MES+MNQ TAKE PRIOR_LOSS_OPPOSED`
  - IS benefit `+0.1630R`
  - BH `q=0.0095`
  - OOS benefit `+0.0123R`

Interpretation:

- not junk
- weaker economics and thinner OOS than the NY handoff
- keep alive, but do not make it the main branch

`LONDON_METALS -> EUROPE_FLOW`

- `MES+MNQ VETO PRIOR_WIN_OPPOSED`
  - IS benefit `+0.0931R`
  - BH `q=0.0152`
  - OOS benefit `+0.4308R`

Interpretation:

- one meaningful survivor
- not enough yet to call the whole family robust
- keep as a watchlist branch, not a program

### Branches To Deprioritize / Kill

Kill or deprioritize for now:

- `SINGAPORE_OPEN -> LONDON_METALS`
  - too many directional failures
- `US_DATA_1000 -> COMEX_SETTLE`
  - weak and unstable
- most `COMEX_SETTLE -> CME_PRECLOSE`
  - weak or directionally inconsistent
- most `CME_PRECLOSE -> NYSE_CLOSE`
  - some strong IS pockets, but too many fail OOS or remain sample-starved
- standalone day-level context filters
  - broad external-context framing did not beat targeted cross-session state

## Proper Decision Calls

### Promote To Round-3

- `NYSE_OPEN -> US_DATA_1000` on `MES+MNQ`

### Active Secondary Research

- `US_DATA_830 -> NYSE_OPEN`

### Watchlist

- `CME_REOPEN -> TOKYO_OPEN`
- `LONDON_METALS -> EUROPE_FLOW`

### Do Not Prioritize

- all other adjacent handoffs unless a new narrowly defined reason emerges

## Round-3 Validation Design

The next pass should not be another giant discovery sweep. It should be a
focused validation pack.

### Round-3 Pack A: NYSE_OPEN -> US_DATA_1000

Validate:

- `MES` alone
- `MNQ` alone
- `MES+MNQ` pooled
- all four state siblings together

Measure:

- trade-count impact vs base strategy
- expectancy impact
- veto benefit vs take-filter benefit
- month-by-month OOS stability
- whether one instrument is carrying pooled performance
- whether state effects survive when measured as:
  - all trades
  - long-only
  - short-only

Fast kill criteria:

- pooled effect disappears when split by instrument
- veto/take edges collapse to one isolated month
- one state survives only because one opposite state is catastrophically bad in
  a small sample

### Round-3 Pack B: US_DATA_830 -> NYSE_OPEN

Validate:

- `MGC`
- `MNQ`
- `MES+MNQ` only where signs agree

Primary lens:

- veto overlays first

Fast kill criteria:

- cross-instrument contradiction
- pooled edge is impossible to state cleanly without hand-waving

### Round-3 Pack C: Watchlist Branches

Keep lean:

- `CME_REOPEN -> TOKYO_OPEN`
- `LONDON_METALS -> EUROPE_FLOW`

These should be treated as:

- cheap follow-up
- no implementation planning yet

## Implementation Guidance

If Round-3 holds, implementation should be framed as an overlay/state machine,
not a generic context filter layer.

Preferred live forms:

- veto overlays
- continuation/reversal state routing
- instrument-specific overlays where pooled logic is not justified

Avoid:

- adding broad “external context” logic everywhere
- importing day-level macro-ish filters because they sound smart
- forcing one pooled rule across instruments with different signs

## Bottom Line

The broad external-context idea was too vague. The data narrowed it to a more
precise direction:

- **cross-session resolved state matters**
- especially `NYSE_OPEN -> US_DATA_1000`
- likely more useful as a structured continuation/reversal state machine than
  as random context filters

That is the honest send.
