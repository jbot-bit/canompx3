# Diversification Research Program

**Date:** 2026-03-15
**Status:** Design doc only
**Scope:** Research-program definition for finding genuinely diversifying edges beyond the current ORB concentration

---

## Why This Exists

The repo already disproved the naive diversification story:

- MNQ and MES at `TOKYO_OPEN` are effectively the same trade (`ROADMAP.md`, P1)
- MGC plus equity micros provides only moderate diversification in the tested overlap
- current strength is still concentrated in a narrow edge family: momentum breakouts with size filters

The next research push should not be "more variants of the same thing."
It should be a disciplined search for **genuinely different edge sources** that can survive institutional scrutiny and improve the portfolio at the daily-PnL level.

This document defines that program.

---

## Program Goal

Find one or more edge families that improve portfolio robustness by adding **structurally different return drivers**, not cosmetic diversity.

Success is not:
- another ticker expressing the same underlying move
- another ORB filter stack that trades the same days
- another backtest with attractive standalone Sharpe but high overlap with existing risk

Success is:
- a candidate with a distinct mechanism
- low enough daily-PnL correlation / overlap to matter
- acceptable cost realism
- evidence that it improves the portfolio, not just the candidate table

---

## Canonical Standards

This program is subordinate to:

- `RESEARCH_RULES.md` for methodology and statistical honesty
- `TRADING_RULES.md` for live trading truth and existing no-go knowledge
- `ROADMAP.md` for current known diversification findings and open direction

Non-negotiables from `RESEARCH_RULES.md` apply here:

- mechanism before narrative
- no in-sample overclaiming
- BH/FDR correction after broad scans
- no sample-size begging
- no "low correlation" storytelling without portfolio-level proof

---

## What Counts As A Genuinely Diversifying Edge

A candidate must satisfy all of the following before it is treated as meaningful:

1. **Mechanism difference**
   - The economic / structural reason for the edge is different from the current ORB-size breakout story.
   - "Same ORB idea on a new symbol" is not enough.

2. **Portfolio difference**
   - The daily-PnL stream is not just a noisy clone of existing slots.
   - Shared break-days, same-session dependency, and same macro-driver concentration all count against it.

3. **Execution realism**
   - The edge survives friction and realistic fill assumptions for the relevant product.
   - Thin pre-cost ideas are dead on arrival.

4. **Research discipline**
   - The finding survives the same sample-size, time-span, multiple-testing, and walk-forward standards used elsewhere in the repo.

5. **Capital-allocation value**
   - It improves portfolio quality: Sharpe, drawdown, return stability, regime coverage, or survivability in lower-gold-vol regimes.

---

## Explicit Anti-Goals

Do not spend research budget on:

- `MES` vs `MNQ` style "different symbol, same trade" setups
- more ORB micro-variants unless they are justified as part of a portfolio-diversification test
- indicator tourism (`RSI`, `MACD`, threshold soup)
- searching for correlation stories first and mechanisms second
- re-opening confirmed ORB no-gos without a materially different model type
- tuning filters to manufacture more trades

---

## Research Hypotheses To Prioritize

Priority order is based on structural difference and diversification potential, not convenience.

### Priority 1: New Asset-Class ORB Tests With Real Structural Separation

Candidate buckets:

- micro rates / bonds
- selected ags
- selected FX only if the mechanism is not already disproven by the M6E ORB result

Why first:

- this is the cleanest path to finding an actually different macro driver
- it directly addresses the failed "equity micro diversification" assumption

Acceptance bar:

- not strongly co-moving with the current gold/equity ORB cluster at the daily-PnL level
- enough session/event structure to support a plausible breakout mechanism

### Priority 2: Non-ORB Intraday Models On Existing Or Nearby Products

Examples:

- failed breakout / failed auction models
- mean-reversion models tied to specific session microstructure
- close / settlement / post-event unwind structures

Why second:

- this can create negative or near-zero correlation to current ORB exposure if the mechanism is genuinely different

Acceptance bar:

- must not be a disguised ORB variant
- must show why it should fire on different days or profit from opposite intraday conditions

### Priority 3: Regime / Event-Conditioned Overlays That Change Portfolio Exposure

Examples:

- add / suppress exposure based on cross-asset event states
- volatility transmission signals that alter sizing or activation rather than spawn a new standalone strategy

Why third:

- overlays may be high leverage, but they are easier to overfit and easier to mistake for "new edge" when they are really just timing wrappers

Acceptance bar:

- evaluated as portfolio modifiers first, not marketed as standalone alpha

---

## Candidate Rejection Rules

Reject early if any of the following are true:

- same asset class, same session, same direction, same trade-day profile as an existing live family
- no credible mechanism beyond "numbers looked good"
- costs are too large relative to the expected move
- sample is too small and there is no plausible path to reaching meaningful N soon
- the edge only appears after broad scan mining and dies under sensitivity checks
- the candidate improves standalone backtest metrics but does not improve the portfolio

---

## Research Funnel

The program should run through a staged funnel.

### Stage 0: Pre-Research Triage

For each candidate market / model:

- define the mechanism in one paragraph
- define why it should diversify the current book
- define why this market/session should exist at all
- define expected failure modes

If the mechanism is weak, stop here.

### Stage 1: Structural Feasibility

Questions:

- does the market have event/session structure compatible with the model?
- are costs acceptable relative to typical range?
- is there enough history to ever reach PRELIMINARY / CORE thresholds?
- is the product practically tradable in the account types you care about?

Output:

- `GO`, `NO-GO`, or `NOT WORTH DATA BUDGET`

### Stage 2: Narrow Discovery

Only after passing Stage 1:

- run a deliberately small search space
- report exact variation count
- apply BH/FDR where appropriate
- compare against a clear baseline

Output:

- promising hypotheses only
- no "validated" language

### Stage 3: Portfolio Relevance Test

Every surviving candidate must be tested against the existing book:

- daily-PnL correlation
- overlap in active days
- drawdown co-movement
- marginal effect on portfolio Sharpe
- marginal effect on portfolio max DD
- regime coverage effect

If it does not improve the portfolio, it is not a diversification win.

### Stage 4: Validation

Before anything is called deployable:

- walk-forward / OOS validation
- parameter sensitivity
- mechanism review
- cost stress
- regime split review where relevant

---

## Measurement Framework

The main mistake to avoid is treating correlation as the whole answer.

Use all of these:

- daily-PnL correlation
- overlap day count
- overlap percentage
- same-day loss clustering
- conditional correlation in stress windows
- portfolio Sharpe delta
- portfolio max drawdown delta
- portfolio trade-frequency contribution
- regime-coverage contribution

Important:

- low raw correlation alone is not enough
- high day overlap with different PnL profile is ambiguous and must be interpreted carefully
- portfolio-level outcome is the final judge

---

## Candidate Universe

This is the starting universe, not an endorsement list.

### Asset-class expansion candidates

- micro rates / bond products
- selected ags with clear event windows
- selected FX only if paired with a mechanism not already disproven by session-open ORB work

### Model-type expansion candidates

- failed breakout / failed auction
- session-close or post-event continuation / reversion
- volatility-state overlays
- non-ORB event-response models

### Lower-priority / caution candidates

- additional equity micro variations
- more gold-adjacent products with the same macro driver
- threshold-heavy indicator models

---

## Program Rules

1. Every candidate starts with a written mechanism and kill thesis.
2. Every scan must declare variation count up front.
3. Every result must state whether it is in-sample or out-of-sample.
4. Every survivor must face a portfolio relevance test.
5. Every no-go gets archived clearly so it is not re-litigated casually.
6. No production wiring, config changes, or schema work until this program produces a reviewed winner.

---

## First Wave Deliverables

These are the first concrete research outputs this program should generate after approval.

1. **Candidate shortlist memo**
   - choose a small initial set of asset-class / model-type candidates
   - include data availability, cost realism, and mechanism summary

2. **Diversification scorecard template**
   - one standard report format for overlap, correlation, stress co-movement, and marginal portfolio value

3. **Stage-1 triage on the first candidate batch**
   - decide which ideas deserve actual discovery work

No code is part of this document. This is the research charter only.

---

## Honest Success Criteria

This program is successful if it does one of the following:

- finds one genuinely diversifying candidate worth deeper validation
- proves that several intuitive candidates are fake diversification and closes them quickly
- narrows the future research agenda so the repo stops spending time on crowded dead ends

Failure would be:

- another large scan with weak mechanism
- another same-family variation dressed up as diversification
- another pile of candidate tables without portfolio-level judgment

---

## Bottom Line

The repo does not need "more strategies."
It needs **different risk drivers**.

The bar for calling something diversifying should be higher than the bar for calling something statistically interesting.
This program exists to enforce that distinction.
