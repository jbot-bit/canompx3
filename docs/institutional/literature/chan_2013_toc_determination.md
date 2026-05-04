# Chan 2013 (*Algorithmic Trading: Winning Strategies and Their Rationale*) — TOC determination

**Source:** `resources/Algorithmic_Trading_Chan.pdf`
**Author:** Ernest P. Chan
**Publication:** Wiley Trading, 2013 (ISBN 978-1-118-46014-6)
**Extracted:** 2026-04-19

**Criticality for our project:** 🟡 MIXED — Ch 6-7 overlap Fitschen Ch 3 on intraday/interday momentum (already grounded). Ch 4 is NOT the level-theory source the HTF prev-month v1 closure criteria hoped for. This note corrects a category error in `docs/audit/results/2026-04-18-htf-path-a-prev-month-v1-scan.md` § Closure recommendation.

---

## Verified table of contents (pp. vii, verbatim)

| Ch | Title | Pages |
|---|---|---|
| Preface | Preface | ix |
| 1 | Backtesting and Automated Execution | 1 |
| 2 | The Basics of Mean Reversion | 39 |
| 3 | Implementing Mean Reversion Strategies | 63 |
| **4** | **Mean Reversion of Stocks and ETFs** | **87** |
| 5 | Mean Reversion of Currencies and Futures | 107 |
| 6 | Interday Momentum Strategies | 133 |
| 7 | Intraday Momentum Strategies | 155 |
| 8 | Risk Management | 169 |
| — | Conclusion | 187 |
| — | Bibliography | 191 |

## What Chan Ch 4 actually covers (per TOC + Preface pp. ix-xii)

From the Preface:
- Book is organized into the **mean-reverting camp** (Ch 2-5) and the **momentum camp** (Ch 6-7).
- Mean-reverting camp covers: multiple statistical tests (ADF, Hurst exponent, Variance Ratio test, half-life) for detecting stationarity, cointegration tests (CADF, Johansen), linear/Bollinger/Kalman filter strategies on mean-reverting portfolios.
- **Ch 4 is the stocks-and-ETFs instantiation of the mean-reversion framework**: pairs trading, triplets, ETFs-vs-components, not breakout / level-break testing.

## Relevance to canompx3 research

### What Chan Ch 4 does NOT ground

- **Level-based breakout filters.** `prev_week_high` / `prev_month_high` / `prev_day_high` / pivot levels — Ch 4 is about mean-reverting portfolio construction, not breakout level theory.
- **HTF level-break filter family.** The 2026-04-18 `docs/audit/results/2026-04-18-htf-path-a-prev-month-v1-scan.md` § Closure recommendation (line 178) listed "Chan Algorithmic Trading Ch 4" among three candidate Pathway-A-qualifying extracts that might unlock t ≥ 3.00 with-theory. That listing was a category error: Ch 4 grounds mean-reversion tests, not level-break theory. Dalton *Mind Over Markets* and Murphy *Technical Analysis* remain the correct (but locally-absent) sources for level-based S/R mechanisms.

### What this book COULD ground (not extracted in this note)

- **Ch 7 "Intraday Momentum Strategies" (pp. 155-168).** Overlaps Fitschen Ch 3 on intraday trend-follow. Could strengthen `docs/institutional/literature/fitschen_2013_path_of_least_resistance.md` with a second source for the ORB intraday-momentum premise. Not extracted here because the single Fitschen Ch 3 citation is already sufficient for Pathway A on the active MES/MNQ mirror work.
- **Ch 1 "Backtesting and Automated Execution" (pp. 1-38).** Per `CLAUDE.md:79-81` corollary about extract-before-dismiss, this chapter has been explicitly cited in the project's postmortem for look-ahead bias (`docs/postmortems/2026-04-07-e2-canonical-window-fix.md` cites "Chan Ch 1 p 4"). A formal extract of Chan Ch 1's look-ahead discussion is queued but not in this note.
- **Ch 8 "Risk Management" (pp. 169-186).** Kelly formula, CPPI, black-swan risk, Monte Carlo on backtest statistical significance. Overlaps Carver Ch 9-10 already extracted at `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md`.

## Action taken by this note

1. Correct the HTF prev-month v1 result doc's § Closure recommendation to remove "Chan Ch 4" as a candidate Pathway-A source for level-based filters. (Follow-up doc edit in the same commit as this extract.)
2. Leave Dalton and Murphy as the remaining listed candidates — both confirmed absent from `resources/` and requiring separate acquisition. Label them "not locally available" rather than "candidate extract."

## Usage rules

- Do NOT cite Chan Ch 4 for Pathway A grounding of level-based filters. Category error.
- DO cite Chan Ch 7 (if extracted) for intraday momentum alongside Fitschen Ch 3.
- DO cite Chan Ch 1 for look-ahead / backtest-methodology claims (already informally in the project postmortem; formal extract pending).

## Audit note

This note was written during the 2026-04-19 session as part of a Phase 4 pre-reg lifecycle audit. The trigger was verification that the HTF prev-month v1 closure criteria (2026-04-18) correctly identified unlockable Pathway A sources. The TOC check revealed Chan Ch 4 is the wrong chapter for the stated purpose. The correct action is to update the HTF closure criteria, not to extract Ch 4 at length.

Chan Ch 7 (Intraday Momentum) extraction is a legitimate future task that would strengthen Fitschen Ch 3 — queued but not done here to avoid scope creep beyond Phase 4.
