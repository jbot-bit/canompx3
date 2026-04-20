# HTF->LTF Intraday Research Routing Decision

**Date:** 2026-04-20
**Branch:** `research/htf-ltf-intraday-program`
**Status:** DESIGN DECISION
**Question:** should the broader "higher-timeframe bias -> lower-timeframe trigger -> retest -> entry" brief be built inside the existing ORB framework, or treated as a separate strategy class?

## Executive decision

This is **not one thing**.

The brief splits into three buckets:

1. **ORB-compatible context overlays**
   - `HTF bias`, `overnight-range position`, `VWAP relation`, `regime`, `calendar`, and similar pre-entry context can be incorporated into ORB as filters, direction gates, confluence inputs, or sizing signals.
2. **ORB-derived entry-model extensions**
   - `breakout -> first retest -> continuation` can live inside ORB **only when the anchor remains the ORB boundary / ORB break** and the trade remains a single-session, single-breakout derivative of the existing ORB setup.
3. **Standalone intraday event strategies**
   - `FVG retest`, `sweep -> reclaim`, `VWAP reclaim after displacement`, and `trend-day pullback after news expansion` are **not** clean ORB filters. They need their own event-detection and backtest surface unless they are deliberately reduced to a simpler ORB-compatible role.

The mistake would be forcing the entire brief into ORB and then calling the result "HTF." That would answer the wrong question.

## Truth check

### What is already proven

- The repo's current canonical engine is ORB-first:
  - `trading_app/outcome_builder.py` computes outcomes **for each `(trading_day, orb_label)` with a break**, across `RR_TARGETS` and `confirm_bars` only. See [trading_app/outcome_builder.py](/mnt/c/Users/joshd/canompx3/trading_app/outcome_builder.py:2).
  - `orb_outcomes` is keyed by `(symbol, trading_day, orb_label, orb_minutes, rr_target, confirm_bars, entry_model)`, which is a strong sign that the canonical truth surface assumes one ORB-centered trade geometry per session/day. See [trading_app/db_manager.py](/mnt/c/Users/joshd/canompx3/trading_app/db_manager.py:89).
- The repo already distinguishes ORB roles from broader strategy roles:
  - `docs/institutional/mechanism_priors.md` explicitly maps ORB-compatible roles such as filter, direction gate, size modifier, target modifier, and entry-model switch, and states that stop/target geometry changes require `outcome_builder` re-simulation and schema work. See [docs/institutional/mechanism_priors.md](/mnt/c/Users/joshd/canompx3/docs/institutional/mechanism_priors.md:71).
- The repo already recognizes a separate non-ORB strategy class:
  - `docs/audit/hypotheses/phase-e-non-orb-strategy-class.md` says level / volatility / timing signals can exist independently of ORB and may open a strategy class orthogonal to ORB. See [docs/audit/hypotheses/phase-e-non-orb-strategy-class.md](/mnt/c/Users/joshd/canompx3/docs/audit/hypotheses/phase-e-non-orb-strategy-class.md:7).
- The repo already has a research-only level-interaction layer:
  - `docs/specs/level_interaction_v1.md` exists to study pass/fail and sweep/reclaim without creating a second truth system.
  - It is explicitly **not** a live-trading spec and explicitly **not** an ICT/FVG ontology. See [docs/specs/level_interaction_v1.md](/mnt/c/Users/joshd/canompx3/docs/specs/level_interaction_v1.md:6).
- The repo already has an ORB-derived retest concept on the shelf:
  - `phase-c-e-retest-entry-model.md` defines an `E_RETEST` branch as a new ORB entry model rather than a standalone system. See [docs/audit/hypotheses/phase-c-e-retest-entry-model.md](/mnt/c/Users/joshd/canompx3/docs/audit/hypotheses/phase-c-e-retest-entry-model.md:1).

### What is not proven

- The broader HTF->LTF brief itself is **UNVERIFIED** in this repo.
- The finished HTF work only killed the narrow `prev_week / prev_month` break-aligned ORB filter family. It did **not** test the broader trigger families.
- No canonical result currently proves that `FVG retest`, `sweep -> reclaim`, `VWAP reclaim`, or `trend pullback after expansion` should be folded into ORB rather than handled as separate strategies.

## Literature routing

### What is literature-grounded for ORB-compatible work

- **Intraday trend-follow / breakout** is grounded locally by Fitschen and Chan:
  - Fitschen grounds intraday trend-follow on stocks and commodities, which supports ORB as a trend-following class, not as a level-theory class. See [docs/institutional/literature/fitschen_2013_path_of_least_resistance.md](/mnt/c/Users/joshd/canompx3/docs/institutional/literature/fitschen_2013_path_of_least_resistance.md:12).
  - Chan explicitly grounds stop-triggered intraday breakout strategies and cascading stop behavior, including intraday entries at support/resistance levels. See [docs/institutional/literature/chan_2013_ch7_intraday_momentum.md](/mnt/c/Users/joshd/canompx3/docs/institutional/literature/chan_2013_ch7_intraday_momentum.md:16).
- **Portfolio / sizing roles** are grounded by Carver in the repo's existing doctrine and already recognized in `mechanism_priors.md`. See [docs/institutional/README.md](/mnt/c/Users/joshd/canompx3/docs/institutional/README.md:64).

### What is literature-grounded online for level-aware standalone work

- **Support/resistance interruptions are real but heterogeneous**, not a blanket edge:
  - Carol Osler, *Support for Resistance: Technical Analysis and Intraday Exchange Rates* (Economic Policy Review / SSRN), finds strong evidence that support/resistance levels can predict intraday trend interruptions, but predictive power varies across instruments and firms.
  - Lin Wang and Paul Wilmott, *Support and Resistance Levels in Financial Markets* (SSRN), argue markets can exhibit richer small-scale structure around support/resistance than a simple random walk.
- **Intraday momentum exists broadly across futures**, which supports continuation-style standalone event studies:
  - Baltussen, Da, Lammers, and Martens, *Hedging Demand and Market Intraday Momentum* (Journal of Financial Economics / SSRN), document strong market intraday momentum across more than 60 futures.
- **Open-driven reversals can also be real after large moves**, which argues against forcing every post-displacement setup into ORB continuation:
  - Grant, Wolf, and Yu, *Intraday Price Reversals in the US Stock Index Futures Market: A 15-Year Study* (SSRN), report significant open-related intraday reversals, with profitability materially reduced once trading costs are considered.

### What is not honestly literature-grounded yet

- There is still no solid repo-grounded or online primary-source case that **ICT/FVG language** is a privileged mechanism category for this project.
- If `FVG` is tested here, it should be treated as a **mechanical encoding of three-bar displacement / imbalance**, not as a protected ontology.
- The repo already warns that level-theory / market-profile style targeting needs Dalton / Steidlmayer class literature, which is still missing locally. See [docs/institutional/mechanism_priors.md](/mnt/c/Users/joshd/canompx3/docs/institutional/mechanism_priors.md:104).

## Routing matrix

| Strategy idea | ORB filter / overlay | ORB entry-model extension | Standalone SC2 |
|---|---|---:|---:|
| 1H / 4H directional bias, EMA slope, overnight-range position | `YES` | `NO` | `NO` |
| HTF bias deciding long-only vs short-only on an ORB lane | `YES` | `NO` | `NO` |
| Calendar / news blackout / volatility regime around ORB | `YES` | `NO` | `NO` |
| ORB breakout -> first retest of ORB level -> continuation | `NO` | `YES` | `NO` |
| ORB stop / target / sizing modified by HTF state | `YES` or `YES_WITH_INFRA` | `NO` | `NO` |
| PDH / PDL sweep -> reclaim reversal | `NO` | `NO` | `YES` |
| Breakout -> retest at non-ORB structural level | `NO` | `NO` | `YES` |
| VWAP reclaim after displacement | `MAYBE` as gate only | `NO` | `YES` |
| FVG retest continuation | `MAYBE` as gate only | `NO` | `YES` |
| Trend-day pullback after news expansion | `MAYBE` as gate only | `NO` | `YES` |

## Why the split is necessary

### 1. ORB filters already have a clean home

The current `StrategyFilter` model already supports binary skip/take and direction gating for pre-entry context. That is the right home for:

- HTF trend direction
- overnight positioning
- prior-day range location
- VWAP relation
- volatility regime
- calendar / event-state overlays

These are about **whether to take an ORB trade** or **which side to take**, not about inventing a new trade path.

### 2. ORB retest entries are a bounded extension, not a whole new strategy class

If the trade is still:

- one ORB session,
- one initial ORB break,
- one later retest of the ORB level,
- one continuation entry derived from the original ORB event,

then it belongs under ORB as a new entry model.

That is exactly why `E_RETEST` already exists as a deferred branch in the repo rather than under Phase E non-ORB. See [docs/audit/hypotheses/phase-c-e-retest-entry-model.md](/mnt/c/Users/joshd/canompx3/docs/audit/hypotheses/phase-c-e-retest-entry-model.md:1).

### 3. The rest need a different truth surface

The broader HTF->LTF brief contains families whose trigger timestamp, invalidation logic, and target are **not determined by the ORB boundary**:

- `sweep -> reclaim`
- `FVG retest`
- `VWAP reclaim after displacement`
- `trend pullback after news expansion`

These strategies usually require:

- arbitrary intraday event detection,
- multiple candidate triggers per session,
- structural stops not tied to the opposite ORB boundary,
- targets other than fixed ORB `RR`,
- potentially more than one trade opportunity per session/day.

That does not fit the current `orb_outcomes` contract without distortion.

## De-tunnel: three legitimate uses of the same idea family

A single chart concept can live in different roles:

1. **Standalone strategy**
   - Example: `PDH sweep -> reclaim -> reversal entry`.
   - Correct when the level interaction itself is the trade.
2. **ORB filter or confluence input**
   - Example: take ORB longs only when 1H trend is up and overnight position is supportive.
   - Correct when the idea only conditions existing ORB quality.
3. **ORB entry timing variant**
   - Example: after ORB break, do not chase; enter first retest of the ORB boundary.
   - Correct when the idea alters execution timing but remains an ORB trade.

The right question is therefore not "is HTF/LTF standalone or ORB?" It is "which role is this specific mechanism playing?"

## What has already been partially explored

- `VWAP` already shows the split cleanly:
  - as an ORB gate, it has meaningful repo evidence;
  - as a standalone VWAP pullback strategy, the archive says `NO-GO`. See [docs/RESEARCH_ARCHIVE.md](/mnt/c/Users/joshd/canompx3/docs/RESEARCH_ARCHIVE.md:556).
- `Sweep/reclaim` already exists as a research-only event study, and the narrow v1 pass produced no primary survivors. That does **not** kill the whole standalone class, but it proves the repo is already able to study these events outside ORB without pretending they are ORB filters. See [docs/audit/results/2026-04-19-sweep-reclaim-v1.md](/mnt/c/Users/joshd/canompx3/docs/audit/results/2026-04-19-sweep-reclaim-v1.md:1).
- `Trend pullback` and `VWAP bounce` have already been prototyped in `research_v3_mechanism.py`, which is further evidence that the repo has a research-only path for non-ORB event logic even though it is not canonicalized. See [research/research_v3_mechanism.py](/mnt/c/Users/joshd/canompx3/research/research_v3_mechanism.py:372).

## Institutional verdict

### Verdict

`CONDITIONAL SPLIT`

The broader HTF->LTF brief should **not** be treated as a single project branch.

### ORB-incorporable now

- HTF directional bias
- pre-entry regime / context filters
- confluence scoring
- allocator / sizing overlays

### ORB-incorporable with bounded infra

- `E_RETEST` style ORB breakout -> retest -> continuation entries
- ORB stop / target geometry modifications

### Standalone by default

- level pass/fail and sweep/reclaim strategies
- FVG retest continuation
- VWAP reclaim after displacement
- news-expansion pullback strategies whose trigger is not the ORB boundary

## Highest-EV next action

Do **not** launch one omnibus "HTF" program.

Launch two narrower programs:

1. **SC1-ORB extension track**
   - objective: improve an existing real edge
   - scope:
     - HTF directional/context overlays that are known before ORB entry
     - `E_RETEST` as the only entry-model extension candidate
2. **SC2-standalone event-study track**
   - objective: discover genuinely different trades
   - scope:
     - level interactions
     - sweep/reclaim
     - breakout-retest at explicit non-ORB levels
     - displacement / VWAP / pullback families

This split preserves honesty:

- ORB research stays ORB.
- standalone research gets a truth surface that matches the actual trade.
- we avoid contaminating the canonical ORB engine with patterns that are not ORB.

## External sources used

- Carol L. Osler, *Support for Resistance: Technical Analysis and Intraday Exchange Rates* — SSRN: https://ssrn.com/abstract=888805
- Lin Wang and Paul Wilmott, *Support and Resistance Levels in Financial Markets* — SSRN: https://ssrn.com/abstract=7484
- Guido Baltussen, Zhi Da, Sten Lammers, Martin Martens, *Hedging Demand and Market Intraday Momentum* — SSRN / JFE: https://ssrn.com/abstract=3760365
- James L. Grant, Avner Wolf, Susana Yu, *Intraday Price Reversals in the US Stock Index Futures Market: A 15-Year Study* — SSRN: https://ssrn.com/abstract=689282
