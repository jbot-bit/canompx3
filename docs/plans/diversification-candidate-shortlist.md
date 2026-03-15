# Diversification Candidate Shortlist

**Date:** 2026-03-15
**Status:** Design doc only
**Purpose:** First-wave candidate ranking for diversification research after the core program definition

---

## Executive View

The repo already learned the most important negative lesson:

- stacking `MNQ` and `MES` is not diversification when the daily-PnL stream is basically the same trade
- moderate cross-asset correlation is not enough by itself to justify portfolio expansion
- broad non-ORB scanning did not produce robust alternatives

So the first wave should be:

1. **small number of structurally different candidates**
2. **primary-source grounded**
3. **fast to kill if the mechanism is weak**

This shortlist intentionally favors **different macro drivers** over convenient backtest opportunities.

---

## Ranking Method

Candidates are ranked on:

1. mechanism clarity
2. expected diversification value versus the current ORB book
3. tradability and contract realism
4. path to meaningful sample size
5. implementation complexity

---

## Wave 1 Candidates

### 1. U.S. Treasury Yield Futures (`2YY`, `5YY`, `10Y`)

**Priority:** Highest
**Why this is here:** This is the cleanest first shot at a genuinely different macro driver.

### Institutional case

- CME Yield futures are cash-settled, priced directly in yield, and built around fixed `$10 DV01` per contract rather than the larger and more variable risk of traditional Treasury futures.
- CME states the contracts reference on-the-run Treasuries, use BrokerTec benchmarks, and trade nearly around the clock.
- The broader rates complex is one of the deepest liquidity pools on CME, with record Treasury and interest-rate volume in 2025.

### Why it may diversify the current book

- current repo strength is concentrated in intraday breakout conditions tied to metals/equity volatility
- front-end and belly rates are driven by a different mix: macro data, Fed path repricing, Treasury auctions, duration demand, and policy shocks
- even if rates are not negatively correlated all the time, they are a different **risk engine**

### What to research first

Do **not** begin with a broad ORB grid.

Start with one narrow model family:

- **event-window response around U.S. macro releases / Treasury event windows**
- choose one tenor family first: `2YY` and `5YY` are the cleanest opening candidates
- test one or two event structures only:
  - immediate continuation after a genuine data shock
  - failed first move / auction-style reversal only if the mechanism is clearly defined

### Why `2YY` / `5YY` before `10Y`

- front-end and belly tenors are more tightly linked to the path of policy repricing
- `10Y` is still strong candidate, but it mixes more term-premium / long-duration behavior
- start where the event mechanism is easiest to explain

### Kill criteria

Kill quickly if:

- price action is too dominated by macro-event noise with no stable event-window behavior
- the candidate only works on a parameter-fragile threshold sweep
- same-day stress clustering still tracks the current book too closely

### Status

**Wave 1A candidate**

---

### 2. Micro Agricultural Futures: `MZC` and `MZS`

**Priority:** High
**Why this is here:** Agriculture gives you a genuinely different fundamental driver set without needing large standard contracts.

### Institutional case

- CME launched Micro Corn (`MZC`) and Micro Soybeans (`MZS`) in February 2025.
- The contracts are `1/10` the size of the standard contracts and are financially settled, which makes them cleaner for research-to-execution transition than physically delivered legacy ag contracts.
- Trading hours match the standard agricultural contracts, including the day session around the primary agricultural trading window.
- CME’s broader agricultural complex continues to show strong benchmark activity, especially in corn and soybeans.

### Why it may diversify the current book

- agriculture is driven by crop cycles, USDA reports, weather, acreage expectations, export flow, and seasonal balance-sheet changes
- that is a very different mechanism set from gold/equity session-open breakouts
- corn and soybeans are also benchmark ag markets, which makes them better first candidates than niche commodities

### What to research first

Do **not** start with 24-hour ORB cloning.

Start with one narrow class of event-driven ideas:

- USDA-report response structure
- day-session open / post-report continuation versus failed first move
- one crop at a time, beginning with `MZC`, then `MZS`

### Important nuance

Micro ag contracts launched recently, so micro-specific history is short.
If research needs a longer sample, use the standard corn / soybean contracts as the research proxy and treat the micro contracts as the eventual execution vehicle.

### Kill criteria

Kill quickly if:

- the usable history for the chosen event structure is too sparse to ever reach meaningful sample size
- the best result only appears after broad parameter mining
- the edge depends on seasonal storytelling more than repeatable event behavior

### Status

**Wave 1A candidate**

---

### 3. Rates-As-Overlay Research For The Existing Book

**Priority:** Medium-High
**Why this is here:** This is not a new standalone strategy. It is a portfolio-modifier candidate tied to a different macro driver.

### Institutional case

- your current book is still concentrated in one broad edge family
- if rates state explains when the current ORB book is strong or fragile, that can improve portfolio quality without pretending to have discovered a brand-new alpha source
- this is more institutionally sane than forcing a standalone strategy where no structural edge exists yet

### Why it may diversify the book

- a regime or event overlay can improve return quality by **changing exposure** under different macro conditions
- that is a legitimate diversification lever if evaluated honestly as a portfolio modifier

### What to research first

Use one tightly-scoped question:

- does a simple rates-state filter materially change the quality of existing ORB slots on key event days or high-rate-volatility days?

Examples of acceptable first-pass framing:

- Treasury yield shock state as risk-on / risk-off regime flag
- front-end rate move threshold as an activation or suppression filter

### What to avoid

- turning this into another giant overlay zoo
- claiming “new edge” when the result is just better activation of the existing book

### Kill criteria

Kill quickly if:

- the overlay improves in-sample metrics but adds no portfolio robustness
- the effect only appears after many threshold variations
- the mechanism cannot be stated cleanly

### Status

**Wave 1B candidate**

---

## Reserve Candidates

These are real possibilities, but they should not be first-wave unless a sharper thesis appears.

### Reserve 1: Micro FX outside the `M6E` logic

The repo already killed `M6E` session-open ORB as a general path.
That does **not** prove all FX research is dead, but it does mean generic FX session-open ORB work should start from a presumption of failure.

If FX is revisited, it should be under a much tighter mechanism:

- `M6A` if the thesis is commodity-linked macro flow
- `MJY` if the thesis is rates / safe-haven regime behavior

But this should be reserve work, not Wave 1.

### Reserve 2: Highly specific failed-auction / failed-breakout models

Broad non-ORB scanning already failed.
If this line is reopened, it should be one sharply defined microstructure thesis, not another 500-test sweep.

---

## Explicit Non-Starters

Do not spend the first-wave budget on:

- more equity micros as “diversifiers”
- more gold-adjacent breakout cloning
- generic FX session-open ORB research
- broad non-ORB archetype scans
- threshold-rich indicator families

---

## Recommended First-Wave Order

### Wave 1A

1. `2YY` / `5YY` event-window research
2. `MZC` event-window research
3. `MZS` only after `MZC` gives a reason to stay in ags

### Wave 1B

4. rates-as-overlay research on the current ORB book

### Reserve

5. one FX candidate only if a stronger mechanism memo is written first
6. one highly specific failed-auction concept only if it is clearly different from the failed broad non-ORB sweep

---

## Source Grounding

Primary sources used for this shortlist:

- CME Group, Yield futures overview and contract structure:
  - https://www.cmegroup.com/articles/2024/understanding-yield-futures.html
- CME Group, 2025 quarterly ADV release showing strong rates, ag, FX and equity benchmark activity:
  - https://www.cmegroup.com/media-room/press-releases/2025/4/02/cme_group_sets_newall-timequarterlyadvrecordof298millioncontract.html
- CME Group, Micro Agricultural futures FAQ and contract details:
  - https://www.cmegroup.com/articles/faqs/faq-micro-agriculture-futures.html
- CME Group, Micro Agricultural futures launch announcement:
  - https://www.cmegroup.com/media-room/press-releases/2025/1/30/cme_group_to_launchmicro-sizedgrainsandoilseedfuturesonfebruary2.html
- CME Group, Micro FX futures overview and contract specifications:
  - https://www.cmegroup.com/markets/microsuite/fx.html

Internal repo grounding:

- `RESEARCH_RULES.md`
- `TRADING_RULES.md`
- `ROADMAP.md`
- `docs/RESEARCH_ARCHIVE.md`
- `docs/plans/diversification-research-program.md`

---

## Bottom Line

The best first shot is not “more products.”
It is **rates first, ags second, overlays third, FX only with a stronger thesis**.

That is the shortest path to a genuinely different risk driver without slipping back into fake diversification.
