# GC / MGC Handling Note

Date: 2026-04-19

## Purpose

Document how `GC` and `MGC` should be handled in this repo as **contracts** and
as exposure to the **gold market**, so future research does not treat them as
just another symbol pair.

This note is guidance for research design and execution assumptions. It is not a
deployment approval and not a new proxy-policy change.

## Bottom line

Treat `GC` and `MGC` as:

- the **same underlying asset class** for broad price discovery and macro
  exposure
- but **different execution surfaces** with different payoff realization
  properties

That means:

- `GC` can still be useful as a **price-safe research proxy** under the current
  repo policy for narrow questions
- `MGC` must remain the **truth surface** for anything involving execution
  quality, scratch behavior, winner size, queue position, or deployability

Our 2026-04-19 translation audit confirms exactly that split:

- price-safe triggers transfer
- payoff shape does not transfer cleanly

## Contract mechanics that matter

### 1. Contract size is different; tick grid is not

Both `GC` and `MGC` trade in U.S. dollars and cents per troy ounce with the
same minimum price fluctuation of `$0.10` per ounce, but:

- `GC` is `100` troy ounces
- `MGC` is `10` troy ounces

So the minimum tick value is:

- `GC`: `$10`
- `MGC`: `$1`

Implication:

- price-level logic and bar geometry can be comparable
- realized dollar and queue economics are not
- any claim that “the edge transfers” must be made in **R** or **price** terms,
  then checked again on native `MGC` execution behavior

Sources:

- CME Gold product overview: <https://www.cmegroup.com/education/lessons/product-gold.html>
- CME Micro Gold contract page: <https://www.cmegroup.com/markets/metals/precious/e-micro-gold.contractSpecs.html>
- COMEX Gold rulebook Chapter 113: <https://www.cmegroup.com/rulebook/COMEX/1a/113.pdf>
- COMEX Micro Gold rulebook Chapter 120: <https://www.cmegroup.com/rulebook/COMEX/1a/120.pdf>

### 2. Delivery mechanics are not identical

`GC` is deliverable as:

- one 100-ounce bar, or
- three 1-kilo bars

`MGC` is not just “small GC delivery.” It delivers through ACEs:

- one `MGC` contract is 10% ownership of a 100-ounce bar
- accumulation/redemption mechanics matter

Implication:

- neither contract should be handled casually into expiration
- delivery-month behavior can distort trading behavior near termination
- rolling/avoidance discipline matters even if the research is “just intraday”

Both contracts stop trading in the deliverable month after the **third last
business day** of the contract month.

Repo implication:

- research and live handling should roll **before** delivery mechanics become
  the dominant influence
- do not hold open positions into delivery month by accident

### 3. `GC` and `MGC` are not the same order book

This is the key operational point.

Even if `GC` and `MGC` are tightly linked in price, they are still distinct
listed contracts with distinct participation, depth, and queue behavior.

That is consistent with our internal finding:

- feature and trigger parity are strong
- but `MGC` realizes smaller winners and weaker expectancy on the same 5-minute
  ORB templates

So:

- **do not** use `GC` to answer an execution question about `MGC`
- **do** use `GC` to answer a narrow price-safe discovery question when policy
  allows it

## Asset-handling: gold is not “just another futures market”

### 1. Gold is a global, cross-venue market

Gold trades across:

- London OTC
- COMEX futures
- Shanghai venues

COMEX is a core derivatives venue, but it is not the whole gold market.

Implication:

- session logic on gold should not be framed only around U.S. cash-equity habits
- Asia and London hours are structurally relevant to price discovery
- gold-specific session assumptions need to respect cross-time-zone liquidity

Sources:

- World Gold Council market structure primer: <https://www.gold.org/goldhub/research/market-primer/gold-market-primer-market-size-and-structure>
- World Gold Council market structure overview: <https://www.gold.org/what-we-do/gold-market-structure/global-gold-market>
- CME article on APAC liquidity: <https://www.cmegroup.com/education/articles-and-reports/comex-gold-futures-and-options-a-look-at-apac-liquidity.html>

### 2. APAC and London liquidity are real, not decorative

CME’s gold education and APAC liquidity materials make two points that matter:

- `GC` trades nearly around the clock
- Asian-hour liquidity has grown materially and execution costs can be comparable
  to the broader trading day for `GC`

Repo implication:

- do not dismiss `TOKYO_OPEN`, `SINGAPORE_OPEN`, or `LONDON_METALS` on gold just
  because they are not U.S. equity-centric hours
- but also do not assume `MGC` microstructure is equally robust there without
  native `MGC` evidence

That distinction matters because our canonical audit showed:

- `GC` is structurally positive in several non-U.S. gold sessions
- `MGC` is still weak on the same broad comparator

So the right question is not “should gold trade in Asia/London?” It clearly
should. The right question is whether **micro-gold payoff realization** is
strong enough in those hours.

### 3. Gold has dual demand and macro-specific drivers

World Gold Council’s recent research emphasizes gold’s “dual nature”:

- consumer/industrial demand
- investment/safe-haven/official demand

CME’s gold overview also highlights the main short-horizon macro drivers:

- U.S. monetary policy
- inflation data
- payrolls
- the U.S. dollar
- political/economic stress across major regions, especially China, Japan,
  Europe, and the Middle East

Implication:

- gold is more event-sensitive than a generic “metals” label suggests
- macro windows and geopolitical shocks are first-class gold inputs
- a gold research path that ignores macro timing is probably underspecified

Sources:

- CME gold overview: <https://www.cmegroup.com/education/lessons/product-gold.html>
- CME gold market developments note: <https://www.cmegroup.com/education/articles-and-reports/comex-gold-futures-a-look-at-recent-market-developments.html>
- World Gold Council liquidity/market structure research:
  - <https://www.gold.org/goldhub/research/market-primer/gold-market-primer-market-size-and-structure>
  - <https://www.gold.org/goldhub/research/relevance-of-gold-as-a-strategic-asset/key-attributes-liquidity>

## Repo-specific handling rules

### 1. Proxy use: narrow and explicit only

Current repo policy already allows `GC` proxy use for price-safe filters on
`MGC` questions.

That remains reasonable, but only for the right question:

- price-based trigger discovery
- regime comparisons
- session ranking

It is **not** reasonable for:

- execution assumptions
- payoff realization
- winner-size / scratch-rate / slippage questions
- volume-based filters
- deployment claims

Current repo truth now sharpens that further:

- trigger parity survives
- payoff translation breaks

So any future `GC` proxy claim must say explicitly which side of that line it is
on.

### 2. Gold research should be session-aware, macro-aware, and contract-aware

Proper gold handling in this repo means:

- session-aware:
  - include Asia/London/U.S. gold-relevant hours in the reasoning
- macro-aware:
  - respect FOMC, CPI, PPI, NFP, and broader geopolitical stress
- contract-aware:
  - separate `GC` price-discovery truth from `MGC` execution truth

### 3. Do not generalize from 5-minute proxy findings to 15/30 minutes

Our current canonical `GC -> MGC` translation proof is 5-minute only.

So:

- do not use this note to claim that 15m/30m `MGC` paths are proxy-grounded
- do not infer that a weak 5m translation means all gold timeframes are dead

### 4. Roll and expiration handling must be explicit

Because both contracts are physically delivered and terminate on the third last
business day of the contract month:

- roll discipline should be explicit in research
- front-month-only research around expiry should be treated carefully
- native `MGC` behavior near delivery month should not be silently assumed from
  `GC`

### 5. Price discovery is concentrated in the active month, not spread evenly across the curve

World Gold Council's COMEX market-structure notes emphasize that trading
activity is primarily concentrated in the **active month** and that the nearest
dated contract often acts as the spot proxy for U.S. futures users.

Implication:

- do not treat all listed `GC` or `MGC` months as interchangeable when framing
  research conclusions
- avoid silently mixing active-month behavior with deferred-month behavior in
  translation claims
- when a `GC -> MGC` proxy argument is made, it should be clear that the claim
  is about the comparable active-month price-discovery surface, not about the
  entire contract strip

## What this means for the next audit

The next correct gold question is now narrower:

- not “is GC proxy valid?”
- not “is gold dead on micro?”
- not “should we reopen broad GC proxy discovery?”

It is:

- where does **5-minute MGC payoff compression** come from in the warm translated
  families, and can lower-RR / exit-shape handling recover it?

That is the right next move because:

- the asset/contract research says gold is globally liquid and macro-sensitive
- the internal audit says trigger transfer is fine
- the remaining break is native `MGC` payoff realization

## Practical do / do not

Do:

- use `GC` for narrow price-safe research questions when explicitly disclosed
- use `MGC` for any execution or deployability question
- treat Asia/London gold sessions as legitimate, not peripheral
- account for delivery/expiry mechanics explicitly
- keep macro-event awareness high on gold

Do not:

- treat `GC` and `MGC` as interchangeable execution surfaces
- revive the retired `GC` shelf because price correlation is high
- assume broad proxy success means broad transfer
- assume weak `MGC` broad baseline means gold itself is dead
- silently extend 5-minute proxy conclusions to 15m/30m
