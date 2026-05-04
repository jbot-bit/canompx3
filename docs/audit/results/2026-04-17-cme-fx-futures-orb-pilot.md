# CME FX Futures ORB Pilot

Stage verdict: **NO_GO**

Locked implementation:
- ORB aperture: `O5`
- Economics baseline: `E2 / CB1 / RR1.0`
- Descriptive companion: `E1 / CB1 / RR1.0`
- Locked round-trip friction: `$29.10`

Coverage audit:
- Raw request metadata claimed a `180`-day pull window, but that is not the realized eligible-session sample.
- Actual candidate coverage is measured from decoded raw bars after front-month filtering, Brisbane 09:00 trading-day assignment, and complete ORB-window enforcement.

Benchmark provenance (per-cell tables below):
- **Live benchmark** (`MNQ TOKYO_OPEN / EUROPE_FLOW / COMEX_SETTLE` E2 ExpR): sourced from same-window recomputation on the canonical `orb_outcomes` table at the corresponding MNQ (session, O5, E2, RR1.0) combination over the FX pilot's coverage window. `@canonical-source`: `gold.db` `orb_outcomes` joined to `daily_features` (orb_minutes=5), same 2025-09-18 → 2026-04-06 window as the pilot for fair delta comparison. Not from `validated_setups` or `strategy_fitness` — deliberately a same-window recompute, not a deployment-grade cross-cite.
- **Dead benchmark** (`M6E TOKYO_OPEN` / `M6E broad ORB family mean`): sourced from the prior dead-FX ORB scan on the M6E family, captured before M6E was retired from `ACTIVE_ORB_INSTRUMENTS` (see `pipeline.asset_configs`). Family-mean figures aggregate `M6E TOKYO_OPEN`, `M6E LONDON_METALS`, `M6E US_DATA_830`, `M6E US_DATA_1000`, `M6E EUROPE_FLOW` on O5 E2 RR1.0. Not deployment-grade-citable — used here as a floor/sanity reference only.

## 6J TOKYO_OPEN

- Verdict: **NO_GO**
- Coverage: raw UTC dates `180`, Brisbane trading days `161`, eligible session days `147`
- Boundary losses: truncated start `none`, truncated end `2026-04-16`, Sunday reopen pseudo-days excluded `12`
- Candidate metrics: double-break `79.6%`, fakeout `57.8%`, continuation E1 `43.5%`, continuation E2 `41.5%`, E2 ExpR `-0.424R`, median risk `$91.60`, friction/risk `31.8%`
- Dead benchmark `M6E TOKYO_OPEN`: double-break `82.8%`, fakeout `58.3%`, continuation E2 `41.7%`, E2 ExpR `-0.397R`
- Live benchmark `MNQ TOKYO_OPEN`: double-break `76.1%`, fakeout `40.5%`, continuation E2 `59.5%`, E2 ExpR `+0.046R`
- Delta vs dead FX: double-break `-3.2%`, fakeout `-0.5%`, continuation E2 `-0.2%`, E2 ExpR `-0.027R`
- Locked gates: cleanliness `False`, economics `False`, live guardrail `False`
- Stability note: 0/8 months positive on E2 RR1.0; worst 2026-03=-0.548R (n=21)

## 6B LONDON_METALS

- Verdict: **NO_GO**
- Coverage: raw UTC dates `180`, Brisbane trading days `161`, eligible session days `145`
- Boundary losses: truncated start `none`, truncated end `2026-04-16`, Sunday reopen pseudo-days excluded `12`
- Candidate metrics: double-break `80.0%`, fakeout `49.7%`, continuation E1 `47.6%`, continuation E2 `50.3%`, E2 ExpR `-0.336R`, median risk `$79.10`, friction/risk `36.8%`
- Dead benchmark `M6E broad ORB family mean`: double-break `78.6%`, fakeout `50.2%`, continuation E2 `47.8%`, E2 ExpR `-0.302R`
- Live benchmark `MNQ EUROPE_FLOW`: double-break `76.9%`, fakeout `41.1%`, continuation E2 `58.9%`, E2 ExpR `+0.043R`
- Delta vs dead FX: double-break `1.4%`, fakeout `-0.5%`, continuation E2 `2.5%`, E2 ExpR `-0.035R`
- Locked gates: cleanliness `False`, economics `False`, live guardrail `False`
- Stability note: 0/8 months positive on E2 RR1.0; worst 2026-01=-0.651R (n=21)

## 6B US_DATA_830

- Verdict: **NO_GO**
- Coverage: raw UTC dates `180`, Brisbane trading days `161`, eligible session days `148`
- Boundary losses: truncated start `none`, truncated end `2026-04-16`, Sunday reopen pseudo-days excluded `12`
- Candidate metrics: double-break `82.4%`, fakeout `49.0%`, continuation E1 `43.2%`, continuation E2 `46.9%`, E2 ExpR `-0.344R`, median risk `$72.85`, friction/risk `39.9%`
- Dead benchmark `M6E broad ORB family mean`: double-break `78.6%`, fakeout `50.2%`, continuation E2 `47.8%`, E2 ExpR `-0.302R`
- Live benchmark `MNQ COMEX_SETTLE`: double-break `78.3%`, fakeout `41.2%`, continuation E2 `58.3%`, E2 ExpR `+0.064R`
- Delta vs dead FX: double-break `3.8%`, fakeout `-1.2%`, continuation E2 `-0.9%`, E2 ExpR `-0.042R`
- Locked gates: cleanliness `False`, economics `False`, live guardrail `False`
- Stability note: 1/8 months positive on E2 RR1.0; worst 2025-11=-0.509R (n=20)

## 6B US_DATA_1000

- Verdict: **NO_GO**
- Coverage: raw UTC dates `180`, Brisbane trading days `161`, eligible session days `147`
- Boundary losses: truncated start `none`, truncated end `2026-04-16`, Sunday reopen pseudo-days excluded `12`
- Candidate metrics: double-break `71.4%`, fakeout `42.2%`, continuation E1 `54.4%`, continuation E2 `54.4%`, E2 ExpR `-0.221R`, median risk `$91.60`, friction/risk `31.8%`
- Dead benchmark `M6E broad ORB family mean`: double-break `78.6%`, fakeout `50.2%`, continuation E2 `47.8%`, E2 ExpR `-0.302R`
- Live benchmark `MNQ COMEX_SETTLE`: double-break `78.3%`, fakeout `41.2%`, continuation E2 `58.3%`, E2 ExpR `+0.064R`
- Delta vs dead FX: double-break `-7.2%`, fakeout `-8.0%`, continuation E2 `6.6%`, E2 ExpR `+0.081R`
- Locked gates: cleanliness `False`, economics `False`, live guardrail `False`
- Stability note: 0/8 months positive on E2 RR1.0; worst 2026-04=-0.648R (n=10)

## 6A LONDON_METALS

- Verdict: **NO_GO**
- Coverage: raw UTC dates `180`, Brisbane trading days `161`, eligible session days `147`
- Boundary losses: truncated start `none`, truncated end `2026-04-16`, Sunday reopen pseudo-days excluded `12`
- Candidate metrics: double-break `86.4%`, fakeout `63.3%`, continuation E1 `40.8%`, continuation E2 `36.7%`, E2 ExpR `-0.508R`, median risk `$74.10`, friction/risk `39.3%`
- Dead benchmark `M6E broad ORB family mean`: double-break `78.6%`, fakeout `50.2%`, continuation E2 `47.8%`, E2 ExpR `-0.302R`
- Live benchmark `MNQ EUROPE_FLOW`: double-break `76.9%`, fakeout `41.1%`, continuation E2 `58.9%`, E2 ExpR `+0.043R`
- Delta vs dead FX: double-break `7.8%`, fakeout `13.1%`, continuation E2 `-11.1%`, E2 ExpR `-0.206R`
- Locked gates: cleanliness `False`, economics `False`, live guardrail `False`
- Stability note: 0/8 months positive on E2 RR1.0; worst 2025-12=-0.582R (n=22)

## 6A US_DATA_830

- Verdict: **NO_GO**
- Coverage: raw UTC dates `180`, Brisbane trading days `161`, eligible session days `146`
- Boundary losses: truncated start `none`, truncated end `2026-04-16`, Sunday reopen pseudo-days excluded `12`
- Candidate metrics: double-break `79.5%`, fakeout `57.9%`, continuation E1 `36.3%`, continuation E2 `36.6%`, E2 ExpR `-0.428R`, median risk `$74.10`, friction/risk `39.3%`
- Dead benchmark `M6E broad ORB family mean`: double-break `78.6%`, fakeout `50.2%`, continuation E2 `47.8%`, E2 ExpR `-0.302R`
- Live benchmark `MNQ COMEX_SETTLE`: double-break `78.3%`, fakeout `41.2%`, continuation E2 `58.3%`, E2 ExpR `+0.064R`
- Delta vs dead FX: double-break `0.8%`, fakeout `7.8%`, continuation E2 `-11.3%`, E2 ExpR `-0.126R`
- Locked gates: cleanliness `False`, economics `False`, live guardrail `False`
- Stability note: 0/8 months positive on E2 RR1.0; worst 2026-04=-0.834R (n=10)

## 6A US_DATA_1000

- Verdict: **NO_GO**
- Coverage: raw UTC dates `180`, Brisbane trading days `161`, eligible session days `148`
- Boundary losses: truncated start `none`, truncated end `2026-04-16`, Sunday reopen pseudo-days excluded `12`
- Candidate metrics: double-break `68.9%`, fakeout `46.3%`, continuation E1 `52.0%`, continuation E2 `51.7%`, E2 ExpR `-0.283R`, median risk `$89.10`, friction/risk `32.7%`
- Dead benchmark `M6E broad ORB family mean`: double-break `78.6%`, fakeout `50.2%`, continuation E2 `47.8%`, E2 ExpR `-0.302R`
- Live benchmark `MNQ COMEX_SETTLE`: double-break `78.3%`, fakeout `41.2%`, continuation E2 `58.3%`, E2 ExpR `+0.064R`
- Delta vs dead FX: double-break `-9.7%`, fakeout `-3.9%`, continuation E2 `3.9%`, E2 ExpR `+0.019R`
- Locked gates: cleanliness `False`, economics `False`, live guardrail `False`
- Stability note: 0/8 months positive on E2 RR1.0; worst 2025-09=-0.569R (n=9)

## Failed Closed

No asset-session met every locked gate. The correct action is to stop here and not rescue the pilot by extending windows, swapping assets, or widening the session list.
