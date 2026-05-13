---
source_pdf: resources/Harris_Trading_Exchanges_Market_Microstructure.pdf
edition: Oxford University Press 2002 hardcover (29 chapters, 656 pages)
author: Larry Harris
author_affiliation: Fred V. Keenan Chair in Finance, Marshall School of Business, USC
year: 2002
pages_cited: [78, 256, 257, 299, 300, 303, 399, 409, 410, 412, 422, 424, 443, 471, 472, 557, 560]
chapters_covered: [4, 11, 14, 19, 20, 21, 22, 28]
criticality: HIGH
role: mechanism_source
project_relevance: [stop_cascade, adverse_selection, volatility_decomposition, order_anticipators, liquidity_bilateral_search, spread_components, implicit_transaction_costs, sample_selection_bias]
extracted: 2026-05-12
last_verified: 2026-05-12
verification_script: scripts/research/verify_harris_quotes.py
supersedes: harris_2002_trading_exchanges_microstructure.md (draft-edition extract, 2026-05-12 morning)
---

# Harris 2002 — Trading and Exchanges: Market Microstructure for Practitioners

**Source:** `resources/Harris_Trading_Exchanges_Market_Microstructure.pdf` (Oxford University Press 2002 hardcover, 29 chapters, 656 PDF pages, locally OCR'd 2026-05-12).
**Author:** Larry Harris, Fred V. Keenan Chair in Finance, Marshall School of Business, USC.
**Extracted:** 2026-05-12 (this revision; supersedes the morning draft-edition extract).

**Edition note (binding):** This extract was rebuilt from the published Oxford 2002 hardcover. The prior revision was sourced from the publicly-circulated March 2002 draft, whose prose was re-written in places before publication. **All verbatim quotes below cite hardcover printed-page numbers** (the printed page = PyMuPDF index − 12). The hardcover renumbers some chapters relative to the draft (notably "Bubbles, Crashes, and Circuit Breakers" is Ch 28 in the hardcover, was Ch 24 in the draft).

**Criticality for our project:** 🟢 **HIGH** — Harris is the canonical microstructure textbook for practitioners and provides the *mechanism* layer that our existing literature (Bailey, Harvey-Liu, Chordia, Carver, Fitschen, Chan) does not. Where those papers establish *whether* an edge survives statistical tests, Harris establishes *why* the edge exists (or doesn't) in terms of order flow, adverse selection, and stop-cascade dynamics. Critical for: (a) mechanism priors on ORB breakouts, (b) understanding adverse-selection cost embedded in our `cost_specs` friction, (c) order-anticipator dynamics that explain why some sessions have stronger breakout follow-through, (d) sample-selection-bias awareness in performance evaluation.

---

## Why Harris belongs in our institutional literature

Our `docs/institutional/mechanism_priors.md` enumerates testable beliefs about *why* ORB-class strategies have edge. Up to now, the supporting literature in `docs/institutional/literature/` has been:

- **Validation methodology** (Bailey 2013, Bailey-LdP 2014, Harvey-Liu 2015, LdP 2020, Chordia 2018, Pepelyshev-Polunchenko 2015).
- **Strategy-design conventions** (Chan 2008/2013, Fitschen 2013, Carver 2015).

Neither set tells us *what microstructural fact produces the edge*. Harris fills that gap. He is the source most commonly cited in academic and industry market-microstructure work, and he writes at a level usable by practitioners (not equation-only).

---

## Key claim 1 — Stop-order cascade is the literature mechanism for breakout momentum

**Source:** Harris hardcover Ch 28 ("Bubbles, Crashes, and Circuit Breakers") § 28.1.1 p.557.

**Verbatim (p.557):**
> "Late buyers especially worry about their positions, and often start selling to stop their losses. Traders who financed their positions on margin may have to sell their positions to satisfy margin calls from their brokers. Other long holders who have placed stop loss orders also will start to sell. Order anticipators may anticipate these margin calls and stop orders, and sell before them. A crash occurs when the combined effect of all their selling causes prices to fall quickly."

**Verbatim (p.560, on the cascade extended into portfolio-insurance dynamics):**
> "Portfolio insurance therefore has the same effect on the market as stop orders. Indeed, many portfolio managers implemented their strategies using stop orders."

**Mechanism implication for canompx3:**

This is the **canonical microstructure source** for the stop-cascade mechanism that Chan 2013 Ch 7 cites as the cause of intraday breakout momentum (already extracted at `chan_2013_ch7_intraday_momentum.md`). Chan attributes the mechanism; Harris is the original-source teacher of it.

**What this anchors in our project:**
- Our **E2 entry model** (stop-market order placed at ORB high+1tick / ORB low-1tick) is *literally an instance of the stop-order class Harris describes*. The stop-order touches the ORB boundary, fires, and is part of the cascade that pushes price further past the level.
- Our E2-look-ahead 41.3% contamination class bug (`memory/recent_findings.md` E2 entry, `docs/postmortems/2026-04-21-e2-break-bar-lookahead.md`) is the direct empirical signature of this cascade: the *first* range-cross fires E2, then the bar-close-outside cross (which is what `daily_features` labels as "the break") happens later in the same cascade. The 41.3% gap between range-cross and close-cross is the cascade window itself.
- Our `Order Anticipators` chapter (Harris Ch 11) describes *exactly* the class of traders who profit by trading ahead of these stop cascades. The reason ORB breakouts have persistent edge is that the cascade is mechanical (stop orders fire at known prices) but the slope after the cascade is conditional (depends on follow-on liquidity).

**Falsification corollary:** if a putative ORB edge does not have a coherent stop-cascade mechanism (e.g., the session has no concentration of stop placements at obvious round-number or visible-level prices), our prior on the edge should be much lower. This is the kind of mechanism-first test that `docs/institutional/mechanism_priors.md` is designed to host.

---

## Key claim 2 — Adverse selection is the dominant component of bid/ask spread for short-horizon traders

**Source:** Harris hardcover Ch 14 ("Bid/Ask Spreads") § 14.2 + § 14.3 p.299-303.

**Verbatim (p.299, on the two-component decomposition):**
> "The transaction cost spread component is the part of the bid/ask spread that compensates dealers for their normal costs of doing business. ... The adverse selection spread component is the part of the bid/ask spread that compensates dealers for the losses they suffer when trading with well-informed traders. This component allows dealers to earn from uninformed traders what they lose to informed traders."

**Verbatim (p.299, on transitory volatility from impatient trading):**
> "Bid/ask bounce is a minor form of price volatility caused by impatient traders who demand immediacy. The transitory spread component is responsible for bid/ask bounce."

**Verbatim (p.303, "the most important lesson"):**
> "Uninformed traders thus ultimately lose to informed traders regardless of how they trade. They lose simply because they trade. They can avoid the problem only by not trading."

**Mechanism implication for canompx3:**

Our `pipeline.cost_model.COST_SPECS` total_friction values (currently MNQ/MES $0.18 round-trip, MGC ~$0.30) are *calibrated* values that bundle commission + slippage. Harris tells us the **slippage portion is dominated by adverse-selection cost** — the dealer (or limit-order trader on the other side of our stop-market fill) suspects we know something they don't, and prices accordingly.

**What this anchors in our project:**
- The reason our friction estimates *cannot* be replaced with broker commission alone is exactly the adverse-selection component. Anyone proposing a lower friction value on the basis of "but the commission is only $X" is missing this lesson.
- The reason E2 entries cost more than E1 entries in real life (a real cost we have not fully empirically validated but suspect on first principles) is that E2 is more aggressive — it's a market-style stop order that takes liquidity. E2 pays the full adverse-selection cost. E1 (limit at ORB level) may offer liquidity and earn part of the spread back.
- Harris's "lose simply because they trade" lesson is the literature foundation for our G6 / G8 filter family (which cuts trade count) and for our preference for high-RR (RR=1.5 over RR=1.0) — fewer trades, larger expected payoff per trade, dilutes per-trade adverse-selection cost.

**Falsification corollary:** any new strategy that proposes *more* trades per session — e.g., multiple intraday ORB-style entries on overlapping windows — must justify why the additional adverse-selection cost is dominated by the additional edge. The default prior, per Harris, is that more trading destroys edge.

---

## Key claim 3 — Volatility decomposes into fundamental + transitory; regimes matter

**Source:** Harris hardcover Ch 20 ("Volatility") § 20.1-20.5 p.410-412 + Ch 28 § 28.1.2 p.557.

**Verbatim (p.410, opening of Ch 20):**
> "In this chapter, we identify the origins of volatility and distinguish between its two types. Fundamental volatility is due to unanticipated changes in instrument values, and transitory volatility is due to trading activity by uninformed traders."

**Verbatim (p.412, on storage costs and commodity volatility — relevant to MGC):**
> "Commodities that are expensive to store are often quite volatile. The high storage costs ensure that producers and distributors generally will not hold large inventories. When demand exceeds supply, buyers can quickly deplete inventories. Prices then spike up until new production can relieve the shortage."

**Verbatim (p.557, Ch 28 reaffirming the decomposition in crash context):**
> "Bubbles and crashes—like all price changes—may be due to fundamental or transitory factors. Unexpected information about fundamental values causes fundamental volatility. The demands for liquidity by uninformed traders cause transitory volatility. Most bubbles and crashes involve both types."

**Mechanism implication for canompx3:**

- **Fundamental volatility** drives the *direction* of intraday momentum on event-bearing sessions (US_DATA_830, US_DATA_1000, COMEX_SETTLE for gold). When new value-relevant information arrives, dealers re-price (sometimes with no trades — the "gap" case), and the new equilibrium drives directional flow.
- **Transitory volatility** drives the *noise* that makes E2 stop fills churn around the ORB boundary. The 41.3% range-cross-but-not-close-cross rate measures *exactly* this — wicks from uninformed liquidity-demand fluctuations.
- **Regime sensitivity** of ORB edge follows from this: when fundamental volatility dominates (e.g., NFP release day = US_DATA_830 session), the breakout is driven by information arrival and the cascade is structural. When transitory volatility dominates (e.g., chop sessions like SINGAPORE_OPEN with limited information flow), the breakout signal is just noise.

**What this anchors in our project:**
- Our **regime classification** (the `regime=HOT/COLD` field on deployed lanes in `live_config`) implicitly captures this distinction. The current MNQ NYSE_OPEN E2 RR1.0 deployed lane has `regime=HOT` precisely because that session has high fundamental information flow from US equity-cash-market open.
- Our preference for *event-anchored sessions* (US_DATA_830, NYSE_OPEN, COMEX_SETTLE) over time-anchored sessions (TOKYO_OPEN, SINGAPORE_OPEN where the local cash market is the relevant information source but our instruments are global futures) follows directly from Harris's volatility decomposition.
- Our MGC sensitivity to *commodity storage and supply* explains why MGC's deployed lanes have not been NYSE_OPEN — gold's fundamental-volatility events are different (FOMC days, DXY moves, COMEX inventory releases), and the NYSE 09:30 ET open is not a primary information event for the gold complex.

**Falsification corollary:** an ORB strategy that survives backtesting only on sessions with low fundamental-information arrival is most likely curve-fitting to transitory volatility. New session candidates should be screened against an explicit information-event prior before formal validation.

---

## Key claim 4 — Order anticipators are the apex predators of stop-cascade markets

**Source:** Harris hardcover Ch 11 ("Order Anticipators") § 11.5 SUMMARY p.256-257.

**Verbatim (p.256):**
> "Order anticipators are speculators who attempt to profit from information about the trades that other traders will make rather than from information about fundamental values. They trade in front of other traders. They profit when other traders move prices to complete their trades."

**Verbatim (p.256):**
> "Order anticipators are parasitic traders because they profit only by exploiting other traders. They generally do not make prices more informative, and they do not make the markets more liquid."

**Verbatim (p.257, on the three sub-types):**
> "Order anticipators differ by what they know about the trades that other traders will make. Front runners know exactly what other traders have decided to do. Sentiment-oriented technical traders try to predict what other traders will decide to do. Squeezers force other traders into making trades at very disadvantageous prices."

**Mechanism implication for canompx3:**

This passage is the **uncomfortable mirror** for our entire ORB premise.

Harris classifies sentiment-oriented technical traders — which is approximately what our ORB breakout E2 entry is — as *order anticipators*. They profit by trading in front of (a) stops about to fire and (b) trend-followers about to chase. He describes them as "parasitic" because their profits come at the expense of the traders whose orders are being anticipated.

**The honest read:**
- Our edge, if it exists, is a small slice of the order-anticipator profit space. We are mid-tier — not the fastest stop-hunters (those are HFT shops), not the slowest trend-followers (those are CTAs with weeks-long horizons). We're trying to identify cascades that are *too large* for the fastest anticipators to fully consume by the time we can fire an E2 stop-market entry on a 1-minute bar.
- The reason our edge is small (annual_r values in the $20-40 range per contract, not $200+) is that we *are* late to the cascade. The fast money has already harvested most of the option value Harris describes.
- The reason our holdout discipline matters is that Harris's "parasitic" framing implies the strategy *generates its own transitory volatility*. Our backtests cannot capture the regime in which enough copycat ORB traders exist to make the cascade less profitable. Live OOS is the only honest measurement.

**Falsification corollary:** if a new candidate strategy proposes to enter *before* the stop cascade (e.g., at a price reflecting "anticipated breakout"), it is competing with HFT order anticipators on their home turf. The prior on retail-grade execution edge in that space is approximately zero. Stay where the cascade has already started.

---

## Key claim 5 — Liquidity is a bilateral search; immediacy is the priced service

**Source:** Harris hardcover Ch 19 ("Liquidity") § 19.1 + § 19.6 p.395-410.

**Verbatim (p.399, defining liquidity):**
> "To summarize, liquidity is the ability to quickly trade large size at low cost. 'Quickly' refers to immediacy; 'size,' to depth; and 'cost,' to width."

**Verbatim (p.409, Ch 19 summary bullets):**
> "Liquidity is the object of a bilateral search problem."

> "Liquidity is the ability to trade when you want to trade, at low cost."

**Mechanism implication for canompx3:**

Our **E2 stop-market entry** is by definition a liquidity-*taking* order. At fill time we are the impatient trader; the limit-order traders or dealers on the other side are the immediacy providers.

**What this anchors in our project:**
- Our `cost_specs` slippage component is the literature-named "price of immediacy" — what we pay to be the impatient side.
- Sessions vary in their bilateral-search depth. NYSE_OPEN has the deepest US-equity-related search because the cash equity market is concurrently open. COMEX_SETTLE has a thinner book for MGC because it's the late-day low-volume window for gold futures. This differential is *why* our friction calibration is per-(instrument, session), not per-instrument.
- **Liquidity holes** are when bilateral search temporarily fails. Our worst expected losses come from sessions with regime breaks where the immediacy provider stops providing — exactly what `live_risk_auditor` agent is for.

**Falsification corollary:** any proposed strategy that increases trade frequency on a thin-liquidity session is paying more for immediacy. New sessions should be ranked by available depth, not just by Sharpe.

---

## Key claim 6 — Stop orders are activation triggers, not execution promises

**Source:** Harris hardcover Ch 4 ("Orders and Order Properties") § 4.5 STOP ORDERS p.78-79.

**Verbatim (p.78):**
> "A stop instruction stops an order from executing until price reaches a stop price specified by the trader. Traders attach stop instructions to their orders when they want to buy only after price rises to the stop price or sell only after price falls to the stop price. Orders with stop instructions are called stop orders."

**Verbatim (p.78):**
> "Traders most commonly use stop orders to stop their losses when prices move against their positions."

**Verbatim (p.78, on stop-order fill quality):**
> "The price at which a stop order executes may not be the stop price."

> "If cotton prices fall quickly, the market order may trade at a price substantially below the 70-cent stop price."

**Mechanism implication for canompx3:**

This is the literature definition of the order type our **E2 entry model** uses. Our E2 stop-market order at ORB high + 1 tick (long side) is *exactly* the construct Harris describes: a stop instruction that activates when price touches the stop, after which the market order takes whatever liquidity is available.

**What this anchors in our project:**
- Our **slippage assumption** for E2 fills must respect Harris's warning that the executed price "may not be the stop price." On thin-liquidity sessions, the gap between stop-price and fill-price can be material. This is why `pipeline.cost_model.COST_SPECS` must remain calibrated empirically per (instrument, session), not theoretically.
- The **E1 (limit-at-ORB-level) entry model** is the structural opposite: it offers liquidity at the ORB boundary and waits to be hit. E2 takes liquidity; E1 provides it. Harris's classification makes this distinction precise.
- The **E3 (stop-with-bar-close-confirmation) entry model** is a stop order whose activation condition is "stop price touched AND bar closes outside ORB." This is a hybrid — Harris-class stop order with an additional confirmation gate before fire. Our E3 vs E2 backtests measure exactly the cost of this confirmation delay.

**Falsification corollary:** any new entry-model proposal that uses a "stop limit" (stop with a fill-price ceiling) on a thin-liquidity session must be backtested with realistic no-fill rates. Pure-stop-market is the worst-case slippage class; stop-limit may have better fills but higher no-fill rates. Harris explicitly enumerates this trade-off; we should not assume one dominates the other without measurement.

---

## Key claim 7 — The bid/ask spread has named components and adverse selection dominates short-horizon trading cost

**Source:** Harris hardcover Ch 14 ("Bid/Ask Spreads") § 14.2-14.3 p.299-303.

**Verbatim (p.299, the two-component model):**
> "The two components taken together constitute the total spread. Dealers never quote both components separately. They simply quote their bid and ask prices. To actually estimate the two spread components, analysts must use econometric methods."

**Verbatim (p.303, empirical dominance result):**
> "These analyses indicate that in most markets the adverse selection spread component accounts for more of the total spread than does the transaction cost spread component."

**Verbatim (p.300, on the random-walk property of the adverse-selection component):**
> "Economists also call the adverse selection spread component the permanent spread component. Price changes due to the adverse selection spread component are permanent in the sense that they do not systematically reverse."

**Mechanism implication for canompx3:**

Harris formally decomposes our `cost_specs` total_friction into two pieces — the *transitory* (transaction cost) component, which mean-reverts (bid/ask bounce), and the *permanent* (adverse selection) component, which does not. Our current cost model bundles both into one `total_friction` value per (instrument, session).

**What this anchors in our project:**
- The reason **slippage on big-news sessions** (US_DATA_830, FOMC days) is empirically worse than the static `total_friction` value suggests is that the adverse-selection component spikes during high-information events. Harris's framework predicts this directly.
- The reason **bid/ask bounce on thin sessions** can falsely look like an edge (or anti-edge) in raw 1-minute return autocorrelation is that the transitory component is mean-reverting. Our pipeline implicitly handles this by trading off the bar close, not off mid-bar ticks — but it's worth knowing this is *why* the design works.
- A future **Stage 2 sophistication** could split `cost_specs` into a transitory + permanent component per (instrument, session), calibrated from order-book data. Carver-style position sizing (already in `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md`) would then size against the permanent component only, not the noisy transitory one.

**Falsification corollary:** if our P&L empirically shows worse-than-modelled slippage on a specific session-event combo, the most likely culprit is adverse-selection spread compression we haven't priced in. The right diagnostic is to compare expected vs realized fill-price-minus-stop-price on that session — not to widen `total_friction` uniformly across all sessions.

---

## Key claim 8 — Implicit transaction costs require benchmark-difference estimators; methodology choice matters

**Source:** Harris hardcover Ch 21 ("Liquidity and Transaction Cost Measurement") § 21.2 p.422-423.

**Verbatim (p.422, on benchmark-based measurement):**
> "Implicit transaction costs and missed trade opportunity costs are harder to measure because they require some benchmark against which to compare trade and no-trade prices. To measure the price impact of a completed trade, analysts must estimate what prices would have been if the trade had not taken place. To measure the opportunity cost of an uncompleted trade, analysts must estimate the average prices at which the trade would have taken place if it had been completed. These estimation problems make transaction cost measurement a difficult and imprecise science."

**Verbatim (p.422, on the canonical methods):**
> "Traders estimate implicit transaction costs by using specified price benchmark methods and econometric transaction cost estimation methods. The price benchmark methods are the most commonly used. They are easier to implement than the econometric methods, and their results are easier to understand."

**Verbatim (p.424, on Perold implementation shortfall):**
> "The transaction cost estimator based on a pre-trade quotation midpoint uses the quotation midpoint at the time the portfolio manager decided to trade. Analysts usually call this method Perold's implementation shortfall (after André Perold, who popularized it in an influential 1988 Journal of Portfolio Management article)."

**Mechanism implication for canompx3:**

This is the literature foundation for **how we should measure live-vs-backtest fill quality**. The right benchmark for an E2 stop-market fill is *not* the ORB high + 1 tick (the stop price) — it's the mid-quote at the moment we submitted the order. The difference is implementation shortfall.

**What this anchors in our project:**
- Our **backtest cost assumption** approximates implementation shortfall as a flat `total_friction` per round-trip. Harris would call this a coarse pre-trade-quote-midpoint benchmark. It's defensible for institutional-scale calibration but should be cross-checked against live fills.
- Our **live execution monitoring** (forthcoming as part of the live-paper / live-real workstream) should record per-fill the (stop_price, fill_price, mid_at_submit) tuple so we can compute Perold-style shortfall empirically. This is the only honest way to know whether our `cost_specs` calibration is too tight or too loose.
- **Missed-trade opportunity costs** (Harris's other half of the chapter) are particularly relevant for our **E3 confirmation-bar entry model** — when E3 doesn't fire because the bar didn't close outside ORB, but E2 would have fired and made money, that's a measurable opportunity cost. Comparing E2 vs E3 P&L on identical days is the canonical Perold-equivalent for our entry-model selection.

**Falsification corollary:** if live fills consistently show negative Perold shortfall (we're getting filled *better* than mid-quote suggests), that's a red flag — it usually means we're slow enough that the market has already moved before our order reaches the book, and the apparent "good fill" reflects information we already missed. Pre-register the shortfall metric and watch its sign.

---

## Key claim 9 — Sample-selection bias systematically corrupts performance inferences

**Source:** Harris hardcover Ch 22 ("Performance Evaluation and Prediction") § 22.6 p.471-472.

**Verbatim (p.471):**
> "The sample selection bias arises when some process selects the information that you see about some object. If the process does not randomly select the information that you see, you will see only selected aspects of the object and your impression of it will not be accurate. Decisions that you make based on a sample of past data will reflect this bias."

**Verbatim (p.443, framing the chapter's stakes):**
> "By the end of this chapter, you will understand why past performance does not necessarily predict future returns. You will also understand how sample selection biases affect the inferences you may make about investment decisions. Failures to understand these issues probably account for more trading losses than any other mistakes traders make."

**Verbatim (p.472, the mutual-fund-family example):**
> "Mutual fund distributors often kill their poorly performing funds, usually by merging them with better-performing funds."

**Mechanism implication for canompx3:**

This is Harris's complementary frame to Bailey-LdP 2014's *deflated Sharpe ratio* (`bailey_lopezdeprado_2014_dsr_sample_selection.md`) and Harvey-Liu 2015's *backtest haircut* (`harvey_liu_2015_backtesting.md`). Where Bailey-LdP gives the *math*, Harris gives the *intuition*.

**What this anchors in our project:**
- Our **`MinBTL` budget enforcement** in `pre_registered_criteria.md` exists precisely because Harris's sample-selection bias applies to discovery scans: of N candidate strategies, the best-looking k=10 will look like edge even under the null. Harris is the literature anchor for *why* we cap K and apply BH-FDR.
- The **2026-04-15 incident** (176 false BH-FDR survivors from look-ahead bias, reduced to 13 honest survivors after temporal-gate fix) is a textbook Harris sample-selection-bias case: the scan was "selecting" features by exposing each to a contaminated input that biased the test toward false positives. Without recognizing this as a Harris-class bias rather than a code bug, the fix could have been "raise the BH threshold" instead of "fix the look-ahead."
- The **grandfathered 124 validated_setups** (per Amendment 2.4 in `pre_registered_criteria.md`) are research-provisional precisely because they were discovered under a brute-force regime that violated Harris's bias controls. They are not invalidated, but they are **not Mode-A-holdout-clean**, and the institutional rigor framework requires we treat them as such.

**Falsification corollary:** any future "look at our deployed lanes, they're all profitable, that proves the strategy works" claim is a textbook Harris sample-selection trap (we only deployed the survivors). The right control is to measure all-discovered-strategies-ever P&L, not just the ones that made it to `validated_setups`. Where the all-strategies population is unavailable, Bailey-LdP DSR is the corrective.

---

## Application map — how Harris mechanisms map onto canompx3 deployed lanes

| Deployed lane | Primary Harris mechanism | Falsification trigger |
|---|---|---|
| MNQ NYSE_OPEN E2 RR1.0 CB1 COST_LT12 | Stop cascade (Ch 28) + fundamental volatility from US cash open (Ch 20) + order-anticipator slipstream (Ch 11) | If NYSE no longer coincides with US cash equity open (regulatory change, market hours change), the fundamental-volatility component vanishes — lane should be re-validated. |
| MNQ COMEX_SETTLE E2 RR1.5 CB1 OVNRNG_100 | Settlement-anchored adverse selection (Ch 14) + overnight-range condition gates the transitory-vs-fundamental ratio (Ch 20) | If MNQ settlement procedure changes (e.g., CME modifies settlement methodology), the adverse-selection structure of the session changes — re-validate. |
| MNQ US_DATA_1000 E2 RR1.5 CB1 VWAP_MID_ALIGNED_O15 | Stop cascade (Ch 28) downstream of US_DATA_830 information shock (Ch 20) — second-wave cascade as 09:30 ET cash equities digest the 08:30 data | If US economic data release schedule moves to non-08:30 ET times, the second-wave cascade timing shifts — re-validate. |

This is the **mechanism-prior structure** that `docs/institutional/mechanism_priors.md` should be expanded to host per deployed lane. Harris is the citation source for the mechanism column.

---

## Cross-references inside canompx3

- `docs/institutional/literature/chan_2013_ch7_intraday_momentum.md` — Chan's stop-cascade attribution citing the same mechanism Harris establishes here.
- `docs/institutional/literature/fitschen_2013_path_of_least_resistance.md` — Fitschen on intraday trend-follow grounded in the same Harris mechanism without attribution.
- `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md` — Carver's volatility-targeting is the *risk-management* counterpart to Harris's volatility decomposition.
- `docs/institutional/literature/bailey_lopezdeprado_2014_dsr_sample_selection.md` — the formal statistical correction for Harris's sample-selection bias.
- `docs/institutional/literature/harvey_liu_2015_backtesting.md` — Harvey-Liu Sharpe haircut, the operational discount factor for sample-selection bias.
- `docs/institutional/mechanism_priors.md` — Live trading-logic doc that should host per-lane Harris-cited mechanism priors going forward.
- `docs/postmortems/2026-04-21-e2-break-bar-lookahead.md` — empirical signature of the stop cascade Harris describes; 41.3% range-cross-without-close-cross rate measures the cascade window directly.
- `pipeline/cost_model.py` `COST_SPECS` — calibrated friction values that bundle Harris's adverse-selection component; cannot be replaced by commission alone.

---

## Open questions raised by Harris that we have not yet addressed

1. **Spread-component decomposition for our actual fills.** Harris (Ch 14) decomposes the spread into transaction-cost + adverse-selection components. We currently treat the bundle as a single `total_friction` value per (instrument, session). Are there sessions where decomposing this would change our cost model in a way that materially affects live edge? Not investigated.

2. **Time-precedence / price-precedence rules.** Harris (Ch 6) covers continuous-auction matching rules in detail. Our E2 stop-market fill assumes our order touches the book and gets filled at the touch price. In reality, in a fast cascade, our order may be queued behind faster orders and fill at a worse price. We have not empirically measured fill-quality slippage as a function of cascade speed. Open research question — possibly already absorbed into `total_friction` but not isolated.

3. **Block-trade externality.** Harris (Ch 15) covers how large institutional trades affect prices. Our deployed lanes trade 2-lot or smaller (TopStep XFA constraint). We are nowhere near block-size, so this is currently moot — but relevant if self-funded scaling produces larger position sizes.

4. **Circuit-breaker behavior under stress.** Harris (Ch 28) discusses regulatory responses to extreme volatility. We have no explicit handling of CME-imposed price-limit halts in our backtest or live risk path. Low-probability tail event; should be flagged in `live_risk_auditor` agent scope.

5. **Perold implementation-shortfall measurement on live fills.** Harris (Ch 21) prescribes the canonical estimator; we have not yet wired live execution monitoring to compute it. Pre-register the metric before live-real deployment.

---

## How to cite this extract in research documents

When a research script, hypothesis YAML, or audit-result MD needs to ground a mechanism claim in Harris, the citation format is:

`Harris 2002 (hardcover, Oxford UP) Ch <N> p.<P> — "<verbatim quote>" — see docs/institutional/literature/harris_2002_trading_exchanges_microstructure.md § "<Key claim N — ...>"`

All quotes in this extract have been verified against the OCR'd PDF in `resources/` by `scripts/research/verify_harris_quotes.py`. To re-run the verification after any edit:

```bash
python scripts/research/verify_harris_quotes.py
```

Expected output: `OK: N/N` where N is the count of verbatim quotes in this file. Any FAIL line indicates the cited page no longer contains the quoted text (e.g., the extract was edited without re-checking, or the PDF source changed).
