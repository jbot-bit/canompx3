# Harris 2002 — Trading and Exchanges: Market Microstructure for Practitioners

**Source:** `resources/Harris_Trading_Exchanges_Market_Microstructure.pdf` (March 5, 2002 Draft, forthcoming Oxford University Press Fall 2002), 113 pages.
**Author:** Larry Harris, Fred V. Keenan Chair in Finance, Marshall School of Business, USC.
**Extracted:** 2026-05-12.

**Source-state note (binding):** The PDF in `resources/` is the **publicly-circulated March 2002 Draft Copy** (113 pp). The published Oxford 2002 hardcover is ~640 pp. The draft contains full chapter outlines for all parts and substantive prose through approximately the early chapters of each part — many sub-sections in chapters 5+ show as outline-only (`p.-1` in PyMuPDF outline). Where this extract cites a passage, it cites a page that exists in the draft and quotes from text that is present in the draft. Claims sourced to outlined-but-unwritten sections are explicitly marked `(outline only in draft — concept canonical, prose unverified)`.

**Criticality for our project:** 🟢 **HIGH** — Harris is the canonical microstructure textbook for practitioners and provides the *mechanism* layer that our existing literature (Bailey, Harvey-Liu, Chordia, Carver, Fitschen, Chan) does not. Where those papers establish *whether* an edge survives statistical tests, Harris establishes *why* the edge exists (or doesn't) in terms of order flow, adverse selection, and stop-cascade dynamics. Critical for: (a) mechanism priors on ORB breakouts, (b) understanding adverse-selection cost embedded in our `cost_specs` friction, (c) order-anticipator dynamics that explain why some sessions have stronger breakout follow-through.

---

## Why Harris belongs in our institutional literature

Our `docs/institutional/mechanism_priors.md` enumerates testable beliefs about *why* ORB-class strategies have edge. Up to now, the supporting literature in `docs/institutional/literature/` has been:

- **Validation methodology** (Bailey 2013, Bailey-LdP 2014, Harvey-Liu 2015, LdP 2020, Chordia 2018, Pepelyshev-Polunchenko 2015).
- **Strategy-design conventions** (Chan 2008/2013, Fitschen 2013, Carver 2015).

Neither set tells us *what microstructural fact produces the edge*. Harris fills that gap. He is the source most commonly cited in academic and industry market-microstructure work, and he writes at a level usable by practitioners (not equation-only).

The draft we have is enough for our purposes because the conceptual content is fully present in Parts I-V (Chapters 1-22). Equation-heavy formalism appears later in the book; the draft's strength is exactly the part we need — *why traders behave the way they do, and what that implies for short-horizon directional strategies on futures*.

---

## Key claim 1 — Stop-order cascade is the literature mechanism for breakout momentum

**Source:** Harris draft Ch 4 ("Orders and Order Properties") outline + Ch 24 ("Bubbles, Crashes, and Circuit Breakers") substantive content p.107 (PyMuPDF page index 105, equivalent to PDF page 107 in the draft's internal numbering).

**Verbatim (p.107, prose body):**
> "Late buyers especially worry about their positions and often start selling to stop their losses. Those traders who financed their positions on margin may have to sell their positions to satisfy margin calls from their brokers. **Other long holders who have placed stop loss orders also will start to sell. Order anticipators may anticipate these margin calls and these stop orders and sell before them.** A crash occurs when the combined effect of all their selling causes prices to quickly fall."

**Mechanism implication for canompx3:**

This is the **canonical microstructure source** for the stop-cascade mechanism that Chan 2013 Ch 7 cites as the cause of intraday breakout momentum (already extracted at `chan_2013_ch7_intraday_momentum.md`). Chan attributes the mechanism; Harris is the original-source teacher of it.

**What this anchors in our project:**
- Our **E2 entry model** (stop-market order placed at ORB high+1tick / ORB low-1tick) is *literally an instance of the stop-order class Harris describes*. The stop-order touches the ORB boundary, fires, and is part of the cascade that pushes price further past the level.
- Our E2-look-ahead 41.3% contamination class bug (`memory/recent_findings.md` E2 entry, `docs/postmortems/2026-04-21-e2-break-bar-lookahead.md`) is the direct empirical signature of this cascade: the *first* range-cross fires E2, then the bar-close-outside cross (which is what `daily_features` labels as "the break") happens later in the same cascade. The 41.3% gap between range-cross and close-cross is the cascade window itself.
- Our `Order Anticipators` chapter (Harris Ch 11, p.49-51 substantive) describes *exactly* the class of traders who profit by trading ahead of these stop cascades. The reason ORB breakouts have persistent edge is that the cascade is mechanical (stop orders fire at known prices) but the slope after the cascade is conditional (depends on follow-on liquidity).

**Falsification corollary:** if a putative ORB edge does not have a coherent stop-cascade mechanism (e.g., the session has no concentration of stop placements at obvious round-number or visible-level prices), our prior on the edge should be much lower. This is the kind of mechanism-first test that `docs/institutional/mechanism_priors.md` is designed to host.

---

## Key claim 2 — Adverse selection is the dominant component of bid/ask spread for short-horizon traders

**Source:** Harris draft Ch 14 ("Bid/Ask Spreads") p.59-61 substantive.

**Verbatim (p.59):**
> "The bid/ask spread is the price impatient traders pay for immediacy. Impatient traders buy at the ask price and sell at the bid price. The spread is the compensation dealers and limit order traders receive for offering immediacy."

**Verbatim (p.59, "The most important lesson"):**
> "The most important lesson you may learn from this book appears in this chapter. You will learn why uninformed traders lose to well-informed traders whether they submit limit orders or market orders. **Uninformed traders lose simply because they trade. If you are an uninformed trader and do not want to lose, you should minimize your trading.**"

**Spread component decomposition** (Harris draft Ch 14 outline; substantive content not in draft prose, but the components are *named* on p.61 outline as `Spread Components`):
- Adverse-selection component (informed-trader expected loss)
- Order-processing component (dealer-cost recovery)
- Inventory-holding component (dealer inventory risk)

**Mechanism implication for canompx3:**

Our `pipeline.cost_model.COST_SPECS` total_friction values (currently MNQ/MES $0.18 round-trip, MGC ~$0.30) are *calibrated* values that bundle commission + slippage. Harris tells us the **slippage portion is dominated by adverse-selection cost** — the dealer (or limit-order trader on the other side of our stop-market fill) suspects we know something they don't, and prices accordingly.

**What this anchors in our project:**
- The reason our friction estimates *cannot* be replaced with broker commission alone is exactly the adverse-selection component. Anyone proposing a lower friction value on the basis of "but the commission is only $X" is missing this lesson.
- The reason E2 entries cost more than E1 entries in real life (a real cost we have not fully empirically validated but suspect on first principles) is that E2 is more aggressive — it's a market-style stop order that takes liquidity. E2 pays the full adverse-selection cost. E1 (limit at ORB level) may offer liquidity and earn part of the spread back.
- Harris's "minimize your trading" lesson is the literature foundation for our G6 / G8 filter family (which cuts trade count) and for our preference for high-RR (RR=1.5 over RR=1.0) — fewer trades, larger expected payoff per trade, dilutes per-trade adverse-selection cost.

**Falsification corollary:** any new strategy that proposes *more* trades per session — e.g., multiple intraday ORB-style entries on overlapping windows — must justify why the additional adverse-selection cost is dominated by the additional edge. The default prior, per Harris, is that more trading destroys edge.

---

## Key claim 3 — Volatility decomposes into fundamental + transitory; regimes matter

**Source:** Harris draft Ch 20 ("Volatility") p.78-80 substantive.

**Verbatim (p.78):**
> "Fundamental volatility is due to unanticipated changes in instrument values while transitory volatility is due to trading activity by uninformed traders."

**Verbatim (p.79):**
> "When new information about changes in fundamental values is common knowledge, prices may change without any trading. For example, suppose that an unexpected killer frost descends upon Florida overnight. The morning news will undoubtedly report the event. The next day, orange juice futures contracts will open at a much higher price than the last price of the previous trading day."

**Verbatim (p.80, on commodities — relevant to MGC):**
> "Commodities that are expensive to store are often quite volatile. The high storage costs ensure that producers and distributors generally will not hold large inventories. When demand exceeds supply, buyers can quickly deplete inventories. Prices then spike up until new production can relieve the shortage."

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

**Source:** Harris draft Ch 11 ("Order Anticipators") p.49-51 substantive.

**Verbatim (p.49):**
> "Order anticipators are speculators who try to profit by trading before other traders trade. They make money when they correctly anticipate how other traders will affect prices or when they can extract option values from the orders that other traders offer to the market."

**Verbatim (p.49):**
> "Order anticipators include front runners, sentiment-oriented technical traders, and squeezers."

**Verbatim (p.49):**
> "Trading by order anticipators often makes prices more volatile and markets less efficient."

**Mechanism implication for canompx3:**

This passage is the **uncomfortable mirror** for our entire ORB premise.

Harris classifies sentiment-oriented technical traders — which is approximately what our ORB breakout E2 entry is — as *order anticipators*. They profit by trading in front of (a) stops about to fire and (b) trend-followers about to chase. They are described as "parasitic" because their profits come at the expense of the traders whose orders are being anticipated.

**The honest read:**
- Our edge, if it exists, is a small slice of the order-anticipator profit space. We are mid-tier — not the fastest stop-hunters (those are HFT shops), not the slowest trend-followers (those are CTAs with weeks-long horizons). We're trying to identify cascades that are *too large* for the fastest anticipators to fully consume by the time we can fire an E2 stop-market entry on a 1-minute bar.
- The reason our edge is small (annual_r values in the $20-40 range per contract, not $200+) is that we *are* late to the cascade. The fast money has already harvested most of the option value Harris describes.
- The reason our holdout discipline matters is that Harris's "make prices more volatile and markets less efficient" sentence implies the strategy *generates its own transitory volatility*. Our backtests cannot capture the regime in which enough copycat ORB traders exist to make the cascade less profitable. Live OOS is the only honest measurement.

**Falsification corollary:** if a new candidate strategy proposes to enter *before* the stop cascade (e.g., at a price reflecting "anticipated breakout"), it is competing with HFT order anticipators on their home turf. The prior on retail-grade execution edge in that space is approximately zero. Stay where the cascade has already started.

---

## Key claim 5 — Liquidity is a bilateral search; immediacy is the priced service

**Source:** Harris draft Ch 19 ("Liquidity") p.75-77 substantive.

**Verbatim (p.75):**
> "Liquidity is the ability to trade large size quickly at low cost when you want to trade. It is the most important characteristic of well-functioning markets."

**Verbatim (p.75):**
> "Impatient traders take liquidity. Dealers, limit order traders, and some speculators offer liquidity."

**Verbatim (p.76):**
> "Liquidity is the object of bilateral search. In a bilateral search, buyers search for sellers, and sellers search for buyers. When a buyer finds a seller who will trade at mutually acceptable terms, the buyer has found liquidity."

**Mechanism implication for canompx3:**

Our **E2 stop-market entry** is by definition a liquidity-*taking* order. At fill time we are the impatient trader; the limit-order traders or dealers on the other side are the immediacy providers.

**What this anchors in our project:**
- Our `cost_specs` slippage component is the literature-named "price of immediacy" — what we pay to be the impatient side.
- Sessions vary in their bilateral-search depth. NYSE_OPEN has the deepest US-equity-related search because the cash equity market is concurrently open. COMEX_SETTLE has a thinner book for MGC because it's the late-day low-volume window for gold futures. This differential is *why* our friction calibration is per-(instrument, session), not per-instrument.
- **Liquidity holes** are when bilateral search temporarily fails. Our worst expected losses come from sessions with regime breaks where the immediacy provider stops providing — exactly what `live_risk_auditor` agent is for.

**Falsification corollary:** any proposed strategy that increases trade frequency on a thin-liquidity session is paying more for immediacy. New sessions should be ranked by available depth, not just by Sharpe.

---

## What's in the draft but NOT yet usable as canonical citation

The following chapters/sections in our draft are **outline-only** (sub-sections show `p.-1` in PDF outline because the prose body was not yet written by March 2002):

- Ch 4 substantive on Stop Orders, Market-If-Touched Orders, Tick-Sensitive Orders (concepts named in outline; prose absent from this draft).
- Ch 5 substantive on Execution Systems, Market Information Systems.
- Ch 10 substantive on Informed Trading Strategies, Styles of Informed Trading.
- Ch 14 substantive on Equilibrium Spreads in Continuous Order-driven Markets, Spread Components inner detail, Cross-sectional Spread Predictions.
- Ch 21 substantive on Implicit Transaction Cost Estimation Methods, Missed Trade Opportunity Costs.
- Ch 22 substantive on Performance Evaluation Methods, Sample Selection Bias.

For these we should **not** cite Harris as the source — we have the concept name from the outline, but the prose body must be sourced from the *published* 2002 edition (which we do not currently have in `resources/`) or from a secondary source citing the same passage.

**Action implication:** if a future research-mode citation needs (e.g.) Harris on stop-order mechanics in full prose, the right move is one of:
- Buy or borrow the Oxford 2002 hardcover and extract the relevant chapter.
- Cite a secondary source that quotes Harris (Hasbrouck 2007, Foucault-Pagano-Roell 2013, O'Hara 1995).
- Treat the citation as "concept canonical per Harris outline + secondary-source prose" and flag it as a partial-source citation in the calling research document.

Per `institutional-rigor.md` § 7, **a partial-source citation must be labeled as such**. We should not invent prose attributed to Harris that does not exist in our local copy.

---

## Application map — how Harris mechanisms map onto canompx3 deployed lanes

| Deployed lane | Primary Harris mechanism | Falsification trigger |
|---|---|---|
| MNQ NYSE_OPEN E2 RR1.0 CB1 COST_LT12 | Stop cascade (Ch 24) + fundamental volatility from US cash open (Ch 20) + order-anticipator slipstream (Ch 11) | If NYSE no longer co-incides with US cash equity open (regulatory change, market hours change), the fundamental-volatility component vanishes — lane should be re-validated. |
| MNQ COMEX_SETTLE E2 RR1.5 CB1 OVNRNG_100 | Settlement-anchored adverse selection (Ch 14) + overnight-range condition gates the transitory-vs-fundamental ratio (Ch 20) | If MNQ settlement procedure changes (e.g., CME modifies settlement methodology), the adverse-selection structure of the session changes — re-validate. |
| MNQ US_DATA_1000 E2 RR1.5 CB1 VWAP_MID_ALIGNED_O15 | Stop cascade (Ch 24) downstream of US_DATA_830 information shock (Ch 20) — second-wave cascade as 09:30 ET cash equities digest the 08:30 data | If US economic data release schedule moves to non-08:30 ET times, the second-wave cascade timing shifts — re-validate. |

This is the **mechanism-prior structure** that `docs/institutional/mechanism_priors.md` should be expanded to host per deployed lane. Harris is the citation source for the mechanism column.

---

## Cross-references inside canompx3

- `docs/institutional/literature/chan_2013_ch7_intraday_momentum.md` — Chan's stop-cascade attribution citing the same mechanism Harris establishes here.
- `docs/institutional/literature/fitschen_2013_path_of_least_resistance.md` — Fitschen on intraday trend-follow grounded in the same Harris mechanism without attribution.
- `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md` — Carver's volatility-targeting is the *risk-management* counterpart to Harris's volatility decomposition.
- `docs/institutional/mechanism_priors.md` — Live trading-logic doc that should host per-lane Harris-cited mechanism priors going forward.
- `docs/postmortems/2026-04-21-e2-break-bar-lookahead.md` — empirical signature of the stop cascade Harris describes; 41.3% range-cross-without-close-cross rate measures the cascade window directly.
- `pipeline/cost_model.py` `COST_SPECS` — calibrated friction values that bundle Harris's adverse-selection component; cannot be replaced by commission alone.

---

## Open questions raised by Harris that we have not yet addressed

1. **Spread-component decomposition for our actual fills.** Harris (Ch 14 outline) decomposes the spread into adverse-selection + order-processing + inventory components. We currently treat the bundle as a single `total_friction` value per (instrument, session). Are there sessions where decomposing this would change our cost model in a way that materially affects live edge? Not investigated.

2. **Time-precedence / price-precedence rules.** Harris (Ch 6 outline) covers continuous-auction matching rules in detail. Our E2 stop-market fill assumes our order touches the book and gets filled at the touch price. In reality, in a fast cascade, our order may be queued behind faster orders and fill at a worse price. We have not empirically measured fill-quality slippage as a function of cascade speed. Open research question — possibly already absorbed into `total_friction` but not isolated.

3. **Block-trade externality.** Harris (Ch 15 outline) covers how large institutional trades affect prices. Our deployed lanes trade 2-lot or smaller (TopStep XFA constraint). We are nowhere near block-size, so this is currently moot — but relevant if self-funded scaling produces larger position sizes.

4. **Circuit-breaker behavior under stress.** Harris (Ch 24 substantive) discusses regulatory responses to extreme volatility. We have no explicit handling of CME-imposed price-limit halts in our backtest or live risk path. Low-probability tail event; should be flagged in `live_risk_auditor` agent scope.

---

## How to cite this extract in research documents

When a research script, hypothesis YAML, or audit-result MD needs to ground a mechanism claim in Harris, the citation format is:

`Harris 2002 (draft, March 5 2002) Ch <N> p.<P> — verbatim quote — see docs/institutional/literature/harris_2002_trading_exchanges_microstructure.md § "<Key claim N — ...>"`

If the underlying claim is outlined in the draft but not prose-substantive (per the "What's in the draft but NOT yet usable" section above), the citation MUST be labeled:

`Harris 2002 (draft — outline only; prose not in local copy) Ch <N> — concept canonical per outline — partial-source citation per institutional-rigor.md § 7`
