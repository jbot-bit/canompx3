# Elite Trader Funding (ETF) Canonical Research Corpus

**Created:** 2026-04-15 (replaces Apex research — Apex is DEAD per `memory/prop_firm_automation_verified_apr5.md`)
**Next re-fetch:** 2026-07-15 (quarterly)
**Source mix:** Elite help center + ETF Terms of Service + third-party reviewers (QuantVPS, PropTradingVibes, FuturesFury, TradersPost)
**Quality note:** ETF's own help center homepage returned 403 to our fetcher but help articles are directly accessible. Pricing and per-plan details cross-referenced.

---

## 1. Account structure

ETF has **6 evaluation models** (more than any other firm we've researched):

| Model | Type | Cost | Account sizes |
|---|---|---|---|
| **1-Step** | Monthly subscription until pass | $165–$655/mo | $50K–$300K |
| **EOD Drawdown** | Monthly subscription | $295–$605/mo | $50K–$300K |
| **Fast Track** | One-time evaluation, 14 calendar-day limit | $75–$175 | $100K–$250K |
| **Static** | One-time, static drawdown | $99–$449 | $50K–$300K |
| **Diamond Hands** | Monthly | $365/mo | varies |
| **Direct to Funded (DTF)** | Skip evaluation, one-time | $599–$699 | $25K/$50K/$100K |

**After passing evaluation: activation fee** for Sim-Funded account — $87/mo OR one-time $177–$307.

Two funded-account streams:
- **Elite Sim-Funded** — post-evaluation funded account
- **DTF** — direct-funded (no evaluation), capped at $25K max total payout before Live

---

## 2. Concurrent account cap — CHANGED September 17, 2025

From https://help.elitetraderfunding.com/help/max-active-elite-sim-funded-accounts:

**Current cap (post Sep 17, 2025):**
> "Traders will be limited to activating a maximum of 5 Elite Sim-Funded accounts at any given time."

**Plus separate DTF cap:**
> "Limit of 5 active DTF account per trader."

**Grandfather clause** for traders who purchased evaluations before Sep 17, 2025:
> Up to 20 active Elite Sim-Funded accounts, but "only 5 can be from passed Fast Track evaluations."

**Interaction between Sim-Funded and DTF caps:** ambiguous in canonical docs. Likely separate buckets — can have 5 Sim-Funded AND 5 DTF concurrently (total 10), but needs support verification.

---

## 3. 1-Step Plan (canonical rules)

From https://help.elitetraderfunding.com/help/how-the-1-step-plan-works:

- Trailing drawdown during evaluation — trails unrealized gains
- Converts to **static drawdown** in Elite Sim-Funded account once safety net reached (drawdown +$100 realized profit)
- **Minimum 5 trading days** in evaluation (waivable with Add-On ODTP)
- **Post-eval: at least 1 trade per week** mandatory to keep account active
- Max positions: **1 mini = 1 position, 10 micros = 1 position** (10:1 micro-to-mini ratio — same as Topstep TopstepX)

**Pricing (1-Step):**
- $50K: **$165/mo** (cheapest)
- $100K: **$205/mo**
- Range $165–$655/mo across sizes

---

## 4. DTF Plan (canonical rules)

From https://help.elitetraderfunding.com/help/how-the-dtf-plan-works:

- **Skip evaluation** entirely, begin live trading immediately
- One-time fee, NO monthly subscription
- **$25,000 maximum total payout cap** per DTF account before forced transition to Live
- **Unusual position rule: 1:1 mini-to-micro** (not 10:1) — DTF treats mini and micro as equal positions
- Drawdown:
  - **$25K & $50K DTF:** EOD trailing drawdown, locks at starting + $100 after realized profit
  - **$100K DTF:** static drawdown, never trails

**ATD (Active Trading Day) requirements — this is the key consistency constraint:**
- $25K DTF: 10 ATDs per cycle at $300 minimum profit/day
- $50K DTF: 15 ATDs per cycle at $600 minimum profit/day
- $100K DTF: 20 ATDs per cycle at $500 minimum profit/day
- Each ATD must meet **38%, 62%, or 50% best-day rule** depending on account size (strict consistency)
- Pricing: $599–$699 one-time

---

## 5. Payout rules

From https://www.quantvps.com/blog/elite-trader-funding-payout-rules-explained-how-trader-payouts-work (canonical scrape blocked):

- **ATD requirements per cycle:**
  - Cycle 1: 8 ATDs
  - Cycles 2–4: 10 ATDs
- An ATD = $200 realized profit AND at least **23% of your best ATD P&L** (lower threshold of $100 for select sizes)
- **23% consistency rule on funded** — TIGHTER than Topstep (50% in combine, none in XFA standard) or Bulenox (40%)
- Profit split: **100% profit split advertised** (ETF marketing claim — likely has tiers)

---

## 6. Automation / bot policy — **RESTRICTED TO APPROVED PARTNERS ONLY**

**This is the most important rule for our deployment.**

From ETF Terms of Service (via QuantVPS cross-reference):
> Using "artificial intelligence, bots, or automated trading systems, trade copiers, or automated trading strategies" that are **not expressly authorized by ETF** is a prohibited activity.

> "Elite Trader Funding prohibits automated trading systems unless expressly authorized in writing, with a **minimum 10-second holding period** required for all trades."

**Approved partners list** exists on ETF's trade-copier-disclaimer page. Known approved:
- **TradersPost** — connects to ETF via Tradovate for automated strategy execution
- **Tradovate** native execution — implicit
- **NinjaTrader** — ETF offers free license key; likely approved
- Specific trade copier partners (not fully enumerated in our research)

**Our bot status:** Our current Rithmic-direct or TopstepX-API direct deployment is NOT an approved ETF partner. To deploy on ETF, we must either:
1. **Route through TradersPost** — adds a layer between our bot and the broker
2. **Route through Tradovate** native execution — requires Tradovate-compatible bot code
3. **Get express written authorization** from ETF for our direct Rithmic integration — non-trivial, requires support contact

**10-second minimum hold time** — our ORB bot trades have median hold time ~15-40 minutes, so this is not a constraint.

---

## 7. Other key rules

- **NO overnight trades.** All positions must be closed 1 minute before market close.
- **23% consistency rule on funded** — best ATD P&L × 0.23 = minimum required per ATD
- **Position sizing:** 1 mini = 10 micros in most plans; DTF is 1:1 (unusual)
- **Position cap exceeded** = evaluation failure or account closure

---

## 8. Live Funded transition

ETF DTF plan: $25K max payout before forced Live transition.
ETF 1-Step plan: Sim-Funded accumulates, eventually offered Live (details not fully scraped).
**Unclear whether ETF Live transition destroys Sim-Funded accounts** (like Topstep LFA does) — needs direct verification.

---

## 9. Implications for our deployment

1. **ETF is the most-restricted of the three firms for bots.** Unlike Bulenox ("not forbidden") or Topstep ("automated allowed"), ETF requires **express authorization or approved partner route**.
2. **TradersPost integration is the likely path.** It's an established ETF-approved partner.
3. **5 Sim-Funded + 5 DTF = 10 potential concurrent ETF accounts** if the caps are separate buckets. Needs verification.
4. **23% consistency rule is the tightest of any firm.** Our bot's Apr 7 2025 -$2,320 day would clobber this — but on upside, a big day creates a LARGE best-day, meaning we must maintain consistent profit to not have subsequent days fail the 23% bar.
5. **DTF's $25K total-payout cap per account** means max ~$125K realized across 5 DTF accounts before each forces a Live transition. Doesn't scale indefinitely.
6. **1-Step $50K @ $165/mo** is more expensive than Topstep 50K @ $49/mo or Bulenox 50K @ $195/mo (similar to Bulenox). Cost efficiency matters.

**ETF is viable but requires TradersPost integration as a prerequisite.** Until that integration is built, ETF is a gated firm, not a drop-in scaling lane.

---

## 10. Revised multi-firm ceiling (correcting audit)

| Firm | Concurrent cap | Bot path | Status |
|---|---|---|---|
| Topstep | 5 XFA pre-LFA | TopstepX API direct | Immediately deployable |
| Bulenox | 3 Master (up to 11 via unlock) | Rithmic API direct | Immediately deployable — most bot-friendly |
| Elite (ETF) | 5 Sim-Funded + up to 5 DTF (separate buckets?) | Requires TradersPost/Tradovate approved route | **Gated** — needs TradersPost integration first |

**Realistic concurrent (direct-bot deployable today): 5 TS + 3 Bulenox = 8 accounts.**
**After TradersPost integration: +5–10 ETF = 13–18 accounts.**
**Eventual (Bulenox unlocks full cap): up to 26 accounts across firms.**

**Apex (previously in audit) is DEAD** per `memory/prop_firm_automation_verified_apr5.md` — bots NO, copy NO. Do not include in scaling plans.

---

## 11. Open questions

1. Exact interaction between 5-Sim-Funded and 5-DTF caps — separate buckets or combined?
2. ETF Live Funded transition rule — does it close Sim-Funded accounts like Topstep LFA?
3. Full list of ETF-approved automation partners (canonical `trade-copier-disclaimer` page not yet fetched)
4. Whether our specific bot (Python direct-to-Rithmic) can get express ETF authorization without going through TradersPost
5. DTF $25K payout cap: does it reset after Live transition, or is it a lifetime cap per DTF account?

---

## 12. Sources

**Canonical ETF help center (scraped 2026-04-15):**
- https://help.elitetraderfunding.com/help/max-active-elite-sim-funded-accounts — concurrent cap change Sep 17 2025
- https://help.elitetraderfunding.com/help/how-the-1-step-plan-works — 1-Step rules, ATDs
- https://help.elitetraderfunding.com/help/how-the-dtf-plan-works — DTF rules
- https://help.elitetraderfunding.com/help/evaluation-and-elite-account-general-faq — (homepage blocked 403)
- https://help.elitetraderfunding.com/help/updates-to-elite-trader-funding-plans
- https://help.elitetraderfunding.com/help/how-do-the-maximum-positions-work
- https://help.elitetraderfunding.com/help/terms-of-service
- https://help.elitetraderfunding.com/help/policy-one-account-per-user-for-etf-registration

**Third-party (cross-referenced):**
- [QuantVPS ETF Payout Rules](https://www.quantvps.com/blog/elite-trader-funding-payout-rules-explained-how-trader-payouts-work)
- [FuturesFury ETF Review 2026](https://futuresfury.com/firms/elite-trader-funding) — 80% OFF discount + pricing
- [PropTradingVibes ETF 6 Models](https://www.proptradingvibes.com/prop-firms/elite-trader-funding)
- [PropTradingVibes ETF Account Types](https://www.proptradingvibes.com/blog/elite-trader-funding-account-types)
- [TradingFinder ETF April 2026 Review](https://tradingfinder.com/props/elite-trader-funding/)
- [TradersPost — Automate ETF](https://blog.traderspost.io/article/automate-elite-trader-funding-with-traderspost) — confirms TradersPost as approved automation route
- [TradersPost ETF Pricing Guide](https://blog.traderspost.io/article/elite-trader-funding-pricing-evaluation-guide)
- [TradingToolsHub ETF Review](https://tradingtoolshub.com/review/elite-trader-funding/)
- [PropFirmApp ETF Review](https://propfirmapp.com/prop-firms/elite-trader-funding)
