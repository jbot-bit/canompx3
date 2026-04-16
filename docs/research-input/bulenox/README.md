# Bulenox Canonical Research Corpus

**Created:** 2026-04-15
**Trigger:** Multi-firm scaling audit (`docs/audit/2026-04-15-topstep-scaling-reality-audit.md`) identified Bulenox as a candidate for parallel deployment. This corpus captures the rules.
**Next re-fetch:** 2026-07-15 (quarterly)
**Source mix:** Bulenox help center (canonical where available) + 3rd-party reviewers (TradingToolsHub, PropTradingVibes, QuantVPS, TradersX) cross-referenced

---

## 1. Account structure

Two-phase like Topstep, but with a key difference: the phases are called **Qualification** (evaluation) → **Master** (funded), and the **Qualification/Master distinction is explicit** rather than XFA/LFA.

### 1a. Qualification Account (evaluation)

From https://bulenox.com/help/qualification-account/:
- Unlimited concurrent Qualification Accounts allowed
- No minimum trading days required to pass
- Profit target to reach and progress to Master phase
- Monthly subscription fee (recurring until you pass)
- Two trading options:
  - **Option 1:** Trailing drawdown, **no** daily loss limit, full contract access from day one
  - **Option 2:** EOD (end-of-day) drawdown, daily loss limit enforced, scaling plan for contracts

### 1b. Master Account (funded)

From https://bulenox.com/help/master-account/:
- **Max 3 Master Accounts simultaneously** — per canonical Bulenox help. External sources (e.g., propfirmapp) cite "up to 11" — this likely refers to sequential/cumulative lifetime accounts or unlocks; the 3-simultaneous cap is the immediate concurrent limit.
- To activate additional Master Accounts beyond the 3 cap: *"reaching the initial starting balance on previously activated accounts"* (so account must grow past its starting balance before you can add another)
- No monthly fee (just one-time activation fee)
- One-time activation fee includes data feed + maintenance
- **No option to reset** Master Accounts (Qualification accounts are resettable)

**Conflict note:** external sources vary on 3 vs 11. Treat 3 as hard concurrent cap; 11 as probable eventual ceiling after earning your way up.

---

## 2. Per-account-size rules

| Size | Profit Target | Max Trailing DD | Daily Loss Limit (Option 2 only) |
|---|---|---|---|
| $10K | $1,000 | $1,000 | $400 |
| $25K | $1,500 | $1,500 | $500 |
| **$50K** | **$3,000** | **$2,500** | **$1,100** |
| $100K | $6,000 | $3,000 | $2,200 |
| $150K | $9,000 | $4,500 | $3,300 |
| $250K | $12,500 | $6,000 | $4,500 |

Drawdown mechanics:
- **Option 1 (Trailing):** trails balance for realized + unrealized, locks when balance hits starting + $100 (after that, drawdown stops moving)
- **Option 2 (EOD):** follows end-of-day balance only

---

## 3. Pricing (2026)

**Qualification monthly subscription** (recurring until pass):
- $50K: **$195/mo**
- $100K: pricing varies with promotional periods; range $115–$535/mo across sizes
- Discount: $50 off $50K, $60 off $100K plans (promo-dependent)

**Master Account one-time activation fee** (paid once upon passing):
- $50K: **$148**
- $100K: **$248**
- Other sizes: scaled proportionally

**No monthly fee in Master phase.** Just the one-time activation.

---

## 4. Payout rules

Canonical (from https://bulenox.com/help/master-account/):

- **Minimum days before first payout: 10 individual trading days**
- **Payout frequency: weekly, Wednesdays**
- **Payout caps (first 3 payouts only):**
  - $25K account: $1,000 per payout
  - $50K: ~$1,500–$2,000 per payout (need to verify exact value)
  - $100K: ~$2,000–$2,500
  - $250K: $2,500 per payout
- **After 3rd payout: no maximum withdrawal cap**

**Profit split:**
- First $10,000 earned: **100% to trader** (no commission)
- Beyond $10,000: **90% trader / 10% Bulenox**

---

## 5. Consistency rule (Master Account)

**40% single-day rule.** From canonical Bulenox: *"The balance in the Master Account must not consist of more than 40% of the total profit balance from a single trading day."*

Equivalent to: best day ≤ 40% of cumulative Master Account profit. Similar to Topstep XFA Consistency variant.

**Not applicable to Qualification Account.**

---

## 6. Automation / bot policy — **FULLY PERMITTED**

From canonical Bulenox help: *"It is not forbidden to use them"* (referring to algorithms, EAs, bots). Company provides no support for third-party software issues.

Cross-referenced (QuantVPS, PropTradingVibes): "EAs, algorithms, trade copiers, and bots allowed on all accounts without restriction." Supported platforms: NinjaTrader EAs, Rithmic API bots, TradingView alert-based.

**Prohibited:**
- **High-frequency trading** (hundreds of orders per minute)
- **Latency arbitrage**
- Specific copy-trading patterns flagged as "flipping" (high-frequency direction reversal)

**Our bot status:** 30-trades/day ORB breakout bot is far below HFT thresholds. **Bulenox is the most bot-friendly firm of the three (TS/Bulenox/Apex).**

---

## 7. Multi-account rules

- Unlimited Qualification Accounts concurrent (but each has monthly fee)
- 3 Master Accounts simultaneously (canonical), eventual max cited as 11 (external) via unlock path
- Copy-trading own accounts: allowed
- Cross-account hedging: monitored, prohibited if flagged as pattern
- Same-direction copies across own accounts: allowed

---

## 8. Implications for our deployment

1. **Most bot-friendly firm.** No ambiguity around automation.
2. **Entry cost higher than Topstep:** $195/mo recurring vs Topstep's $49/mo. Need to pass fast or cost compounds.
3. **Profit split MORE generous than Topstep** on first $10K (100% vs 90%). Good for early payouts.
4. **Weekly Wednesday payout cadence** is predictable.
5. **3-Master concurrent cap** means real ceiling on Bulenox is 3 parallel bot copies initially, scaling to 11 over time. Not 11-from-day-one as external sources implied.
6. **Drawdown mechanics similar to Topstep** ($2.5K trailing on 50K), but Option 1 has NO daily loss limit — slightly more bot-friendly than Topstep's XFA trailing.
7. **40% consistency rule** applies on Master Account only. Our bot's 1ct distribution shows max day ~$1,949 over 7yr. At a Master with, say, $5K cumulative profit, that's 39% — right at the line. Worth monitoring. At 2ct the tail day hits 4x = blow-up.

---

## 9. Open questions

- Exact $50K and $100K payout cap values (first 3 payouts). Sources give ranges.
- Whether the 3-simultaneous-Master rule allows you to run 3 × $50K Master + 3 × $100K Master concurrently (i.e., 3 per size), or 3 total across all sizes.
- Bulenox response to bot running on 3 Master Accounts with identical signals (copy pattern flagging risk).
- Whether the 11-account cap cited by external sources is a Bulenox official number or speculation.

---

## 10. Sources

**Canonical Bulenox help center (quality: high where available):**
- https://bulenox.com/help/qualification-account/
- https://bulenox.com/help/master-account/
- https://bulenox.com/help/frequently-asked-questions/
- https://bulenox.com/help/
- https://bulenox.com/ (homepage)
- https://bulenox.com/wa-data/public/site/data/bulenox.com/Terms_of_Use.pdf (to fetch later)

**Third-party (cross-referenced, good consistency across sources):**
- [TradingToolsHub Bulenox Pricing Guide 2026](https://tradingtoolshub.com/blog/bulenox-pricing-guide-2026/) — pricing
- [QuantVPS Bulenox Activation Fee](https://www.quantvps.com/blog/bulenox-activation-fee) — fees
- [QuantVPS Bulenox Consistency Rule](https://www.quantvps.com/blog/bulenox-consistency-rule) — 40% rule
- [QuantVPS Topstep vs Apex vs Bulenox](https://www.quantvps.com/blog/topstep-vs-apex-vs-bulenox) — comparison
- [PropTradingVibes Qualification Account Rules 2026](https://www.proptradingvibes.com/blog/bulenox-qualification-account-rules)
- [PropTradingVibes Rules Overview 2026](https://www.proptradingvibes.com/blog/bulenox-rules-overview)
- [PropTradingVibes Permitted Strategies](https://www.proptradingvibes.com/blog/bulenox-permitted-strategies)
- [TradersX Bulenox Detailed Overview](https://tradersx.io/prop-firms/bulenox)
- [TradingFinder Bulenox Rules 2026](https://tradingfinder.com/props/bulenox/rules/)
- [Trinity Trading Bulenox Review 2026](https://blog.trinitytrading.io/bulenox-review-2026/)
- [PropJournal Bulenox Drawdown Guide](https://propjournal.net/prop-firms/bulenox/rules)
