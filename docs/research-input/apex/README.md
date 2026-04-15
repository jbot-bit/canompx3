# Apex Trader Funding Canonical Research Corpus

**Created:** 2026-04-15
**Trigger:** Multi-firm scaling audit (`docs/audit/2026-04-15-topstep-scaling-reality-audit.md`) identified Apex as highest-ceiling candidate (20 accounts). This corpus captures the rules.
**Next re-fetch:** 2026-07-15 (quarterly)
**Source mix:** Apex help center (ZenDesk — blocks WebFetch) + 3rd-party reviewers (QuantVPS, PropTradingVibes, DamnPropFirms, TradeTanto) cross-referenced
**Quality note:** Canonical pages at support.apextraderfunding.com return HTTP 403 to our fetcher. All rules below are from secondary sources; verify on Apex's own help center before acting live.

---

## 1. Account structure

Two-phase like Topstep and Bulenox:
- **Evaluation** (test phase) — monthly subscription, pass once
- **Performance Account (PA)** (funded) — after passing evaluation
- Plus a separate **Live Funded** tier (real money, promoted from PA — analogous to Topstep LFA)

---

## 2. Per-account-size rules

**Profit target: 6% of starting balance flat across all sizes.**

| Size | Profit Target (6%) | Max Trailing DD | DD post-funded |
|---|---|---|---|
| $25K | $1,500 | $1,000 | starting + $100 (static after funded) |
| **$50K** | **$3,000** | **$2,000** | **starting + $100 static** |
| $100K | $6,000 | $3,000 | starting + $100 static |
| $150K | $9,000 | $4,000 | starting + $100 static |
| $250K | ~$15,000 | $5,000 | starting + $100 static |
| $300K | ~$18,000 | $7,500 | starting + $100 static |

**Drawdown mechanics:**
- In evaluation: **trailing** for realized + unrealized (typical)
- Once funded (PA): **static at starting balance + $100** — drawdown stops trailing and locks

**No minimum trading days in evaluation.** Can pass in 1 day if profit target + rules met within 30 calendar days.

---

## 3. Payout rules — 6-step ladder (2026)

Progressive cap structure by account size:

| Size | Step 1 cap | Step 6 cap | After Step 6 |
|---|---|---|---|
| $25K | $1,000 | $1,000 | No cap |
| **$50K** | **$1,500** | **$3,000** | **No cap** |
| $100K | $2,000 | $4,000 | No cap |
| $150K | $2,500 | $5,000 | No cap |

**Eligibility per payout:**
- Minimum **8 trading days**, with **5 qualifying days** ($100–$300 daily profit range)
- **No single day ≥ 50% of total net profit** (consistency equivalent)
- Account balance must exceed **Safety Net threshold** for first 3 payouts
- Minimum withdrawal request: $500
- Cadence: weekly or twice monthly; 2–4 business day processing

**Profit split:**
- First **$25,000** earned: **100% to trader** (vs Topstep's $10K cutoff — Apex is 2.5x more generous)
- Beyond $25,000: **90% trader / 10% Apex**

---

## 4. Automation / bot policy — **PARTIALLY RESTRICTED**

This is the most important Apex rule for our deployment.

**What's ALLOWED:**
- **Bots/auto-strategies permitted during evaluation** — full autonomous OK
- **DCA (Dollar-Cost Averaging) bots explicitly permitted** on funded PA accounts as of 2025
- **Copy trading your own accounts** allowed (e.g., Tradovate Group Copier)
- Automated algos on **NinjaTrader and TradingView** using Rithmic or Tradovate (per 2026 updates)

**What's PROHIBITED:**
- **"Fully automated trading" prohibited on Performance Account (PA) and Live accounts.** Funded phase requires human oversight.
- Must "actively manage all trades" on funded accounts
- "Traders are expected to actively monitor their systems to ensure proper functionality and performance"
- Trading during major economic events is prohibited unless bot is programmed to avoid
- HFT / mass orders

**Interpretation:** The "fully automated" ban is ambiguous in practice. The explicit allowance of NinjaTrader/TradingView/Tradovate algos via Rithmic contradicts a blanket ban. Likely Apex's position: bots may execute but a trader must be actively monitoring (not "set and forget"). Copy trading your own accounts across your PAs is explicitly permitted.

**As of March 1, 2026:** "Hard-Stop Enforcement" via Rithmic/Tradovate introduced. Forces discipline on bot stops.

**Bot deployment risk for our ORB bot:**
- Evaluation phase: fully clear to use bot
- Funded PA: technically requires human monitoring. Running 20 bot copies with no human on duty during sessions would likely be flagged.
- Copy-trading signals from a Lead account to Follower PAs: allowed
- This is a meaningfully tighter rule than Bulenox ("not forbidden") or Topstep ("automated permitted with disclaimer")

---

## 5. Multi-account rules

- **Up to 20 Performance Accounts simultaneously** — the largest cap of any major firm
- Each account needs its own evaluation pass
- Copy trading across your own 20 accounts explicitly permitted via Tradovate/NinjaTrader copiers
- Different account sizes can be mixed

**Scale advantage:** "$300 scalp × 20 accounts = $6,000 from one trade idea" — this is the core multi-account value prop.

---

## 6. Prohibited activities

Canonical list at https://apextraderfunding.com/help-center/getting-started/prohibited-activities/ (not accessible via our fetcher — 403).

Cross-referenced prohibited:
- HFT / mass orders
- Latency arbitrage
- Fully automated trading on PA/Live without oversight
- Trading during restricted news events (bot must auto-avoid)
- Coordinated trading with other people
- Account stacking (hit DD, switch to next account, repeat) — implied, same as Topstep

---

## 7. 2026 updates (recent as of April)

- **March 1, 2026:** "Hard-Stop Enforcement" via Rithmic/Tradovate
- Major payout ladder changes making payouts easier and faster
- DCA bot permission carried over from 2025

---

## 8. Implications for our deployment

1. **20-account ceiling is real but gated by the automation ambiguity.** Running bot on 20 PAs without human oversight = likely flag. Running bot with a trader logged in monitoring = OK.
2. **Evaluation phase is bot-clear.** Can stress-test the bot on Apex evaluations cheaply.
3. **Static DD post-funded (starting + $100)** is more forgiving than Topstep's XFA trailing behavior. Once funded, the account doesn't "creep up" on you.
4. **6% profit target flat** is easier than Topstep's ladder ($3K for 50K = same as Topstep but easier structure — no best-day 50% consistency during evaluation per search results).
5. **$25K 100% split tier** is very generous — more than 2x Topstep's $10K.
6. **Tradovate copier + Rithmic API** is the safest deployment path.
7. **Our 30-trades/day bot is NOT HFT** — well below any threshold.
8. **Copy trading across own 20 accounts is explicitly permitted** — this is the legal vehicle for multi-copy deployment.

---

## 9. Open questions / verification before live

1. **What exactly counts as "fully automated" vs "monitored bot"?** Apex's own Prohibited Activities page is blocked from our fetcher. Must verify directly via login or support contact.
2. **Exact monthly evaluation cost per size** not yet pulled. External sources suggest ~$85–$85+ per size; need canonical.
3. **Activation/reset fees** per size — need canonical.
4. **Safety Net threshold values** per size — need canonical.
5. **Whether our bot's copy pattern (6 lanes, identical signals to all 20 accounts) triggers "account stacking" flag.**
6. **Whether Apex "Live Funded" promotion works like Topstep's LFA consolidation** (destroys all PAs) or runs alongside.

---

## 10. Sources

**Canonical Apex help center (ZenDesk — blocked from our fetcher, verify manually):**
- https://support.apextraderfunding.com/hc/en-us/articles/31519769997083-Legacy-Evaluation-Rules
- https://support.apextraderfunding.com/hc/en-us/articles/31519788944411-Performance-Account-PA-and-Compliance
- https://support.apextraderfunding.com/hc/en-us/articles/40507212951451-Legacy-PA-Payout-Parameters
- https://support.apextraderfunding.com/hc/en-us/articles/30306093336603-All-Apex-Trading-Account-Rules
- https://support.apextraderfunding.com/hc/en-us/articles/40463668243099-Prohibited-Activities
- https://support.apextraderfunding.com/hc/en-us/articles/4408610260507-How-Does-the-Trailing-Static-Drawdown-Threshold-Work-Master-Course
- https://apextraderfunding.com/help-center/getting-started/prohibited-activities/
- https://apextraderfunding.com/resources/trading-market-analysis/mastering-day-trading-rules-with-apex-trader-funding-a-comprehensive-guide/

**Third-party (cross-referenced):**
- [QuantVPS Apex PA Rules](https://www.quantvps.com/blog/apex-pa-account-rules) — drawdown, contract limits, DLL
- [QuantVPS Apex Bot Policy](https://www.quantvps.com/blog/apex-trader-funding-automated-trading-bots) — bot rules
- [DamnPropFirms Apex 2026 Rules](https://damnpropfirms.com/prop-firms/apex-trader-funding-profit-target-and-payout-ladder-rules-2026/) — 6-step ladder, profit target
- [Livestream Trading Apex Review 2026](https://livestreamtrading.com/apex-funded-trader-review/)
- [PropFirmApp Apex Review](https://propfirmapp.com/prop-firms/apex-trader-funding)
- [OnlyPropFirms Automated Futures Trading](https://onlypropfirms.com/articles/automated-futures-trading) — supports automation
- [PropTradingVibes Apex Overview 2026 (after 4.0)](https://proptradingvibes.com/blog/apex-trader-funding-rules-overview)
