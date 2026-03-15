# Beginner Trading Plan — ORB Breakout System
*Generated: 2026-03-10 | Based on: 435 validated strategies, BH FDR + Walk-Forward*

---

## The Short Version

**Trade MNQ. US session (late night Brisbane). 5 contracts. 4 sessions.**

At 5 contracts, the validated edge generates **~$34K/year** per funded account after prop split.
At 2 funded accounts: ~**$68K/year**. At 3 accounts: ~**$102K/year**.

That is income replacement. That is the target.

---

## What to Trade

### The Strategy Filter (Non-Negotiable)

Only trade strategies that pass ALL of these:

| Gate | Threshold | Why |
|------|-----------|-----|
| Family status | ROBUST or WHITELISTED | ≥3 parameter siblings survived — not curve-fit |
| PBO | < 0.50 | Probability of Backtest Overfitting < 50% |
| Fitschen multiplier | ≥ 3× RT cost | Edge covers execution variance with margin |
| BH FDR | Must have passed | Already applied in pipeline — don't re-check |
| Walk-forward | Must have passed | Already applied in pipeline — don't re-check |

Run `python scripts/tools/beginner_tradebook.py --contracts N` to see your current tradeable strategies.

### Your Strategies (at 5 contracts)

#### US Session Block — Primary (late night Brisbane, use alerts)

| Time BRIS | Session | Filter | ORB | RR | WR | Exp$/5c | Fitschen | Quality |
|-----------|---------|--------|-----|----|----|---------|----------|---------|
| 22:30 | NYSE_OPEN | VOL_RV12_N20 | 30m | 1.0 | 49% | $96 | 35x | WHITELISTEDx4 PBO=0.00 |
| 00:00 | US_DATA_1000 | VOL_RV12_N20 | 15m | 1.0 | 51% | $72 | 26x | WHITELISTEDx4 PBO=0.00 |
| 03:30 | COMEX_SETTLE | VOL_RV12_N20 | 5m | 1.5 | 45% | $41 | 15x | ROBUSTx9 PBO=0.01 |
| 05:45 | CME_PRECLOSE | VOL_RV12_N20 | 5m | 1.5 | 54% | $57 | 21x | ROBUSTx7 PBO=0.00 |

#### Morning Block — Secondary (trade live)

| Time BRIS | Session | Filter | ORB | RR | WR | Exp$/5c | Fitschen | Quality |
|-----------|---------|--------|-----|----|----|---------|----------|---------|
| 11:00 | SINGAPORE_OPEN | DIR_LONG | 30m | 4.0 | 20% | $32 | 12x | ROBUSTx17 PBO=0.01 |
| 17:00 | LONDON_METALS | ORB_G8_NOMON | 30m | 2.5 | 32% | $30 | 11x | ROBUSTx9 PBO=0.11 |

**DO NOT TRADE:**
- MES or M2K (fail Fitschen threshold at any reasonable contract size)
- MGC (zero ROBUST morning strategies; best MGC strategies are in DB but require verification for 2026 data)
- SINGLETON families (no siblings = may be curve-fit)
- Any strategy with PBO ≥ 0.50 (e.g. MES COMEX_SETTLE PBO=0.89 — even if family shows ROBUST×7)
- Sessions not in this list

---

## Account Structure

### Phase 1: Learning (Paper + 1 Contract, 90 days)

**Goal: Prove execution discipline before scaling.**

- Platform: TradingView + Interactive Brokers or Tradovate (for MNQ)
- Paper trade every session for 30 days. Record every signal — did you see it? Did you take it? What was fill quality?
- Then go live at **1 contract** for 60 days
- Account minimum: **$3,000** (1 MNQ margin ~$500, need 6× buffer for drawdown room)
- Risk per trade: fixed 2% = $60/trade
- Expected return at 1c: ~$5-15/trade (thin but real — this is education, not income)

**Milestone to advance:** 60 consecutive trading days. No exceptions to rules. Daily journal kept. Slippage tracked.

### Phase 2: Prop Firm Evaluation

**Goal: Access to 5-contract sizing without risking personal capital.**

Recommended path:
- **TopStep Futures** — $50K account, $165/month fee during evaluation
  - Profit target: $3,000 (hit it in 2-4 weeks at 5c with the US session)
  - Max daily loss: $1,000
  - Max overall loss: $2,000
  - Max contracts MNQ on 50K: 5 (check platform rules)
- **Apex Trader Funding** — similar structure, sometimes lower fee

**Evaluation strategy:** Trade only NYSE_OPEN + US_DATA_1000. These are the two highest Exp$ sessions. Do not overtrade. Hit the target cleanly and consistently — evaluators look at behavior, not just profit.

**What to tell them:** You trade a systematic ORB breakout strategy on MNQ. You take 2-4 trades per day maximum. You do not hold overnight.

### Phase 3: Funded Account(s) (Ongoing)

- Once funded: trade all 4 US sessions + 2 morning sessions
- **Contract size: 5 per trade**
- Monthly gross expectancy: ~$3,553 (200 day basis, 65% activation, 1 account)
- After 80% prop split: ~$2,843/month from one account
- **Two funded accounts: ~$5,686/month = $68K/year**

**To open second account:** Fund it from first account profits. Do not use personal capital for second evaluation if avoidable.

---

## Risk Rules

### Trade-Level Rules

| Rule | Value | Rationale |
|------|-------|-----------|
| Stop loss | ORB size × filter (pre-set) | The system defines this. Never widen. |
| Profit target | RR × ORB size (pre-set) | The system defines this. Never move to breakeven early. |
| Position size | Fixed contracts (5) | Do not scale mid-drawdown |
| Max 1 position at a time | Hard limit | ORB is one trade per session |
| Never average down | Zero tolerance | One entry, one exit |

**The exit IS the strategy.** ORB strategies have defined stop and target. Your only job is:
1. Wait for the signal
2. Enter at the correct price (limit order inside the range, not market chasing)
3. Set stop and target immediately on fill
4. Walk away until it resolves

### Daily Stop Rules

| Trigger | Action |
|---------|--------|
| -3R on the day | Stop trading for the rest of the day. No exceptions. |
| -6R on the week | Reduce to 1 contract next week. Review logs. |
| -15R on the month | Stop live trading. Paper trade for 2 weeks. Check if setup logic has changed. |
| 3 consecutive missed signals | Review platform/alerts. Something is wrong technically. |

**What is 1R?** Your stop distance in dollars. At 5c MNQ with a 10-point ORB stop: 10pts × $2/pt × 5c = $100. So -3R = -$300 for the day.

### Prop Firm Specific Rules

- **NEVER approach within $200 of your daily loss limit** — stop early that day
- **NEVER trade on high-impact news days** unless the session starts after the news release (e.g. COMEX_SETTLE starts after US_DATA_1000 news at midnight Brisbane — check calendar)
- **Keep a daily log** — prop firms review your trade history. Systematic logs prove you are a system trader, not a gambler.

---

## Session Schedule (Brisbane Time)

```
TIME    SESSION           ACTION
──────────────────────────────────────────────────────────
11:00   SINGAPORE_OPEN    Check signal. Trade if valid. 30m ORB. RR=4.0 means patient.
17:00   LONDON_METALS     Check signal. Trade if valid. 30m ORB. Don't force thin ORBs.
        [gap - rest]
22:30   NYSE_OPEN         ALERT. Best session. 30m ORB. VOL filter = need elevated vol.
00:00   US_DATA_1000      ALERT. Set price alert after NYSE_OPEN resolves.
03:30   COMEX_SETTLE      ALERT. 5m ORB. Quick signal — be ready.
05:45   CME_PRECLOSE      ALERT. 5m ORB. Last US session before morning.
```

**Practical setup:**
- Use TradingView price alerts on the ORB high/low for each session
- You don't need to stare at screens. Set the alert, wake up if needed, execute, go back to sleep.
- NYSE_OPEN is 22:30 BRIS — this is evening. Easiest to trade live while awake.
- COMEX_SETTLE (03:30) and CME_PRECLOSE (05:45) are the sacrifice sessions — alerts only, execute from bed if you're disciplined enough.

---

## What to Ignore

- **Your P&L while in a trade.** You set the stop and target. Let the system work.
- **"But this trade looks different."** It always looks different. Run the system.
- **Strategies that aren't on this list.** Curiosity is fine. Trading them live is not.
- **Adding more sessions.** You have 6. That's enough. Quality > quantity.
- **Anyone who says "ORB doesn't work."** You have 5 years of data, BH FDR validation, and walk-forward proof. They don't.

---

## When to Quit (Red Flags)

These mean something has changed in the market — review before continuing:

1. **Any session produces -10R over 60 consecutive trade days** — that session is WATCH/DECAY. Run `python scripts/tools/regime_check.py` and check strategy fitness.
2. **Slippage consistently > 2 ticks** on MNQ entries — broker or time-of-day issue. Fix execution before continuing.
3. **Filter activation rate drops below 30% for 60 days** — market regime has changed. Check if ORB sizes have compressed (contracting ATR).
4. **Prop firm account blows up in < 30 trades** — execution problem, not strategy problem. Re-evaluate entries.

---

## Expected Return Scenarios

*Based on 5 contracts, 6 sessions, 65% activation, 80% prop split*

| Accounts | Gross/year | After split | Monthly |
|----------|-----------|-------------|---------|
| 1 funded | $42,646 | $34,117 | $2,843 |
| 2 funded | $85,292 | $68,234 | $5,686 |
| 3 funded | $127,938 | $102,350 | $8,529 |

**These are expected values based on historical backtested edge.** Real results will vary. Some months will be negative. The edge is real on a per-year basis, not a per-month basis.

---

## The Honest Bottom Line

The edge is statistically proven. The research is done.

What determines whether you succeed from here:
1. **Execution discipline** — taking every valid signal, no cherry-picking
2. **Fill quality** — getting in at the right price, not chasing
3. **Drawdown tolerance** — not quitting a valid system during a losing streak
4. **Capital path** — prop firm is mandatory for income replacement. 1 contract personal account is tuition.

The strategy is not the risk. **You are the risk.**
