# Prop Account Allocation Design

**Status:** V3 — corrected after codex institutional review + proper sim pass
**Date:** 2026-03-16
**Purpose:** Scale validated ORB edge to $100K/year via copy-traded prop firm account stacks

---

## Core Thesis

The edge is real. The question is scaling it. Single-account optimization was the wrong lens — the answer is **account stacking via copy trading.**

**$100K/year = 20 Apex EOD 50K accounts × 2 micros × best all-hours pair**

---

## Firms & Rules (Current March 2026)

| Rule | TopStep $50K | Apex EOD $50K | Apex Intraday $50K | Tradeify $50K | MFFU Rapid $50K |
|------|-------------|---------------|--------------------|--------------|-----------------|
| **Trailing DD** | $2,000 (EOD) | $2,000 (EOD) | $2,000 (real-time) | $2,000 (EOD) | $2,000 (intraday) |
| **DD locks?** | No | Yes at +$2,100 | Yes at +$2,100 | Yes at $50,100 | Yes at starting |
| **Daily loss limit** | Optional | None | None | None | None |
| **Consistency (funded)** | None | None | None | None (Select Flex) | 40% (Core) / None (Rapid) |
| **News restriction** | None | None | None | None | T1: FOMC/NFP/CPI |
| **Max accounts** | **5 Express + 1 Live** | **20** | **20** | **5** | Unknown |
| **Contract limit** | 50 micros | 4 mini / 40 micro | 4 mini / 40 micro | 4 mini / 40 micro | Scaling |
| **Close-by time** | 3:10 PM CT | Session close | Session close | 4:59 PM ET | Session close |
| **Profit split** | 90/10 | 100% first $25K, 90/10 | 100% first $25K, 90/10 | 90/10 | 90/10 |
| **Fee after funded** | **$0** | **$0** (lifetime $160) | **$0** (lifetime) | **$0** | **$129/month FOREVER** |
| **Copy trading** | Yes (own accts) | Yes | Yes | Yes (5 accts) | Yes |
| **Auto trading** | Yes (TopstepX API) | Yes (Tradovate) | Yes (Tradovate) | Yes (own bots) | Yes |
| **Metals (MGC)** | Yes | **BANNED** | **BANNED** | Yes | Yes |

### Key Corrections (from codex review)
- Apex legacy products ($2,500 DD) **discontinued March 1, 2026**. Current is $2,000.
- TopStep new accounts (after Jan 12, 2026): 90/10 split from start (not 100% first $10K).
- MFFU: $129/month never stops. No lifetime option. **Dead for this plan.**
- Apex bans metals (MGC) — critical constraint for morning MGC strategies.

Sources:
- https://support.apextraderfunding.com/hc/en-us/articles/47204516592795-EOD-Performance-Accounts-PA
- https://support.apextraderfunding.com/hc/en-us/articles/47206242141979-Intraday-Trailing-Drawdown-Performance-Accounts-PA
- https://support.apextraderfunding.com/hc/en-us/articles/42526785111699-Legacy-Products-Overview
- https://help.topstep.com/en/articles/10799569-xfa-faq
- https://help.topstep.com/en/articles/9208217-topstep-pricing
- https://help.tradeify.co/en/articles/10495897-rules-trailing-max-drawdowns

---

## Firm Ranking for Scale

| Rank | Firm | Why | Best For |
|------|------|-----|----------|
| 1 | **Apex EOD** | 20 accounts, copy trading, $0 after funded, EOD trailing | Main scaling vehicle |
| 2 | **Tradeify** | $0 fees, EOD trailing, clean rules, auto allowed | Secondary diversification (5 acct cap) |
| 3 | **TopStep** | Best for MGC (Apex bans metals), EOD trailing, API | Morning MGC lane |
| 4 | **MFFU** | — | **Dead.** $129/month forever kills economics |

---

## Strategy Selection

### Sim Corrections Applied
The original sim (tmp_prop_sim.py) had 7 bugs identified by codex review:
1. ~~Legacy Apex DD $2,500~~ → corrected to $2,000
2. ~~Naive 0.75x (scale losses only)~~ → corrected via `apply_tight_stop()` from config.py
3. ~~No MFE-aware trailing~~ → Apex trails on peak unrealized, modeled via mfe_r
4. ~~Combined metrics not matched-path~~ → fixed
5. ~~Wrong day pool for TopStep~~ → uses full calendar including zero-trade days
6. ~~2024+ claim but pulls 2020+~~ → made explicit
7. ~~TopStep 100% first $10K~~ → corrected to 90/10 for new accounts

Corrected sim: `scripts/tmp_prop_sim_v2.py`. Codex proper pass: `scripts/tmp_prop_firm_proper_pass.py`.

### Best Bundles by Firm (from corrected sims)

**Apex EOD 50K (scaling vehicle):**
- Best all-hours pair: MNQ CME_PRECLOSE + MNQ COMEX_SETTLE
- ~$214/mo at 1 micro, ~$423/mo at 2 micros, ~99.7% survival at 2 micros
- Both are overnight AUTO sessions

**TopStep (morning MGC lane):**
- Best manual morning pair: MGC CME_REOPEN ORB_G5 + MGC TOKYO_OPEN ORB_G6_CONT_S075
- ~$61/mo at 1 micro, 100% survival
- ORB gate mandatory: skip if MGC ORB > 26 points

**Tradeify (secondary):**
- Same equity pairs as Apex (MES/MNQ), or MGC morning if metals allowed
- ~$33-57/mo at 1 micro depending on pair

**Apex (morning fallback, no metals):**
- Best non-MGC morning pair: MES TOKYO_OPEN ORB_G6_CONT + MNQ CME_REOPEN VOL_RV12_N20
- ~$36/mo at 1 micro, 100% survival

### 0.75x vs 1.0x Stop
**0.75x is definitively better for prop.** Every strategy survives better. The win-kill rate (15-25% of 1.0x wins killed by tighter stop) is offset by 25% smaller losses. Net effect: survival improves +6-16% per account, income drops $20-60/month. On prop where DD = death, survival wins.

---

## Income Plans

### Plan A: $100K/year (Realistic)
- **Apex EOD 50K × 20 accounts × 2 micros**
- Best all-hours pair (CME_PRECLOSE + COMEX_SETTLE)
- Copy-traded: same execution effort as 1 account
- ~$423/mo per account × 20 = **~$8,460/month = ~$101K/year**
- Upfront: 20 × ($35 eval + $160 lifetime) = ~$3,900
- Monthly ongoing: **$0**
- **Requires: auto-execution for overnight sessions**

### Plan B: $200K/year (Achievable)
- Same Apex stack at higher per-account sizing (if validated)
- OR Apex 20 accounts + Tradeify 5 accounts + TopStep 5 accounts = 30 total
- ~$16-17K/month across all
- **Requires: proven live results + auto-execution**

### Plan C: $100K/month (Moonshot)
- Not supported by current validated edge as base case
- Would require: larger account sizes, additional firm capacity, own capital + IBKR auto
- Self-funded IBKR at 10-20c on $100K capital: sims show $112-225K/year
- Stretch target, not current operating plan

---

## Deployment Phases

### Phase 1: Prove It Live (Months 1-3)
- **1-2 Apex EOD 50K accounts**, manual, 1 micro
- Morning sessions you can trade while awake + NYSE at 11:30pm if willing
- Goal: 2+ payouts, confirm backtest-to-live match
- Expected: $200-400/month
- Cost: $195-390 upfront, $0 ongoing

### Phase 2: Scale Accounts (Months 3-6)
- Copy-trade proven strategy to **5-10 Apex accounts**
- Still 1 micro per account
- Add TopStep for MGC morning lane (diversification)
- Expected: $1,000-4,000/month
- Cost: $195 per additional account

### Phase 3: Scale Contracts (Months 6-9)
- Move to **2 micros per account** on proven bundles
- Only after survival confirmed at 1 micro
- Expected: $4,000-8,500/month
- Cost: $0 (same accounts, more contracts)

### Phase 4: Full Stack + Auto (Months 9-12)
- **20 Apex accounts × 2 micros**, auto-execution for overnight sessions
- Add overnight sessions (CME_PRECLOSE, COMEX_SETTLE — the highest E$/mo strategies)
- Expected: **$8,000-10,000/month** (~$100K/year)
- Auto infra: ProjectX/Tradovate already built. Wire to Apex API.

### Phase 5: Self-Funded (Month 12+)
- ONLY after 3+ months of proven prop results
- IBKR self-funded, $50-100K capital, 10-20c, all 9 sessions
- No DD death, no trailing, no rules
- Expected: $9,000-19,000/month ($112-225K/year)
- Build IBKR broker integration (~700 lines, broker ABC pattern ready)

---

## Key Rules

1. **0.75x stop on all prop accounts.** Non-negotiable. Survival > income.
2. **ORB gate on MGC (TopStep).** Skip if ORB > 26 points (risk > $200).
3. **Apex EOD, never Intraday.** EOD trailing = safer, especially overnight.
4. **Always pick Apex lifetime fee.** $160 one-time vs $85-137/month. Break-even: 1.6 months.
5. **Never MFFU.** $129/month forever = dead money.
6. **Scale horizontally (more accounts), not vertically (more contracts).** 20 × 1c >> 1 × 20c.
7. **Copy trade = same effort as 1 account.** The whole point of account stacking.
8. **Prop first, capital later.** Prove the edge live before risking own money.

---

## Variable Risk Management

### ORB Size Distributions (0.75x stop, filtered trades, 2024+)
| Session | Inst | Median$ | P75$ | P95$ | Max$ |
|---------|------|---------|------|------|------|
| CME_REOPEN | MGC | $57 | $113 | $278 | $749 |
| TOKYO_OPEN | MGC | $59 | $86 | $172 | $447 |
| NYSE_OPEN | MNQ | $84 | $115 | $177 | $368 |
| US_DATA_1000 | MNQ | $66 | $97 | $170 | $401 |
| COMEX_SETTLE | MNQ | $33 | $48 | $93 | $534 |
| CME_PRECLOSE | MNQ | $33 | $50 | $92 | $292 |
| SINGAPORE_OPEN | MNQ | $38 | (TBD) | $123 | (TBD) |
| EUROPE_FLOW | MNQ | $25 | (TBD) | $59 | (TBD) |
| LONDON_METALS | MNQ | $59 | (TBD) | $146 | (TBD) |

### Max Risk Per Trade Gate
Rule: ORB_size × point_value × 0.75 must be < 10% of DD budget.
- Apex $50K ($2K DD): max $200 → MGC 26.4 pts, MNQ 133 pts
- Only MGC needs the gate. MNQ at $2/point naturally stays under.

---

## Auto-Execution Status

### What Exists
- Full live trading pipeline: DataFeed → BarAgg → ExecutionEngine → OrderRouter → TradeJournal
- ProjectX (TopstepX) integration: complete (auth, data, orders, positions)
- Tradovate integration: complete (auth, data, orders, positions)
- Safeguards: CircuitBreaker, PositionTracker state machine, CUSUM drift detection, max concurrent positions

### What's Missing
- **IBKR integration: NOT BUILT** (zero code, but broker ABC pattern ready — ~700 lines to add)
- Webhook risk hardening (Task #12)
- pysignalr graceful stop (Task #10)
- Tradovate feed tests (Task #11)

### Mar 14 Audit Status
- Trade journal: ✅ DONE (was CRITICAL)
- Exit order retry: ✅ DONE (was HIGH)
- Remaining: 3 HIGH tasks (webhook, pysignalr, tradovate tests)

Design doc: `docs/plans/2026-03-14-live-trading-hardening-design.md`

---

## Open Questions

- [x] ~~Monte Carlo with correct 0.75x mechanics~~ — DONE (tmp_prop_sim_v2.py)
- [x] ~~Apex lock-in strategy~~ — Go full size from day 1
- [x] ~~Firm fee comparison~~ — Apex lifetime wins, MFFU dead
- [x] ~~0.75x vs 1.0x for prop~~ — 0.75x wins on survival
- [x] ~~Strategy selection optimization~~ — Best pair varies by firm (codex proper pass)
- [ ] Codex detailed 100K/200K/moonshot blueprints (exact account counts, sessions)
- [ ] Copy trading setup mechanics (Tradovate Group Trading for Apex)
- [ ] Apex EOD vs Intraday survival comparison with corrected sims
- [ ] Live trading infra hardening (3 remaining HIGH tasks)
- [ ] IBKR broker integration design (Phase 5)
- [ ] Tax implications of 20 prop firm payouts
- [ ] Payout cadence optimization (Apex payout rules)
