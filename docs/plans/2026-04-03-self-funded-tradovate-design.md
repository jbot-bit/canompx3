# Self-Funded Tradovate Deployment — Design Spec

**Date:** 2026-04-03
**Status:** APPROVED FOR IMPLEMENTATION
**Author:** Josh + Claude (brainstorming session)

---

## 1. Overview

Deploy the ORB breakout portfolio on a personal Tradovate brokerage account using own capital. Runs parallel to prop firm accounts (TopStep, Tradeify) — prop proves edge with someone else's money, self-funded captures 100% profit split with zero restrictions.

**Why self-funded:**
- 100% profit split (vs 90% prop)
- No payout caps ($5-6K limits on prop)
- No winning day requirements
- No consistency rules
- No trailing drawdown ratchet
- Automation always allowed
- Multi-contract scaling (prop caps at micro)
- Tradovate intraday margin: $50/contract (vs IBKR $2,000-6,000)

**Why Tradovate:**
- API already built (6 modules: auth, contracts, http, order_router, positions, __init__)
- $50 intraday margin per micro contract
- Free plan: $0/mo, $1.22/RT all-in for MNQ
- Profile infrastructure already exists (`self_funded_tradovate` in prop_profiles.py)

---

## 2. The Portfolio — 11 Honest Lanes

Stress-tested against full history + 2025-2026. Two historically negative lanes (MES_CME_PRECLOSE, MES_US_DATA_830) dropped. Remaining 11 are positive in both 2025 and 2026 (9/11) or positive full-history with acceptable 2026 start (2/11 on watch).

### Lane Book

| # | Strategy ID | Instrument | Session | RR | ORB Cap | 2025 P&L | 2026 P&L | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | MGC_CME_REOPEN_E2_RR2.5_CB1_ORB_G6 | MGC | CME_REOPEN | 2.5 | 30pt | +$695 | +$4,630 | CORE |
| 2 | MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_COST_LT12 | MNQ | SINGAPORE_OPEN | 2.0 | 90pt | +$1,433 | +$1,271 | CORE |
| 3 | MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100 | MNQ | COMEX_SETTLE | 1.5 | 150pt | +$3,464 | +$706 | CORE |
| 4 | MNQ_EUROPE_FLOW_E2_RR3.0_CB1_COST_LT10 | MNQ | EUROPE_FLOW | 3.0 | 120pt | +$1,981 | +$909 | CORE |
| 5 | MNQ_TOKYO_OPEN_E2_RR2.0_CB1_COST_LT10 | MNQ | TOKYO_OPEN | 2.0 | 80pt | +$903 | +$1,082 | CORE |
| 6 | MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR70_VOL | MNQ | NYSE_OPEN | 1.0 | 70pt | +$5,389 | +$2,325 | **UNDEPLOYABLE** |
| 7 | MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR70_S075 | MNQ | US_DATA_1000 | 1.0 | 65pt | +$1,928 | +$398 | **UNDEPLOYABLE** |
| 8 | MGC_US_DATA_1000_E2_RR1.0_CB1_ORB_G6 | MGC | US_DATA_1000 | 1.0 | 15pt | +$2,273 | -$223 | WATCH |
| 9 | MES_US_DATA_1000_E2_RR1.0_CB1_VOL_RV15_N20_S075 | MES | US_DATA_1000 | 1.0 | 20pt | +$533 | +$466 | CORE |
| 10 | MNQ_CME_PRECLOSE_E2_RR1.0_CB1_VOL_RV20_N20 | MNQ | CME_PRECLOSE | 1.0 | 50pt | +$1,451 | -$43 | WATCH |
| 11 | MNQ_CME_REOPEN_E2_RR1.0_CB1_VOL_RV30_N20 | MNQ | CME_REOPEN | 1.0 | 50pt | +$577 | +$513 | CORE |

### Dropped Lanes (negative full-history)
- MES_CME_PRECLOSE_E2_RR1.0_CB1_VOL_RV20_N20_S075: full-history -$689, negative 3/5 years
- MES_US_DATA_830_E2_RR1.0_CB1_VOL_RV20_N20_S075: full-history -$9,626, only worked 2025
- MNQ_LONDON_METALS: $415/yr, avg $1.30/trade — not worth the slot
- MES_LONDON_METALS: -$1,584/yr — confirmed dead
- MNQ_US_DATA_830: -$180/yr — negative
- MES_NYSE_OPEN: -$134/yr — negative

### Session Schedule (Brisbane Time)

| Brisbane Time | Session | Lanes |
|---|---|---|
| 03:30 | COMEX_SETTLE | MNQ #3 |
| 05:45 | CME_PRECLOSE | MNQ #10 |
| 08:00 | CME_REOPEN | MGC #1, MNQ #11 |
| 11:00 | SINGAPORE_OPEN | MNQ #2 |
| 11:00 | TOKYO_OPEN | MNQ #5 |
| 18:00 | EUROPE_FLOW | MNQ #4 |
| 23:30 | NYSE_OPEN | MNQ #6 |
| 00:00 | US_DATA_1000 | MNQ #7, MGC #8, MES #9 |

---

## 3. Performance (Stress-Tested from gold.db — WITH REAL FILTERS)

**IMPORTANT:** These numbers use the actual strategy filters (COST_LT, OVNRNG, ATR70_VOL, VOL_RV, ORB_G6) applied via daily_features triple-join.

**Stage 4 finding (Apr 4 2026):** Filter columns are CORRECT (atr_20_pct, rel_vol, cross_atr_MES_pct all exist and work). However, 2 lanes are **UNDEPLOYABLE** — not pipeline-validated:
- **Lane 6 (NYSE_OPEN ATR70_VOL):** Not in experimental_strategies or validated_setups. Never ran through discovery. Validated alternative: `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_50` (N=1441, ExpR=0.112, Sharpe=1.15).
- **Lane 7 (US_DATA_1000 X_MES_ATR70 S075):** In experimental but fdr_significant=False (ExpR=0.019, Sharpe=0.22). Too weak for BH FDR. Validated alternative: `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT10` (N=1941, ExpR=0.078, Sharpe=0.73).
- **Action required:** Replace these lanes with validated alternatives before deployment. Replacing will change the portfolio P&L estimate.

### 2025-2026 Backtest (329 trading days, 1ct per lane)

| Metric | Value |
|---|---|
| Total P&L | $23,817/yr |
| Monthly avg | $1,985 |
| Daily avg | $72.39 |
| Trades/day | 4.9 |
| Win days | 62% |
| Losing months | 1/16 (Jun 2025: -$857) |
| Worst single day | -$499 |
| Max drawdown | -$1,237 (4.1% of $30K) |
| MNQ all-lanes-lose days | 0.6% |

### Filter impact

Filters reject ~50% of trades. Rejected trades are net positive (+$1,574/yr in 2025-2026) BUT filters improve per-trade edge on every lane. Filters were validated by BH FDR over full 16-year history — they protect against cold regimes where unfiltered trades turn negative. In a hot regime (2025-2026), unfiltered trades look profitable. That's selection bias.

### Scaling

| Contracts | Annual (gross) | Commission | Net Annual | Monthly | Worst Day | Max DD | DD % |
|---|---|---|---|---|---|---|---|
| 1ct | $23,817 | $1,639 | **$22,178** | $1,848 | -$499 | -$1,237 | 4.1% |
| 2ct | $47,634 | $3,134 | **$44,500** | $3,708 | -$998 | -$2,474 | 8.2% |
| 3ct | $71,452 | $4,629 | **$66,822** | $5,569 | -$1,497 | -$3,710 | 12.4% |

### Year-by-Year Consistency (filtered, 2025-2026 per lane)

Lanes with proper filters applied:
- 7/11 positive in both 2025 and 2026
- 2 UNVERIFIED: MNQ_NYSE_OPEN (filter column issue), MNQ_US_DATA_1000 (cross-instrument filter not in daily_features)
- 2 WATCH: MES_US_DATA_1000 (2026 barely -$21), MNQ_CME_REOPEN VOL_RV30 (2026 -$118, only 29 trades)
- Kill criterion for WATCH lanes: if negative for 3 consecutive months, remove from book

---

## 4. Risk Rules (Replace Prop DD)

These are YOUR rules. No firm enforces them — you must enforce them yourself or code them into the bot.

### Per-Trade

| Rule | Limit | Enforcement |
|---|---|---|
| Max risk per trade | $300 | ORB cap on each lane (coded in profile). Reject if risk_dollars > 300. |
| Stop multiplier | S0.75 | Same as prop — validated, don't change. |
| Max ORB size | Per-lane caps (see lane book) | Coded in DailyLaneSpec.max_orb_size_pts |

### Per-Day

| Rule | Limit | What Happens |
|---|---|---|
| Daily loss limit | -$600 (2% of $30K) | Bot stops opening new trades for the day. Open positions run to target/stop. |
| Daily loss HALT | -$1,000 | Bot flatlines ALL open positions and halts for the day. Manual review required. |

### Per-Week

| Rule | Limit | What Happens |
|---|---|---|
| Weekly loss limit | -$1,500 (5% of $30K) | Bot enters observation mode for rest of week. No new trades. |

### Drawdown

| Rule | Limit | What Happens |
|---|---|---|
| Drawdown warning | -$2,000 from peak (6.7%) | Alert. Review all WATCH lanes. Consider removing weakest. |
| Drawdown HALT | -$3,000 from peak (10%) | FULL STOP. No trading for 1 week minimum. Review everything. |
| Recovery rule | — | Only scale UP at new equity high. Never add contracts during drawdown. |

### Key Difference from Prop DD

Prop trailing DD ratchets against you — if you make $500, your DD floor moves up $500 permanently. One good day followed by one bad day and you're blown.

Self-funded rules DON'T ratchet. Daily/weekly limits reset. The drawdown halt measures from your peak, but you can recover without the floor chasing you. This is strictly better risk management than prop trailing DD.

---

## 5. Costs

### Tradovate Free Plan (recommended at 4 trades/day)

| Item | Cost |
|---|---|
| Monthly subscription | $0 |
| MNQ commission | $0.39/side + $0.22 exchange = $1.22/RT |
| MGC commission | $0.39/side + $0.22 exchange = $1.22/RT |
| MES commission | $0.39/side + $0.22 exchange = $1.22/RT |
| Market data (CME Level 1) | ~$12/mo |
| Annual cost (~5 trades/day) | ~$1,639/yr |

Note: our cost model assumes $2.74/RT for MNQ and $5.74/RT for MGC. Tradovate Free plan is cheaper ($1.22/RT). Backtests are CONSERVATIVE — real costs are lower.

### Breakeven: Free vs Monthly vs Lifetime

| Plan | Annual All-In | Breakeven vs Free |
|---|---|---|
| Free ($0/mo) | $1,374 | — |
| Monthly ($99/mo) | $2,360 | Need 12+ trades/day |
| Lifetime ($1,499 once) | $768/yr after year 1 | Pays for itself in 2.5 years |

Recommendation: Start Free. Switch to Lifetime after 6 months if keeping the account.

---

## 6. Account Structure

### Single Account

One personal Tradovate account, $30K starting capital. Multiple accounts are possible (confirmed via Tradovate forum — subaccounts under one user) but unnecessary at 1ct per lane with $50 intraday margin.

$30K / ($50 × 11 lanes) = 54× margin coverage. Margin is not the constraint. Drawdown risk is.

### Prop Parallel

| Account | Purpose | Status |
|---|---|---|
| topstep_50k_mnq_auto | Prove edge, 90% split, 5 lanes | ACTIVE |
| self_funded_tradovate | Scale edge, 100% split, 11 lanes | PENDING (this spec) |
| tradeify_50k | MNQ scaling lane, 90% split | INACTIVE until auth fixed |

Both TopStep and self-funded run the same 5 core lanes. Self-funded adds 6 extra lanes. This is a live A/B comparison — if prop and self-funded diverge, investigate.

---

## 7. Phased Deployment

### Phase 1: Prove (Month 1-3)

- Deploy 5 core lanes only (same as TopStep)
- 1 contract per lane
- S0.75 stops, all caps active
- Daily limit -$600, weekly -$1,500
- **Goal:** Bot runs clean on real money. Verify fills match backtest expectations.
- **Kill criterion:** If net P&L < -$1,500 after 60 trades, STOP and review.

### Phase 2: Expand (Month 4-6)

- If Phase 1 net positive >$2K, add 6 extra lanes (NYSE_OPEN, US_DATA_1000, CME_PRECLOSE, MNQ_CME_REOPEN)
- Raise COMEX_SETTLE cap from 80→150pt
- Still 1 contract per lane
- **Goal:** Full 11-lane book running. Verify diversification benefit (should see fewer all-loss days).
- **Kill criterion for WATCH lanes:** If negative for 3 consecutive months, remove.

### Phase 3: Scale (Month 7+)

- If account equity >$35K (new high watermark), scale top 3 lanes to 2ct
- Top 3 by trailing 6-month P&L (not by backtest — by LIVE performance)
- Adjust daily limit to 2% of current equity
- **Goal:** $5K+/month income.
- **Max:** 3ct per lane until account reaches $50K.

---

## 8. Implementation Gaps to Close

| # | Gap | Fix | Effort |
|---|---|---|---|
| 1 | Profile config (account_size, payout_policy, caps) | Update `self_funded_tradovate` in prop_profiles.py | 30 min |
| 2 | Add daily/weekly loss limits to HWM tracker | Extend `check_halt()` with daily_loss_limit and weekly_loss_limit params | 2 hours |
| 3 | Tradovate personal account auth | Create account, get API creds, test auth flow | 30 min (manual) |
| 4 | COMEX_SETTLE cap raise 80→150pt | Update DailyLaneSpec in profile | 5 min |
| 5 | Add 6 extra lane definitions to profile | New DailyLaneSpec entries | 30 min |
| 6 | Per-trade max risk rejection | Add risk_dollars > max_risk_per_trade guard in session_orchestrator | 1 hour |

Total code work: ~4 hours. Auth setup: 30 min manual.

---

## 9. Risks and Honest Caveats

### Known Risks

1. **MNQ concentration:** 8/11 lanes are MNQ. A structural Nasdaq change hits most of the book simultaneously. The Jun 2025 -$930 month was exactly this.

2. **2025-2026 is a hot regime:** Gold at $4,675, NQ vol elevated. The 16-year history shows lean years (2023 was -$378 on SINGAPORE_OPEN, -$450 on TOKYO_OPEN). Current results may not persist.

3. **MGC_CME_REOPEN is regime-dependent:** 77.6% of trades from 2025-2026 (gold vol explosion). Only 79 total trades — thin sample. The $4,630 in 2026 could be a vol artifact.

4. **Self-discipline risk:** No firm forces you to stop. If you override your own rules during a losing streak, you lose real money with no safety net.

5. **Tradovate broker risk:** Not SIPC-protected (futures). NFA requires segregated customer funds but broker bankruptcy is a tail risk. Mitigate: don't keep more than $50K in any single futures brokerage.

### What Could Kill This

- Gold returns to $1,800 → MGC lanes die (G6+ filter passes <5 days/year)
- NQ enters sustained bear → MNQ baselines go negative
- Tradovate raises intraday margin → still fine at $500/contract, problem at $2K+
- CME changes session times → SESSION_CATALOG needs update (event-based design handles this)

### Tax Advantage (Section 1256)

Self-funded futures: 60% taxed as long-term capital gains, 40% as short-term. At $32K/yr income, estimated tax savings vs ordinary income (prop payouts): ~$2,000-3,000/yr depending on bracket. This is NOT financial advice — consult a CPA.

---

## 10. Decision

**Recommended:** Approach C (Graduated Hybrid) with 11 honest lanes.

- Phase 1: 5 lanes, 1ct, prove on real money (same lanes as prop)
- Phase 2: 11 lanes, 1ct, full diversified book
- Phase 3: Scale to 2-3ct on best performers

Expected path: $1,423/mo → $2,722/mo → $5,443/mo over 7+ months.

Prop accounts run in parallel as zero-risk edge validation.
