# Prop Account Allocation Design

**Status:** DRAFT — iterating before implementation
**Date:** 2026-03-16
**Purpose:** Assign strategies to prop firm accounts for maximum net income with clean per-account tracking

---

## Firms & Rules (Verified March 2026)

| Rule | TopStep $50K | Apex $50K | MFFU Rapid $50K |
|------|-------------|-----------|-----------------|
| Trailing DD | $2,000 (EOD) | $2,500 (real-time) | $2,000 (intraday) |
| DD locks? | No (always trails) | Yes at $50,100 | Yes at starting balance |
| Daily loss limit | Optional (you set it) | None | None |
| Consistency rule | Combine only, not funded | None | None |
| News restriction | None | None | T1 only (~40 days/yr) |
| Max accounts | 5 Express + 1 Live | 20 | Unknown |
| Contract limit (50K) | 50 micros | 5 micros (10 after $52,600) | Scaling (micro) |
| Close-by time | 3:10 PM CT | Session close | Session close |
| Profit split | 90/10 | 100% (legacy PAs) | 90/10 |
| Monthly fee | ~$49 | ~$37 | $129 |
| Copy trading | Yes (own accounts) | Yes | Yes (funded) |
| Auto trading | Yes (API/ProjectX) | Yes (Tradovate) | Yes (since Jul 2025) |

### T1 News Events (MFFU only)
Must be flat 2 minutes before and after: FOMC, FOMC Minutes, Employment/NFP, CPI.
That's ~40 event days/year. Sessions at risk: COMEX_SETTLE (3:30am Bris ≈ 1:30pm ET, near FOMC 2pm ET).

### TopStep 3:10 PM CT Close
Positions must be flat by 3:10 PM CT (≈5:10-6:10 AM Brisbane depending on DST).
Sessions at risk: CME_PRECLOSE (5:45am), NYSE_CLOSE (6:00am).

### Apex Safety Net
DD trails real-time until peak unrealized balance hits $52,600. Then floor locks at $50,100 forever.
Danger zone: first ~2 weeks until $2,600 profit accumulated. After lock, account is nearly unkillable.

Sources:
- https://help.topstep.com/en/articles/8284204-what-is-the-maximum-loss-limit
- https://help.topstep.com/en/articles/8284215-express-funded-account-parameters
- https://support.apextraderfunding.com/hc/en-us/articles/40463582041371-Trailing-Drawdown-Rule
- https://help.myfundedfutures.com/en/articles/8230009-news-trading-policy
- https://www.proptradingvibes.com/blog/myfundedfutures-rules-overview

---

## Strategy Ranking (Real Data)

All 30 prop-book strategies ranked by expected $/month at 1 contract, 0.75x stop.
Expected $/month = median_risk_dollars × ExpR × trades_per_month (2024+ data from gold.db).

| # | Session | Inst | ORB | RR | Filter | ExpR | WR | Med$ | Tr/mo | E$/mo | Manual? |
|---|---------|------|-----|-----|--------|------|----|------|-------|-------|---------|
| 1 | CME_PRECLOSE | MNQ | 15m | 1.0 | VOL_RV12_N20 | 0.266 | 60% | $76 | 13.1 | $264 | AUTO |
| 2 | NYSE_OPEN | MNQ | 5m | 1.5 | VOL_RV12_N20 | 0.132 | 40% | $84 | 18.7 | $208 | YES |
| 3 | US_DATA_1000 | MNQ | 5m | 1.0 | VOL_RV12_N20 | 0.149 | 54% | $66 | 18.7 | $185 | AUTO |
| 4 | CME_REOPEN | MGC | 5m | 1.5 | ORB_G4_FAST10 | 0.365 | 54% | $57 | 5.9 | $123 | YES |
| 5 | LONDON_METALS | MNQ | 30m | 3.0 | ORB_G6_NOMON | 0.112 | 24% | $59 | 18.5 | $122 | YES |
| 6 | SINGAPORE_OPEN | MNQ | 30m | 3.0 | DIR_LONG | 0.155 | 26% | $38 | 18.5 | $108 | YES |
| 7 | COMEX_SETTLE | MNQ | 5m | 2.5 | VOL_RV12_N20 | 0.181 | 31% | $33 | 17.9 | $106 | AUTO |
| 8 | TOKYO_OPEN | MGC | 15m | 2.0 | ORB_G5_FAST10 | 0.189 | 37% | $69 | 7.0 | $91 | YES |
| 9 | SINGAPORE_OPEN | MGC | 5m | 3.0 | ORB_G5 | 0.179 | 27% | $66 | 6.6 | $79 | YES |
| 10 | EUROPE_FLOW | MNQ | 5m | 1.0 | VOL_RV12_N20 | 0.144 | 57% | $25 | 18.7 | $68 | YES |
| 11 | TOKYO_OPEN | MGC | 5m | 3.0 | ORB_G5 | 0.298 | 30% | $59 | 3.8 | $66 | YES |
| 12 | BRISBANE_1025 | MNQ | 5m | 4.0 | ORB_G6 | 0.123 | 21% | $19 | 14.7 | $35 | YES |
| 13 | NYSE_CLOSE | MES | 15m | 1.0 | VOL_RV12_N20 | 0.070 | 54% | $25 | 10.6 | $18 | AUTO |

Manual trading hours: 7am-midnight Brisbane.
AUTO sessions: US_DATA_1000 (midnight), COMEX_SETTLE (3:30am), CME_PRECLOSE (5:45am), NYSE_CLOSE (6am).

---

## Scenario A: Manual Only ($576/month net)

No auto-execution. Only sessions tradeable while awake (7am-midnight Brisbane).

### Account 1: TopStep $50K — "MGC Morning"
| Time | Strategy | E$/mo | Med Risk | P95 Risk |
|------|----------|-------|----------|----------|
| 08:00 | CME_REOPEN MGC 5m E2 ORB_G4_FAST10 RR1.5 0.75x | $123 | $57 | $274 |
| 10:00 | TOKYO_OPEN MGC 15m E1 ORB_G5_FAST10 RR2.0 0.75x | $91 | $69 | $187 |

**Gross:** $214/mo | **Net:** $144/mo (90% split - $49 fee)
**Why TopStep:** MGC has wide ORB variance ($32-$749). EOD trailing gives intraday breathing room on spiky gold mornings. Both sessions finish by noon — no 3:10 PM CT risk.
**Max risk per trade:** 3.5% DD median, 13.7% DD at P95.
**Co-occurrence:** Both fire only 13% of nights. Most nights = 0-1 trades.
**Trade duration:** CME_REOPEN median 34min. TOKYO median 36min. 7.2% overlap risk.

### Account 2: Apex $50K #1 — "MNQ Night+Evening"
| Time | Strategy | E$/mo | Med Risk | P95 Risk |
|------|----------|-------|----------|----------|
| 23:30 | NYSE_OPEN MNQ 5m E2 VOL_RV12_N20 RR1.5 0.75x | $208 | $84 | $177 |
| 18:00 | LONDON_METALS MNQ 30m E2 ORB_G6_NOMON RR3.0 0.75x | $122 | $59 | $146 |

**Gross:** $330/mo | **Net:** $293/mo (100% split - $37 fee)
**Why Apex:** Top E$/mo session. 100% payout. DD locks at $50,100 after ~$2,600 profit.
**Max risk per trade:** 3.4% DD median, 7.1% DD at P95.
**Co-occurrence:** Different time blocks (6pm vs 11:30pm) — no overlap.

### Account 3: Apex $50K #2 — "MNQ Day Sessions"
| Time | Strategy | E$/mo | Med Risk | P95 Risk |
|------|----------|-------|----------|----------|
| 11:00 | SINGAPORE_OPEN MNQ 30m E2 DIR_LONG RR3.0 0.75x | $108 | $38 | $123 |
| 17:00 | EUROPE_FLOW MNQ 5m E2 VOL_RV12_N20 RR1.0 0.75x | $68 | $25 | $59 |

**Gross:** $176/mo | **Net:** $139/mo (100% split - $37 fee)
**Why Apex not MFFU:** Same pair on MFFU Rapid ($129 fee, 90% split) = $29/month net. On Apex ($37 fee, 100% split) = $139/month net. Apex is strictly better for manual.
**Max risk per trade:** 1.5% DD median, 4.9% DD at P95. Lowest risk account.

### Why Not MFFU for Manual?
MFFU Rapid costs $129/month with 90% split. For the SINGAPORE+EUROPE pair:
- MFFU net: $176 × 0.90 - $129 = **$29/month**
- Apex net: $176 × 1.00 - $37 = **$139/month**

MFFU only makes economic sense when gross income justifies the fee (>$500/month = overnight auto).

---

## Scenario B: With Auto-Execution ($928/month net)

Same 3 manual accounts PLUS overnight auto:

### Account 4: MFFU Rapid $50K — "Overnight AUTO"
| Time | Strategy | E$/mo | Med Risk | P95 Risk |
|------|----------|-------|----------|----------|
| 05:45 AUTO | CME_PRECLOSE MNQ 15m E2 VOL_RV12_N20 RR1.0 0.75x | $264 | $76 | $191 |
| 03:30 AUTO | COMEX_SETTLE MNQ 5m E2 VOL_RV12_N20 RR2.5 0.75x | $106 | $33 | $92 |

**Gross:** $370/mo | **Net:** $204/mo (90% split - $129 fee)
**Why MFFU for overnight auto:** EOD trailing means a bad overnight trade doesn't kill you until end of day — you can wake up and manage. Apex's real-time trailing is dangerous when sleeping (can't monitor P&L).
**T1 risk:** COMEX_SETTLE at 1:30pm ET is near FOMC (2pm ET). Skip COMEX on ~8 FOMC days/year.
**CME_PRECLOSE:** 3:45pm ET — after all T1 releases. No conflict.
**Trade duration:** CME_PRECLOSE resolves in median 4 minutes (!). COMEX median 23 minutes. 3.3% overlap.

### Account 5: Apex $50K #3 — "Overnight AUTO"
| Time | Strategy | E$/mo | Med Risk | P95 Risk |
|------|----------|-------|----------|----------|
| 00:00 AUTO | US_DATA_1000 MNQ 5m E2 VOL_RV12_N20 RR1.0 0.75x | $185 | $66 | $170 |

**Gross:** $185/mo | **Net:** $148/mo (100% split - $37 fee)
**Why not MFFU:** US_DATA_1000 (midnight = 10am ET) could conflict with CPI (8:30am ET) — but CPI is actually before this session, so no real issue. However Apex has no restriction at all and 100% split.
**Note:** 40% overlap with NYSE_OPEN (same account? no — NYSE is on Apex #1). If running independently, no issue.

### Auto Execution Readiness
Live infra exists (DataFeed → BarAgg → Engine → OrderRouter) but Mar 14 audit found:
- 1 CRITICAL: no trade journal
- 3 HIGH: pysignalr stop, webhook risk, exit retry
- Test gaps

**Recommendation:** Start Scenario A (manual). Fix auto infra. Deploy Scenario B after 2-3 months.

---

## Variable Risk Management

### The ORB Changes Every Trade
Dollar risk = ORB_size × point_value × 0.75 + commission. Not fixed.

Real ORB size distributions (2024+, filtered trades only):

| Session | Inst | Min$ | Median$ | P75$ | P95$ | Max$ |
|---------|------|------|---------|------|------|------|
| CME_REOPEN | MGC | $32 | $57 | $113 | $278 | $749 |
| TOKYO_OPEN | MGC | $39 | $59 | $86 | $172 | $447 |
| NYSE_OPEN | MNQ | $16 | $84 | $115 | $177 | $368 |
| US_DATA_1000 | MNQ | $10 | $66 | $97 | $170 | $401 |
| COMEX_SETTLE | MNQ | $10 | $33 | $48 | $93 | $534 |
| CME_PRECLOSE | MNQ | $12 | $33 | $50 | $92 | $292 |
| LONDON_METALS | MNQ | (TBD) | $59 | (TBD) | $146 | (TBD) |
| SINGAPORE_OPEN | MNQ | (TBD) | $38 | (TBD) | $123 | (TBD) |
| EUROPE_FLOW | MNQ | (TBD) | $25 | (TBD) | $59 | (TBD) |

### Max Risk Per Trade Gate
Institutional standard: no single trade > 10% of DD budget.

| Account | DD | 10% Cap | Max ORB (pts) | Trades Skipped |
|---------|-----|---------|---------------|----------------|
| TopStep $50K | $2,000 | $200 | MGC 26.4 pts | 13% of CME_REOPEN |
| Apex $50K | $2,500 | $250 | MNQ 166 pts | <2% |
| MFFU $50K | $2,000 | $200 | MNQ 133 pts | <1% |

**Only MGC CME_REOPEN needs the gate** (13% skip rate). MNQ at $2/point naturally stays under.

**Rule:** Before entering, check ORB size. If ORB_size × point_value × 0.75 > 10% of DD budget, skip.

### Trade Duration & Overlap Risk
| Session pair | Gap | Overlap % | Risk |
|-------------|-----|-----------|------|
| CME_REOPEN → TOKYO_OPEN | 120 min | 7.2% | Low |
| NYSE_OPEN → US_DATA_1000 | 30 min | 40.5% | HIGH — 2 concurrent MNQ 40% of nights |
| COMEX_SETTLE → CME_PRECLOSE | 135 min | 3.3% | Low |

### Co-occurrence (Both Strategies Fire Same Night)
| Account | Both fire | One fires | Neither |
|---------|-----------|-----------|---------|
| TopStep (CME_REOPEN + TOKYO) | 13% | ~20% | ~67% |
| Apex #1 (NYSE + LONDON) | independent | — | — |
| Apex #2 (SINGAPORE + EUROPE) | independent | — | — |
| MFFU (COMEX + CME_PRECLOSE) | 85% | ~0% | ~15% |
| Apex #3 (US_DATA only) | — | 89% | 11% |

---

## Deployment Plan

### Phase 1: Prove It (Months 1-3)
- Open TopStep $50K only
- Trade MGC CME_REOPEN + TOKYO_OPEN manually
- Goal: 2+ payouts, prove the edge is real live
- Expected: $144/month net, 0-2 trades/day
- Cost: $49/month

### Phase 2: Scale Manual (Months 3-5)
- Add Apex $50K #1 (NYSE_OPEN + LONDON_METALS)
- Two accounts, two instruments, two time blocks
- Expected: $437/month net
- Cost: $86/month total

### Phase 3: Full Manual (Months 5-7)
- Add Apex $50K #2 (SINGAPORE_OPEN + EUROPE_FLOW)
- Three accounts running
- Expected: $576/month net
- Cost: $123/month total

### Phase 4: Auto-Execution (Months 7+)
- Fix live trading infra (journal, pysignalr, exit retry)
- Deploy MFFU Rapid + Apex #3 for overnight auto
- Expected: $928/month net
- Cost: $289/month total

### Phase 5: Copy Trade Scaling (Months 9+)
- Duplicate winning accounts (TopStep allows 5, Apex allows 20)
- Copy trade from master to follower accounts
- 3x multiplier on proven strategies

---

## Open Questions / Things To Refine

- [ ] MFFU Core vs Rapid vs Pro — detailed comparison for overnight auto
- [ ] Copy trading mechanics (Tradovate Group Trading, TraderSyncer)
- [ ] Exact Apex contract limits at $50K before Safety Net ($52,600)
- [ ] Live trading infra gaps — what needs fixing before auto?
- [ ] IBKR self-funded allocation (all strategies at 1.0x, position sizing)
- [ ] Max ORB gate implementation in trade sheet generator
- [ ] Trade journal integration (the CRITICAL gap from Mar 14 audit)
- [ ] FOMC/NFP/CPI calendar integration for MFFU T1 auto-skip
- [ ] Worst-case scenario analysis: 3 consecutive all-loss nights per account
- [ ] Re-run Monte Carlo sims with THIS specific allocation (not the old 8-session book)
- [ ] Tax implications of multiple prop firm payouts
- [ ] Which sessions benefit most from IBKR's full 1.0x stop vs 0.75x prop
- [ ] Trailing DD lock-in strategy: early weeks conservative, then open up after lock?
