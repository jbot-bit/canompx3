# PROP PLAYS — Manual Trading Playbook
_Apex Trader Funding — Tradovate FULL accounts. 1 trade per session window._
_Last updated: 2026-02-23_

---

## Confirmed Apex Rules (verified)

- **News/events**: No blackout. Can trade ORB at equity open normally. ✅
- **Consistency rule**: No single day's profit can exceed **30% of total profit balance** when requesting a payout
- **Static 100K**: $625 max DD — **DO NOT USE**. System max DD is $1,900. It will blow.
- **Platform**: Tradovate native on FULL accounts ✅
- **Daily DD**: None on FULL accounts ✅ (important for ORB trading)
- **Scaling plan**: None ✅

---

## Account Decision

| Account | DD | Fee | Contracts | Net/mo | Safety | Verdict |
|---|---|---|---|---|---|---|
| FULL 25K | $1,500 trail | $187 | 0ct | — | -27% | KILL — DD < system max |
| **FULL 50K** | **$2,500 trail** | **$197** | **1ct** | **$263** | **24%** | **START HERE** |
| FULL 100K | $3,000 trail | $297 | 1ct | $163 | 37% | Higher safety but fee hurts net |
| **FULL 150K** | **$5,000 trail** | **$397** | **2ct** | **$523** | **24%** | **GRADUATE TO THIS** |
| FULL 250K | $6,500 trail | $397 | 3ct | $983 | 12% | Safety margin too tight |
| FULL 300K | $7,500 trail | $397 | 4ct | $1,443 | -1% | Will blow historically |
| Static 100K | $625 hard | $297 | — | — | -204% | KILL — confirmed trap |
| MFF Pro 150K | $4,500 EOD | $477 | 2ct | $443 | 16% | EOD better mechanics but fee kills it |

**Path: Start FULL 50K → validate live for 2-3 months → add FULL 150K once funded**

System baseline: max DD $1,900 (1 micro) / $3,800 (2 micros) | ~$460/month realistic per contract

---

## Consistency Rule — Payout Timing

30% rule means first payout takes longer than you'd expect.

- Single biggest trade (MGC_0900 at 2.5R): $185 → need $617 total before payout
- Typical best day (2-3 wins at 1ct): ~$200–350 → need $667–1,167 total
- At $460/month realistic: **first payout ~2–3 months in** at 1 contract

Not a blocker — you keep trading while the rule resolves itself. Just don't expect a withdrawal in week 1.

---

## Trailing DD — How to Think About It

With trailing DD, the floor **rises as you profit** but never falls. This means:

- If you make $500, your floor rises $500 — buffer shrinks temporarily
- If a trade briefly floats +$200 unrealized then stops out at -$100, floor rose $200 AND balance fell $100 = $300 buffer lost on one trade

**How to protect yourself:**
- Trade conservatively early (treat floor as $0 until $1K+ ahead)
- Don't let open profits run too far on losing trades — exit cleanly at target or stop
- Adaptive rule: if portfolio DD hits $1,500, go to half-size until it recovers

---

## Manual Session Schedule (Brisbane time)

One pick per window. When slots clash — pick highest ExpR.

### Morning (Brisbane)

| Time (BRIS) | Trade | ExpR | WR% | RR | Note |
|---|---|---|---|---|---|
| **9:00am** | **MGC_1800** | **+0.333** | **49%** | **2.0** | Gold ORB. Solo. Set alarm. |

### Afternoon (Brisbane)

| Time (BRIS) | Trade | ExpR | WR% | RR | Note |
|---|---|---|---|---|---|
| **11:30am** | **MNQ_0030** | **+0.201** | **42%** | **2.0** | NQ ORB. Solo. |

### Night / US Session (Brisbane — main cluster)

| Time (BRIS) | Trade | ExpR | WR% | RR | Note |
|---|---|---|---|---|---|
| 11:30pm | MNQ_CME_OPEN | +0.130 | 61% | 1.0 | Solo. OK to skip if tired. |
| **12:00am** | **MGC_0900** | **+0.443** | **46%** | **2.5** | Gold NY open. Solo. Best risk-adj. |
| **12:30am** | **MES_US_EQUITY_OPEN** | **+0.585** | **43%** | **1.5** | A0 — only validated slot. Skip MNQ/M2K. |
| **1:00am** | **MGC_1000** | **+0.311** | **50%** | **2.0** | Pick over MES_1000/MNQ_1000. |
| 1:30am | ~~MES_US_POST_EQUITY~~ | +0.077 | 59% | 1.0 | Too weak. Skip. |
| **2:00am** | **MNQ_1100** | **+0.249** | **47%** | **2.0** | Solo. Easy to watch. |
| **3:00am** | **MES_CME_CLOSE** | **+0.206** | **70%** | **1.0** | Pick over MNQ_CME_CLOSE. |

### Evening (Brisbane — optional)

| Time (BRIS) | Trade | ExpR | WR% | RR | Note |
|---|---|---|---|---|---|
| 6:00pm | MNQ_LONDON_OPEN | +0.092 | 59% | 1.0 | Weakest. Skip unless nothing else on. |

---

## Realistic Daily Plans

### Option A — Morning (light, 2 sessions)
- 9:00am MGC_1800
- 11:30am MNQ_0030
- ~$115/month realistic at 1ct

### Option B — US Session cluster (night owl)
- 12:00am MGC_0900
- 12:30am MES_US_EQUITY_OPEN (A0)
- 1:00am MGC_1000
- 2:00am MNQ_1100
- 3:00am MES_CME_CLOSE
- ~$240/month realistic at 1ct

### Option C — Hybrid (best balance)
- 9:00am + 11:30am + 12:00am + 12:30am
- ~$210/month realistic at 1ct

Full 15-slot coverage: ~$460/month at 1ct (automated or very disciplined manual)

---

## DD Rules

| Situation | Action |
|---|---|
| Normal | Trade your plan |
| Down 3R in a day | Stop for the day |
| Portfolio DD hits $1,000 | Reduce to half-size, review |
| Portfolio DD hits $1,500 | Stop trading, reassess |
| Portfolio DD hits $2,000 | Hard stop — protect the account |

---

## Correlated Pairs (count as 1 risk unit if both fire)

- MES_1000 + MNQ_1000 (r=0.59) — already resolved by picking MGC_1000 at 1am
- MES_CME_CLOSE + MNQ_CME_CLOSE (r=0.55) — pick MES
- MES_US_POST_EQUITY + MNQ_US_POST_EQUITY (r=0.54) — skipped (weak)

---

## Scaling Path

1. **FULL 50K eval**: Trade Option B or C (US session). 1 micro. Prove it works live.
2. **Funded on 50K**: Continue same plan. First payout ~2-3 months.
3. **Once $500+ ahead on 50K**: Open FULL 150K eval alongside. Start building track record.
4. **Funded on 150K**: Graduate to 2 micros per slot. ~$523/month net.
5. **Never**: Scale contracts until 3+ months of positive live P&L on current size.

---

## Validation Status

| Slot | Status |
|---|---|
| **MES_US_EQUITY_OPEN (A0)** | PROMOTE — passed no-leakage WF + stress + falsification |
| All other slots | Backtested — forward gate needs 60+ live trades each |

Only A0 is fully validated. Other slots are strong backtests. Add them to live trading as they accumulate forward gate evidence.

---

## Research Merge Status

All research is in this workspace. One terminal. No merge needed.
- `research/output/a0_final_validation_report.md` — A0 PROMOTE verdict
- `research/output/forward_gate_status_latest.md` — live forward gate tracker
- `PROP_PLAYS.md` — this file

Next: A0 execution-latency test (+1/+2 bar delay) to confirm edge survives real entry conditions.
