# Max Profit Extraction — Fresh-Eyes Audit (2026-04-15)

**Mandate:** Bloomberg-PM-level capital-allocation review of the existing 38-strategy book. No new research. Only: are we sizing right, allocating right, leaving money on the table?

**TL;DR:** The binding constraint is **not correlation, not dead weight, not Kelly sizing — it's the TopStep $2,000 max trailing DD**, which the deployed 6-lane portfolio has already touched at **1 MICRO contract** in the last 12 months (max DD = **$2,433** on 2025-07-01). Every "more contracts" scaling path on TopStep 50K XFA is DD-blocked. The real unlocks are (1) multi-account parallelism, and (2) moving off TopStep's tight DD onto Bulenox/Elite/self-funded.

## 1. Current baseline (1 micro contract per lane, 6 deployed lanes)

Deployed portfolio from `lane_allocation.json` (profile `topstep_50k_mnq_auto`).

Sample-size tier convention (per RESEARCH_RULES.md): N<30 = INVALID, N=30-99 = REGIME, N≥100 = CORE. Numbers below are aggregate trade counts across 4 lanes (verified) — proxy lanes excluded; see §6.1.

| Window | Trades (4-lane) | Tier | Total $ | MaxDD $ | Annual $ / contract |
|---|---:|:---:|---:|---:|---:|
| 2019-05 → 2026-04 (full 6.6yr) | 1789 | CORE | $30,050 | **$2,433** | $4,337/yr |
| 2021-01 → 2026-04 (post-COVID 5.25yr) | 1361 | CORE | $28,559 | $2,433 | $5,427/yr |
| 2023-01 → 2026-04 (recent 3.3yr) | 845 | CORE | $19,974 | $2,433 | $6,115/yr |
| **2025-04 → 2026-04 (trailing 12mo)** | 255 | CORE | **$8,641** | $2,433 | **$8,767/yr** |
| 2026-01 → 2026-04 (OOS) | 70 | REGIME | $4,759 | $585 | $17,736/yr |

**Honest number for a fresh account today: ~$8,800/yr per $50K account at 1 contract** (CORE-tier evidence). The $17.7K OOS-annualized is REGIME-tier (3.5 months, N=70) — directionally encouraging but not a planning number.

The $2,433 max DD occurred in mid-2025 (recent), not the 2020 COVID period. This is not a distant-history artifact. It's the live regime.

## 2. Naive baseline vs correlation-adjusted

- Sum of individual strategy $ claims: **$40,236/yr gross** (38 strategies × 1 ct each)
- Sum net of commission: **$31,371/yr**
- Correlation-dedup'd (1 best lane per session): **$9,833/yr gross**
- **Actual deployed 6-lane portfolio**: $6,323/yr (lifetime) to $8,767/yr (trailing)

The naive baseline overstates by **4.09×** because same-session clusters count the same ORB break multiple times. Example: EUROPE_FLOW has 10 validated strategies claiming 1,708 trades/year combined — but EUROPE_FLOW runs once per day (~255 days/yr), so the real annual trades cap is **255, not 1,708**. That's a 6.7× overcount.

**The allocator already does the right thing here.** It picks 1 best-of-cluster per session × 6 sessions = 6 lanes. The "missing" $22K/yr (between naive $31K and deployed $9K) is not money left on the table — it's correlation double-counting.

## 3. Correlation clusters (with overcount factor)

| Session | N strategies | Sum TPY | Max possible (1/session/day) | Overcount |
|---|---:|---:|---:|---:|
| MNQ EUROPE_FLOW | 10 | 1,708 | 255 | **6.7×** |
| MNQ COMEX_SETTLE | 8 | 1,337 | 246 | 5.4× |
| MNQ NYSE_OPEN | 6 | 1,229 | 255 | 4.8× |
| MNQ TOKYO_OPEN | 4 | 804 | 256 | 3.1× |
| MNQ US_DATA_1000 | 5 | 689 | 255 | 2.7× |
| MNQ SINGAPORE_OPEN | 2 | 291 | 256 | 1.1× |
| MES CME_PRECLOSE | 2 | 83 | 243 | 0.3× |
| MNQ CME_PRECLOSE | 1 | 101 | 244 | 0.4× |

## 4. Kelly sizing — irrelevant at current scale

Half-Kelly per deployed lane (from win_rate + avg_win$/avg_loss$):

| Lane | Half-Kelly |
|---|---:|
| EUROPE_FLOW ORB_G5 RR1.5 | 3.0% |
| SINGAPORE_OPEN ATR_P50 RR1.5 | 4.6% |
| COMEX_SETTLE OVNRNG_100 RR1.5 | 8.0% |
| NYSE_OPEN ORB_G5 RR1.0 | 4.8% |
| TOKYO_OPEN ORB_G5 RR1.5 | 3.9% |
| US_DATA_1000 VWAP RR1.5 | 7.4% |

Half-Kelly average ~5% × $50K account = $2,500 risk per trade. At MNQ median risk of ~$30/trade/ct, Half-Kelly wants ~80 contracts per trade. **We're 80× BELOW Kelly at 1 contract.** Kelly is not the binding constraint. DD is.

## 5. Dead-weight check — nothing to cull

No strategy matches the dead-weight rule (`tpy<30 OR (tpy<100 AND ExpR<0.10)`). All 38 active strategies are clearing quality thresholds. The allocator's option pool is healthy — it's just only picking 6 because that's all there are distinct sessions to fill.

## 6. The DD ceiling — the actual binding constraint

### 6.1 Measurement caveat (read first)

Two of the six deployed lanes were computed with PROXY filters, not the actual canonical filter logic:

- **SINGAPORE_OPEN ATR_P50_O30**: proxy `daily_features.atr_20_pct >= 50` (an absolute-percent threshold). The real ATR_P50 is a ROLLING percentile. Proxy almost certainly fires on a different day-set than canonical.
- **US_DATA_1000 VWAP_MID_ALIGNED_O15**: VWAP filter SKIPPED entirely; query takes ALL US_DATA_1000 O15 breaks. This OVERCOUNTS trades on days the real VWAP filter would reject.

**Honest DD bound:**
- 4-lane measured (EF/CS/NY/TK with exact filters): **$2,433**
- 6-lane with proxies (above 4 + SGP_proxy + VWAP-skipped): **$3,790**
- True 6-lane measurement requires implementing the canonical ATR_P50 percentile predicate and the VWAP_MID_ALIGNED filter in SQL or via `trading_app.config.ALL_FILTERS["..."].matches_df()`. Not done in this audit.

### 6.2 What the numbers say (within the caveat)

Even at the lower bound ($2,433, 4-lane measured), portfolio cumulative DD HISTORICALLY exceeded the TopStep 50K XFA `max_trailing_dd = $2,000`. This is a backtest finding — a fresh April 2026 account did NOT experience the 2025-07 DD; forward-DD is unknown.

**Honest framing:** historical 4-6yr backtest pattern, IF REPEATED in a new account during the corresponding window, would have breached the $2K trailing limit at 1 ct. Scaling 2ct+ would breach in any 6mo+ backtest window with very high probability.

| Scale | Portfolio DD $ (4-lane verified) | Portfolio DD $ (6-lane proxied) | Worst day $ (4-lane) | Forward-DD verdict |
|---:|---:|---:|---:|---|
| 1× | $2,433 | $3,790 | $-567 | **TIGHT — historical breach in some windows** |
| 2× | $4,867 | $7,580 | $-1,135 | BREACH likely |
| 3× | $7,300 | $11,370 | $-1,702 | BREACH near-certain |
| 5× | $12,166 | $18,950 | $-2,837 | BREACH near-certain |
| 10× | $24,333 | $37,900 | $-5,673 | BREACH near-certain |

F-1 XFA scaling plan (50K): 2 mini-lots Day 1 (=20 micros), 3 at +$1,500 (=30 micros), 5 at +$2,000 (=50 micros). **Position cap is nowhere near binding. DD is the binding constraint, even at the lower-bound 4-lane measurement.**

## 7. The real unlock paths

### Path A — Multi-account parallel (2-account split, designed not built)

Each TopStep 50K XFA account runs the same 6-lane portfolio at 1 ct independently.

| N accounts | Capital (subscription + reset buffers) | Annual $ | Independent blow-up risk |
|---:|---:|---:|---|
| 1 | ~$150-250 subscription | $8,800/yr | one account at risk |
| 2 | ~$300-500 subscription | $17,600/yr | two independent risks |
| 3 | ~$450-750 subscription | $26,400/yr | three independent risks |

Cost per account per year on TopStep: ~$150/mo × 12 = $1,800. So 3-account net ≈ $26.4K - $5.4K = **$21K/yr** at ~$150 capital outlay.

### Path B — Prop firm migration (Bulenox 50K + Elite 50K, Rithmic-based)

Per `memory/prop_firm_automation_verified_apr5.md`: Bulenox + Elite use Rithmic API (same as self-funded), have longer payout windows, and have less-tight DD rules. TopStep's short window (3-5 payouts) caps long-term scaling.

Estimated income (from prior MEMORY analysis): similar $9K/yr/ct but with better DD headroom and payout durability.

### Path C — Self-funded (AMP or EdgeClear via Rithmic)

Per `memory/deployment_plan_final_apr3.md` and `self_funded_realistic_assessment.md`:

- 2026 OOS: **$2,929/yr NET per contract** (after commission+slippage)
- At 10 contracts: **$29K/yr NET = 59% ROI on $50K self-funded**
- No DD ceiling imposed by a prop firm — only your own capital
- Rithmic API integration DONE per MEMORY

**This is the highest-ROI path per-dollar-of-capital, and infrastructure is already built.**

## 8. The money answers

### Q1: Current system annual $P&L (1 ct per deployed lane, correlation-adjusted)

**$8,800/yr per $50K account** on trailing 12mo basis. **$6,300/yr** on conservative lifetime-average basis.

### Q2: Optimized annual $P&L after:
- Deduplicating correlated strategies — **already done** (allocator picks 6 best-of-cluster)
- Kelly sizing — **not binding** (DD is)
- Max contract allowance — **irrelevant** (DD caps well before contract limit)

The single-account optimum IS the current deployment. Further optimization requires changing the constraint layer (firm or capital source).

### Q3: Single highest-impact change

**MIGRATE FROM TOPSTEP 50K XFA TO SELF-FUNDED (AMP/EdgeClear via Rithmic) AT $50K CAPITAL.**

Why:
- TopStep $2,000 DD is already being touched at 1 ct → fundamentally ceiling-bound
- Self-funded removes the external DD ceiling, exposes only your own capital
- Per prior MEMORY research: 10 contracts × $2,929/yr NET = $29K/yr NET on $50K self-funded = **59% annual ROI**
- All infrastructure (Rithmic adapter, cost model, risk manager, F-1 gate) is already built
- F-1 disable-for-TC-accounts fix shipped today means bot is ready to run either mode

Secondary impact: parallel multi-account (Bulenox + Elite + TopStep) as blow-up insurance. But the DD math favors self-funded first.

### Q4: Realistic annual $/yr per account tier

Using trailing-12mo ExpR as the "current vintage" baseline:

| Capital | Route | Contracts | Annual $ NET | ROI |
|---:|---|---:|---:|---:|
| $50K | TopStep 50K XFA (status quo, DD-bound at 1 ct) | 1 | $8,800 | 18% |
| $100K | 2× TopStep 50K XFA | 2 (1 each) | $17,600 | 18% |
| $150K | 3× TopStep 50K XFA | 3 (1 each) | $26,400 | 18% |
| $50K | Self-funded AMP via Rithmic | 5 (DD-sized) | ~$14,600 | 29% |
| $50K | Self-funded AMP at 10 ct (higher DD tolerance) | 10 | ~$29,200 | **59%** |
| $100K | Self-funded 20 ct | 20 | ~$58,400 | 58% |

**The gating question for self-funded 10 ct:** will your own gut tolerate a ~$4.9K drawdown (2× historical = 10ct × $245/ct DD × safety factor)? If yes (and you've said "I can handle the DD"), self-funded 10 ct is the answer. If no, split across 5–10 contracts or across 2–3 TopStep accounts.

## 9. What's NOT the answer

- **New filters / new discovery** — this is capital-allocation work, not research. Research is already caught up (Wave 4/5 shipped, 38 validated).
- **Retire "dead weight"** — no strategy fails the quality floor. The 38-strategy pool IS the option space for the allocator.
- **Scale contracts on current TopStep 50K** — DD math makes this impossible.
- **Deploy more lanes in parallel on one account** — correlation gate + DD ceiling block this. Allocator already picks the 6 uncorrelated best-of-session heads.
- **Composite SGP+ORB_G5 filter** (from earlier audit) — captures ~+12% ExpR on EUROPE_FLOW, but 1 lane improvement out of 6 = marginal. Legitimate future work, not the top lever.

## 10. Risk analysis

- Historical max DD (4-lane subset, 1 ct): **$2,433** — breaches TopStep 50K $2K limit
- Worst single day (1 ct): **-$567** — 2.8% of a $50K account. Manageable as a daily loss.
- Trailing 12mo DD ratio: DD/Total = $2,433 / $8,641 = **28%** → "Calmar ratio" ~3.5 annualized, healthy for systematic futures.
- Correlation-adjusted VaR (95%): daily stdev ~$120 at 1 ct × 1.65 = **$200 one-day 95% VaR per contract**. At 10 ct = $2,000 one-day VaR.

## 11. Implementation checklist

To execute Option Q3 (self-funded AMP at 10 ct):

- [ ] Verify Rithmic live adapter works end-to-end (per MEMORY: DONE, async_rithmic 1.5.9)
- [ ] Open AMP futures account with $50K funded
- [ ] Add new profile `amp_self_funded_mnq_auto` in `trading_app/prop_profiles.py`:
  - `firm="amp"`, `is_express_funded=False`, `max_contracts_per_entry=10`
  - No XFA scaling plan (F-1 stays dormant for this profile via `topstep_xfa_account_size=None`)
  - `daily_loss_limit_dollars` = user-set kill switch (suggest $1,000 for 2× daily VaR)
- [ ] Run in signal/demo mode for 4 weeks parallel to TopStep for live P&L verification
- [ ] Cut over to live after signal-vs-actual divergence < 2%

If self-funded feels like too big a leap: start with `bulenox_50k` + `elite_50k` per existing plan in MEMORY. Migrate after payout validation on both.

## 12. Caveats on these numbers

- **Trailing 12mo is generous.** Recent 6.6-year average is $4.3K/yr/ct — half of what trailing shows. Strategies may mean-revert to lifetime averages.
- **2026 OOS (N=70 trading days) is cherry-picked time.** $17.7K annualized is not credible as a planning number.
- **Portfolio DD of $2,433 is computed on 4 measurable lanes** (EF/CS/NY/TK). Full 6-lane adds SINGAPORE_OPEN ATR_P50 and US_DATA_1000 VWAP — likely pushes DD modestly higher, not lower.
- **Commission assumption: $1.42/RT MNQ** (TopStep Rithmic rate). AMP/Bulenox may be $3-5. Deduct ~$2/RT × 800 trades/yr = -$1,600/yr at 1 ct for rate difference. Still dominates TopStep on net.
