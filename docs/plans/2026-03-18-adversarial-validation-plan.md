# Adversarial Validation Plan — Breaking the Echo Chamber

**Date:** 2026-03-18
**Status:** PLAN (not implemented)
**Problem:** The entire system was built by one person + AI assistants that were trained on that person's assumptions. Nobody external has challenged the core thesis, statistical methodology, or implementation correctness.

---

## 1. External Human Review ($200-500)

**What:** Hire a quant practitioner for a 2-hour adversarial review of the methodology.

**Brief to post (QuantConnect forums / Toptal / Upwork):**

> I built a systematic futures trading system (micro gold, Nasdaq, S&P, Russell) using ORB (Opening Range Breakout) strategies across 12 intraday sessions. I'm looking for someone to tear apart my methodology — not the code, just the statistical approach. I WANT you to find problems.
>
> **What I did:**
> - Grid search over ~2,376 strategy combinations (3 entry models × 4 instruments × 12 sessions × 3 ORB apertures × 3 confirm bars × filter types × RR targets)
> - Benjamini-Hochberg FDR correction at α=0.05 across all combinations
> - Walk-forward validation: 6-month rolling OOS windows, WFE > 50% required
> - Classification: CORE (≥100 trades), REGIME (30-99), INVALID (<30)
> - Cost model: spread + slippage + commission per instrument from broker data
> - Edge families: cluster overlapping strategies by trade-day hash, pick head by ExpR
> - Fitness monitoring: 18-month rolling Sharpe, WATCH/DECAY classification
>
> **What I'm worried about:**
> - Is BH FDR at α=0.05 sufficient for 2,376 tests, or should I use something stricter?
> - Is 100 trades enough for CORE classification with daily micro futures data?
> - My walk-forward uses calendar windows — should I use trade-count windows?
> - I have 5-7 years of data depending on instrument. Is that enough?
> - The "mechanism test" (every finding needs a market structure reason) — is this real rigor or just storytelling with extra steps?
> - Am I fooling myself?
>
> **Data available for your review:**
> - RESEARCH_RULES.md (full statistical methodology)
> - Sample FDR results (before/after correction)
> - Walk-forward efficiency distribution
> - List of NO-GOs (things I tested and rejected)
>
> Budget: $200-500 for 2 hours of written adversarial feedback.

**Where to post:**
- QuantConnect community forums (free, but may get generic advice)
- Elite Trader "Strategy Development" subforum (free, experienced traders)
- Toptal quant network ($$$, but professional)
- Upwork "quantitative finance" category (mid-range)
- r/algotrading (free, variable quality)

---

## 2. Fresh AI Adversarial Review (Zero Context)

**What:** Use a brand new AI session with NO project memory, rules, or assumptions. Feed it only the methodology documents and ask it to find holes.

**Procedure:**
1. Open a new Claude session (not this project — different working directory, no CLAUDE.md)
2. Or use a different model entirely (GPT-4, Gemini, Grok) for cross-model diversity
3. Paste in ONLY:
   - RESEARCH_RULES.md
   - A summary of the grid search approach (parameters, counts, FDR method)
   - The NO-GO list (what was rejected)
   - 5-10 example validated strategies with their stats
4. Prompt: "You are an adversarial reviewer hired to find flaws. Assume everything is wrong until proven otherwise. Find the statistical, methodological, and logical holes in this approach. Do not be polite."

**Why this works:** The fresh session has no memory files, no TRADING_RULES.md, no "this session works because Asian gold demand" stories. It judges the methodology cold.

**What to check across models:**
- Do different models find the same holes? (convergence = real problem)
- Do different models find different holes? (divergence = blind spots)
- Does any model challenge the fundamental premise (ORB breakouts in micros)?

---

## 3. Independent Backtester Cross-Validation

**What:** Run the same signals through a completely separate backtesting engine to verify the numbers match.

**Options:**
- **QuantConnect Lean** (C#, free, institutional-grade) — import the same trade signals and verify P&L matches
- **Zipline / Zipline-reloaded** (Python, free) — closest to our stack
- **Backtrader** (Python, free) — simple, quick to set up
- **Manual spot-check** — pick 10 specific trades from the backtest, manually verify entry/exit prices against raw 1-minute bar data

**What to verify:**
- Do the same entry signals fire on the same bars?
- Are fills at the same prices? (E2 stop-market vs E1 limit — fill assumptions matter)
- Is the P&L per trade within $1 of our pipeline's calculation?
- Are cost deductions identical?

**Minimum viable check:**
Pick 20 trades across 4 instruments, manually trace each one through raw bars. If even 1 trade has a wrong fill price, wrong stop, or wrong P&L, the pipeline has a bug that invalidates aggregate stats.

---

## 4. Paper Trading Reality Check (3 Months)

**What:** Run live paper trading for 3 months and compare every fill against backtest predictions.

**Metrics to track:**
- **Fill rate:** What % of backtest signals actually execute in paper? (If backtest shows 100% and paper shows 70%, there's a fill assumption problem)
- **Slippage delta:** Actual fill price vs theoretical entry price. If consistently worse, cost model is wrong.
- **Signal match rate:** Do the same sessions fire the same direction on the same days?
- **P&L per trade:** Backtest prediction vs paper actual. Track the distribution of the gap.

**Kill criteria:**
- If paper P&L is < 50% of backtest P&L after 100+ trades → methodology is likely overfit
- If fill rate < 80% → entry model assumptions are wrong
- If slippage is > 2x modeled → cost model is wrong
- If signal match rate < 90% → there's a data/timing bug

**Already planned:** Apex Phase 1 (manual, 1 account, 1 micro). This IS the paper trading phase. But it needs the tracking infrastructure above, not just "trade and hope."

---

## 5. Statistical Methodology Audit

**What:** Independent verification that the statistical methods are correctly implemented.

### Tests to run:
1. **FDR injection test:** Insert 50 known-null strategies (random entry/exit) into the grid. Run FDR. Verify that all 50 are rejected. If any survive, the FDR implementation has a bug.

2. **Walk-forward shuffled baseline:** Shuffle trade dates randomly (break temporal structure). Run walk-forward. If shuffled strategies still pass WF, the validation is not detecting overfitting.

3. **Monte Carlo null distribution:** Generate 10,000 random strategies. How many pass the full pipeline (FDR + WF + min samples)? This is the false discovery rate of the PIPELINE ITSELF, not individual tests. If > 5% pass, the pipeline is too permissive.

4. **Sharpe ratio audit:** Compute Sharpe using the simple formula AND the Jobson-Korkie corrected formula. Compare. If they diverge significantly, the reported Sharpes are misleading.

5. **Autocorrelation check:** Test whether consecutive trade outcomes are autocorrelated (Durbin-Watson or Ljung-Box). If yes, the effective sample size is smaller than N, and all significance tests are inflated.

---

## 6. Core Premise Challenge

**The question nobody has asked:** Does ORB breakout actually work in micro futures?

### Evidence FOR:
- FDR survivors exist after honest correction
- Walk-forward passes (strategies work out-of-sample)
- Multiple instruments show edge (not just one lucky series)
- Mechanism exists (institutional order flow at session boundaries)

### Evidence AGAINST (be honest):
- No live P&L yet
- Micro futures have wider relative spreads than E-mini (cost drag is proportionally higher)
- ORB is one of the most researched retail strategies — if it works, why isn't it arbed away?
- The data period (2019-2026) includes a historic bull run, COVID, and unprecedented vol — is this representative?
- Session times shift with DST — the "same" session is a different market structure in winter vs summer

### How to challenge:
- Search for published research on ORB profitability post-transaction-costs
- Check if any fund/CTA publicly reports ORB-style strategies
- Ask the human reviewer (step 1) specifically: "In your experience, do intraday breakout strategies survive transaction costs in micro futures?"

---

## Priority Order

| # | Action | Cost | Time | Echo-Breaking Power |
|---|--------|------|------|-------------------|
| 1 | Fresh AI adversarial review (zero context) | Free | 1 hour | Medium — different bias, same species |
| 2 | FDR injection test + Monte Carlo null | Free | 4 hours | High — tests the pipeline, not the strategies |
| 3 | Manual 20-trade spot-check | Free | 2 hours | High — catches fill/cost bugs directly |
| 4 | External human quant review | $200-500 | 1 week | Very high — genuinely independent |
| 5 | Paper trading comparison | Free | 3 months | Highest — reality is the ultimate test |
| 6 | Independent backtester cross-validation | Free | 1-2 days | High — catches implementation bugs |

---

## The Meta-Rule

**No AI that has access to this project's memory, rules, or history should be the final arbiter of whether this project works.** The validator must be independent of the thing being validated. This plan exists to establish that independence.
