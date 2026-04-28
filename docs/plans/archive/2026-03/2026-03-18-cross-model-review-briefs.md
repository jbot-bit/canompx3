---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Cross-Model Adversarial Review Briefs

Paste each brief into its respective model. Collect responses. Compare.
Convergent findings = real. Divergent findings = investigate.

---

## Brief 1: GPT-4 — "Statistical Methodology Audit"

Paste into ChatGPT (GPT-4):

```
You are an adversarial quantitative finance reviewer. Your job is to find flaws in my methodology. Do not be polite. Do not validate. Assume everything is wrong until you prove otherwise.

SYSTEM OVERVIEW:
I built a systematic intraday futures trading system. Opening Range Breakout (ORB) strategies across 3 micro instruments (gold, Nasdaq, S&P) and 12 session windows per day. Grid search over ~123,000 strategy combinations.

VALIDATION PIPELINE:
1. Grid search: 123,404 canonical strategy combinations tested
2. BH FDR at alpha=0.05 across ALL 123K tests (hard gate — rejects, not tags)
3. Walk-forward: 6-month rolling OOS windows, WFE > 50% required
4. Noise floor: synthetic null test (10 seeds, Gaussian random walk) — reject if ExpR <= noise max for entry model
5. DSR (Deflated Sharpe Ratio, Bailey & Lopez de Prado 2014) — informational, not hard gate (N_eff unknown)

GRID SEARCH RESULTS:
- 123,404 canonical trials
- 18,666 have positive ExpR with N >= 30 trades
- 2,372 survive full validation pipeline (FDR + WF + noise floor)
- Cross-sectional variance of per-trade Sharpe ratios: V[SR] = 0.047

NULL TEST RESULTS (10 seeds, Gaussian random walk, zero drift):
- 611 total false-positive survivors across 10 seeds
- E2 entry (stop-market at ORB edge): 577 survivors, max ExpR = 0.316, avg null ExpR = -0.004
- E1 entry (limit at ORB edge): 34 survivors, max ExpR = 0.243, avg null ExpR = -0.118
- Noise floor set at noise MAX: E1 >= 0.25, E2 >= 0.32

DSR RESULTS:
- SR0 at N_eff=253 (edge family count): 0.618 per-trade Sharpe
- SR0 at N_eff=10: 0.343
- Zero production strategies pass DSR > 0.95 at any N_eff >= 10
- 4 pass at N_eff=5 (all from one session: CME_PRECLOSE)

PRODUCTION BEST STRATEGIES:
- MGC TOKYO_OPEN E2 RR4.0 ORB_G4_FAST5: ExpR=0.553, Sharpe_ann=1.08, N=89
- MNQ CME_PRECLOSE E2 RR1.0 VOL_RV12_N20: ExpR=0.374, Sharpe_ann=2.29, N=142
- Best E1: MGC TOKYO_OPEN E1 RR2.0 ORB_G5_FAST10: ExpR=0.270, Sharpe_ann=0.74, N=151

QUESTIONS:
1. My noise floor uses the MAX of 10 null seeds as the acceptance threshold. Is this the correct threshold for White's Reality Check, or should I use a percentile? How many seeds do I need for stable tail estimation?
2. The DSR kills everything at N_eff >= 10 but passes 4 strategies at N_eff = 5. My N_eff is unknown (somewhere between 5 and 611). How should I estimate it? Is the edge family count (611 total, 253 non-purged) the right proxy?
3. The Gaussian random walk null has no fat tails, no volatility clustering, no microstructure. Is this null too easy (meaning my noise floor is too LOW and I should be stricter) or too hard (meaning it creates patterns that real markets don't)?
4. BH FDR at alpha=0.05 across 123K correlated tests — is this valid? The strategies share underlying price data (correlation structure). Should I use BHY instead? Or is FDR the wrong framework entirely for "is my best strategy real?"
5. The top strategies have N=89-151 trades over 5+ years. Is this enough to distinguish real edge from noise? What minimum sample size does the literature recommend for intraday futures strategies?
6. Am I fooling myself? What's the probability this entire system is noise given the numbers above?

Be specific. Give me numbers, not platitudes. Reference papers if relevant.
```

---

## Brief 2: Gemini — "Entry Model Structural Bias Deep Dive"

Paste into Gemini:

```
You are an adversarial statistical reviewer specializing in multiple testing corrections and selection bias. I need you to evaluate a specific structural issue in my trading strategy validation.

THE PROBLEM:
I have two entry models for intraday futures breakout trades:
- E2 (stop-market): enters when price hits the ORB boundary. On random walk data, avg PnL = -0.004R (near breakeven). The entry mechanism costs almost nothing on noise.
- E1 (limit): enters at the ORB boundary via limit order. On random walk data, avg PnL = -0.118R. The entry mechanism costs real money on noise.

I ran 10 synthetic null tests (Gaussian random walk, zero drift, 6 years each). Results:
- E2: 577 false positive strategies survive the full pipeline. Max ExpR = 0.316.
- E1: 34 false positive strategies survive. Max ExpR = 0.243.

My current approach: set separate noise floors per entry model:
- E2 floor = 0.32 (ceil of noise max 0.316)
- E1 floor = 0.25 (ceil of noise max 0.243)

I also computed the Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014):
- V[SR] across 123K strategies = 0.047 (per-trade Sharpe variance)
- At N_eff=253 (edge family count), SR0 = 0.618 per-trade — NOTHING passes DSR > 0.95
- At N_eff=5 (absurdly generous), SR0 = 0.260 — 4 strategies pass

PRODUCTION DATA:
- E2: 1,912 strategies, avg ExpR=0.114, max=0.553 (MGC TOKYO_OPEN, N=89)
- E1: 460 strategies, avg ExpR=0.114, max=0.363
- Grid searched 123,404 combinations total

QUESTIONS:
1. Is setting separate noise floors per entry model methodologically correct? Or does the structural difference in E2 vs E1 null expectations mean I need a fundamentally different approach (e.g., standardizing ExpR by the entry model's null mean before comparing)?
2. The E2 null mean (-0.004R) means E2 strategies start from near-zero. A strategy with ExpR=0.10 has only overcome 0.104R of "friction" vs E1 which overcomes 0.218R. Should I define a "noise-adjusted ExpR" = ExpR - null_mean for cross-model comparison?
3. With 577 E2 noise survivors from 10 seeds, I can estimate the E2 noise distribution reasonably well. But 34 E1 survivors from 10 seeds — is that enough? What confidence interval should I put around the E1 noise max of 0.243?
4. The DSR formula uses a single V[SR] across all strategies. But E1 and E2 have very different Sharpe distributions (E1 mean SR ≈ -0.14, E2 mean SR ≈ 0.02). Should I compute V[SR] separately per entry model?
5. My best E2 strategy (ExpR=0.553, N=89) has an annualized Sharpe of 1.08. The null can produce Sharpe_ann up to ~2.09 (from noise!). The best E1 (ExpR=0.270, N=151) has Sharpe_ann of 0.74. Which one has stronger statistical evidence of being real, adjusted for their different null distributions?
6. The E2 structural bias (-0.004R) suggests stop-market entry at a breakout level is nearly free on random data. Does this mean ALL E2 breakout strategies are suspect by construction? Or does it mean E2 is a legitimate low-cost entry that provides edge when combined with real market structure?

Be precise. I want formulas, not intuition.
```

---

## Brief 3: Grok — "Kill-or-Keep Decision Framework"

Paste into Grok:

```
I need you to help me make a binary decision. No hedging, no "it depends." Kill or keep.

CONTEXT:
I built a systematic intraday futures trading system (micro gold, micro Nasdaq, micro S&P). Opening Range Breakout strategies, grid-searched across 123,404 combinations, validated with BH FDR, walk-forward, and a synthetic null test.

THE NUMBERS:
- 123,404 strategy combinations tested
- 2,372 pass all validation gates
- 253 independent edge families (clustered by overlapping trade days)
- Best strategy: ExpR=0.553 R per trade, Sharpe_ann=1.08, on 89 trades over 5 years
- Best family head: ExpR=0.270, Sharpe_ann=0.74, on 151 trades

NULL TEST (10 seeds, random walk):
- 611 fake strategies survive from pure noise
- Noise MAX ExpR: E2=0.316, E1=0.243
- After applying noise floor gate: 1 family survives (MGC TOKYO_OPEN E1, ExpR=0.270)
- The Deflated Sharpe Ratio (analytical) says ZERO pass at any reasonable N_eff

THE CONFLICT:
Two independent methods agree most is noise. But they disagree on the threshold:
- Noise floor (empirical, 10 seeds): 1-34 families survive depending on MAX vs P95
- DSR (analytical): 0-4 strategies survive depending on N_eff assumption
- The one family that survives the noise MAX has ExpR=0.270 but only 151 trades

ADDITIONAL CONTEXT:
- Null model is Gaussian random walk — too easy (no fat tails, no GARCH). Real noise would produce MORE false positives.
- 10 seeds isn't enough for stable tail estimation (need 50-100+)
- N_eff for DSR is unknown (could be 5 or 253 — massive uncertainty)
- I've already killed 4 instruments, 1 entry model, calendar effects, break quality metrics, and multiple filter types as no-go

THE DECISION:
Given these numbers, should I:
A) KEEP the project — trade the 1-34 surviving families, accept that most was noise, focus on what's real
B) KILL the project — the evidence is insufficient, the methodology has too many gaps, the honest answer is "not proven"
C) PAUSE — run more seeds (50-100), implement block bootstrap with real data, estimate N_eff properly, then decide

For each option, give me:
1. The probability you'd assign to "this system has real edge" given the data
2. What a quant fund would do with these numbers
3. What the decision looks like in 12 months under each scenario

No sugarcoating. I've spent 2+ years on this. Sunk cost doesn't matter. What do the numbers say?
```

---

## How to Use

1. Open ChatGPT (GPT-4), paste Brief 1. Save response.
2. Open Gemini, paste Brief 2. Save response.
3. Open Grok, paste Brief 3. Save response.
4. Bring all 3 responses back here. I'll synthesize convergent vs divergent findings.

Convergent findings across 3 different model families = high confidence.
Divergent findings = the models have different biases, investigate each.
