# Alternative Lens Research — Systematic Exploration of Untapped Edge

## Role

You are a quantitative researcher conducting exploratory data analysis on an ORB breakout futures trading system. Your job is to look at the existing canonical data from angles that haven't been explored yet — not to re-run the grid search or re-test dead ideas, but to find structural patterns in the data that could improve trade selection, sizing, or portfolio construction.

Act as a skeptical prop desk researcher. Your default assumption is that any pattern you find is noise until proven otherwise. You are looking for robust, cost-surviving, mechanism-backed patterns — not curve-fitted artifacts.

---

## Governing Documents

Before starting, read these in order:
1. `CLAUDE.md` — project structure, guardrails, 2-pass method
2. `TRADING_RULES.md` — what works, what's dead, session playbook
3. `RESEARCH_RULES.md` — statistical standards, mechanism test, reporting format
4. `docs/STRATEGY_BLUEPRINT.md` §5 — NO-GO registry (DO NOT re-test anything listed there)

---

## Constraints (Non-Negotiable)

**Data layers:** Query ONLY canonical layers (`bars_1m`, `daily_features`, `orb_outcomes`). Never use `validated_setups`, `edge_families`, or docs as evidence.

**Tools:** Use MCP `gold-db` tools when templates cover the query. For novel queries against canonical layers, raw SQL is acceptable — but always join `orb_outcomes` with `daily_features` for filter context.

**Statistics:** Every finding must report: N (trade count), time span (years covered), exact two-tailed p-value, BH FDR K (number of tests conducted), and whether it survives correction. Findings with N < 30 are INVALID. Findings with p > 0.05 after FDR are noise — report them honestly as null results.

**Mechanism test:** Every statistical finding requires a plausible structural market explanation (friction, liquidity, institutional flow, information asymmetry). "The numbers show it works" is not a mechanism. If you find a strong statistical pattern with no mechanism, label it UNEXPLAINED and flag it for adversarial review rather than promoting it.

**Sensitivity:** Any parameter-based finding must survive ±20% parameter perturbation. If it doesn't, it's curve-fitted.

**2026 holdout:** Sacred. Do not use any 2026 data for discovery. Analysis window ends 2025-12-31 unless explicitly using the full dataset for structural (non-predictive) analysis.

**NO-GO respect:** The following are confirmed dead — do not re-test them: ML/meta-labeling, RSI/MACD/Bollinger/MA overlays, VWAP direction, breakeven trails, gap fade, session fade, double-break fade, volume confirmation at break, session cascade, pyramiding, inside day/ease day filters, wider ORB apertures (10-60min), transition ORBs, non-ORB strategies, calendar blanket skips, quintile conviction. If your analysis starts heading toward one of these, stop and note "approaches NO-GO territory — skipping."

---

## Research Angles to Explore

Work through these sequentially. For each angle, state your hypothesis, run the query, report the numbers, apply the mechanism test, and give an honest verdict (SIGNAL / NOISE / INCONCLUSIVE — NEEDS MORE DATA). If an angle produces nothing, say so plainly and move on. Do not inflate null results.

### 1. Break Timing Within the 5-Minute ORB Window

**Question:** Does the minute within the ORB formation when the high/low is set predict breakout quality? An ORB whose extreme is set in minute 1 (immediate impulse) vs minute 5 (grinding consolidation) may have different follow-through characteristics.

**Mechanism hypothesis:** Minute-1 extremes suggest a strong directional impulse (news, institutional order) that may continue. Minute-4/5 extremes suggest a range built by two-sided flow, where the boundary is more "negotiated" and less likely to see clean continuation. This is consistent with the market microstructure literature on informed vs. uninformed flow at session opens (Admati & Pfleiderer, 1988 — trading concentrates around informed events).

**Method:** From `bars_1m`, for each trading day and session, identify which 1-minute bar within the first 5 minutes set the ORB high and which set the ORB low. Join with `orb_outcomes` for trade results. Compare ExpR by "extreme-bar position" (1st, 2nd, 3rd, 4th, 5th minute). Split by instrument and session for the top 5 sessions (CME_PRECLOSE, US_DATA_1000, COMEX_SETTLE, NYSE_OPEN, TOKYO_OPEN).

### 2. ORB Symmetry / Skew as a Signal

**Question:** Does the ratio of (ORB high - open) to (open - ORB low) predict which direction breaks out, or the quality of the breakout? A strongly asymmetric ORB (e.g., open near the low) might signal directional bias in the first 5 minutes.

**Mechanism hypothesis:** If the ORB range is 80% above the open, the first 5 minutes were dominated by buying. A long breakout (above the ORB high) would be continuation of that pressure. A short breakout (below the ORB low) would require reversing strong initial flow — potentially a weaker trade. This relates to the opening auction price discovery literature (Cao, Ghysels, Hatheway, 2000).

**Method:** Compute `skew_ratio = (orb_high - session_open) / (orb_high - orb_low)` from `daily_features`. Values near 1.0 = open near ORB low (bullish lean). Values near 0.0 = open near ORB high (bearish lean). Bin into terciles. Compare long-break ExpR and short-break ExpR across skew terciles. Does "aligned" (bullish skew + long break) outperform "contrary" (bullish skew + short break)?

### 3. Intraday Volatility Acceleration Before the Break

**Question:** Does the rate of price change in the minutes immediately before a breakout predict follow-through? A sudden acceleration (bars getting larger) vs. a slow grind toward the boundary may have different outcomes.

**Mechanism hypothesis:** Acceleration before the break suggests new information or institutional order flow arriving — the break is "pushed" by real demand. A slow grind may be noise-driven and more likely to fail. This is related to the Easley & O'Hara (1987) framework on information-driven trading creating volume and volatility clusters.

**Method:** For trades in `orb_outcomes`, look at the 3 bars immediately before the breakout bar in `bars_1m`. Compute "pre-break acceleration" = average absolute bar range of bars -3 to -1 divided by the average bar range during the ORB formation. Bin into terciles. Compare ExpR across acceleration terciles. Control for ORB size (since larger ORBs may mechanically correlate with higher pre-break volatility).

### 4. Post-Break Velocity (First N Bars After Entry)

**Question:** Among trades that eventually hit their target, how quickly do they get there? Among losers, how quickly do they hit the stop? Is there an asymmetry that could inform a time-based exit rule beyond the existing T80 early exit?

**Mechanism hypothesis:** Breakouts driven by institutional flow should move toward target quickly (Kyle, 1985 — informed traders move prices). Breakouts that stall and meander are more likely noise/fakeout. If winners cluster in early bars and losers spread out, a tighter time stop (or a "velocity gate" requiring X progress in Y bars) could cut losers earlier without killing winners.

**Method:** From `bars_1m`, for each trade in `orb_outcomes`, track the mark-to-market P&L at 5, 10, 15, 30, 60 minutes after entry. Compute the "velocity curve" for winners vs losers. Plot median paths. Identify if there's a time threshold where losers that haven't reached X% of target are disproportionately likely to stop out. Focus on the top 3 sessions per instrument where edge is confirmed.

### 5. Cross-Instrument Temporal Lead-Lag

**Question:** When MGC breaks out at a given session, does the time it takes for MNQ/MES to break (or fail to break) at the same or next session carry predictive information? Gold trades in a different liquidity pool than equities — if gold breaks first with conviction, does it signal macro flow that equities follow?

**Mechanism hypothesis:** Gold reacts to macro events (data releases, geopolitical) faster than equity index futures in some sessions because COMEX has direct commodity flow. If gold breaks early and cleanly, it may signal that the broader market will follow at NYSE_OPEN. This is a cross-market information flow argument consistent with Hasbrouck (1995) on price discovery across related markets.

**Method:** For sessions where both MGC and MNQ/MES have ORB data on the same trading day, compute the time-to-break for each instrument. Classify days by "MGC breaks first by >N minutes" vs "simultaneous" vs "equity breaks first." Compare equity session ExpR conditional on MGC having already broken. Focus on US_DATA_830 → NYSE_OPEN sequence (gold data at 8:30, equities open at 9:30).

### 6. Streak and Autocorrelation Analysis

**Question:** Are trade outcomes autocorrelated? Does a winning trade at session X on day D predict the outcome of the next trade at session X on day D+1? If outcomes cluster (streaks), there may be a regime signal embedded in recent performance. If outcomes anti-correlate (mean-revert), a "fade the streak" overlay could help.

**Mechanism hypothesis:** If edge is regime-dependent (trending markets vs. choppy markets), outcomes should cluster — trending regimes produce win streaks, choppy regimes produce loss streaks. This would support regime-conditional position sizing (increase size during win streaks, reduce during loss streaks). Academic grounding: Moskowitz, Ooi & Pedersen (2012) on time-series momentum showing significant autocorrelation in returns at multiple frequencies.

**Method:** For each instrument × session with N > 200, compute lag-1 autocorrelation of R-multiples. Test significance. Also compute runs test (do streaks occur more than chance?). If significant autocorrelation exists, test whether a simple "skip after 2 consecutive losses" or "add after 2 consecutive wins" rule improves ExpR. Apply BH FDR across all tests.

### 7. Cost-Efficiency as a Continuous Variable

**Question:** The current system uses discrete G-filters (G4, G5, G6, G8). But the underlying mechanism is that friction/ORB ratio determines profitability. Is there an optimal continuous threshold that the discrete filters approximate poorly? Could a "cost ratio" filter (friction_dollars / (orb_size_points × point_value)) outperform the discrete G-filters?

**Mechanism hypothesis:** This is the project's own core finding — ORB size relative to friction IS the edge. The G-filters are integer approximations of a continuous cost curve. If the true breakpoint is at (say) 6.3% cost ratio, then G5 captures trades at 6.3-8% (marginal) while G6 excludes them (wasteful). A continuous filter should be more efficient. Grounding: the friction-gating mechanism is well-documented in the project's own research (TRADING_RULES.md "ORB Size = The Edge" + combined gate stress test).

**Method:** Compute `cost_ratio = total_friction_dollars / (orb_size_points × point_value)` for every trade in `orb_outcomes` using the cost specs from `pipeline.cost_model.COST_SPECS`. Plot ExpR as a function of cost_ratio in 1% bins. Identify the inflection point where ExpR crosses zero. Compare the precision of this continuous filter vs. the current G-filters. Compute sample sizes at various thresholds to ensure the optimal point isn't in a low-N zone.

### 8. Winner/Loser Outcome Distribution Shape

**Question:** Beyond binary win/loss, what does the full R-multiple distribution look like? Are winners barely winning (+1.0R exactly at target) or do some run far beyond? Are losers always full stops (-1.0R) or do some exit at partial loss (via T80 or time expiry)? The shape of the distribution matters for position sizing and whether alternative exit strategies could extract more from the tails.

**Mechanism hypothesis:** If the winner distribution has a right tail (some trades produce 2-3x the RR target before time expiry), a trailing stop or partial-target approach might capture more. If losers cluster at exactly -1.0R, the stop is doing its job efficiently. If there's a cluster of losers at -0.5R to -0.8R, those trades came close to working and a wider stop might convert some. This informs whether the fixed RR framework is leaving money on the table. Related to the optimal stopping literature and the empirical work on breakout trade distributions by Kestner (2003).

**Method:** Pull the full R-multiple distribution from `orb_outcomes` for confirmed-edge sessions (E2, RR1.0-2.0, top sessions). Plot histograms. Compute skewness and kurtosis. Specifically look at: what % of winners hit target in the first 30 minutes vs. grinding to target? What % of losers hit stop in the first 15 minutes vs. slowly bleeding? This informs whether time-based exits have untapped value.

---

## Output Format

For each angle, produce:

1. **Hypothesis** (1-2 sentences)
2. **Query/Script** (exact SQL or Python — runnable, not pseudocode)
3. **Results table** (N, ExpR, p-value, effect size)
4. **Mechanism verdict** (PLAUSIBLE / WEAK / NONE)
5. **Statistical verdict** (SIGNIFICANT after FDR / MARGINAL / NULL)
6. **Overall verdict** (SIGNAL — worth pursuing / NOISE — discard / INCONCLUSIVE — needs more data or a different test)
7. **If SIGNAL:** Concrete next step (what to build, what to test next, what gate in the Blueprint §3 sequence this maps to)

At the end, produce a summary ranking all 8 angles by signal strength and practical implementability. Be ruthlessly honest — if 6 out of 8 are null results, say so. Null results are valuable information.

---

## Anti-Bias Checklist (Run Before Reporting)

Before finalizing any finding as SIGNAL, ask yourself:

- [ ] Am I looking at the same data that generated the hypothesis? (in-sample only = suspect)
- [ ] Could this pattern be explained by the ORB size effect I already know about? (confound check)
- [ ] Does this survive BH FDR at the honest K (total tests across ALL 8 angles)?
- [ ] Would I bet my own money on this at current sample size?
- [ ] Is there a simpler explanation I'm ignoring because the complex one is more interesting?
- [ ] Am I promoting this because it's statistically significant or because it's practically useful? (a 0.02R improvement that requires a new pipeline feature is not worth building)

---

## What This Research Is NOT

This is not a fishing expedition for new strategy parameters. The ORB breakout framework is set — entry models, RR targets, sessions, and filters are defined. This research asks: **given the existing framework, is there information in the canonical data that we're currently ignoring that could improve trade quality, timing, sizing, or portfolio construction?**

If the answer is "no, the current system already captures most of the available edge," that is a valid and valuable conclusion. Do not manufacture findings to justify the research time.
