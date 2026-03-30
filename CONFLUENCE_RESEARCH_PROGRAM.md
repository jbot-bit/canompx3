# Confluence Research Program — Complete Execution Prompt

## Context for Claude

You are executing a structured research program on a futures ORB breakout trading system. The system declared "ML is DEAD" after V2 meta-labeling (RandomForest, 5 features, 108 configs) produced 0/12 BH FDR survivors. A full audit (ML_AUDIT_PROMPT.md) revealed that the individual features were never tested as standalone signals — the system jumped from "features in RF" to "ML is dead" without the intermediate step.

This prompt runs that intermediate step — and everything that follows from it — as a single sequential research program with hard gates between phases. No phase starts until the prior phase passes its gate. No decisions are made from memory — every claim requires a query, a number, and a statistical test.

**Read before executing:** `CLAUDE.md`, `TRADING_RULES.md`, `RESEARCH_RULES.md`, `docs/STRATEGY_BLUEPRINT.md`, `ML_AUDIT_PROMPT.md` (the audit that generated this program).

**Governing constraint:** 2026 is a sacred holdout year. All discovery uses pre-2026 data only. No exceptions.

---

## PHASE 0: Establish Ground Truth (30 min)

**Purpose:** Before testing anything, confirm what data exists, which sessions have positive baselines, and what features are available. Every later phase depends on this.

### Step 0.1 — Baseline census

Query `orb_outcomes` for each (instrument, session, orb_minutes, entry_model, rr_target) combination. For each:
- N (total trades with non-NULL outcome, pre-2026 only)
- Win rate
- ExpR (mean pnl_r)
- t-stat and p-value (one-sample t-test, H0: ExpR = 0)

**Output:** A table of ALL baselines. Mark each as POSITIVE (ExpR > 0, p < 0.10) or NEGATIVE.

**Gate 0.1:** Identify the set of POSITIVE baselines. These are the ONLY populations used in Phases 1-3. If fewer than 5 positive baselines exist across all instruments, STOP — the system doesn't have enough positive ground to test confluences on.

### Step 0.2 — Feature availability census

For each feature below, query `daily_features` to confirm:
- Column exists (not just in schema — has non-NULL values)
- Coverage: what % of positive-baseline trading days have non-NULL values?
- Distribution: min, p25, median, p75, max (sanity check — are values reasonable?)

**Feature list (11 core + 6 proximity + 2 secondary = 19 total):**

V2 core (5):
1. `orb_{SESSION}_size` (normalized by `atr_20` → `orb_size_norm`)
2. `atr_20_pct`
3. `gap_open_points` (normalized by `atr_20` → `gap_open_points_norm`)
4. `orb_{SESSION}_pre_velocity` (normalized by `atr_20` → `orb_pre_velocity_norm`)
5. `prior_sessions_broken` (cross-session count)

Previously in ML but dropped (6):
6. `orb_{SESSION}_volume`
7. `rel_vol_{SESSION}` (relative volume at break bar)
8. `orb_{SESSION}_compression_z`
9. `atr_vel_ratio` (compressed spring)
10. `prev_day_range` (normalized by `atr_20`)
11. `orb_{SESSION}_vwap`

Level proximity (6 — from `features.py`, not in `daily_features` directly; must be computed):
12. `nearest_level_to_high_R`
13. `nearest_level_to_low_R`
14. `levels_within_1R`
15. `levels_within_2R`
16. `orb_nested_in_prior`
17. `prior_orb_size_ratio_max`

Session-conditional (2 — clean for US sessions only):
18. `overnight_range` (ONLY for sessions starting after 17:00 Brisbane: LONDON_METALS through NYSE_CLOSE)
19. `overnight_range_pct`

**Gate 0.2:** Any feature with < 80% coverage on positive-baseline days is flagged SPARSE. Sparse features can still be tested but results carry a caveat. Features with < 50% coverage are EXCLUDED.

### Step 0.3 — Look-ahead verification

For EACH feature that passed Gate 0.2, verify it has NO look-ahead contamination for the session being tested:
- Is the feature value determined BEFORE the ORB window opens? (Pre-session features: gap, ATR, prev_day — always clean)
- Is the feature value determined BEFORE the break signal? (ORB-window features: orb_size, orb_volume, orb_vwap — clean at ORB close, before break)
- Does the feature use information from AFTER the break? (break_bar_volume, break_delay — AT-BREAK, excluded from this program)
- Does the feature use information from a concurrent or later session? (cross-session features — verify chronological ordering)
- Session-conditional features (overnight_range): explicitly confirm which sessions are clean and which are contaminated.

**Output:** A verified feature × session eligibility matrix. Each cell is CLEAN, CONTAMINATED, or EXCLUDED.

**Gate 0.3:** Any feature × session pair marked CONTAMINATED is permanently excluded from all later phases. No exceptions.

---

## PHASE 1: Univariate Signal Scan (2-3 hours)

**Purpose:** For each feature, test whether it has standalone predictive signal on positive-baseline populations. This is the step that was skipped before ML.

### Step 1.1 — Quintile split

For each (feature, positive-baseline) pair:

1. Take ALL trades from `orb_outcomes` joined to `daily_features` (triple-join: trading_day + symbol + orb_minutes) where:
   - `trading_day < '2026-01-01'`
   - Feature value is not NULL
   - The session has a POSITIVE baseline (from Step 0.1)

2. Split into 5 equal-sized quintiles by feature value. Record quintile boundaries.

3. For each quintile, compute:
   - N (trade count)
   - ExpR (mean pnl_r)
   - Win rate
   - Median MAE (median mae_r)
   - Median MFE (median mfe_r)
   - Sum R (total pnl_r)

4. Compute the **monotonic spread**: ExpR(Q5) - ExpR(Q1). Sign matters — positive means higher feature values predict better outcomes.

5. Statistical test: Two-sample t-test on per-trade `pnl_r` between Q5 and Q1. Record t-stat, p-value (two-tailed), and Cohen's d (effect size).

**Output:** A table with one row per (feature, session, aperture, rr_target) showing: Q1 ExpR, Q5 ExpR, spread, t-stat, p-value, Cohen's d, N per quintile.

### Step 1.2 — Multiple testing correction

Apply Benjamini-Hochberg FDR at q = 0.10 across ALL (feature × session × aperture × rr_target) tests from Step 1.1.

**K = total number of tests run.** Report K explicitly. Do NOT adjust K after the fact.

For each test, report:
- Raw p-value
- BH rank
- BH threshold (rank/K × q)
- PASS / FAIL

### Step 1.3 — Monotonicity check

For features that pass BH FDR:
- Is the Q1 → Q5 progression monotonic? (Strictly increasing or decreasing ExpR across quintiles)
- Or is the signal concentrated in one extreme quintile? (Q5 >> Q4 ≈ Q3 ≈ Q2 ≈ Q1 = "threshold effect")
- A threshold effect suggests a binary filter (top quintile vs rest), not a continuous signal.

**Gate 1:**
- **0 features pass BH FDR** → Features genuinely have no standalone signal. ML is truly dead. Close the confluence research program. Update NO-GO registry.
- **1-3 features pass** → Proceed to Phase 2 with ONLY the passing features.
- **4+ features pass** → Proceed, but flag potential overfitting (many discoveries at q=0.10 suggests some are false positives). Consider tightening to q=0.05 and re-running.

---

## PHASE 2: Distributional Analysis (1-2 hours)

**Purpose:** For features that passed Phase 1, test whether they shift the SHAPE of the PnL distribution — not just the mean. This answers: "Do confluences change trade quality in ways that binary win/loss classification misses?"

### Step 2.1 — MAE distribution comparison

For each surviving feature:
1. Split trades into HIGH (top quintile) and LOW (bottom quintile) by feature value
2. Compare MAE distributions:
   - KDE plot (overlay HIGH vs LOW)
   - Two-sample KS test (p-value)
   - Median MAE difference
   - 90th percentile MAE difference (tail behavior)

**Interpretation:** If HIGH-feature trades have systematically lower MAE (less drawdown before resolution), the confluence improves position quality even if it doesn't change win rate.

### Step 2.2 — MFE distribution comparison

Same as 2.1 but for MFE (maximum favorable excursion):
- Do HIGH-feature trades reach further into profit before reversing?
- Is the MFE distribution fatter-tailed for HIGH-feature trades? (More home runs)

### Step 2.3 — PnL distribution comparison

Full PnL distribution (not just mean):
- KDE overlay
- KS test
- Skewness comparison (do HIGH-feature trades have more positive skew?)
- Kurtosis comparison (thinner tails = more predictable outcomes)

**Gate 2:**
- Features where KS p < 0.05 on at least one of (MAE, MFE, PnL): **CONFIRMED distributional signal.** Proceed to Phase 3.
- Features where all KS p > 0.05: Mean shift only (already captured by Phase 1). Still proceed to Phase 3 for filter design, but note the signal is MEAN-ONLY.

---

## PHASE 3: Filter Design & Walk-Forward Validation (3-4 hours)

**Purpose:** Convert surviving features into deployable binary filters. Validate out-of-sample using walk-forward.

### Step 3.1 — Threshold optimization (in-sample)

For each surviving feature:
1. Use the quintile boundaries from Phase 1 as candidate thresholds
2. For each threshold (e.g., "feature >= Q4 boundary"):
   - Compute: N_pass, N_fail, ExpR_pass, ExpR_fail, Win%_pass, Win%_fail
   - Compute the COST of the filter: how many positive-ExpR trades does it exclude?
   - Compute the BENEFIT: how much negative-ExpR mass does it remove?

3. Select the threshold that maximizes (ExpR_pass × sqrt(N_pass)) — balancing edge improvement against sample loss.

**Do NOT optimize on Sharpe.** Sharpe is biased under multiple testing (per RESEARCH_RULES.md).

### Step 3.2 — Walk-forward validation

For each filter from Step 3.1:
1. Split data into 3 non-overlapping time blocks (approximately equal, chronological)
2. Train (pick threshold) on block 1+2, test on block 3
3. Train on block 1, test on block 2
4. Train on block 2+3, test on block 1
5. Record WFE (Walk-Forward Efficiency) = OOS_ExpR / IS_ExpR

**Gate 3:**
- **WFE >= 0.50:** Filter is DEPLOYABLE. Proceed to Phase 4.
- **WFE 0.30-0.49:** Filter is MARGINAL. Flag for monitoring but do not deploy.
- **WFE < 0.30:** Filter is OVERFIT. Kill it. Record in NO-GO registry with evidence.

### Step 3.3 — Interaction test (conditional on Phase 3 survivors)

If 2+ filters survive Walk-Forward:
1. Test pairwise: does Filter A + Filter B together improve on either alone?
2. Check for redundancy: if Filter A and Filter B select nearly the same trades (Jaccard overlap > 0.80), keep only the one with higher WFE.
3. Check for conflict: if Filter A and Filter B select nearly opposite trades (Jaccard < 0.20), they may represent different regimes — test each independently.

---

## PHASE 4: Integration Decision (1 hour)

**Purpose:** Decide what to do with the results.

### Decision tree:

**Path A: 0 features survive Phase 1 (no univariate signal)**
→ Confluences genuinely don't have standalone signal in this data.
→ "ML is dead" becomes "confluences are dead" — a stronger, more honest verdict.
→ Update STRATEGY_BLUEPRINT.md NO-GO registry.
→ Focus shifts to: more data (tick data for order flow), more instruments, or execution improvement.
→ Do NOT revisit for 12 months or until 2x data is available.

**Path B: Features survive Phase 1 but fail Phase 3 (signal exists, not deployable)**
→ The features have statistical signal but it doesn't survive walk-forward.
→ Most likely: the signal is real but too weak to be profitable after costs, OR it's regime-dependent and the regime shifted between IS and OOS periods.
→ Record the feature signals in TRADING_RULES.md as "CONFIRMED SIGNAL, NOT DEPLOYABLE" with exact stats.
→ Revisit when data doubles (organic growth ~2029, or tick data purchase).
→ Do NOT attempt ML on these features — if they can't survive as simple filters, RF won't help.

**Path C: 1+ features survive Phase 3 (deployable filter)**
→ Integrate into the existing filter grid (alongside G8, ATR70_VOL, etc.).
→ Run `strategy_discovery.py` with new filter(s) included.
→ Run full validation pipeline (`strategy_validator.py`).
→ Pre-register for 2026 forward test (if not already using all 3 pre-registered slots).
→ Update TRADING_RULES.md, STRATEGY_BLUEPRINT.md, and config.py.
→ ONLY THEN consider ML: if 3+ deployable filters exist, meta-labeling with these as binary features (not continuous) may outperform individual filters. But this is Phase 5 — and it requires its own pre-registration.

**Path D: Distributional signal only (Phase 2 passes, Phase 1 marginal)**
→ The feature shifts MAE/MFE distributions but doesn't reliably shift the mean.
→ This is a POSITION MANAGEMENT signal, not an entry signal.
→ Design as: early exit modifier or position sizing input, not a trade filter.
→ Requires separate research track (not covered by this program).

---

## EXECUTION RULES

1. **Sequential, not parallel.** Phase N must complete and pass its gate before Phase N+1 starts. No skipping.

2. **Every number comes from a query.** Do not cite stats from memory, docs, or prior sessions. Run the query. Show the output. Then interpret.

3. **Pre-2026 data only.** All queries filter `trading_day < '2026-01-01'`. No exceptions. 2026 is sacred holdout.

4. **Triple-join always.** Every query joining `orb_outcomes` to `daily_features` must join on ALL THREE keys: `trading_day`, `symbol`, `orb_minutes`. Missing `orb_minutes` triples the row count and creates fake correlations.

5. **Report K honestly.** When applying BH FDR, K = total tests run in that phase. Do not subset K to "the interesting ones." Do not add tests after seeing results (that changes K).

6. **Show your work.** For each statistical test: state the hypothesis, show the query, show the numbers, compute the test statistic, report the p-value, apply correction, state the conclusion. No shortcuts.

7. **Null results are results.** If Phase 1 finds zero signal, that's a valuable finding — it upgrades "ML is dead" to "confluences are dead." Report it with the same rigor as a positive finding. Do not spin it.

8. **No scope creep.** This program tests the 19 features listed in Phase 0. If you discover a new feature idea during execution, write it down in a "FUTURE IDEAS" section at the bottom of the output. Do NOT test it in this run — it would inflate K without being pre-registered.

9. **Feature × session eligibility matrix governs everything.** If a feature is marked CONTAMINATED for a session in Step 0.3, it stays excluded for that session in ALL later phases. No second-guessing.

10. **Save all intermediate outputs.** Every phase produces a CSV or table that the next phase consumes. Save to `research/output/confluence_program/` with phase prefix (e.g., `phase0_baselines.csv`, `phase1_quintiles.csv`). This creates a reproducible audit trail.

---

## WHAT THIS PROGRAM DOES NOT COVER

- **Tick data / order flow features:** Not available in current data. Separate research track.
- **Multi-timeframe trend alignment:** Not computed in daily_features. Requires new feature engineering.
- **Volume profile / POC / value area:** Requires tick data. Not available.
- **ML reopening:** Explicitly deferred until Phase 4 decision. If you reach Phase 4 Path C with 3+ deployable filters, ML reopening becomes a separate pre-registered program.
- **2026 forward testing:** Covered by existing pre-registration (3 strategies). New filters from this program need their own pre-registration before touching 2026 data.
- **Execution modeling improvements:** Separate track. Not blocked by this program.

---

## SUCCESS CRITERIA

This program succeeds if it produces ONE of these outcomes:

1. **"Confluences are dead"** — No feature has univariate signal on positive baselines. Clean kill. The system's edge is ORB size + ATR regime + session selection, full stop. All future ML attempts are blocked without new data sources.

2. **"N features have signal, M are deployable"** — Specific features identified, thresholds set, walk-forward validated. Ready to integrate into the filter grid and pre-register for forward test.

3. **"Distributional signal exists for position management"** — Features shift MAE/MFE without reliably shifting the mean. Separate research track opened for early-exit or sizing research.

Any of these is a win. The only failure mode is ambiguity — "maybe there's signal but we're not sure." This program is designed to make that impossible by requiring hard statistical gates at every phase.
