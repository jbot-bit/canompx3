# Confluence Research Program — Complete Execution Prompt

## Context for Claude

You are executing a structured research program on a futures ORB breakout trading system. The system declared "ML is DEAD" after V2 meta-labeling (RandomForest, 5 features, 108 configs) produced 0/12 BH FDR survivors. A full audit revealed that individual features were never tested as standalone signals — the system jumped from "features in RF" to "ML is dead" without the intermediate step.

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

**Feature list (19 candidates):**

V2 core (5):
1. `orb_{SESSION}_size` / `atr_20` → `orb_size_norm`
2. `atr_20_pct`
3. `gap_open_points` / `atr_20` → `gap_open_points_norm`
4. `orb_{SESSION}_pre_velocity` / `atr_20` → `orb_pre_velocity_norm`
5. `prior_sessions_broken` (cross-session count — compute from break_dir columns)

Previously in ML but dropped (6):
6. `orb_{SESSION}_volume`
7. `rel_vol_{SESSION}`
8. `orb_{SESSION}_compression_z`
9. `atr_vel_ratio`
10. `prev_day_range` / `atr_20`
11. `orb_{SESSION}_vwap`

Level proximity (6 — computed by `trading_app/ml/features.py`, not stored in daily_features):
12. `nearest_level_to_high_R`
13. `nearest_level_to_low_R`
14. `levels_within_1R`
15. `levels_within_2R`
16. `orb_nested_in_prior`
17. `prior_orb_size_ratio_max`

Session-conditional (2 — clean for US sessions only, starting LONDON_METALS onward):
18. `overnight_range`
19. `overnight_range_pct`

**Gate 0.2:** Features with < 80% coverage = SPARSE (flagged). Features with < 50% coverage = EXCLUDED.

### Step 0.3 — Look-ahead verification

For EACH feature that passed Gate 0.2, verify no look-ahead contamination for the session being tested. Build a feature × session eligibility matrix where each cell is CLEAN, CONTAMINATED, or EXCLUDED.

Key checks:
- Pre-session features (gap, ATR, prev_day): always clean
- ORB-window features (orb_size, volume, vwap): clean at ORB close, before break
- Cross-session features (prior_sessions_broken): verify chronological ordering handles concurrent sessions (TOKYO_OPEN/BRISBANE_1025 overlap)
- Session-conditional (overnight_range): CONTAMINATED for CME_REOPEN, TOKYO_OPEN, SINGAPORE_OPEN. CLEAN for LONDON_METALS through NYSE_CLOSE.

**Gate 0.3:** CONTAMINATED cells are permanently excluded from all later phases. No exceptions.

---

## PHASE 1: Univariate Signal Scan (2-3 hours)

**Purpose:** Test whether each feature has standalone predictive signal on positive-baseline populations. This is the step that was skipped before ML.

### Step 1.1 — Quintile split

For each (feature, positive-baseline session, aperture, rr_target) where the feature × session cell is CLEAN:

1. Query `orb_outcomes` JOIN `daily_features` (triple-join: trading_day + symbol + orb_minutes) where:
   - `trading_day < '2026-01-01'`
   - Feature value is not NULL
   - Baseline is POSITIVE (from Step 0.1)

2. Split into 5 equal-sized quintiles by feature value. Record boundaries.

3. Per quintile compute: N, ExpR, Win%, Median MAE, Median MFE, Sum R

4. Monotonic spread: ExpR(Q5) − ExpR(Q1). Sign matters.

5. Statistical test: Two-sample t-test on per-trade `pnl_r` between Q5 and Q1. Report t-stat, p-value (two-tailed), Cohen's d.

**Output:** Table with one row per (feature, session, aperture, rr_target): Q1 ExpR, Q5 ExpR, spread, t-stat, p, d, N per quintile.

### Step 1.2 — BH FDR correction

Apply Benjamini-Hochberg at q = 0.10 across ALL tests from Step 1.1.

**K = total number of tests run.** Report K explicitly. Do not adjust K after seeing results.

Per test report: raw p, BH rank, BH threshold (rank/K × q), PASS/FAIL.

### Step 1.3 — Monotonicity check

For features passing BH:
- Is Q1→Q5 monotonic? (Continuous signal)
- Or concentrated in one extreme? (Threshold effect → binary filter candidate)

**Gate 1:**
- **0 features pass BH** → Features have no standalone signal. ML is truly dead. STOP. Update NO-GO. This is a valid, valuable result.
- **1-3 pass** → Proceed to Phase 2 with ONLY passing features.
- **4+ pass** → Proceed but flag potential FDR leakage. Consider tightening to q=0.05.

---

## PHASE 2: Distributional Analysis (1-2 hours)

**Purpose:** Test whether surviving features shift the SHAPE of the outcome distribution — not just the mean. This answers the question meta-labeling couldn't: "Do confluences change trade quality?"

### Step 2.1 — MAE comparison

For each surviving feature, split trades into HIGH (top quintile) and LOW (bottom quintile):
- KDE overlay plot (HIGH vs LOW MAE distributions)
- Two-sample KS test (p-value)
- Median MAE difference
- 90th percentile MAE difference (tail risk)

**Interpretation:** Lower MAE in HIGH-feature trades = confluence reduces drawdown before resolution.

### Step 2.2 — MFE comparison

Same split. Compare MFE distributions:
- Do HIGH-feature trades reach further into profit?
- Is the MFE distribution fatter-tailed? (More runners)

### Step 2.3 — Full PnL distribution

- KDE overlay
- KS test
- Skewness comparison
- Kurtosis comparison

**Gate 2:**
- KS p < 0.05 on any of (MAE, MFE, PnL) → **CONFIRMED distributional signal.** Proceed to Phase 3.
- All KS p > 0.05 → Mean-shift only (captured by Phase 1). Still proceed to Phase 3 for filter design, but note signal is MEAN-ONLY.

---

## PHASE 3: Filter Design & Walk-Forward (3-4 hours)

**Purpose:** Convert surviving features into deployable binary filters. Validate out-of-sample.

### Step 3.1 — Threshold optimization (in-sample)

For each surviving feature:
1. Use quintile boundaries as candidate thresholds
2. Per threshold: N_pass, N_fail, ExpR_pass, ExpR_fail, Win%_pass, Win%_fail
3. Compute COST (excluded positive-ExpR trades) and BENEFIT (removed negative-ExpR mass)
4. Select threshold maximizing ExpR_pass × sqrt(N_pass)

**Do NOT optimize on Sharpe.** (Biased under multiple testing per RESEARCH_RULES.md)

### Step 3.2 — Walk-forward validation

3-block chronological walk-forward:
- Train on blocks 1+2, test on block 3
- Train on block 1, test on block 2
- Train on blocks 2+3, test on block 1
- WFE = mean(OOS_ExpR) / mean(IS_ExpR)

**Gate 3:**
- **WFE ≥ 0.50** → DEPLOYABLE. Proceed to Phase 4.
- **WFE 0.30-0.49** → MARGINAL. Flag, don't deploy.
- **WFE < 0.30** → OVERFIT. Kill. Record in NO-GO with evidence.

### Step 3.3 — Interaction test (if 2+ survive)

- Pairwise: does A + B beat either alone?
- Redundancy: Jaccard overlap > 0.80 → keep higher-WFE only
- Conflict: Jaccard < 0.20 → separate regime signals, test independently

---

## PHASE 4: Integration Decision (1 hour)

### Decision tree:

**Path A: 0 features survive Phase 1**
→ "Confluences are dead." Stronger verdict than "ML is dead."
→ Update STRATEGY_BLUEPRINT.md NO-GO. System edge = ORB size + ATR regime + session selection, full stop.
→ Future work: tick data (order flow), not more features on 1m bars.
→ Do NOT revisit for 12 months or until 2× data available.

**Path B: Features survive Phase 1, fail Phase 3**
→ Statistical signal exists, not deployable (too weak after costs, or regime-dependent).
→ Record in TRADING_RULES.md as "CONFIRMED SIGNAL, NOT DEPLOYABLE" with stats.
→ Do NOT attempt ML on these — if simple filters fail walk-forward, RF won't help.
→ Revisit when data doubles.

**Path C: 1+ features survive Phase 3**
→ Integrate into filter grid alongside G8, ATR70_VOL, etc.
→ Run `strategy_discovery.py` with new filters → `strategy_validator.py`.
→ Pre-register for 2026 forward test.
→ ONLY THEN consider ML: if 3+ deployable filters exist, meta-labeling with binary filter features (not continuous) becomes a separate pre-registered program.

**Path D: Distributional signal only (Phase 2 passes, Phase 1 marginal)**
→ Feature shifts MAE/MFE but not the mean reliably.
→ This is a POSITION MANAGEMENT signal (early exit modifier / sizing input), not an entry filter.
→ Separate research track.

---

## EXECUTION RULES

1. **Sequential.** Phase N gate must pass before Phase N+1 starts.
2. **Every number from a query.** No stats from memory/docs/prior sessions.
3. **Pre-2026 only.** `trading_day < '2026-01-01'` on every query. Sacred holdout.
4. **Triple-join always.** `orb_outcomes` ↔ `daily_features` on trading_day + symbol + orb_minutes.
5. **Report K honestly.** K = total tests in that phase. No post-hoc subsetting.
6. **Show work.** Hypothesis → query → numbers → test statistic → p-value → correction → conclusion.
7. **Null results are results.** "No signal" is as valuable as "signal found." Report with equal rigor.
8. **No scope creep.** Test only the 19 listed features. New ideas go in FUTURE_IDEAS appendix, not into this run's K.
9. **Feature × session matrix governs.** CONTAMINATED = excluded everywhere, no second-guessing.
10. **Save intermediates.** Each phase → CSV in `research/output/confluence_program/` (e.g., `phase0_baselines.csv`, `phase1_quintiles.csv`). Reproducible audit trail.

---

## WHAT THIS DOES NOT COVER (EXPLICITLY OUT OF SCOPE)

- Tick data / order flow → separate track, requires data purchase
- Multi-timeframe trend → not in daily_features, requires new engineering
- Volume profile / POC → requires tick data
- ML reopening → deferred until Phase 4 decision
- 2026 forward testing → existing pre-registration governs
- Execution modeling → separate track, not blocked by this program

---

## SUCCESS CRITERIA

This program produces ONE of four outcomes:

1. **"Confluences are dead"** — Clean kill. No feature signal. All ML closed.
2. **"N features signal, M deployable"** — Thresholds set, walk-forward validated, ready for filter grid.
3. **"Distributional signal for position management"** — Separate research track opened.
4. **"Signal exists, not yet deployable"** — Recorded with stats, revisit conditions defined.

Any of these is a win. The only failure mode is ambiguity — and the hard gates at every phase make that impossible.
