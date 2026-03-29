# Statistical Hardening Spec — Phase 1-3

**ARCHIVED:** This spec is no longer active. Phase 1A targeted ML wrong-fit fixes; ML was declared DEAD 2026-03-27 (0/12 BH FDR survivors at K=12). Phase 1B/2/3 statistical fixes were either completed or superseded by later pipeline work. See memory ml_v2_final_verdict.md.

## Context
Audit of 15 papers (nb resources/) + ML module wrong-fit review against futures ORB trading.
Date: 2026-03-03. Detailed gap analysis: memory/paper_audit_gaps.md.

## Scope
Two categories:
1. **Wrong-fit fixes** — recently added ML code using equity assumptions that don't apply to futures ORB
2. **Paper gap implementations** — statistical guardrails recommended by the literature

---

## PHASE 1A — ML Wrong-Fit Fixes (immediate, small effort)

### FIX 1: Remove sqrt(252) Annualization from ML Sharpe
- **Files:** `trading_app/ml/evaluate.py`, `trading_app/ml/meta_label.py`, `trading_app/ml/evaluate_validated.py`
- **Problem:** `sqrt(252)` assumes 252 daily returns/year. Our data is per-trade (not per-day).
  Trade frequency varies by filter (G0 ~500/yr, G8 ~50/yr). Absolute Sharpe values are meaningless.
- **Fix:** Remove annual_factor parameter. Use per-trade Sharpe: `mean(pnl_r) / std(pnl_r)`.
  No annualization — we only compare within the same scale (relative comparisons unchanged).
- **Impact:** Sharpe values will be smaller numbers but correctly comparable across filters/sessions.
- **Tests:** Update any assertions on Sharpe magnitude in test_features.py.

### FIX 2: Remove NO-GO Features from ML Feature List
- **Files:** `trading_app/ml/config.py` (GLOBAL_FEATURES and SESSION_FEATURE_SUFFIXES)
- **Problem:** Three features are confirmed NO-GO from our own research:
  - `is_nfp_day` — calendar overlay, 0 BH survivors at q=0.10 (re-verified Mar 3 with TZ fix)
  - `is_opex_day` — calendar overlay, same result
  - `compression_z` — pre-ORB compression, 90+ tests, 0 BH survivors for break quality
- **Fix:** Remove `is_nfp_day` and `is_opex_day` from GLOBAL_FEATURES.
  Remove `compression_z` from SESSION_FEATURE_SUFFIXES.
  Add comment explaining WHY they were removed (research provenance).
- **Impact:** Feature matrix shrinks by 3 columns. RF may improve slightly (less noise).
- **NOT removing:** `day_of_week`, `is_friday`, `is_monday` — these are 0 BH survivors for
  general DOW effects, but RF can learn session-specific DOW interactions that linear BH can't detect.
  Keep for now; revisit if feature importance confirms they're useless.
- **Tests:** No test changes needed (feature list is dynamic).

### FIX 3: Label Threshold Optimization Bias in Docstrings
- **Files:** `trading_app/ml/meta_label.py`
- **Problem:** Threshold optimized on same 20% holdout used for OOS AUC reporting.
  Only CPCV AUC is truly unbiased.
- **Fix (Phase 1A):** Add docstring warnings that OOS metrics are biased by threshold selection.
  The CPCV AUC is the only honest metric.
- **Fix (Phase 2):** 3-way split or CPCV-derived threshold. Deferred — more complex.

### NOT FIXING (justified deferrals):
- **CPCV purge=1 day:** Over-conservative for intraday trades but pessimistic = safer. No harm.
- **`d.*` look-ahead columns:** Managed by LOOKAHEAD_BLACKLIST. Working correctly. Risk is
  blacklist going stale if schema changes — already covered by drift checks.
- **DOW features:** Keeping for RF interaction learning. Will audit feature importance later.

---

## PHASE 1B — Paper Gap Fixes (immediate, small effort)

### FIX 4: Deflated Sharpe Non-Normality Correction
- **File:** `trading_app/strategy_discovery.py` → `_compute_haircut_sharpe()`
- **Paper:** Bailey & Lopez de Prado (2014) — deflated-sharpe.pdf
- **Problem:** Uses simplified `V_null = 1/T`, ignoring skewness/kurtosis. ORB outcomes are
  bimodal (+RR / -1R), NOT normal. Skewness and kurtosis matter significantly.
- **Formula (Mertens 2002):**
  ```
  V[SR] = (1/T) * (1 - skew*SR + ((kurt-3)/4)*SR^2)
  ```
- **We already store:** `skewness`, `kurtosis_excess` in experimental_strategies.
- **Fix:** Update `_compute_haircut_sharpe()` to use full Mertens formula.
- **Effort:** ~10 lines in 1 function.

### FIX 5: Record K (Total Trials) in Database
- **File:** `pipeline/init_db.py` (schema), `trading_app/strategy_discovery.py` (write)
- **Paper:** Chordia et al (2018) — Two Million Strategies
- **Problem:** `total_combos` computed at runtime but never persisted. Can't audit trial count.
- **Fix:** Add `n_trials_at_discovery INT` column to `experimental_strategies`.
  Write `total_combos` value during discovery. One schema change + one line in writer.
- **Effort:** Tiny.

### FIX 6: False Strategy Theorem Hurdle
- **File:** `trading_app/strategy_discovery.py` (new function + store)
- **Paper:** Bailey & Lopez de Prado (2018) — false-strategy-lopez.pdf
- **Problem:** For K=2772 combos with zero skill, expected max SR ≈ 3.2.
  Any strategy below this is noise-floor indistinguishable.
- **Formula:** `E[max{SR}] ≈ (1 - γ)*Φ^(-1)(1 - 1/K) + γ*Φ^(-1)(1 - 1/(K*e))`
  where γ ≈ 0.5772 (Euler-Mascheroni), Φ^(-1) is inverse normal CDF.
- **CAVEAT:** Our per-trade SR ≠ annualized equity curve SR. Must document this distinction.
  FST applies to the Sharpe computed from the same distribution used in discovery.
- **Fix:** Add `compute_fst_hurdle(n_trials)` function. Store as column or metadata.
- **Effort:** Small (1 function + 1 column).

---

## PHASE 2 — Stress Testing & Operational (near-term)

### FIX 7: BHY as Optional Stress Test
- **File:** `trading_app/strategy_validator.py` → `benjamini_hochberg()`
- **Paper:** Benjamini-Yekutieli (2001)
- **Problem:** BH assumes independence/PRDS. BHY divides by c(m)=sum(1/k, k=1..m)≈8.6 for m=2772.
- **Our case:** BH is likely valid (PRDS holds for shared-data strategies). BHY is a conservative check.
- **Fix:** Add `bhy_mode=True` parameter to `benjamini_hochberg()`, ~5 lines.
- **Usage:** Run BHY as stress test. If all BH survivors also survive BHY → extra confidence.

### FIX 8: Strategy Rejection Rate Logging
- **File:** New table `validation_run_log` in `init_db.py`, writes in `strategy_discovery.py`
- **Paper:** Man AHL Advisory Board (2015)
- **What:** Track combos_tested → phase1_survivors → fdr_survivors → validated → per run.
- **Why:** Detect pipeline drift. If rejection rate suddenly changes → something changed.

### FIX 9: Walk-Forward Efficiency Metric
- **File:** `trading_app/walkforward.py`
- **Paper:** Pardo — Evaluation and Optimization
- **What:** WF Efficiency = OOS avgR / IS avgR. Should be >0.5 for healthy strategy.
- **Fix:** Compute and store during walk-forward. Small addition.

### FIX 10: ML Threshold Optimization Debiasing
- **File:** `trading_app/ml/meta_label.py`
- **What:** Split holdout into threshold-optimization set (10%) and true OOS test (10%).
  Or derive threshold from CPCV folds (more complex but cleaner).
- **Deferred from Phase 1A** — requires rethinking the data split.

---

## PHASE 3 — Research Depth (when ready)

### FIX 11: Synthetic Null Data Pipeline Test
- **File:** New script `scripts/tools/null_data_pipeline_test.py`
- **Paper:** Man AHL (2015)
- **What:** Generate fake bars with zero edge → run full pipeline → assert 0 survivors.
- **Hard part:** Generating synthetic 1m OHLCV that matches real microstructure
  (volatility clustering, overnight gaps, session-specific volume) but has zero trend in ORB windows.

### FIX 12: Probability of Backtest Overfitting (PBO)
- **File:** New module
- **Paper:** Bailey et al (2014)
- **What:** Combinatorial symmetric CV → rank correlation IS vs OOS performance.
- **Requires:** S≥8 data partitions, each spanning meaningful time periods.

---

## NOT DOING (justified)

| Item | Paper | Why Skip |
|------|-------|----------|
| 1/e optimal stopping | Bailey 2014 | Not relevant — we systematically test full grid + BH FDR |
| Carver portfolio sizing | Systematic Trading | Future phase — multi-strategy portfolio allocation |
| Chan equity strategies | Algo/Quant Trading | Wrong asset class — equity mean-reversion |
| White's Reality Check | Aronson 2007 | Superseded by BH FDR (more powerful, less conservative) |

---

## Execution Order

1. **Phase 1A** (this session): FIX 1-3 (ML wrong-fit)
2. **Phase 1B** (this or next session): FIX 4-6 (paper gaps)
3. **Phase 2** (next session): FIX 7-10
4. **Phase 3** (when ready): FIX 11-12
5. **Retrain all 4 instrument models** after Phase 1A (features changed)
