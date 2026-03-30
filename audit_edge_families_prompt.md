# AUDIT PROMPT: Edge Family & Strategy Classification — Full Adversarial Review

**Objective:** Audit the entire edge family and strategy classification pipeline for correctness, bias, look-ahead contamination, and grounding in quantitative finance literature. This is a read-only audit — no code changes. Produce a findings report.

**Mode:** DESIGN / AUDIT ONLY. Do NOT edit any files. Do NOT implement fixes. Report findings with evidence.

---

## PHASE 1: Data-First Inventory (query before reading code)

Run these queries against gold.db (use MCP tools where templates exist, raw SQL otherwise):

1. **Table counts:** `query_trading_db(template="table_counts")` — how many rows in each table?
2. **Validated setups breakdown:**
   ```sql
   SELECT instrument, entry_model, COUNT(*) as n_strategies,
          COUNT(DISTINCT orb_label) as n_sessions,
          COUNT(DISTINCT filter_type) as n_filters,
          MIN(sample_size) as min_N, MAX(sample_size) as max_N,
          AVG(expectancy_r) as avg_expr, AVG(sharpe_ann) as avg_sharpe
   FROM validated_setups
   GROUP BY instrument, entry_model
   ORDER BY instrument, entry_model
   ```
3. **Edge families breakdown:**
   ```sql
   SELECT instrument, robustness_status, trade_tier, COUNT(*) as n_families,
          AVG(member_count) as avg_members, AVG(median_expectancy_r) as avg_median_expr,
          AVG(avg_sharpe_ann) as avg_sharpe, AVG(cv_expectancy) as avg_cv,
          MIN(min_member_trades) as min_trades, AVG(pbo) as avg_pbo
   FROM edge_families
   GROUP BY instrument, robustness_status, trade_tier
   ORDER BY instrument, robustness_status, trade_tier
   ```
4. **Family head selection audit — are heads actually median?**
   ```sql
   SELECT ef.family_hash, ef.instrument, ef.member_count, ef.median_expectancy_r,
          vs.strategy_id as head_id, vs.expectancy_r as head_expr,
          ABS(vs.expectancy_r - ef.median_expectancy_r) as head_deviation
   FROM edge_families ef
   JOIN validated_setups vs ON vs.family_hash = ef.family_hash AND vs.is_family_head = TRUE
   WHERE ef.member_count > 1
   ORDER BY head_deviation DESC
   LIMIT 20
   ```
5. **Orphan check — strategies in validated_setups with no edge family:**
   ```sql
   SELECT vs.strategy_id, vs.instrument, vs.orb_label, vs.family_hash
   FROM validated_setups vs
   LEFT JOIN edge_families ef ON vs.family_hash = ef.family_hash
   WHERE ef.family_hash IS NULL
   ```
6. **FDR audit — are p-values and BH corrections consistent?**
   ```sql
   SELECT instrument, entry_model,
          COUNT(*) as total,
          SUM(CASE WHEN fdr_significant = TRUE THEN 1 ELSE 0 END) as fdr_pass,
          MIN(p_value) as min_p, MAX(p_value) as max_p,
          MIN(fdr_adjusted_p) as min_adj_p, MAX(fdr_adjusted_p) as max_adj_p,
          MAX(n_trials_at_discovery) as K_used
   FROM validated_setups
   GROUP BY instrument, entry_model
   ```
7. **Walk-forward efficiency distribution:**
   ```sql
   SELECT instrument,
          COUNT(*) as total,
          SUM(CASE WHEN wfe > 0.5 THEN 1 ELSE 0 END) as wfe_pass,
          SUM(CASE WHEN wfe <= 0.5 THEN 1 ELSE 0 END) as wfe_fail,
          SUM(CASE WHEN wfe IS NULL THEN 1 ELSE 0 END) as wfe_null,
          AVG(wfe) as avg_wfe, MIN(wfe) as min_wfe
   FROM validated_setups
   GROUP BY instrument
   ```
8. **Yearly robustness — any strategy with a losing year still in validated?**
   ```sql
   SELECT strategy_id, instrument, orb_label, entry_model, yearly_results
   FROM validated_setups
   LIMIT 50
   ```
   Then parse `yearly_results` JSON for any year with negative ExpR. Report how many validated strategies have losing years and whether regime waivers explain them.

9. **Concentration risk — how many families share the same session?**
   ```sql
   SELECT instrument, orb_label, COUNT(DISTINCT family_hash) as n_families,
          SUM(CASE WHEN robustness_status = 'ROBUST' THEN 1 ELSE 0 END) as robust,
          SUM(CASE WHEN robustness_status = 'WHITELISTED' THEN 1 ELSE 0 END) as whitelisted
   FROM edge_families ef
   JOIN validated_setups vs ON vs.family_hash = ef.family_hash AND vs.is_family_head = TRUE
   GROUP BY instrument, orb_label
   ORDER BY n_families DESC
   ```

**Record all numbers. These are the ground truth for the rest of the audit.**

---

## PHASE 2: Code Audit — Classification Logic

Read and verify these files against the data from Phase 1:

### 2A: Edge Family Builder (`scripts/tools/build_edge_families.py`)

Check for:
- [ ] **Trade-day hash correctness:** Is the hash computed from `strategy_trade_days` table? Could two strategies with different trade outcomes but same trade calendar get merged? (This would be a BUG — same days but different directions/RRs shouldn't merge.)
- [ ] **Hash includes direction?** If strategy A is LONG on day X and strategy B is SHORT on day X, do they get different hashes? They MUST.
- [ ] **Hash includes RR target?** RR=1.0 and RR=2.0 on the same session have different exit points and different trade calendars. Verify they don't merge.
- [ ] **Median head election:** Verify the code actually picks median, not mean or max. Check the tiebreaker logic.
- [ ] **Robustness thresholds:** Do ROBUST (N≥5), WHITELISTED (3-4 + ShANN≥0.8 + CV≤0.5 + min_trades≥50), SINGLETON (N=1 + min_trades≥100 + ShANN≥1.0) match what's documented? Any undocumented exceptions?
- [ ] **PBO calculation:** Is Probability of Backtest Overfitting (Bailey et al. 2014) computed correctly? Does it use combinatorial symmetric cross-validation (CSCV)? Or is it a simplified proxy?
- [ ] **PURGED families:** What happens to them? Are they deleted or just flagged? Can they leak into downstream queries?

### 2B: Strategy Validator (`trading_app/strategy_validator.py`)

Check for:
- [ ] **Phase ordering:** Do phases run in the documented order (sample→cost→yearly→stress→WF→FDR)? Or can a strategy skip a phase?
- [ ] **FDR K value:** What K is actually used? Global K (all strategies across all instruments) or per-instrument K? Compare to what's stored in `n_trials_at_discovery`. If K is wrong, ALL p-value adjustments are wrong.
- [ ] **Walk-forward window construction:** Are windows truly anchored-expanding (growing IS, fixed OOS)? Or sliding? Anchored-expanding is correct for regime-spanning data. Sliding windows can miss regime shifts.
- [ ] **WFE calculation:** Is it `OOS_Sharpe / IS_Sharpe`? What happens when IS_Sharpe is negative or zero? Division by zero guard?
- [ ] **Look-ahead in yearly robustness:** When checking "positive in all years," does the code use the full sample including 2026? If 2026 holdout is sacred, yearly robustness should exclude it.
- [ ] **DST split logic:** When splitting winter/summer for DST-affected sessions, does it correctly join on `(trading_day, symbol, orb_minutes)` or is there a row-tripling bug from missing the `orb_minutes` join condition?
- [ ] **Regime waivers:** What strategies get regime waivers for the yearly robustness check? Is the waiver logic documented and auditable, or is it ad-hoc?

### 2C: Strategy Discovery (`trading_app/strategy_discovery.py`)

Check for:
- [ ] **Grid search completeness:** Does the grid actually cover all documented combinations (E1/E2 × 12 ORBs × 6 RRs × 5 CBs × 11 filters)? Or are some silently excluded?
- [ ] **Filter application correctness:** When a strategy uses `ORB_G6`, is the filter applied to `daily_features` BEFORE computing outcomes, or AFTER? Post-hoc filtering is look-ahead.
- [ ] **Canonicalization logic:** When multiple strategies share the same trade-day hash, the highest-specificity filter is canonical. Verify the specificity ranking is correct and that aliases properly point to the canonical head.
- [ ] **Sharpe haircut (Bailey & López de Prado 2014):** Is the deflated Sharpe implemented correctly? Does it adjust for skewness, kurtosis, AND number of trials? Or just some of these?
- [ ] **p-value computation:** Two-tailed t-test on what? On raw R-multiples? On daily returns? The choice matters for autocorrelation and heteroscedasticity.

### 2D: Outcome Builder (`trading_app/outcome_builder.py`)

Check for:
- [ ] **No look-ahead in outcome computation:** When computing whether a target/stop is hit, does it use only bars AFTER the entry bar? Or does the entry bar's close/high/low contaminate?
- [ ] **Fill assumption:** For E1 (limit entry at ORB edge), what fill price is assumed? For E2 (stop entry on break), what slippage is applied?
- [ ] **Time-stop implementation:** Is the early exit at `EARLY_EXIT_MINUTES` using the correct bar? Is it marking the exit at the correct price (bar close vs mid vs open)?
- [ ] **Cost model application:** Are costs applied ONCE per trade, not doubled? Is the cost model using the correct instrument's specs?

### 2E: Config (`trading_app/config.py`)

Check for:
- [ ] **Threshold provenance:** Are CORE_MIN_SAMPLES=100, REGIME_MIN_SAMPLES=30 grounded in literature? (Common references: Carver "Systematic Trading" suggests N≥100 for reliable Sharpe; Bailey & López de Prado suggest higher bars for multiple testing contexts.)
- [ ] **WF_START_OVERRIDE for MGC (2022-01-01):** Is this justified by regime analysis or is it data-snooping to exclude bad years?
- [ ] **NOISE_FLOOR disabled (all zeros):** Was this a conscious decision with justification, or drift?
- [ ] **Entry model definitions:** Are E1/E2 clearly defined with no ambiguity about fill prices, slippage, or timing?

---

## PHASE 3: Bias & Look-Ahead Checklist

For each item, provide PASS/FAIL/SUSPECT with evidence:

### 3A: Selection Bias
- [ ] **Survivorship bias in instrument selection:** MGC, MNQ, MES are active. M2K, MCL, SIL, M6E, MBT are dead. Were the dead instruments killed BEFORE or AFTER the strategy pipeline was built? If after — was the pipeline re-run on all instruments with identical methodology, or were dead instruments tested with an older/different pipeline version?
- [ ] **Session selection bias:** How were the 12 sessions chosen? Were they chosen BEFORE seeing outcome data, or did outcome data inform which sessions to include?
- [ ] **Filter specificity bias:** Higher-specificity filters (G8, G6) mechanically reduce sample size. Are the FDR corrections using K that includes ALL filter levels tested, or only the filter level that "won"?

### 3B: Look-Ahead Bias
- [ ] **`orb_outcomes` uses only bars within the session window?** No future bars after the session close leak into outcome computation?
- [ ] **`daily_features` ORB columns computed from bars within the aperture only?** The 5m/15m/30m ORB high/low uses only bars in that window?
- [ ] **Break detection:** When a break is detected (price crosses ORB high/low), is the break timestamp the FIRST crossing, or a retrospective detection?
- [ ] **Double-break filter:** STRATEGY_BLUEPRINT.md marks this as killed. Verify it's actually removed from outcome_builder and not just disabled. Double-break inherently requires seeing the second break, which is look-ahead from the first break's perspective.
- [ ] **ATR/RSI in daily_features:** Are these computed from data PRIOR to the session, or do they include the session's own bars?

### 3C: Multiple Testing / Data Mining Bias
- [ ] **BH FDR K is honest:** The grid searches ~5,500 combinations. Is K=5500 used, or a smaller K? Report the actual K from the data (Phase 1, query 6).
- [ ] **Researcher degrees of freedom:** Beyond the formal grid, how many informal decisions were made (which instruments to include, which sessions, which date ranges, which entry models)? Each informal decision is an untested hypothesis that inflates effective K.
- [ ] **Family-level testing vs strategy-level:** If 50 strategies form 5 families, is the FDR correction applied to 50 tests or 5? (It should be applied at the strategy level FIRST, then families are formed from survivors. If families are formed first and then tested, the effective K is wrong.)

### 3D: Overfitting Indicators
- [ ] **In-sample vs out-of-sample decay:** For strategies with WFE data, what's the distribution of IS_Sharpe vs OOS_Sharpe? If median decay > 60%, the system may be overfit even if individual strategies pass WFE > 0.50.
- [ ] **Parameter cliff:** For the top 10 families, check if neighboring parameter combinations (±1 RR step, ±1 CB step) are also profitable. If the edge exists ONLY at exact parameter values, it's likely curve-fitted.
- [ ] **PBO distribution:** For families with PBO computed, what's the distribution? PBO > 0.50 means more than half of backtest configurations would have been unprofitable. How many families have PBO > 0.50?

---

## PHASE 4: Literature Grounding Check

For each methodological choice, verify it's grounded in published quant finance literature:

| Method | Claimed Reference | Verify |
|--------|------------------|--------|
| BH FDR correction | Benjamini & Hochberg (1995) | Is the implementation correct? Monotonicity enforcement? |
| Deflated Sharpe | Bailey & López de Prado (2014) | Does the haircut use their exact formula including skew/kurtosis/trials? |
| PBO (Probability of Backtest Overfitting) | Bailey et al. (2014) CSCV | Is it real CSCV or a simplified version? If simplified, document the deviation. |
| Walk-Forward Efficiency > 0.50 | Pardo "The Evaluation and Optimization of Trading Strategies" | Is 0.50 the correct threshold from Pardo, or is it project-specific? |
| Median head election | Anti-Winner's Curse | Is there a specific reference, or is this project-invented? If project-invented, is the reasoning sound? |
| CORE=100, REGIME=30 thresholds | Carver "Systematic Trading" / standard practice | Are these sample sizes sufficient given the number of parameters? Rule of thumb: N > 10× free parameters. |
| CV ≤ 0.5 for WHITELISTED | Carver Ch.4 | Verify this is actually from Carver and not misattributed. |
| Cost buffer +50% stress test | Industry practice | Is 50% standard or arbitrary? What do Kissell, Chan, or Aldridge suggest? |

Check `resources/` directory for local academic PDFs. If a reference is claimed but the PDF can't be read, say so explicitly — do not fake the grounding.

---

## PHASE 5: Cross-Consistency Checks

1. **Config ↔ Code:** Do the thresholds in `config.py` match what the code actually uses? `grep -r` for hardcoded thresholds that bypass config.
2. **Docs ↔ Data:** Do the counts in TRADING_RULES.md, STRATEGY_BLUEPRINT.md, and memory files match the actual DB? (They probably don't — report every discrepancy.)
3. **Robustness status ↔ Thresholds:** For every WHITELISTED family, verify it actually meets ALL 4 criteria (3-4 members, ShANN≥0.8, CV≤0.5, min_trades≥50). For every SINGLETON, verify (1 member, min_trades≥100, ShANN≥1.0).
4. **Dead instruments:** Query `validated_setups` and `edge_families` for M2K, MCL, SIL, M6E, MBT. Are they truly empty, or is there residual data that could contaminate?
5. **2026 holdout integrity:** Query `orb_outcomes` for any 2026 trading days. Then check if any validated_setups or edge_families were computed using 2026 data. If yes, the holdout is breached.

---

## DELIVERABLE

Produce a structured findings report with:

1. **Summary verdict:** CLEAN / ISSUES FOUND / CRITICAL ISSUES (with count of each severity)
2. **Per-check results:** PASS / FAIL / SUSPECT for each checkbox above, with evidence (query output, code line numbers, or specific discrepancies)
3. **Literature grounding gaps:** Any method that lacks proper grounding or deviates from the cited reference
4. **Look-ahead contamination:** Any confirmed or suspected look-ahead, with specific code paths
5. **Recommended fixes:** Prioritized list of what to fix (CRITICAL → HIGH → MEDIUM → LOW), but DO NOT implement them

**Format:** Write the report to `docs/audits/edge_family_audit_YYYY-MM-DD.md`. Use the current date.

**Time budget:** This is a thorough audit. Take your time. Query first, read code second, conclude third. Do not skip queries to save time. Do not assume code does what the docstring says — verify.
