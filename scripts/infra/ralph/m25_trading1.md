<think>
Let me analyze these two files in detail to identify any real bias risks, overfitting concerns, or data quality issues.

## File 1: outcome_builder.py

This file pre-computes outcomes for all RR targets x confirm_bars combinations.

Key observations:
1. **DuckDB usage**: The file uses DuckDB with direct DataFrame references (`batch_df`). Per the system prompt, this is intentional and correct - DuckDB natively supports this pattern.
2. **Multiple RR targets**: The file computes outcomes for multiple RR targets (1.0-4.0) and confirm bars (1-5). This is a grid search that generates many hypotheses.
3. **Cost model**: Uses `cost_model.py` for transaction costs - this is the correct place per system prompt.
4. **Entry models**: Handles E1, E2, E3 as specified. E3 is soft-retired but still present for historical purposes.
5. **Break speed filter removal**: There's a comment about F-03 AUDIT FIX removing hardcoded C3 break speed filter. This is interesting - they're acknowledging a previous issue.

Let me check for real issues:

1. **No BH FDR correction in this file**: Correct - per system prompt, FDR correction happens in `strategy_validator.py`. This file just computes outcomes.

2. **Data snooping concern**: The file generates outcomes for ALL combinations of RR targets, confirm bars, entry models, and sessions. This creates a large hypothesis space. However, the system prompt explicitly says this is intentional - FDR correction happens downstream.

3. **Fill bar exit logic**: Lines 57-93 handle the case where the fill bar itself hits stop or target. This seems correct.

4. **Time-stop annotation**: The `_annotate_time_stop` function handles early exit logic. This appears correct.

5. **Checkpoint/resume**: The file supports checkpoint/resume via `computed_days` set (lines 356-361). This prevents re-computation.
   
## File 2: strategy_discovery.py

This file performs the grid search and computes performance metrics.

Key observations:
1. **BH FDR**: Not present in this file either - correct, it lives in `strategy_validator.py`.
2. **FST hurdle**: The file implements the False Strategy Theorem hurdle (lines 153-186) to account for multiple testing bias.
3. **Haircut Sharpe**: Implements Deflated Sharpe Ratio per Bailey & Lopez de Prado (2014) - lines 188-247.
4. **Autocorrelation adjustment**: Lines 339-354 adjust for lag-1 autocorrelation in returns.
5. **p-value**: Lines 356-363 compute statistical significance.
6. **Temporal holdout**: The file supports `holdout_date` parameter (lines 508-517) to prevent look-ahead bias.

The strategy_discovery.py implements multiple statistical corrections for overfitting: FST hurdle adjusts for the number of strategy combinations tested, haircut Sharpe applies the Deflated Sharpe Ratio to account for selection bias, autocorrelation adjustment corrects for serial dependence in returns, and p-values provide formal significance testing. The temporal holdout feature also prevents data leakage by reserving a portion of the historical data for out-of-sample validation. The `make_strategy_id` function doesn't include trade_day_hash in the identifier itself, which could create ambiguity across different trading day samples. However, the system properly handles this through the canonical/alias distinction managed by `_mark_canonical`.

For potential issues in these files: outcome_builder.py appears solid with no critical concerns, while strategy_discovery.py has a MEDIUM issue where the FST hurdle calculation uses the full `total_combos` count rather than accounting for filtered combinations. In strategy_discovery.py, the full grid is computed upfront with total_combos representing every combination, then n_trials_at_discovery and fst_hurdle are calculated using this total. This means even when filters reduce the actual tested combinations, the hurdle still reflects the entire potential search space.

There's a subtle issue here: strategies discovered after filtering should technically have a lower FST hurdle since fewer combinations were actually tested. However, this overestimates the difficulty—which is the conservative direction. The grid already supports checkpoint/resume functionality.

Looking at outcome_builder.py, there's potential inefficiency with daily feature queries. The query selects all ORB columns for all sessions, but not all sessions are enabled for every instrument, so this could be optimized.

In strategy_discovery.py, the filter specificity ranking system uses a predefined dictionary to order filters by specificity. There's also a minor formatting concern with the _t_test_pvalue function that implements a regularized incomplete beta function.

The relative volume computation in _compute_relative_volumes appears sound—it normalizes timestamps to UTC before extracting the minute-of-day, then fetches historical volumes and computes the relative volume correctly. The _ts_minute_key function also properly handles timezone conversion.

However, there's a potential issue in the outcome builder's entry model loop. When iterating through ENTRY_MODELS for each session and orb_label combination, the code handles E2 separately by using break-touch detection with confirm_bars fixed at 1, then calls _compute_outcomes_all_rr to process the outcomes.

For E1 and E3, the approach differs—E3 restricts confirm_bars to [1] since higher values produce identical outcomes, while E1 explores the full CONFIRM_BARS_OPTIONS range. The outcome computation then loops through RR_TARGETS to generate outcomes for each risk-reward ratio.

This design appears sound, with E3's constraint being an intentional optimization since it always enters at the ORB level regardless of confirm bars. The time threshold check comparing timestamps seems correct, as both `result.get("exit_ts")` and `ts_bar_ts` should be comparable datetime objects.

I notice that `outcome_builder.py` doesn't validate whether break_ts actually occurs within the trading day's bounds, but that's likely handled upstream in the feature building phase. The session filtering in strategy_discovery.py correctly respects enabled_sessions, and the DST regime handling includes proper guards for when sessions aren't DST-affected.

Looking at the filter_days computation, it creates separate sets for each filter-orb_label combination by iterating through features and checking if each strategy_filter matches the row for that specific orb_label. This appears sound since filter_keys are unique per session through get_filters_for_grid.

When multiple filters are applied to the same session, like "ORB_G4" + "ORB_G6" in grid search, both get their own distinct day sets. The approach is valid.

The batch writing implementation uses INSERT OR REPLACE with DuckDB's reference to batch_df, which works correctly per the system prompt. Similarly, the outcome_builder's INSERT OR REPLACE pattern is the same and functions properly.

I'm not seeing critical or high-severity issues in these files. There are a few minor observations worth noting. The FST hurdle calculation uses the total grid size rather than filtered combinations, which is conservative but not technically precise—though it's a minor issue since the conservative bias is favorable. The relative volume computation loads all historical bars for each minute-of-day, which could be optimized but performs acceptably.

The checkpoint mechanism in outcome_builder.py appears sound. Looking at the overall system, strategy_discovery.py correctly uses outcomes_by_key grouped by (orb_label, entry_model, rr_target, confirm_bars), which matches how outcome_builder.py inserts outcomes—ensuring the data flows properly through the pipeline.

The time-stop annotation with ts_outcome and ts_pnl_r gets written to the database without any apparent issues. The system handles edge cases appropriately: E3 confirm bars are constrained to [1], E2 uses break-touch detection, and time thresholds default to None when not applicable. No look-ahead bias appears in the discovery phase since outcomes are grouped by (orb, rr, cb) and filtered by matching days from features without any future information leakage.

The holdout_date parameter correctly constrains outcome loading to only pre-holdout data when specified, creating a proper temporal split for discovery. Since outcome_builder computes outcomes for all RR targets and confirm bars without filtering, and strategy_discovery applies filters afterward, there's no risk of post-hoc filtering bias. The files implement the system as intended.
</think>

Looking at these two files in detail, I find:

---

## 1. WELL-DONE — Specific Bias Prevention Measures

### outcome_builder.py
- **Lines 164-169**: Removed hardcoded C3 break speed filter (F-03 AUDIT FIX) — prevents invisible parameter embedding that would create research degrees of freedom
- **Lines 350-361**: Checkpoint/resume mechanism prevents re-computation bias when re-running discovery on same date range
- **Lines 336-348**: Pre-loads ALL bars for date range in single query — avoids per-day query bias where slow days might be selectively excluded
- **Lines 286-310**: `_compute_outcomes_all_rr` shares entry detection across all RR targets — eliminates redundant detection that could create optimization bias

### strategy_discovery.py
- **Lines 153-186**: Implements False Strategy Theorem (FST) hurdle — proper multiple-testing correction framework per Bailey & Lopez de Prado (2018)
- **Lines 188-247**: Deflated Sharpe Ratio with Mertens (2002) non-normality correction — accounts for bimodal ORB payoff distribution
- **Lines 339-354**: Lo (2002) autocorrelation adjustment (`sharpe_ann_adj`) — prevents inflated Sharpe from positively autocorrelated trades
- **Lines 356-363**: One-sample t-test p-value (F-04) — formal statistical significance test
- **Lines 508-517**: True temporal holdout (`holdout_date` parameter) — prevents data snooping by using only pre-holdout outcomes for discovery
- **Lines 620-626**: DST-aware strategy splitting with explicit `dst_regime` parameter — handles DST contamination properly
- **Lines 80-108**: Canonical vs alias marking — handles filter degeneracy correctly with specificity-based ranking

---

## 2. FINDINGS

### MEDIUM — FST Hurdle Uses Full Grid Size, Not Filtered Combinations

**Location**: `strategy_discovery.py`, lines 565-568 and 683-684

```python
total_combos += nf * len(RR_TARGETS) * len(CONFIRM_BARS_OPTIONS)
# ...
sharpe_haircut = _compute_haircut_sharpe(..., n_trials=total_combos, ...)
fst_hurdle = compute_fst_hurdle(n_trials)  # Uses total_combos
```

**Issue**: The FST hurdle and Deflated Sharpe Ratio use `total_combos` (full grid size) rather than the actual number of strategy combinations that survived filtering. When `matching_day_set` is empty or DST regime filtering removes outcomes, fewer trials are actually run, but the hurdle assumes the full grid was tested.

**Why it's MEDIUM**: This creates a **conservative bias** (higher hurdle = harder to pass), which is the safe direction. However, it's technically inconsistent — a strategy that passes the FST hurdle with 10,000 theoretical combos but only 500 tested combinations is being judged too harshly. The effect is small since most filters don't dramatically reduce combinations.

**Recommendation**: Track `actual_trials` incrementally as combinations are tested and use that for FST/haircut calculations.

---

### LOW — Relative Volume Computation Loads All Historical Minutes

**Location**: `strategy_discovery.py`, lines 424-448`

**Issue**: The `_compute_relative_volumes` function loads ALL historical volume data for each unique minute-of-day in the dataset, then filters in Python. This is O(total_bars) regardless of lookback period.

**Why it's LOW**: It works correctly and is fail-closed (missing data → filter rejects). Performance is acceptable for the data sizes involved. Could be optimized with SQL window functions but not a correctness issue.

---

### LOW — Checkpoint Resume Uses Only Trading Day, Not Full Signature

**Location**: `outcome_builder.py`, lines 356-361`

```python
computed_days = {
    r[0] for r in con.execute(
        "SELECT DISTINCT trading_day FROM orb_outcomes "
        "WHERE symbol = ? AND orb_minutes = ?",
        [instrument, orb_minutes],
    ).fetchall()
}
```

**Issue**: Resume only checks if a trading day was computed, not whether the full grid (all RR targets, confirm bars, entry models) was computed. If the grid parameters changed mid-run, stale outcomes could persist.

**Why it's LOW**: The `--force` flag exists to handle this. For production runs with fixed grid parameters, this is not a practical issue.

---

## 3. RECOMMENDATIONS

| Priority | Finding | Fix |
|----------|---------|-----|
| 1 (MEDIUM) | FST hurdle uses full grid size | Track `actual_trials` incrementally in the loop (lines 565-680) and pass to `compute_metrics` |
| 2 (LOW) | Relative volume loading | Accept as-is; optimization not worth complexity |
| 3 (LOW) | Checkpoint uses only day | Accept as-is; `--force` flag handles parameter changes |

---

**Summary**: These files implement the ORB outcome computation and strategy discovery pipeline correctly. The only real issue is the FST hurdle being conservatively biased (overestimates trials), which is safe but slightly inconsistent. No look-ahead bias, survivorship bias, or transaction cost issues found — all are properly handled in downstream modules as designed.