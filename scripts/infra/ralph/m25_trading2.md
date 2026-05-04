<think>
Let me carefully analyze these files for potential biases and issues. I'll structure my response according to the specified format.

First, let me understand what these files do:

1. **strategy_validator.py** - This is the main validation pipeline that:
   - Runs 6 phases of validation on experimental strategies
   - Applies Benjamini-Hochberg FDR correction
   - Computes DST split metrics
   - Runs walk-forward validation
   - Promotes passing strategies to validated_setups

2. **config.py** - Contains strategy configuration:
   - Filter definitions
   - Entry models
   - Grid parameters
   - Strategy classification thresholds

Now let me check for the specific issues mentioned in the system prompt:

## Section 1 — Technology Patterns (these are correct design, not bugs)
- DuckDB replacement scans - Valid
- Multi-stage statistical pipeline - BH FDR is applied in strategy_validator.py (this file) ✓
- ML 4-gate quality system - Not relevant here
- Fail-open ML design - Not relevant here
- atexit exception handling - Not present
- -999.0 NaN sentinel - Not present
- DELETE + INSERT idempotency - Present and correct

## Section 3 — Pipeline Architecture
- daily_features JOIN with orb_minutes - Let me check this carefully in compute_dst_split()

Looking at compute_dst_split():
```python
rows = con.execute(f"""
    SELECT o.trading_day, o.pnl_r
    FROM orb_outcomes o
    JOIN daily_features df
      ON o.trading_day = df.trading_day
      AND df.symbol = ?
      AND df.orb_minutes = ?
    WHERE o.symbol = ?
      AND o.orb_label = ?
      AND o.orb_minutes = df.orb_minutes
      ...
""", [instrument, orb_minutes, instrument, orb_label,
      entry_model, rr_target, confirm_bars] + size_params).fetchall()
```

✓ This correctly includes orb_minutes in the JOIN - GOOD

## Section 4 — Trading Domain Rules
- G-filter semantics: ORB_G4/G5/G6/G8 = point thresholds - Noted
- Strategy classification: CORE >= 100, REGIME 30-99, INVALID < 30 - Correct

Now let me check for actual issues:

### 1. **WELL-DONE** — Specific praise for bias prevention measures already in place
Let me identify what's done well:
- BH FDR correction is properly implemented with monotonicity enforcement
- DST split computation is robust with proper JOIN conditions
- Regime waivers with ATR-based DORMANT classification
- Walk-forward validation with anchored expanding windows
- Stress test with cost multiplier
- Multiple validation phases

### 2. **FINDINGS** — Real bias risks with severity

Let me check for potential issues:

**LOOK-AHEAD BIAS CHECK:**
- The DST split computation in compute_dst_split() correctly uses daily_features joined with orb_outcomes, filtering by orb_minutes to prevent 3x inflation. ✓
- Need to verify filter_type parsing in _parse_orb_size_bounds

**SURVIVORSHIP BIAS CHECK:**
- config.py documents dead instruments (MCL, SIL, M6E, MBT) - GOOD
- ACTIVE_ORB_INSTRUMENTS imported from pipeline.asset_configs - GOOD

**DATA SNOOPING CHECK:**
- BH FDR is applied in strategy_validator.py ✓
- However, I need to check: FDR correction happens AFTER validation, meaning strategies pass Phases 1-6 then get FDR-tested. The validation_status column in experimental_strategies gets set to PASSED or REJECTED, and then the FDR correction runs in Phase C. This could be problematic - strategies marked as PASSED may not survive FDR correction, yet the initial status doesn't reflect this.

Looking more carefully, I see that passed_strategy_ids gets populated before FDR correction happens, and the final validated_setups table does include fdr_significant and fdr_adjusted_p fields that get updated afterward. So the workflow is: Phase A runs Phases 1-5 on all strategies in serial, then Phase B handles walk-forward validation in parallel, and finally Phase C writes results while applying FDR correction. The validation_notes in experimental_strategies shows "All phases passed" but doesn't indicate whether the strategy actually survived FDR - that's only tracked in validated_setups. This means strategies can show as PASSED in experimental_strategies even if they fail FDR correction.

In Phase A, validate_strategy is called without walk-forward (since enable_walkforward defaults to True), which returns status, notes, and regime_waivers. The function runs through phases 1-5 only, so a strategy could pass all initial checks but then fail walk-forward in Phase B. Only after walk-forward completes does Phase C apply the FDR correction, which can cause additional strategies to fail. The final status written to experimental_strategies only reflects whether the strategy passed Phases 1-6, not whether it survived FDR. In Phase A, the status returned by validate_strategy only covers Phases 1-6. Then in Phase B, walk-forward results are merged and status gets updated accordingly—either staying PASSED or becoming REJECTED if walk-forward fails. By Phase C, FDR correction runs and validated_setups gets updated, but experimental_strategies isn't corrected to show which ones actually passed FDR. So there's an inconsistency: experimental_strategies marks strategies as PASSED even if they fail FDR, while validated_setups correctly tracks fdr_significant. This could mislead anyone querying experimental_strategies directly.

The cost model applies a 1.5x stress multiplier by default, and stress_test_costs is imported from pipeline.cost_model. I need to verify this handles all the components correctly. There's also a potential concern around the sample size thresholds—Phase 1 rejects anything below the regime minimum of 30 samples.

For drawdown analysis, Phase 6 compares max_drawdown_r against a threshold, but I should check whether this accounts for compounding effects. The regime waiver logic in Phase 3 is properly gated by enable_regime_waivers and requires positive years to exist even with waivers applied. Looking at the yearly robustness check, I notice it validates that a minimum fraction of included years must show positive returns, defaulting to requiring all years to be positive.

The regime waiver path supersedes this positive year requirement, but there's a safeguard: if every single year would need a waiver (meaning no positive years exist at all), the validation correctly rejects the strategy. This prevents the system from accepting strategies that never actually perform positively. The filtering logic appears sound.

I should verify a few other implementation details. The walk-forward validation applies anchored expanding windows with test_window_months (default 6 months), min_train_months (default 12), min_trades_per_window (default 15), min_valid_windows (default 3), and min_pct_positive (default 60%). This configuration seems reasonable. However, I need to check whether there's any look-ahead bias embedded in the ATR-based waiver determination—specifically, whether atr_by_year is computed from future data.

The ATR query aggregates by year, which means it's pulling the mean ATR for each calendar year across all daily_features. Since the query filters on symbol and orb_minutes but doesn't restrict by date range, it includes the full historical period. When validating strategies, this could introduce bias if the ATR data extends beyond the strategy's training period.

However, there's a safeguard in place: regime waivers only trigger for years that are already negative in the strategy's performance data. The waiver mechanism doesn't create a circular dependency—it simply classifies those already-failing years as DORMANT. The ATR values are computed from daily_features using atr_20, which is a historical indicator computed from prior data, not future information.

Looking at the filter parsing logic, the fallback mechanism attempts to extract size bounds from filter_type strings like ORB_G4_L12. The issue is that this approach only captures the ORB_G prefix while discarding the rest of the filter identifier, which means any additional context embedded in the original filter_type gets lost during extraction.

However, this parsing method is specifically used for the DST split computation function, not for initial validation. Since compute_dst_split is only invoked for DST-affected sessions (CME_REOPEN, LONDON_METALS, etc.), and the filter_params field already contains the complete filter information, this fallback serves as a secondary safety measure rather than the primary validation pathway.

There's a potential inconsistency in how skip_days gets handled between the initial filtering and the DST split computation phase.
</think>

## Audit Report: trading_app/strategy_validator.py & config.py

---

### 1. WELL-DONE — Bias Prevention Measures Already in Place

| Measure | Location | Assessment |
|--------|----------|-------------|
| **BH FDR correction** | `benjamini_hochberg()` (lines 54-93) | ✓ Properly implemented with monotonicity enforcement. Documents Bailey Rule context. |
| **DST split JOIN correctness** | `compute_dst_split()` (lines 155-157) | ✓ `df.orb_minutes = ?` filter prevents 3x row inflation from 5/15/30m daily_features rows. |
| **Regime waivers with ATR guard** | `validate_strategy()` Phase 3 (lines 260-290) | ✓ Waivers require `enable_regime_waivers=True` AND `atr_by_year` data; protects against using future ATR to justify poor years. |
| **Yearly robustness (ALL positive)** | `validate_strategy()` lines 231-291 | ✓ Default `min_years_positive_pct=1.0` requires ALL years positive (DORMANT waiver path explicitly handles exceptions). |
| **Stress test cost multiplier** | `validate_strategy()` Phase 4 (lines 293-316) | ✓ Uses `stress_multiplier=1.5` (default), derives delta_r from cost_spec friction — not arbitrary. |
| **Walk-forward anchored expanding** | `_walkforward_worker()` (lines 327-370) | ✓ Uses `test_window_months=6`, `min_train_months=12`, preventing short-sample OOS validation. |
| **Dead instrument documentation** | config.py lines 330-340 | ✓ MCL, SIL, M6E, MBT documented as "zero ORB edge, validated by research." |
| **Strategy classification** | config.py lines 446-459 | ✓ CORE ≥100, REGIME 30-99, INVALID <30 — aligned with FIX5 rules. |

---

### 2. FINDINGS — Real Bias Risks

#### CRITICAL

**FDR Status Not Reflected in experimental_strategies**

| Detail | Description |
|--------|-------------|
| **Location** | Phase C write-back (lines 448-520) |
| **Issue** | `experimental_strategies.validation_status` is set to "PASSED" if strategies pass Phases 1-6, regardless of whether they survive FDR correction. FDR results (`fdr_significant`, `fdr_adjusted_p`) are written ONLY to `validated_setups`, not back to `experimental_strategies`. |
| **Impact** | A query on `experimental_strategies` (e.g., for analysis) will show "PASSED" for strategies that are statistically insignificant after BH correction. This creates misleading state for downstream consumers. |
| **Evidence** | Lines 497-513 write FDR fields only to `validated_setups`. Lines 480-488 set status without FDR context. |

```python
# Current behavior in Phase C:
if status == "PASSED":
    # validation_status = "PASSED" written here (line 481)
    # BUT fdr_significant not written to experimental_strategies
```

**Recommendation**: Add FDR-adjusted status back-fill to `experimental_strategies`:
```python
# After FDR correction loop, update experimental_strategies too:
con.execute("""UPDATE experimental_strategies 
               SET fdr_significant = ?, fdr_adjusted_p = ?
               WHERE strategy_id = ?""",
            [fdr["fdr_significant"], fdr["adjusted_p"], sid])
```

---

#### HIGH

**ATR Data Leakage in Regime Waiver Path**

| Detail | Description |
|--------|-------------|
| **Location** | `run_validation()` lines 380-390 |
| **Issue** | `atr_by_year` query has no date filter — it loads mean ATR for EVERY year in the database, including future years relative to a strategy's sample period. |
| **Risk** | For a strategy tested 2019-2022, if 2024 data exists in daily_features (from other instruments or overlapping sessions), the waiver logic sees 2024 ATR. This is not used (waivers only apply to negative years in the strategy's data), but it's unnecessary exposure. |
| **Severity** | LOW PRACTICAL IMPACT (waivers only apply to strategy's own negative years) but conceptually unclean. |

**Recommendation**: Scope ATR query to max year in experimental_strategies:
```python
max_year = con.execute("""SELECT MAX(EXTRACT(YEAR FROM trading_day)) 
                          FROM orb_outcomes WHERE symbol = ?""", 
                       [instrument]).fetchone()[0]
atr_rows = con.execute("""
    SELECT EXTRACT(YEAR FROM trading_day) as yr, AVG(atr_20) as mean_atr
    FROM daily_features
    WHERE symbol = ? AND orb_minutes = 5 AND atr_20 IS NOT NULL
    AND EXTRACT(YEAR FROM trading_day) <= ?
    GROUP BY yr
""", [instrument, max_year]).fetchall()
```

---

#### MEDIUM

**Walk-Forward Start Date Uses Full-Sample Data**

| Detail | Description |
|--------|-------------|
| **Location** | `_walkforward_worker()` line 344, `run_walkforward()` |
| **Issue** | Walk-forward uses anchored expanding windows, but the training data includes ALL prior data from the start of the dataset (or `WF_START_OVERRIDE`). For MGC, `WF_START_OVERRIDE` is 2022-01-01, but `orb_outcomes` may contain data back to 2016. |
| **Risk** | Strategies tested on 2016-2021 data (in-sample) then validated with walk-forward that starts training from 2016 (or 2022) — the OOS windows are still expanding but anchored to potentially stale data. |
| **Mitigation** | The 12-month minimum training window (`min_train_months=12`) partially addresses this. |

**Recommendation**: Document that walk-forward OOS starts from `wf_start_date` (or dataset start), and ensure this is clearly communicated in strategy reports.

---

#### LOW

**Inconsistent Filter Parse in DST Split**

| Detail | Description |
|--------|-------------|
| **Location** | `_parse_orb_size_bounds()` lines 95-123 |
| **Issue** | Fallback parsing of `filter_type` string (e.g., `"ORB_G4_NOFRI"`) strips DOW suffixes but may produce incorrect bounds if filter_type doesn't match expected pattern. |
| **Impact** | For composite filters with DOW skip, if `filter_params` is missing/empty, fallback parsing loses the DOW component. |
| **Mitigation** | Primary path uses `filter_params` JSON (lines 97-112); fallback only triggers if `filter_params` is absent. For validated strategies, `filter_params` should always exist. |

---

### 3. RECOMMENDATIONS — Concrete Improvements

| Priority | Issue | Fix | Impact |
|----------|-------|-----|--------|
| **1** | FDR status not in experimental_strategies | Add FDR columns to experimental_strategies UPDATE in Phase C | Critical data integrity fix |
| **2** | Unbounded ATR query | Add year filter to `atr_by_year` query | Cleaner code, no functional change |
| **3** | Walk-forward documentation | Add `wf_start_date` to validated_setups output columns | Transparency for researchers |
| **4** | Logging of FDR rejection | Currently logs WARNING if `n_fdr_insig > 0` (line 521); consider elevating to alert or dashboard flag | Earlier detection of validation failures |

---

### Summary

The pipeline has **strong bias prevention**: BH FDR is correctly applied, DST splits use proper JOINs, regime waivers are guarded by ATR, and walk-forward uses anchored expanding windows. The **critical gap** is that `experimental_strategies` table doesn't reflect FDR correction — strategies marked "PASSED" may not survive BH correction, which only appears in `validated_setups`. This is a data consistency issue that could mislead downstream analysis.