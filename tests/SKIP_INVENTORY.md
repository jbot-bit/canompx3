# Test Skip Inventory

This document catalogs all conditional test skips in the test suite. These tests are skipped when certain runtime conditions are not met (e.g., database not available, artifacts missing, insufficient data).

**Last Updated:** 2026-02-21
**Total Skips:** 8 (across 4 files)

---

## test_trader_logic.py (5 skips)

### 1. Database Availability Gate (`_skip_if_no_db()`)

**Function:** `_skip_if_no_db()` (lines 592-605)
**Affected Test Classes:**
- `TestRandomOutcomeMath` (7 tests)
- `TestRandomStrategyMath` (3 tests)

**Skip Conditions:**
1. **Line 595:** `gold.db not available` - Database file does not exist at expected path
2. **Line 603:** `gold.db has no orb_outcomes table` - Database exists but missing required table
3. **Line 605:** `gold.db locked by another process` - Database file is locked (concurrent access)

**Tests Using This Gate:**
- `test_risk_points_recompute`
- `test_win_pnl_consistent_with_cost`
- `test_loss_pnl_exactly_minus_one`
- `test_target_price_recompute`
- `test_win_pnl_r_recompute`
- `test_stop_is_opposite_orb_level`
- `test_e1_entry_not_at_orb_level`
- `test_e3_entry_at_orb_level`
- `test_win_rate_recompute`
- `test_expectancy_recompute`
- `test_max_drawdown_recompute`

**Purpose:** These tests validate stored outcomes in `gold.db` against independent recomputation from first principles. They sample random rows and verify mathematical consistency (risk calculation, PnL formulas, strategy metrics).

---

### 2. Walk-Forward Artifacts Missing

**Test Class:** `TestRandomWalkForwardIntegrity`

#### Skip 4: `test_oos_trade_count_matches_outcomes`
- **Line:** 1078
- **Condition:** `walk-forward artifacts not present`
- **Path Checked:** `artifacts/walk_forward/walk_forward_results.json`
- **Purpose:** Verify OOS trade counts in walk-forward results match actual eligible outcomes in test folds

#### Skip 5: `test_no_train_test_overlap_in_results`
- **Line:** 1102
- **Condition:** `walk-forward artifacts not present`
- **Path Checked:** `artifacts/walk_forward/walk_forward_results.json`
- **Purpose:** Verify no fold has overlapping train/test date ranges (detect temporal leakage)

---

## test_corpus.py (1 skip)

### Skip 6: Database Fixture

**Test Class:** `TestSchemaDefinitions`
**Fixture:** `db_path` (lines 55-60)
**Line:** 59
**Condition:** `gold.db not available`

**Affected Tests:**
- `test_get_schema_definitions`
- `test_get_db_stats`

**Purpose:** These tests verify schema extraction and database statistics functions in `trading_app.ai.corpus`. They require a real `gold.db` with production schema.

**Note:** These tests use a fixture-level skip, so the skip message appears once per test class, not per test method.

---

## test_integration.py (1 skip)

### Skip 7: Insufficient Validated Strategies

**Test Class:** `TestPipelineFull`
**Test:** `test_promoted_strategy_has_all_fields`
**Line:** 162
**Condition:** `No strategies passed validation with test data`

**Purpose:** Verify that strategies promoted to `validated_setups` have all required fields populated correctly. Skipped when synthetic test data doesn't produce any strategies that pass validation gates.

**Trigger:** Occurs when validation thresholds (min_sample, Sharpe, etc.) reject all experimental strategies generated from 20-day synthetic data.

---

## test_integration_l1_l2.py (1 skip)

### Skip 8: No Strategies With Trades

**Test Class:** `TestFullL1L2`
**Test:** `test_data_contract_chain`
**Line:** 429
**Condition:** `No strategies with trades in synthetic data`

**Purpose:** Trace a single strategy back through the full pipeline (strategy → outcomes → daily_features → bars_1m) to verify data integrity across all layers.

**Trigger:** Occurs when synthetic data generation doesn't produce any strategies with `sample_size > 0` (no eligible break days matched filter criteria).

---

## Skip Categories

### Category A: Runtime Dependencies (6 skips)
- Database availability (3 conditions in `test_trader_logic.py`, 1 in `test_corpus.py`)
- Walk-forward artifacts (2 in `test_trader_logic.py`)

**Characteristics:**
- Check for external resources (files, databases)
- Pass in CI when resources are present
- Required for production validation

### Category B: Data-Dependent (2 skips)
- Synthetic data insufficient for validation (`test_integration.py`)
- Synthetic data produces no trades (`test_integration_l1_l2.py`)

**Characteristics:**
- Depend on statistical properties of generated test data
- May pass/fail non-deterministically based on random seed
- Could be stabilized with deterministic fixture design

---

## Investigation Priority

**High Priority (Spec Target):**
1. `test_trader_logic.py` database skips (lines 595, 603, 605) — 4 skipped tests in recent run
2. Walk-forward artifact skips (lines 1078, 1102) — Need artifact generation

**Low Priority (Expected Behavior):**
- `test_corpus.py` fixture skip — Correctly skips when DB unavailable
- Data-dependent skips — Acceptable for synthetic fixture tests

---

## Notes

1. **Database Skip Logic:** The `_skip_if_no_db()` helper has 3 exit points but is called once per test class (in a shared fixture or setup). This means a single invocation can produce different skip messages depending on which condition triggers first.

2. **CI Behavior:** In CI environments with pre-seeded `gold.db`, all database-dependent skips should pass. Walk-forward skips require `artifacts/` directory population.

3. **Local Dev:** Database skips are expected when running tests without a populated `gold.db` (e.g., fresh clone, scratch environment).

4. **Skip vs. Xfail:** All skips here are conditional runtime skips (`pytest.skip()`), not expected failures (`@pytest.mark.xfail`). They represent "test cannot run" rather than "test is expected to fail."
