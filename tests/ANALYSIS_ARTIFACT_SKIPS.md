# Walk-Forward Artifact Availability Skip Analysis

**Analysis Date:** 2026-02-21
**Scope:** Category 2 - Artifact Availability Skips (2 skip conditions)
**Decision:** **KEEP AS INTENTIONAL** - Both skips are correct build-dependent guards

---

## Executive Summary

Both walk-forward artifact availability skips are **intentional runtime guards** that enable tests to validate walk-forward analysis results when artifacts exist, while gracefully skipping when the computationally expensive walk-forward analysis has not been run. These skips represent correct fail-safe behavior for optional research artifacts.

**Decision Rationale:**
- ✅ Tests validate walk-forward result integrity when artifacts are available
- ✅ Tests gracefully skip when artifacts are unavailable (expected in CI, fresh clones, environments without walk-forward analysis)
- ✅ Skip messages are clear and actionable
- ✅ Walk-forward artifacts are **optional research outputs**, not required for core pipeline operation
- ✅ No code changes required — behavior is correct as-is

---

## Skip Inventory

### Skip 1: Walk-Forward Trade Count Validation
- **Location:** `tests/test_trader_logic.py:1078`
- **Test:** `TestRandomWalkForwardIntegrity::test_oos_trade_count_matches_outcomes`
- **Condition:** `not wf_json.exists()` where `wf_json = artifacts/walk_forward/walk_forward_results.json`
- **Message:** `"walk-forward artifacts not present"`
- **Purpose:** Verify OOS trade counts in walk-forward results match independent recomputation from database

### Skip 2: Walk-Forward Train/Test Overlap Validation
- **Location:** `tests/test_trader_logic.py:1102`
- **Test:** `TestRandomWalkForwardIntegrity::test_no_train_test_overlap_in_results`
- **Condition:** `not wf_json.exists()` where `wf_json = artifacts/walk_forward/walk_forward_results.json`
- **Message:** `"walk-forward artifacts not present"`
- **Purpose:** Verify no train/test leakage in walk-forward fold date ranges (causality check)

---

## Root Cause Analysis

### What is walk_forward_results.json?

**Nature:** Optional research artifact containing out-of-sample validation results for validated strategies

**Schema:** JSON array of `WalkForwardResult` objects
```json
{
  "strategy_id": "MGC_0900_E1_RR1.0_CB1_ORB_G5",
  "filter_type": "ORB_G5",
  "folds": [
    {
      "fold_idx": 1,
      "train_start": "2021-02-05",
      "train_end": "2023-12-29",
      "test_start": "2024-01-02",
      "test_end": "2024-12-31",
      "train_days": 851,
      "test_days": 293,
      "trade_count": 6,
      "win_rate": 0.4,
      "expectancy_r": -0.293,
      "sharpe_ratio": -0.3025,
      "max_drawdown_r": 1.4648
    }
  ],
  "oos_trade_count": 6,
  "oos_win_rate": 0.4,
  "oos_expectancy_r": -0.293,
  "oos_sharpe_ratio": -0.3025,
  "oos_max_drawdown_r": 1.4648,
  "fold_imbalance_ratio": null,
  "fold_imbalanced": false
}
```

**Contents:**
- 150+ strategy evaluations (validated strategies only)
- Per-fold metrics: train/test periods, trade counts, performance metrics
- Aggregate OOS metrics: total OOS trades, win rate, expectancy, Sharpe, max drawdown
- Fold imbalance detection (identifies strategies with uneven performance across folds)

**Location:** `artifacts/walk_forward/walk_forward_results.json`

**Size:** ~12K lines, ~330KB

**Generation Command:**
```bash
python trading_app/walk_forward.py --instrument MGC --walk-forward --train-years 3 --test-years 1
```

**Lifecycle:**
- **Prerequisites:** Requires `gold.db` with populated `validated_setups` table
- **Generation Time:** 5-30 minutes (depends on strategy count and data range)
- **Frequency:** Generated on-demand for walk-forward analysis research
- **Version Control:** **NOT committed to git** (research artifact, regenerated as needed)
- **Environment:** Local-only (CI does not generate walk-forward artifacts)

**Related Artifacts:**
- `artifacts/walk_forward/walk_forward_summary.csv` - Summary table of OOS results
- `artifacts/walk_forward/walk_forward_folds.csv` - Per-fold detailed metrics

---

### What is Walk-Forward Analysis?

**Purpose:** Out-of-sample validation technique to detect overfitting

**Method:**
1. Split timeline into (train, test) folds
2. Each fold uses past data for training window (e.g., 3 years)
3. Evaluate strategy on test window (e.g., 1 year) that follows training
4. Slide window forward, repeat
5. Aggregate results across all test folds

**Example Timeline:**
```
Fold 1: Train 2021-2023 → Test 2024
Fold 2: Train 2022-2024 → Test 2025
Fold 3: Train 2023-2025 → Test 2026
```

**Key Properties:**
- **No peeking:** Test periods never overlap with train periods
- **Temporal causality:** Train always precedes test
- **Forward-only:** Each fold uses only past data (no future leakage)
- **Multiple windows:** Multiple test periods reduce sample size bias

**Module:** `trading_app/walk_forward.py`

**Integration with Pipeline:**
```
gold.db:validated_setups (937 strategies)
  → walk_forward.py (OOS evaluation)
  → artifacts/walk_forward/walk_forward_results.json
  → Tests validate result integrity
```

---

### Why Tests Require walk_forward_results.json

**Test 1: `test_oos_trade_count_matches_outcomes`**

**Purpose:** Validate walk-forward trade count arithmetic

**What It Checks:**
- For each strategy in walk-forward results:
  - Sum trade counts across all folds: `sum(fold["trade_count"] for fold in folds)`
  - Compare to aggregate `oos_trade_count` field
  - Assert: `oos_trade_count == sum_of_fold_trade_counts`

**Why It Matters:**
- Catches aggregation bugs in walk-forward computation
- Ensures OOS counts are correctly totaled from individual folds
- Validates data contract between fold-level and aggregate-level metrics

**Example Assertion:**
```python
total_oos = sum(fold["trade_count"] for fold in wf["folds"])
assert wf["oos_trade_count"] == total_oos, (
    f"OOS count mismatch for {wf['strategy_id']}: "
    f"aggregate={wf['oos_trade_count']}, sum_folds={total_oos}"
)
```

**Test 2: `test_no_train_test_overlap_in_results`**

**Purpose:** Validate temporal causality in walk-forward folds

**What It Checks:**
- For each fold in each strategy:
  - Extract `train_end` and `test_start` dates
  - Assert: `train_end < test_start` (no overlap, no same-day)

**Why It Matters:**
- Prevents lookahead bias (future data leaking into training)
- Validates core walk-forward assumption: train precedes test
- Catches fence-post errors in fold construction

**Example Assertion:**
```python
train_end = date.fromisoformat(fold["train_end"])
test_start = date.fromisoformat(fold["test_start"])
assert train_end < test_start, (
    f"Leakage in {wf['strategy_id']} fold {fold['fold_idx']}: "
    f"train_end={train_end} >= test_start={test_start}"
)
```

**Why These Tests Can't Use Synthetic Data:**

| Reason | Explanation |
|--------|-------------|
| **Real fold structure** | Tests validate actual walk-forward fold generation logic from `walk_forward.py` |
| **Real strategy IDs** | Tests validate real strategy references match database entries |
| **Real date ranges** | Tests validate actual train/test splits from 2021-2026 timeline |
| **Integration validation** | Tests validate the artifact generation pipeline, not isolated logic |
| **Scale validation** | Tests sample 20 random strategies from 150+ to catch edge cases |

**These are integration tests for research artifact integrity, not unit tests.**

---

### When are walk_forward_results.json Unavailable?

**Expected Scenarios (Normal):**
1. **Fresh repository clone** - New developer, CI runner, fresh machine
2. **CI environment** - Walk-forward analysis is expensive and optional for CI
3. **Pre-walk-forward workflow** - Developer working on pipeline, hasn't run walk-forward yet
4. **Selective research** - User only interested in strategy discovery, not walk-forward validation
5. **Different artifact path** - User specified custom `--output-dir` in walk-forward command

**Unexpected Scenarios (Investigate):**
1. **Artifact deleted** - File was present, now missing (possible manual deletion)
2. **Walk-forward failed** - Command ran but crashed before writing artifact
3. **Wrong working directory** - Tests looking for artifact in wrong location

**Artifact Availability by Environment:**

| Environment | walk_forward_results.json Expected? | Reason |
|-------------|-------------------------------------|--------|
| **Local dev (full pipeline)** | ✅ Yes | User has run walk-forward analysis |
| **Local dev (basic pipeline)** | ❌ No | User only ran ingest/features, not walk-forward |
| **CI/GitHub Actions** | ❌ No | Walk-forward analysis not part of CI (too expensive) |
| **Fresh clone** | ❌ No | Artifacts not version controlled |
| **Production** | ⚠️ Maybe | Depends on whether walk-forward analysis has been run |

---

### Why Skips are Correct Behavior

**Design Philosophy: Build-Dependent Testing**

The test suite supports **two execution modes for walk-forward tests:**

| Mode | walk_forward_results.json State | Behavior | Use Case |
|------|--------------------------------|----------|----------|
| **Research Validation** | Available (generated via walk_forward.py) | Tests run, validate artifact integrity | Local research environment after running walk-forward |
| **Standard Pipeline** | Unavailable (walk-forward not run) | Tests skip gracefully | CI, fresh clones, developers not doing walk-forward research |

**Benefits:**
- ✅ Fast test feedback without expensive walk-forward computation (5-30 min)
- ✅ Walk-forward integrity validation when artifacts exist
- ✅ No false negatives (tests don't fail when optional artifact is missing)
- ✅ Clear distinction between "test failed" vs "artifact not generated"
- ✅ Supports incremental workflow (run pipeline → optionally run walk-forward → tests validate if available)

**Alternative Approaches Rejected:**

| Approach | Why Rejected |
|----------|--------------|
| **Always require artifacts** | Breaks CI, breaks fresh clones, forces all users to run expensive walk-forward |
| **Generate artifacts in test setup** | Walk-forward takes 5-30 min per instrument — unacceptable for test suite |
| **Mock walk-forward results** | Defeats purpose of validating real walk-forward artifact structure |
| **@pytest.mark.skip decorator** | Would always skip, even when artifacts are available |
| **Remove tests entirely** | Loses critical walk-forward integrity validation |
| **Synthetic minimal JSON** | Doesn't validate real fold generation logic or date range handling |

---

## Per-Skip Decision Rationale

### Skip 1: `test_oos_trade_count_matches_outcomes` (line 1078)

**Decision:** **KEEP - Intentional build artifact gate**

**Reasoning:**
- Walk-forward artifacts are **optional research outputs**, not core pipeline requirements
- Test provides value only when validating real walk-forward results
- Clear, actionable message tells user exactly what's missing
- Expected to trigger in CI, fresh clones, environments without walk-forward analysis
- Prevents false negative (test failure when artifact simply hasn't been generated)

**Test Value When It Runs:**
- Catches aggregation bugs: `oos_trade_count ≠ sum(fold_trade_counts)`
- Validates data contract between fold-level and aggregate metrics
- Samples 20 random strategies to catch edge cases across instruments/sessions

**Resolution Path for Users:**
```bash
# If you need this test to run:
# 1. Ensure gold.db is populated with validated strategies
python trading_app/strategy_validator.py --instrument MGC --min-sample 50

# 2. Run walk-forward analysis (5-30 min depending on strategy count)
python trading_app/walk_forward.py --instrument MGC --walk-forward --train-years 3 --test-years 1

# 3. Test now runs
python -m pytest tests/test_trader_logic.py::TestRandomWalkForwardIntegrity::test_oos_trade_count_matches_outcomes -v
# ✅ PASSED
```

**Verification:**
- ✅ Skip message is descriptive and actionable
- ✅ Guides user to resolution (implies: run walk_forward.py)
- ✅ Test runs when artifact exists
- ✅ Test validates real integrity (not synthetic data)
- ✅ No silent failures or false positives

---

### Skip 2: `test_no_train_test_overlap_in_results` (line 1102)

**Decision:** **KEEP - Intentional build artifact gate**

**Reasoning:**
- Same artifact dependency as Skip 1 (walk_forward_results.json)
- Test validates temporal causality in walk-forward folds (train_end < test_start)
- Critical validation: catches lookahead bias if fold generation logic is broken
- Expected to skip in same environments as Skip 1

**Test Value When It Runs:**
- Catches train/test leakage: `train_end >= test_start` (would invalidate walk-forward results)
- Validates core walk-forward assumption: training always precedes testing
- Prevents fence-post errors in fold date range construction
- Checks ALL folds for ALL strategies (exhaustive validation, not sampled)

**Why This Test Exists:**
Walk-forward analysis is only valid if train/test periods are strictly ordered. If a fold has overlapping dates, the entire analysis is invalid (future data leaked into training). This test is a **critical guardrail** against lookahead bias.

**Resolution Path for Users:**
```bash
# Same as Skip 1 — requires walk_forward_results.json
python trading_app/walk_forward.py --instrument MGC --walk-forward --train-years 3 --test-years 1
python -m pytest tests/test_trader_logic.py::TestRandomWalkForwardIntegrity::test_no_train_test_overlap_in_results -v
# ✅ PASSED
```

**Verification:**
- ✅ Skip message matches Skip 1 (consistency)
- ✅ Test covers critical causality invariant
- ✅ Exhaustive check (all folds, not sampled)
- ✅ Test runs when artifact exists
- ✅ Clear what's being validated (train/test temporal ordering)

---

## Testing the Skips

### Reproduction Steps

**Verify Skip (Artifact Missing):**
```bash
# Temporarily move artifacts
mv artifacts/walk_forward artifacts/walk_forward.bak

# Run affected tests
python -m pytest tests/test_trader_logic.py::TestRandomWalkForwardIntegrity -v -rs

# Expected Output:
# test_oos_trade_count_matches_outcomes SKIPPED - walk-forward artifacts not present
# test_no_train_test_overlap_in_results SKIPPED - walk-forward artifacts not present

# Restore
mv artifacts/walk_forward.bak artifacts/walk_forward
```

**Verify Tests Run (Artifact Present):**
```bash
# Ensure artifacts exist (generate if needed)
python trading_app/walk_forward.py --instrument MGC --walk-forward --train-years 3 --test-years 1

# Run tests
python -m pytest tests/test_trader_logic.py::TestRandomWalkForwardIntegrity -v

# Expected Output:
# test_oos_trade_count_matches_outcomes PASSED
# test_no_train_test_overlap_in_results PASSED
```

**Verify Assertions Work (Intentionally Break Data):**
```bash
# Edit walk_forward_results.json to introduce count mismatch
# Change first strategy's oos_trade_count to 999

python -m pytest tests/test_trader_logic.py::TestRandomWalkForwardIntegrity::test_oos_trade_count_matches_outcomes -v

# Expected: FAILED - OOS count mismatch (assertion caught the error)
# This proves the test is actually validating, not just passing blindly

# Restore original file
git checkout artifacts/walk_forward/walk_forward_results.json
```

---

## Relationship to Core Pipeline

**Walk-Forward Analysis Position in Workflow:**

```
REQUIRED (Core Pipeline):
├── pipeline/ingest_dbn.py          → bars_1m
├── pipeline/build_bars_5m.py       → bars_5m
├── pipeline/build_daily_features.py → daily_features
├── trading_app/outcome_builder.py  → orb_outcomes
├── trading_app/strategy_discovery.py → experimental_strategies
└── trading_app/strategy_validator.py → validated_setups

OPTIONAL (Research & Analysis):
└── trading_app/walk_forward.py     → walk_forward_results.json
    └── Tests in TestRandomWalkForwardIntegrity
```

**Key Distinction:**
- **Core pipeline tests** (outcome math, entry logic, cost model) → **Always run** (use gold.db or fixtures)
- **Research artifact tests** (walk-forward integrity) → **Conditionally run** (skip if artifacts unavailable)

**This is correct architectural separation.** Walk-forward analysis is a **validation and research tool**, not a core requirement for strategy discovery or live trading.

---

## Recommendations

### Short-Term (No Changes Required)
- ✅ Skips are working as designed
- ✅ Messages are clear and actionable
- ✅ Test behavior is correct in all environments
- ✅ Artifact generation is appropriately separated from core pipeline
- ✅ No code changes needed

### Long-Term (Optional Enhancements)

If walk-forward testing becomes more critical, consider:

1. **CI Artifact Caching**
   - Generate walk-forward artifacts in dedicated CI job
   - Cache in GitHub Actions artifacts
   - Restore before test runs
   - Trade-off: Faster tests vs longer CI setup + storage costs

2. **Minimal Fixture for Smoke Tests**
   - Create `tests/fixtures/walk_forward_minimal.json` with 2-3 synthetic strategies
   - Use fixture when real artifacts unavailable
   - Trade-off: Some test coverage vs not validating real artifact structure

3. **Skip Reason Documentation**
   - Add URL to this document in skip messages
   - Example: `pytest.skip("walk-forward artifacts not present — see tests/ANALYSIS_ARTIFACT_SKIPS.md")`

4. **Test Categorization**
   - Add pytest marker: `@pytest.mark.requires_walk_forward_artifacts`
   - Allow users to run: `pytest -m "not requires_walk_forward_artifacts"` for core tests only
   - Document in `CLAUDE.md`: "Test markers and what they mean"

**None of these are necessary** — current behavior is correct and follows best practices for optional build artifacts.

---

## Conclusion

**DECISION: KEEP BOTH SKIPS AS-IS**

These skips represent **correct fail-safe behavior** for optional research artifacts. They enable:
- ✅ Walk-forward result validation when artifacts are available
- ✅ Fast test feedback without expensive walk-forward computation (5-30 min)
- ✅ Clear distinction between test failures and unavailable artifacts
- ✅ Proper separation of core pipeline vs optional research tools

**No code changes required.** The skip messages are clear, the conditions are correct, and the test behavior matches design intent.

**For future reference:** If these tests skip unexpectedly, check:
1. Does `artifacts/walk_forward/walk_forward_results.json` exist?
2. If not, has walk-forward analysis been run for this instrument?
3. Was a custom `--output-dir` specified in walk_forward.py command?
4. Is the working directory correct? (Tests look for artifacts relative to project root)

**Walk-forward artifacts are OPTIONAL.** They are:
- Generated on-demand for research
- Not required for core pipeline operation
- Not version controlled (too large, research-specific)
- Environment-specific (local research environments only)

**These tests validate walk-forward integrity when you choose to run walk-forward analysis.** They are **not** a requirement for basic pipeline operation or CI.

**Sign-off:** Ready for QA verification per spec requirements.
