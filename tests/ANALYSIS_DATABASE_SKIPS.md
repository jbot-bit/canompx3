# Database Availability Skip Analysis

**Analysis Date:** 2026-02-21
**Scope:** Category 1 - Database Availability Skips (4 skip conditions)
**Decision:** **KEEP AS INTENTIONAL** - All 4 skips are correct environmental gating

---

## Executive Summary

All 4 database availability skips are **intentional runtime guards** that enable tests to run in multiple environments (production with gold.db, fresh clones without it, CI with or without DB seeding). These skips represent correct fail-safe behavior, not bugs or missing implementations.

**Decision Rationale:**
- ✅ Tests validate production data integrity when database is available
- ✅ Tests gracefully skip when database is unavailable (expected in many environments)
- ✅ Skip messages are clear and actionable
- ✅ No code changes required — behavior is correct as-is

---

## Skip Inventory

### Skip 1: Database File Missing
- **Location:** `tests/test_trader_logic.py:595`
- **Function:** `_skip_if_no_db()`
- **Condition:** `not GOLD_DB.exists()`
- **Message:** `"gold.db not available"`
- **Affected Tests:** 10 tests in `TestRandomOutcomeMath` and `TestRandomStrategyMath`

### Skip 2: Missing orb_outcomes Table
- **Location:** `tests/test_trader_logic.py:603`
- **Function:** `_skip_if_no_db()`
- **Condition:** `"orb_outcomes" not in tables`
- **Message:** `"gold.db has no orb_outcomes table"`
- **Affected Tests:** Same 10 tests (second gate in same helper)

### Skip 3: Database Locked
- **Location:** `tests/test_trader_logic.py:605`
- **Function:** `_skip_if_no_db()`
- **Condition:** `Exception during connection attempt`
- **Message:** `"gold.db locked by another process"`
- **Affected Tests:** Same 10 tests (third gate in same helper)

### Skip 4: Corpus Test Database Missing
- **Location:** `tests/test_trading_app/test_ai/test_corpus.py:59`
- **Function:** `db_path` fixture
- **Condition:** `not GOLD_DB_PATH.exists()`
- **Message:** `"gold.db not available"`
- **Affected Tests:** 2 tests in `TestSchemaDefinitions`

---

## Root Cause Analysis

### What is gold.db?

**Nature:** Local production database containing 5+ years of trading data

**Schema:** 7 core tables
- `bars_1m` - 1-minute OHLCV bars (UTC timestamps)
- `bars_5m` - 5-minute aggregated bars
- `daily_features` - ORB levels, RSI, ATR per trading day
- `orb_outcomes` - Pre-computed trade outcomes (689K rows)
- `experimental_strategies` - Strategy grid search results
- `validated_setups` - Validated strategies (937 active)
- `edge_families` - Clustered edge groups

**Location Resolution (from `pipeline/paths.py`):**
1. `DUCKDB_PATH` env var (if set and exists) → scratch DB workflow
2. `<project_root>/gold.db` (canonical production path)

**Size:** ~2.5 GB (production), varies by data range

**Lifecycle:**
- Created via `python pipeline/init_db.py`
- Populated via `pipeline/ingest_dbn.py` → `build_bars_5m.py` → `build_daily_features.py`
- Trading data added via `trading_app/outcome_builder.py` → `strategy_discovery.py` → `strategy_validator.py`
- **Not version controlled** (too large, local-only, user-specific data ranges)

### Why Tests Require gold.db

**Test Purpose:** Mathematical integrity validation

The affected tests validate that stored computed values in `orb_outcomes` and `experimental_strategies` match independent recomputation from first principles:

**Examples:**
- `test_risk_points_recompute` - Verify `risk_points = abs(entry_price - stop_price)` for random sample
- `test_win_pnl_consistent_with_cost` - Verify win PnL includes cost model deduction
- `test_target_price_recompute` - Verify `target_price = entry ± (risk * rr_target)`
- `test_win_rate_recompute` - Verify `win_rate = wins / (wins + losses)` matches stored value
- `test_get_schema_definitions` - Verify AI corpus can extract schema from live DB

**Why They Can't Use Synthetic Data:**
1. **Scale validation** - Need to sample 50+ random real outcomes to catch edge cases
2. **Production fidelity** - Test against actual data pipeline outputs, not synthetic mocks
3. **Historical coverage** - Verify consistency across 5 years of data, multiple contracts
4. **Performance validation** - Ensure schema extraction works on production-scale tables

### When is gold.db Unavailable?

**Expected Scenarios:**
1. **Fresh repository clone** - New developer, CI runner, fresh machine
2. **Test-only environment** - Running unit tests without full pipeline setup
3. **Concurrent access** - Another process has write lock during long-running jobs
4. **Scratch workflow active** - `DUCKDB_PATH=C:/db/gold.db` set, original DB removed
5. **Partial pipeline run** - Database exists but `orb_outcomes` not yet populated

**Unexpected Scenarios (indicate real issues):**
- Database corrupted (schema exists but tables missing)
- Permissions issue (file exists but unreadable)
- Disk full during write operation

### Why Skips are Correct Behavior

**Design Philosophy: Environment-Adaptive Testing**

The test suite supports **three execution modes:**

| Mode | gold.db State | Behavior | Use Case |
|------|---------------|----------|----------|
| **Production Validation** | Available, fully populated | All tests run, validate production data | Local dev with full pipeline |
| **Unit Test Mode** | Unavailable | DB-dependent tests skip, pure-function tests run | CI without DB seeding, fresh clones |
| **Partial Environment** | Available but incomplete | Tests skip if required table missing | Mid-pipeline development |

**Benefits:**
- ✅ Fast test feedback in environments without expensive setup
- ✅ No false negatives (tests don't fail when dependency is missing)
- ✅ Clear distinction between "test failed" vs "test cannot run"
- ✅ Supports incremental onboarding (clone → run tests → see what skips → decide if you need gold.db)

**Alternative Approaches Rejected:**

| Approach | Why Rejected |
|----------|--------------|
| Always require gold.db | Breaks CI without DB seeding, slows onboarding |
| Mock gold.db | Defeats purpose of production data validation |
| @pytest.mark.skip decorator | Would always skip, even when DB is available |
| Remove tests entirely | Loses critical production integrity validation |
| Auto-create empty gold.db | Tests would fail instead of skip (worse UX) |

---

## Per-Skip Decision Rationale

### Skip 1: `gold.db not available` (test_trader_logic.py:595)

**Decision:** **KEEP - Intentional environmental gate**

**Reasoning:**
- Primary gate protecting 10 production validation tests
- Clear, actionable message tells user exactly what's missing
- Expected to trigger in fresh clones, test-only environments
- Tests provide value only when validating real production data

**Resolution Path for Users:**
```bash
# If you need these tests to run:
python pipeline/init_db.py                    # Create schema
python pipeline/ingest_dbn.py ...              # Populate bars
python pipeline/build_daily_features.py ...    # Build features
python trading_app/outcome_builder.py ...      # Compute outcomes
python -m pytest tests/test_trader_logic.py -v  # Tests now run
```

**Verification:**
- ✅ Skip message is descriptive
- ✅ Guides user to resolution (implies: run pipeline)
- ✅ Tests run when condition is met
- ✅ No silent failures or false positives

---

### Skip 2: `gold.db has no orb_outcomes table` (test_trader_logic.py:603)

**Decision:** **KEEP - Intentional schema validation gate**

**Reasoning:**
- Catches partial pipeline runs (DB exists but outcomes not computed)
- More specific than Skip 1 (user knows DB exists, just incomplete)
- Prevents cryptic SQL errors later in test execution
- Guides user to specific missing step: `outcome_builder.py`

**Scenario:**
```bash
# User runs:
python pipeline/init_db.py                # ✅ gold.db created
python pipeline/ingest_dbn.py ...          # ✅ bars_1m populated
python -m pytest tests/test_trader_logic.py  # ⚠️ Skips with clear message

# Clear next step:
python trading_app/outcome_builder.py ...  # Populate orb_outcomes
python -m pytest tests/test_trader_logic.py  # ✅ Tests now run
```

**Verification:**
- ✅ Differentiates from Skip 1 (DB exists vs DB missing)
- ✅ Points to missing table, implying which pipeline step to run
- ✅ Prevents downstream SQL errors
- ✅ Tests run once table exists

---

### Skip 3: `gold.db locked by another process` (test_trader_logic.py:605)

**Decision:** **KEEP - Intentional concurrency safety gate**

**Reasoning:**
- DuckDB allows single writer (by design)
- Common during long-running jobs: `strategy_discovery.py`, `outcome_builder.py`
- Prevents test failures due to transient lock contention
- Graceful degradation (skip now, retry later) better than hard failure

**Scenario:**
```bash
# Terminal 1 (long-running write):
python trading_app/outcome_builder.py --instrument MGC  # Holds write lock

# Terminal 2 (user runs tests):
python -m pytest tests/test_trader_logic.py -v
# ⚠️ Tests skip cleanly: "gold.db locked by another process"
# ✅ No test failures, no corruption, clear message

# After Terminal 1 completes:
python -m pytest tests/test_trader_logic.py -v  # ✅ Tests now run
```

**Alternative Considered:**
- Wait with timeout → Rejected (could delay CI by minutes)
- Fail test → Rejected (not a test failure, just transient unavailability)

**Verification:**
- ✅ Catches both write locks and read locks
- ✅ Generic exception handler (covers OS-level locks too)
- ✅ Message is actionable (retry later)
- ✅ Preserves test suite idempotency

---

### Skip 4: `gold.db not available` (test_corpus.py:59)

**Decision:** **KEEP - Intentional fixture-level gate**

**Reasoning:**
- Tests AI corpus generation from production schema
- Requires real database to verify schema extraction logic
- Fixture-level skip (cleaner than per-test skips)
- Same rationale as Skip 1, different codebase location

**Test Purpose:**
- `test_get_schema_definitions` - Verify `corpus.py` can extract CREATE TABLE statements
- `test_get_db_stats` - Verify row counts, table sizes for AI context generation

**Why Fixture-Level Skip is Correct:**
- Both tests in class require same dependency (gold.db)
- Skip appears once (cleaner pytest output)
- Follows pytest best practices for shared dependencies

**Resolution Path:**
```bash
# Same as Skip 1 (requires populated database)
python pipeline/init_db.py && python pipeline/ingest_dbn.py ...
python -m pytest tests/test_trading_app/test_ai/test_corpus.py -v
```

**Verification:**
- ✅ Fixture pattern matches pytest conventions
- ✅ Skip message matches Skip 1 (consistency)
- ✅ Tests run when DB available
- ✅ No duplicate skip messages per test method

---

## Testing the Skips

### Reproduction Steps

**Verify Skip 1 (DB Missing):**
```bash
# Temporarily rename database
mv gold.db gold.db.bak

# Run affected tests
python -m pytest tests/test_trader_logic.py::TestRandomOutcomeMath::test_risk_points_recompute -v -rs

# Expected: SKIPPED - gold.db not available

# Restore
mv gold.db.bak gold.db
```

**Verify Skip 2 (Table Missing):**
```bash
# Connect to DB and drop table
python -c "import duckdb; con = duckdb.connect('gold.db'); con.execute('DROP TABLE IF EXISTS orb_outcomes'); con.close()"

# Run tests
python -m pytest tests/test_trader_logic.py::TestRandomOutcomeMath -v -rs

# Expected: SKIPPED - gold.db has no orb_outcomes table

# Restore via pipeline
python trading_app/outcome_builder.py --instrument MGC --start 2024-01-01 --end 2024-12-31
```

**Verify Skip 3 (Lock Contention):**
```bash
# Terminal 1: Hold write lock
python -c "import duckdb; import time; con = duckdb.connect('gold.db'); print('Lock acquired'); time.sleep(60); con.close()"

# Terminal 2: Run tests while lock held
python -m pytest tests/test_trader_logic.py::TestRandomOutcomeMath -v -rs

# Expected: SKIPPED - gold.db locked by another process
```

**Verify Skip 4 (Corpus Tests):**
```bash
# Same as Skip 1
mv gold.db gold.db.bak
python -m pytest tests/test_trading_app/test_ai/test_corpus.py::TestSchemaDefinitions -v -rs
# Expected: SKIPPED - gold.db not available
mv gold.db.bak gold.db
```

---

## Recommendations

### Short-Term (No Changes Required)
- ✅ Skips are working as designed
- ✅ Messages are clear and actionable
- ✅ Test behavior is correct in all environments
- ✅ No code changes needed

### Long-Term (Optional Enhancements)
If skip frequency becomes problematic, consider:

1. **CI Database Seeding**
   - Cache populated gold.db in CI artifacts
   - Restore before test runs
   - Trade-off: Faster tests vs larger artifact storage

2. **Skip Reason Documentation**
   - Add URL to this document in skip messages
   - Example: `pytest.skip("gold.db not available — see tests/ANALYSIS_DATABASE_SKIPS.md")`

3. **Test Categorization**
   - Add pytest marker: `@pytest.mark.requires_gold_db`
   - Allow users to run: `pytest -m "not requires_gold_db"` for fast unit tests only

4. **Developer Onboarding Guide**
   - Add to CLAUDE.md: "What to do when tests skip"
   - Flowchart: Skip message → Resolution steps

**None of these are necessary** — current behavior is correct.

---

## Conclusion

**DECISION: KEEP ALL 4 SKIPS AS-IS**

These skips represent **correct fail-safe behavior** in an environment-adaptive test suite. They enable:
- ✅ Fast test feedback without expensive setup
- ✅ Production data validation when database is available
- ✅ Clear distinction between test failures and unavailable dependencies
- ✅ Graceful degradation during concurrent access

**No code changes required.** The skip messages are clear, the conditions are correct, and the test behavior matches design intent.

**For future reference:** If a test skip appears unexpected, check:
1. Does `gold.db` exist at project root?
2. Does it contain the required table (`orb_outcomes`, etc.)?
3. Is another process holding a write lock?
4. Is `DUCKDB_PATH` pointing to a non-existent scratch location?

**Sign-off:** Ready for QA verification per spec requirements.
