# Phase 3 Implementation Plan: trading_app

> **HISTORICAL DOCUMENT** — Phase 3 is COMPLETE. This plan was used during
> initial implementation. For current state, see `CLAUDE.md` and `ROADMAP.md`.
> LOC counts, test counts, and row counts below are from the original session
> and may not reflect current values.

## Status Snapshot (2026-02-07)

### DONE (verified working)
| File | Tests | Status |
|------|-------|--------|
| `trading_app/__init__.py` | - | Module init |
| `trading_app/config.py` | 34 tests (test_config.py) | Filters, ALL_FILTERS registry |
| `trading_app/execution_spec.py` | 17 tests (test_execution_spec.py) | ExecutionSpec dataclass |
| `trading_app/entry_rules.py` | 15 tests (test_entry_rules.py) | Confirm bars detection |
| `trading_app/db_manager.py` | 8 tests (test_db_manager.py) | Schema for 4 tables |
| `pipeline/check_drift.py` | 29 tests (test_check_drift.py) | Checks 1-11 (incl 3 trading_app) |

**Baseline: 241 tests, all passing. 11 drift checks, all passing.**

### BUG FIXES APPLIED THIS SESSION
1. `db_manager.py`: Fixed broken import (`get_db_path` → `GOLD_DB_PATH`)
2. `config.py`: Removed dead dict comprehension in ALL_FILTERS
3. **GC REFACTOR**: Switched from MGC to GC outrights for price data (better coverage)
   - `ingest_dbn_mgc.py`: `GC_OUTRIGHT_PATTERN`, `prefix_len=2`
   - `ingest_dbn_daily.py`: Same changes
   - All tests updated (contract selection, ingest daily)
   - CLAUDE.md, CANONICAL_LOGIC.txt, MEMORY.md all updated
   - **DATABASE NEEDS RE-INGEST** (Step 0 below)

---

## Step 0 (PREREQUISITE): Re-ingest with GC bars

**Why**: Current gold.db was ingested from MGC bars (~78% coverage). ORBs are inaccurate.
GC has ~100% coverage. Pipeline code is already updated to use GC_OUTRIGHT_PATTERN.

**Commands**:
```bash
python pipeline/init_db.py --force        # Wipe bars_1m, bars_5m, daily_features
python pipeline/ingest_dbn_daily.py --start 2021-02-05 --end 2026-02-05
python pipeline/build_bars_5m.py --instrument MGC --start 2021-02-05 --end 2026-02-05
python pipeline/build_daily_features.py --instrument MGC --start 2021-02-05 --end 2026-02-05
```

**AUDIT GATE**: After re-ingest, verify:
- bars_1m row count HIGHER than before (was 1,751,302)
- source_symbol values start with 'GC' not 'MGC'
- daily_features still has ~1,460 rows
- All tests pass, all drift checks pass

---

## 2-PASS AUDIT GATE (runs after EVERY step)

Every step ends with this mandatory 2-pass audit before proceeding.

### Pass 1: Mechanical Verification
Run these commands. ALL must succeed.
```
1. python -m pytest tests/ -x -q                    # ALL tests pass
2. python pipeline/check_drift.py                   # ALL 11 drift checks pass
3. python -c "from trading_app.<module> import ..."  # Every new module imports cleanly
4. python -c "from tests.test_trading_app.test_<module> import *"  # Every new test file imports
```

### Pass 2: Honest Self-Audit Checklist
Answer each question. If ANY answer is NO, fix before proceeding.

| # | Question | Required |
|---|----------|----------|
| 1 | Does every `duckdb.connect()` have a matching `finally: con.close()`? | YES |
| 2 | Does every import resolve to a real function/class that exists? | YES |
| 3 | Are there any TODO/FIXME/stub functions left unimplemented? | NO |
| 4 | Does the import map in this plan still match reality? | YES |
| 5 | Does every new file follow the one-way dependency rule? | YES |
| 6 | Are there any dead code paths (unreachable branches, empty dicts)? | NO |
| 7 | Do test counts match what the plan predicted (within ±3)? | YES |
| 8 | Can every CLI entry point run with `--help` without crashing? | YES |

If the audit reveals issues, document them in the "BUG FIXES" section above and fix them before moving on. This saves tokens by catching problems early instead of rewriting later.

---

## REMAINING WORK — 4 Steps

### Step 1: outcome_builder.py + tests
**Purpose**: Pre-compute outcomes for every (trading_day, orb_label, rr_target, confirm_bars) combination. This populates the `orb_outcomes` table.

**File**: `trading_app/outcome_builder.py` (~350-450 LOC)

**Logic**:
1. For each trading day + orb_label in daily_features:
   - Skip if no break (break_dir IS NULL)
   - For each rr_target in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
     - For each confirm_bars in [1, 2, 3]:
       - Call `detect_entry_with_confirm_bars()` on bars_1m
       - If entry triggered:
         - Compute target_price from entry + rr_target * risk
         - Scan bars_1m forward to determine outcome (win/loss/scratch)
         - Track MAE/MFE using `pnl_points_to_r()` from cost_model
       - Write row to orb_outcomes (INSERT OR REPLACE)

**Inputs**:
- `daily_features` table (break_dir, break_ts, orb_high, orb_low)
- `bars_1m` table (for confirm bars detection + excursion tracking)
- `pipeline.cost_model` (for R-multiple conversion)

**CLI**: `python trading_app/outcome_builder.py --instrument MGC --start 2024-01-01 --end 2024-12-31 [--orb-minutes 5] [--dry-run]`

**Tests**: `tests/test_trading_app/test_outcome_builder.py` (~15 tests)
- Long win at RR=2.0 with confirm_bars=1
- Short loss scenario
- No-break day produces no rows
- No-confirm produces NULL entry fields
- MAE/MFE computation correctness
- Dry-run produces no DB writes
- Idempotent (re-running doesn't duplicate)
- RR targets grid coverage

**Dependencies**: entry_rules.py, cost_model.py, db_manager.py (all tested)

**Acceptance**: `orb_outcomes` table populated, tests pass, drift checks pass.

**AUDIT GATE**: Run 2-pass audit. Specifically verify:
- `outcome_builder.py` imports match plan import map
- `duckdb.connect()` has `finally: con.close()`
- `--help` flag works
- Test count ~15 (±3)

---

### Step 2: setup_detector.py + strategy_discovery.py + tests
**Purpose**: Filter trading days by market conditions, run backtests over all strategy variants, save results to `experimental_strategies`.

**File 1**: `trading_app/setup_detector.py` (~120-150 LOC)

**Logic**:
- `detect_setups(con, filter, orb_label, instrument, start, end)` → list of matching trading_days
- Builds SQL query against daily_features using filter.matches_row() or SQL WHERE clause
- Returns list of (trading_day, row_dict) tuples

**File 2**: `trading_app/strategy_discovery.py` (~350-400 LOC)

**Logic**:
1. Grid search over:
   - 6 orb_labels x 6 rr_targets x 3 confirm_bars x 8 filters = 864 combos (upper bound)
2. For each combo:
   - Query orb_outcomes WHERE filter matches (via setup_detector)
   - Compute metrics: sample_size, win_rate, avg_win_r, avg_loss_r, expectancy_r
   - Compute sharpe_ratio and max_drawdown_r
   - Compute yearly breakdown (JSON)
   - Generate strategy_id = f"{instrument}_{orb_label}_RR{rr_target}_CB{confirm_bars}_{filter_type}"
   - INSERT into experimental_strategies

**CLI**: `python trading_app/strategy_discovery.py --instrument MGC --start 2024-01-01 --end 2024-12-31 [--orb-minutes 5] [--dry-run]`

**Tests**: `tests/test_trading_app/test_setup_detector.py` (~8 tests)
- Filter correctly restricts days
- NoFilter returns all days
- OrbSizeFilter boundary behavior
- Empty result for impossible filter

**Tests**: `tests/test_trading_app/test_strategy_discovery.py` (~12 tests)
- Win rate computation
- Expectancy computation (E = WR*AvgWin - LR*AvgLoss)
- Sharpe ratio computation
- Max drawdown computation
- Yearly breakdown structure
- Strategy ID format
- Dry-run mode
- Grid produces correct number of strategies

**Dependencies**: outcome_builder.py (Step 1 must be done first)

**Acceptance**: `experimental_strategies` table populated, tests pass, drift checks pass.

**AUDIT GATE**: Run 2-pass audit. Specifically verify:
- Both `setup_detector.py` and `strategy_discovery.py` import cleanly
- No SQL injection (all params via ? placeholders)
- Expectancy formula matches CANONICAL_LOGIC.txt section 4: `E = (WR * AvgWin_R) - (LR * AvgLoss_R)`
- Test count ~20 (±3)

---

### Step 3: strategy_validator.py + tests
**Purpose**: 6-phase validation framework per CANONICAL_LOGIC.txt section 9. Promotes passing strategies to `validated_setups`.

**File**: `trading_app/strategy_validator.py` (~250-300 LOC)

**Logic**:
1. **Phase 1 — Sample size**: Reject if < 30, warn if < 100
2. **Phase 2 — Post-cost expectancy**: ExpR > 0 using $8.40 RT friction
3. **Phase 3 — Yearly robustness**: Positive in ALL years (from yearly_results JSON)
4. **Phase 4 — Stress test**: ExpR > 0 at +50% costs (via `stress_test_costs()`)
5. **Phase 5 — Sharpe ratio** (optional quality filter, configurable threshold)
6. **Phase 6 — Max drawdown** (optional risk filter, configurable threshold)

**Outputs**:
- Update `experimental_strategies.validation_status` = 'PASSED' | 'REJECTED'
- Update `experimental_strategies.validation_notes` with rejection reason
- Promote passing strategies: INSERT into `validated_setups` with denormalized params
- Archive retired strategies: INSERT into `validated_setups_archive`

**CLI**: `python trading_app/strategy_validator.py --instrument MGC [--min-sample 100] [--stress-multiplier 1.5] [--dry-run]`

**Tests**: `tests/test_trading_app/test_strategy_validator.py` (~15 tests)
- Sample size < 30 → REJECT
- Sample size 30-99 → WARN but continue
- Sample size >= 100 → PASS
- Negative ExpR → REJECT
- One year negative → REJECT (yearly robustness)
- Stress test failure → REJECT
- All phases pass → promoted to validated_setups
- Already-validated strategy not re-validated
- Validation notes contain rejection reason

**Dependencies**: strategy_discovery.py (Step 2 must be done first)

**Acceptance**: Validator rejects/promotes correctly, tests pass, drift checks pass.

**AUDIT GATE**: Run 2-pass audit. Specifically verify:
- Validation phases match CANONICAL_LOGIC.txt section 9 exactly (4 mandatory checks)
- `stress_test_costs(multiplier=1.5)` is called, not hand-rolled
- Promoted strategies have ALL denormalized fields populated (no NULLs where NOT NULL)
- Test count ~15 (±3)

---

### Step 4: Integration test + final verification
**Purpose**: End-to-end test of the full pipeline: daily_features → outcome_builder → strategy_discovery → strategy_validator.

**File**: `tests/test_trading_app/test_integration.py` (~5 tests)

**Tests**:
- End-to-end: synthetic data → outcomes → discovery → validation
- Rejected strategy does NOT appear in validated_setups
- Promoted strategy has correct denormalized fields
- Re-running is idempotent
- Yearly breakdown JSON is valid and parseable

**AUDIT GATE**: Run 2-pass audit. Additionally:
- Verify total test count across ALL files (target: 300+)
- Verify ALL trading_app modules import cleanly in a single script
- Run `python trading_app/db_manager.py --verify` against gold.db
- Cross-check: every function referenced in the plan's import map actually exists

**Final checklist**:
- [ ] All tests pass (target: 300+ tests)
- [ ] All 11 drift checks pass
- [ ] `python trading_app/db_manager.py --verify` passes
- [ ] CLAUDE.md updated to reflect new files
- [ ] ROADMAP.md Phase 3 marked DONE
- [ ] MEMORY.md updated
- [ ] Commit all changes

---

## Architecture Rules (enforced by drift checks)

1. **One-way dependency**: `trading_app/` CAN import from `pipeline/`. `pipeline/` NEVER imports from `trading_app/`. (Drift check 9)
2. **Connection cleanup**: All `duckdb.connect()` must have `finally: con.close()`. (Drift checks 7, 10)
3. **No hardcoded paths**: Use `pipeline.paths.GOLD_DB_PATH`. (Drift checks 6, 11)
4. **Fail-closed**: Unknown instruments → ValueError. Invalid params → ValueError. Missing data → skip (don't crash).

## Import Map

```
trading_app/outcome_builder.py imports:
  - from pipeline.paths import GOLD_DB_PATH
  - from pipeline.cost_model import get_cost_spec, pnl_points_to_r
  - from pipeline.init_db import ORB_LABELS
  - from trading_app.entry_rules import detect_entry_with_confirm_bars
  - from trading_app.db_manager import init_trading_app_schema

trading_app/setup_detector.py imports:
  - from pipeline.paths import GOLD_DB_PATH
  - from trading_app.config import ALL_FILTERS, StrategyFilter

trading_app/strategy_discovery.py imports:
  - from pipeline.paths import GOLD_DB_PATH
  - from pipeline.cost_model import get_cost_spec
  - from pipeline.init_db import ORB_LABELS
  - from trading_app.config import ALL_FILTERS
  - from trading_app.setup_detector import detect_setups

trading_app/strategy_validator.py imports:
  - from pipeline.paths import GOLD_DB_PATH
  - from pipeline.cost_model import get_cost_spec, stress_test_costs
  - from trading_app.db_manager import init_trading_app_schema
```

## Estimated Row Counts

| Table | Expected Rows |
|-------|---------------|
| orb_outcomes | ~50,000-80,000 (not all ORBs break, not all confirm) |
| experimental_strategies | ~500-864 strategy variants |
| validated_setups | < 50 (highly selective) |
| validated_setups_archive | 0 initially |

## Order of Execution

```
Step 1 → AUDIT → Step 2 → AUDIT → Step 3 → AUDIT → Step 4 → FINAL AUDIT
  ↓                 ↓                 ↓                 ↓
outcome          discovery         validator        integration
builder          + setup           framework          test
                 detector
```

Each step is independently testable. Each step MUST pass the 2-pass audit before proceeding to the next. No exceptions.
