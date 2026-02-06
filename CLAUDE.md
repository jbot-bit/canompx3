# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Gold (MGC) Data Pipeline for building a clean, replayable local dataset of Micro Gold futures (MGC / MGC1!) for ORB-based discretionary trading, systematic backtesting, and session statistics analysis.

**Primary Focus**: 09:00, 10:00, 11:00 ORBs
**Secondary**: 18:00, 23:00, 00:30 ORBs

---

## 🚨 CRITICAL: LIVE TRADING TEST REQUIREMENTS

**MANDATORY STANDING RULE**: Any code that runs in LIVE mode must pass a "boundary + state" test suite.

This rule is **NON-NEGOTIABLE** and applies to:
- Entry/exit logic
- Position sizing calculations
- Risk management code
- Data validation
- State management
- API integrations
- Database queries
- Price/P&L calculations

### What This Means:
1. **Boundary Tests**: Test empty data, null values, edge cases, extreme inputs
2. **State Tests**: Test symbol changes, connection loss, race conditions, state corruption
3. **Coverage Requirement**: 80%+ test coverage for live trading code
4. **Enforcement**: All PRs must include tests; CI/CD blocks merges without tests

### Quick Reference:
- See: `LIVE_TRADING_TEST_REQUIREMENTS.md` (complete specification)
- Test Templates: `tests/templates/boundary_test_template.py`, `tests/templates/state_test_template.py`
- Run Tests: `python -m pytest tests/boundary/ tests/state/ -v`

### Why This Exists:
Ghost audit found 53 bugs, 40% were boundary/state issues. Live trading requires zero tolerance for crashes. Real money at risk.

**Before writing ANY live trading code, read `LIVE_TRADING_TEST_REQUIREMENTS.md`.**

---

## 🛡️ CRITICAL RULE: Strategy Family Isolation

**All analysis, validation, and conclusions apply ONLY to the active STRATEGY_FAMILY.**

**Cross-family inference is FORBIDDEN unless explicitly promoted.**

This means:
- ✅ CORRECT: "5min filter validated for ORB_L4 family (1000 ORB)"
- ❌ WRONG: "5min filter works for ORB_L4, so it should work for ORB_RSI"
- ❌ WRONG: "L4 validated, so all session types should work"

**Strategy Families** (see `strategy_families/` directory):
1. **ORB_L4** - L4_CONSOLIDATION filter (0900, 1000 ORBs)
2. **ORB_BOTH_LOST** - Sequential failure pattern (1100 ORB)
3. **ORB_RSI** - Momentum exhaustion (1800 ORB)
4. **ORB_NIGHT** - Night sessions (2300, 0030 ORBs) - RESEARCH ONLY

Each family is a memory container. Findings stay inside. No leakage.

**When working on a strategy:**
1. Identify its STRATEGY_FAMILY
2. Open `strategy_families/{FAMILY}.md`
3. Work ONLY on that family
4. Document findings in the family file
5. DO NOT apply findings to other families

This prevents:
- Cross-contamination of findings
- "Does this apply to X?" confusion
- Re-discovering old findings (API waste)
- Overwhelm from tracking multiple contexts

**See**: `strategy_families/README.md` for full explanation and template.

---

## ⚡ Skills Integration

This project uses Anthropic skills for specialized tasks. **IMPORTANT:** These skills are designed to AUTO-ACTIVATE. Claude should use them proactively, not wait for explicit requests.

### ⚡ PROACTIVE SKILL USAGE (MANDATORY)

**Claude MUST use these skills automatically in these situations:**

1. **BEFORE editing ANY protected file** → Activate code-guardian
   - Pipeline files, trading_app files, config.py, execution_engine.py, schema.sql
   - Validated_setups database
   - NO EXCEPTIONS - show protection warnings FIRST

2. **WHEN user asks navigation questions** → Activate quick-nav
   - "Where is...", "Find...", "Locate...", "Show me..."
   - Give ONE clear answer, not multiple options

3. **WHEN user seems overwhelmed** → Activate focus-mode
   - Multiple unrelated questions
   - "What should I do?"
   - Context switching mid-conversation
   - Offer structure and prioritization

4. **WHEN analyzing research results** → Activate code-guardian (meta-logic protection)
   - Edge discovery outputs
   - Optimization results
   - Before trusting any "this looks good" moment
   - Run validation checklists

5. **WHEN root directory is messy** → Offer project-organizer
   - If user mentions "can't find things"
   - If they seem frustrated with file count
   - Ask once per session if files are organized
   - **CRITICAL:** ALWAYS check for imports before moving Python files

6. **WHEN validating trading strategies** → Activate strategy-validator **[AUTO-ACTIVATES]**
   - User says "validate strategies", "test setups", "approve strategies"
   - User asks "are these strategies correct?" or "do strategies work?"
   - User updates validated_setups database
   - User mentions TCA, expectancy, or stress testing
   - Run autonomous 6-phase validation framework

**These are NOT optional. Use the skills automatically when the conditions match.**

### 🚨 MANDATORY FILE MOVE SAFETY (NEVER SKIP THIS)

**BEFORE moving ANY Python file (during organization or refactoring):**

```bash
# Check if file is imported elsewhere
grep -r "from [filename_no_py]\|import [filename_no_py]" \
  pipeline/ trading_app/ analysis/ *.py 2>/dev/null
```

**If ANY imports found:**
- ❌ DO NOT MOVE
- ⚠️ WARN USER immediately
- 🛡️ Either keep in place or update imports first

**This check is MANDATORY. Moving imported files breaks production code.**

### 🛡️ Code Guardian (`skills/code-guardian/`) **[AUTO-ACTIVATES]**
**CRITICAL - Use BEFORE any edit to protected files**

**Auto-activates when:**
- Editing pipeline/, trading_app/, execution_engine.py, config.py
- Modifying validated_setups database
- Changing ORB calculations
- Updating schema.sql
- ANY change to production trading logic

**Also activates when:**
- Analyzing edge discovery results (overfitting check)
- Optimizing strategy parameters (curve-fitting check)
- Trusting new research (validation gate)
- Adding strategies to validated_setups (integrity checklist)

**Key protections:**
- Prevents breaking database/config sync
- Backs up files before edits
- Runs test_app_sync.py automatically
- Validates research integrity
- Documents assumptions
- Stress tests strategies

**Usage:** This skill activates automatically. Claude will show protection warnings BEFORE making dangerous changes.

### 📍 Quick Nav (`skills/quick-nav/`) **[AUTO-ACTIVATES]**
**ADHD-optimized instant navigation**

**Auto-activates when user asks:**
- "Where is X?"
- "Find Y"
- "Locate Z"
- "Show me..."
- "I can't find..."
- "Navigate to..."

**Key features:**
- ONE clear answer (no walls of text)
- Exact file paths with line numbers
- Common commands to use
- Visual markers (📍 🔧 ⚠️)
- Project structure mental map

**Usage:** Automatically helps with navigation questions. Keeps responses < 5 lines.

### 🗂️ Project Organizer (`skills/project-organizer/`) **[AUTO-ACTIVATES]**
**Cleans up root directory**

**Auto-activates when:**
- User says "organize", "clean up", "too many files"
- Root directory has 50+ files (auto-detects)
- User seems overwhelmed by file count
- User asks "where should I put this?"

**Key features:**
- Moves check_*.py to scripts/check/
- Moves test_*.py to scripts/test/
- Moves analyze_*.py to scripts/analyze/
- Reduces root to < 20 essential files
- ADHD-friendly: one category at a time

**Usage:** Automatically offers to organize when needed. Always asks before moving files.

### 🎯 Focus Mode (`skills/focus-mode/`) **[AUTO-ACTIVATES]**
**ADHD task management and focus system**

**Auto-activates when user:**
- Says "what should I do?", "I'm stuck", "overwhelmed"
- Asks multiple unrelated questions in one message
- Switches topics mid-conversation
- Shows decision paralysis
- Says "too many things"

**Key features:**
- ONE TASK at a time
- 25-minute focus blocks (Pomodoro)
- Context saving for task switching
- Decision paralysis helpers
- Progress celebration

**Usage:** Automatically detects when user needs focus help. Provides structure and prioritization.

### ✅ Strategy Validator (`skills/strategy-validator/`) **[AUTO-ACTIVATES]**
**Autonomous 6-phase validation framework for trading strategies**

**Auto-activates when user:**
- Says "validate strategies", "test setups", "approve strategies"
- Asks "are these strategies correct?" or "do strategies work?"
- Updates validated_setups database
- Mentions TCA, expectancy, or stress testing
- Asks about strategy approval or rejection

**Key features:**
- **Phase 1:** Ground truth discovery (reverse engineer filters from database)
- **Phase 2:** Data integrity validation (check for NULL/invalid values)
- **Phase 3:** Single-trade reconciliation (verify CANONICAL formulas)
- **Phase 4:** Statistical validation (sample size, expectancy at $8.40 MANDATORY)
- **Phase 5:** Stress testing (+25%, +50% costs)
- **Phase 6:** Iterative correction & documentation

**Approval thresholds:**
- **EXCELLENT:** ExpR >= +0.15R at $8.40 AND survives +50% stress
- **MARGINAL:** ExpR >= +0.15R at $8.40 AND survives +25% stress only
- **WEAK:** ExpR >= +0.15R at $8.40 but fails stress tests
- **REJECTED:** ExpR < +0.15R at $8.40 OR N < 30

**Key principles:**
- HONESTY OVER OUTCOME (reject if fails, regardless of past claims)
- NO ASSUMPTIONS (reverse engineer from ground truth data)
- MANDATORY $8.40 RT costs (honest double-spread accounting, lower costs for comparison only, NOT approval)
- INDIVIDUAL validation (each strategy tested independently)
- CANONICAL formulas (CANONICAL_LOGIC.txt lines 76-98)

**Usage:** Run `python scripts/audit/autonomous_strategy_validator.py` or skill auto-activates when validation requested.

**Reference documents:**
- `CANONICAL_LOGIC.txt` (calculation formulas)
- `TCA.txt` (transaction cost analysis)
- `audit.txt` (meta-audit principles)
- `COST_MODEL_MGC_TRADOVATE.txt` ($8.40 specification - honest double-spread accounting)
- `scripts/audit/VALIDATION_METHODOLOGY.md` (complete framework)

### Database Design (`skills/database-design/`)
**When to use:** Designing database schemas, planning migrations, optimizing queries, selecting databases/ORMs.
- Read `skills/database-design/SKILL.md` for database design thinking
- Read `skills/database-design/schema-design.md` for normalization and relationships
- Read `skills/database-design/migrations.md` for safe schema transitions
- Read `skills/database-design/indexing.md` for index strategies
- Read `skills/database-design/optimization.md` for query performance

**Key principles:**
- Think before copying SQL patterns
- Choose appropriate database for context (DuckDB for analytical, SQLite for embedded)
- Plan indexes strategically
- Safe migrations (no data loss)
- Avoid N+1 queries

**Current project database:**
- DuckDB (analytical, OLAP workload)
- Primary tables: bars_1m, bars_5m, daily_features, validated_setups
- Always test migrations with test_app_sync.py

### Code Review Pipeline (`skills/code-review-pipeline/`)
**When to use:** Reviewing critical code changes before commit, especially trading logic, database changes, or config updates.
- Read `skills/code-review-pipeline/SKILL.md` for multi-agent review methodology
- Uses 4 specialized agents in parallel: Code Reviewer, Security Auditor, Architect Reviewer, Test Analyzer
- Cross-validation boost: issues flagged by multiple agents get elevated severity
- 3-stage pipeline: Automated checks → Parallel agent reviews → Consensus report
- Takes 2-3 minutes but catches 90%+ of critical bugs before production

**When to ALWAYS use:**
- Trading strategy logic changes (ORB calculations, entry/exit rules)
- Database schema migrations or validated_setups updates
- Config file changes (trading_app/config.py)
- Financial calculations (R-multiples, P&L, position sizing)
- Session window or timezone handling changes

**Key principles:**
- No CRITICAL issues allowed (blocks merge)
- HIGH issues must be resolved or acknowledged
- Always run test_app_sync.py for trading logic changes
- Security vulnerabilities get immediate escalation
- Database/config sync violations are CRITICAL
- Idempotency and timezone handling verified

**Severity levels:**
- CRITICAL: Financial loss risk, data corruption, production crash → Blocks merge
- HIGH: Likely production bug, logic flaw, sync violation → Fix before deploy
- MEDIUM: Code quality concern, minor bug potential → Fix recommended
- LOW: Style issue, minor optimization → Optional

### Brainstorming (`skills/brainstorming/`)
**When to use:** Planning new features, redesigning components, exploring architectural changes, preventing feature bloat.
- Read `skills/brainstorming/SKILL.md` for structured design process
- Uses 3-phase process: Understanding → Design → Validation
- Prevents bloat through YAGNI principles and incremental validation
- Collaborative dialogue approach

**Process:**
1. **Understanding** - Examine current state, ask questions sequentially
2. **Design** - Propose minimal viable version, get user feedback
3. **Validation** - Check assumptions, verify constraints

**Key principles:**
- Ask questions sequentially (not all at once)
- Design smallest viable version first
- Validate before building
- Prevent feature creep through YAGNI

**Usage:** Invoke when planning new features or significant changes to existing code.

### Reflect (`skills/reflect.md`)
**When to use:** Session-end learning, capturing insights, improving workflow, continuous skill improvement.
- Analyzes conversation for learnings and patterns
- Auto-applies high-confidence improvements to skills
- Stages medium/low confidence learnings in `_memory_log.md` for review
- Continuous improvement cycle

**What it learns:**
- Project-specific patterns (coding conventions, common operations)
- Workflow optimizations (frequent tasks, efficiency improvements)
- Error patterns (common mistakes, prevention strategies)
- User preferences (communication style, tool preferences)

**Confidence levels:**
- HIGH: Auto-applied to skill files immediately
- MEDIUM: Staged in _memory_log.md, requires approval
- LOW: Noted for observation, may become MEDIUM after repetition

**Usage:** Currently manual (`/reflect` command). Auto-hook was planned but not yet implemented.

**Future:** Will auto-run at session end via hook (when Claude Code supports skill invocation from bash hooks).

---

### UPDATE14 ExecutionSpec Checks (MANDATORY)

**After ANY changes to execution spec system, ALWAYS run:**

```bash
python scripts/check/check_execution_spec.py
python test_app_sync.py
python scripts/check/app_preflight.py
```

**What these check:**
- ExecutionSpec system integrity (spec creation, validation, serialization)
- Contract validation (required columns, tables, fields)
- Entry rule implementations (limit_at_orb, 1st_close_outside, 5m_close_outside)
- Universal invariants (no lookahead, entry after ORB, structural consistency)
- All 6 test categories must PASS

**When to run:**
- After modifying trading_app/execution_spec.py
- After modifying trading_app/execution_contract.py
- After modifying trading_app/entry_rules.py
- After updating ExecutionSpec presets
- After changing entry rule logic or validation

**Sync guard (fail-closed):**
- If execution spec files exist, check_execution_spec.py MUST exist and pass
- This prevents silent drift
- test_app_sync.py will FAIL if guard is triggered

See UPDATE14_COMPLETE.md for full ExecutionSpec documentation.

---

## Key Commands

### Backfilling Data

**Databento (primary source for historical data):**
```bash
python backfill_databento_continuous.py YYYY-MM-DD YYYY-MM-DD
```
- Example: `python backfill_databento_continuous.py 2024-01-01 2026-01-10`
- Automatically selects front/most liquid contract per day
- Stitches contracts into continuous series
- Safe to interrupt and re-run (idempotent)
- Can run forward or backward
- Automatically calls `build_daily_features.py` after backfill

**ProjectX (alternative source, not used for deep history):**
```bash
python backfill_range.py YYYY-MM-DD YYYY-MM-DD
```
- Example: `python backfill_range.py 2025-12-01 2026-01-09`
- Handles contract rollovers automatically
- Also calls `build_daily_features.py` after backfill

### Feature Building

```bash
python pipeline/build_daily_features.py YYYY-MM-DD
```
- Example: `python pipeline/build_daily_features.py 2025-01-10`
- Automatically called by backfill scripts
- Computes session stats (Asia/London/NY), ORBs, RSI
- Safe to re-run (upserts)
- Writes to `daily_features` table (canonical table)

### Database Operations

**Initialize database schema:**
```bash
python pipeline/init_db.py
```

**Wipe all MGC data (bars_1m, bars_5m, daily_features):**
```bash
python pipeline/wipe_mgc.py
```

**Check database contents:**
```bash
python pipeline/check_db.py
```

**Query features:**
```bash
python analysis/query_features.py
```

### Testing & Inspection

**Inspect DBN files:**
```bash
python pipeline/inspect_dbn.py
```
- Configured to read from `dbn/` folder
- Shows schema, dataset, symbols, record counts

**Validate data:**
```bash
python pipeline/validate_data.py
```
- Validates data integrity and completeness

## Architecture

### Canonical Realized RR

**CRITICAL:** This system uses **realized RR with costs embedded**, not theoretical RR.

**Why this matters:**
- Old system: RR=2.0 assumed $20 reward per $10 risk (ignoring costs)
- New system: RR=2.0 → realized_rr=1.238 after $8.40 friction (MGC, honest double-spread)
- **Realized expectancy is MORE accurate than theoretical** (uses real P&L distribution)

**Single Source of Truth:**
```
pipeline/cost_model.py
  ├─ Contract specs (MGC: $10/point, $8.40 friction)
  ├─ Honest accounting: commission $2.40 + spread_double $2.00 + slippage $4.00
  ├─ Realized RR formulas
  └─ Used by execution_engine.py and build_daily_features.py
```

**Data Flow (Option B - Locked In):**
```
bars_1m
   │
   ├──> execution_engine (rr=1.0) ──> daily_features (1R cache)
   │
   └──> execution_engine (rr=1.5/2.0/2.5/3.0) ──> validated_setups (realized expectancy)
```

**Key Rules:**
1. **Apps read from validated_setups.realized_expectancy** (NOT daily_features)
2. **daily_features = 1R cache** (for historical analysis only)
3. **cost_model.py is the ONLY source** for friction values (no hard-coded constants)
4. **Expectancy is strategy-level** (requires win rate across trades, not per-trade)

### ⚠️ CRITICAL: daily_features Stores RR=1.0 Outcomes ONLY

**NEVER use daily_features outcomes directly for RR > 1.0 backtesting!**

The `orb_XXXX_outcome` columns (WIN/LOSS) and `orb_XXXX_r_multiple` columns are calculated at **RR=1.0 only**.
- WIN = price hit 1R target (NOT 4R target!)
- r_multiple = +1.0 for WIN, -1.0 for LOSS

**BUG THAT WAS FOUND (2026-02-04):**
Validation scripts read these RR=1.0 outcomes and incorrectly applied RR=4.0 multipliers.
A trade that barely hit 1R got credit for hitting 4R, inflating ExpR from +0.08R to +0.55R.

**THE ONLY CORRECT WAY to backtest at RR > 1.0:**
```python
from trading_app.strategy_discovery import StrategyDiscovery, DiscoveryConfig

discovery = StrategyDiscovery()
config = DiscoveryConfig(
    instrument='MGC',
    orb_time='0900',
    rr=4.0,  # Correctly re-simulates at this RR
    sl_mode='HALF',
    orb_size_filter=None
)
result = discovery.backtest_configuration(config)
# result.avg_r is CORRECT
```

**DO NOT write SQL queries that read daily_features outcomes for backtesting!**
- Test: `tests/test_rr_calculation_integrity.py` catches this bug
- See: `scripts/validation/validate_strategies_canonical.py` for correct approach

**Files to check:**
- `pipeline/cost_model.py` - Authoritative specs and formulas
- `strategies/execution_engine.py` - Calls cost_model
- `pipeline/build_daily_features.py` - Stores realized_rr columns
- `validated_setups` table - Has realized_expectancy column

**Testing:**
- Run `tests/test_cost_model_sync.py` after any cost_model changes
- Run `tests/test_realized_rr_sync.py` after build_daily_features runs
- Run `python test_app_sync.py` after validated_setups updates

---

### Data Flow

```
Source → Normalize → Store → Aggregate → Feature Build
```

1. **Source**: Databento (GLBX.MDP3) or ProjectX API
2. **Normalize**: Convert to standard format with timezone handling
3. **Store**: Insert into DuckDB (`gold.db`)
4. **Aggregate**: Build 5-minute bars from 1-minute bars
5. **Feature Build**: Calculate daily ORBs, session stats, indicators

### Database Schema (DuckDB)

**bars_1m** (primary raw data):
- Columns: `ts_utc`, `symbol`, `source_symbol`, `open`, `high`, `low`, `close`, `volume`
- Primary key: `(symbol, ts_utc)`
- `symbol`: 'MGC' (continuous logical symbol)
- `source_symbol`: actual contract (e.g., 'MGCG4', 'MGCM4')

**bars_5m** (derived):
- Same columns as bars_1m
- Deterministically aggregated from bars_1m
- Bucket = floor(epoch(ts)/300)*300
- Fully rebuildable at any time

**daily_features** (canonical table):
- One row per local trading day
- Primary key: `(date_local, instrument)` - ready for multi-instrument support
  - Currently: instrument always = 'MGC'
- Session high/low (Asia 09:00-17:00, London 18:00-23:00, NY 23:00-02:00)
- Pre-move travel (pre_ny_travel, pre_orb_travel)
- **All 6 ORBs stored**: Each ORB has 8 columns (high, low, size, break_dir, outcome, r_multiple, mae, mfe)
  - `orb_0900_*`: 09:00-09:05 ORB
  - `orb_1000_*`: 10:00-10:05 ORB
  - `orb_1100_*`: 11:00-11:05 ORB
  - `orb_1800_*`: 18:00-18:05 ORB
  - `orb_2300_*`: 23:00-23:05 ORB
  - `orb_0030_*`: 00:30-00:35 ORB
- RSI at ORB (RSI_LEN=14, computed for 00:30 ORB)
- Missing ORBs stored as NULL (no crashes on weekends/holidays)
- **IMPORTANT**: Outcomes computed with RR=1.0 targets only (for higher RR setups, use execution_engine.py)

**validated_setups** (active strategies):
- Current production strategies for all instruments
- Primary key: `id`
- Columns: instrument, orb_time, rr, sl_mode, orb_size_filter, win_rate, expected_r, sample_size, notes, status
- **ONLY table to query for trading decisions**
- Current count: 2 setups (1 ACTIVE: MGC 1000 RR=4, 1 MARGINAL: MGC 0900 RR=3) as of 2026-02-05
- NQ/MPL blocked until broker specs available for cost model

**validated_setups_archive** (historical only):
- **⚠️ ARCHIVE ONLY - DO NOT USE IN PRODUCTION**
- Preserves old strategy versions for audit/rollback
- Same columns as validated_setups + archive metadata (archive_id, archived_at, archived_reason, replaced_by_id, version_tag)
- **NEVER query in production systems unless explicitly debugging**

**Archive Warning**: Production systems (trading apps, config_generator, execution_engine) must ONLY query `validated_setups`. The archive exists for historical research and rollback safety only. Archived strategies used optimistic assumptions and have been superseded by stress-tested conservative approaches.

### Time & Calendar Model (CRITICAL)

**Trading Day Definition:**
- Local timezone: `Australia/Brisbane` (UTC+10, no DST)
- Trading day window: **09:00 local → next 09:00 local**
- All session windows (Asia/London/NY/ORBs) are evaluated inside that trading-day cycle
- Consistent across backfills, aggregations, and feature building

**Expected 1-Minute Counts:**
- Full weekday: ~1440 rows
- Partial holidays/roll days: fewer
- Weekends: 0 rows (expected)

### Futures Contract Handling

**Why you see MGCG4, MGCM4 when you trade MGC1!:**
- **MGC1!** = continuous front-month symbol (charting/broker convention)
- Databento returns **real contracts** (MGCG4, MGCM4, MGCV4, MGCG6, etc.)
- Pipeline automatically:
  - Selects front/most liquid contract per day (highest volume, excludes spreads)
  - Stitches them into continuous series
  - Stores under `symbol='MGC'` with `source_symbol=actual contract`
- This builds a tradeable continuous series required for proper historical backtesting

### ORB Break Rules

- Break detected when CLOSE is outside the ORB range (not touch)
- Direction: UP, DOWN, or NONE
- **Uses 1-minute closes for detection** (from bars_1m with confirm_bars=1)
- Entry happens at FIRST 1-minute close outside ORB range (NOT 5-minute close!)

### Idempotency & Resume Behavior

All operations are safe to re-run:
- Backfills use `INSERT OR REPLACE` on primary key (will overwrite same timestamps, not duplicate)
- 5m aggregation: DELETE then INSERT for date range
- Feature building: upserts on `(date_local)`
- No duplicate rows possible

**Resume / Backwards backfill:**
- Re-running the same date range overwrites existing data (safe)
- To continue from where you stopped, run a new date range that picks up after the last successful day
- Backward backfill: run earlier start/end date ranges
- No automatic checkpoint - you control the date range on each invocation

## Configuration (.env)

Required environment variables:
- `DATABENTO_API_KEY`: Databento API key
- `DATABENTO_DATASET`: Default "GLBX.MDP3"
- `DATABENTO_SCHEMA`: Default "ohlcv-1m"
- `DATABENTO_SYMBOLS`: Default "MGC.FUT"
- `DUCKDB_PATH`: Default "gold.db"
- `SYMBOL`: Default "MGC"
- `TZ_LOCAL`: Default "Australia/Brisbane"
- `PROJECTX_USERNAME`, `PROJECTX_API_KEY`, `PROJECTX_BASE_URL`: For ProjectX backfills
- `PROJECTX_LIVE`: "false" for historical data

## Important Notes

1. **Databento availability**: `backfill_databento_continuous.py` has a hardcoded `AVAILABLE_END_UTC` to prevent 422 errors. Update this when Databento extends the dataset.

2. **Contract selection**: The pipeline automatically handles futures contract rolls by selecting the most liquid contract (highest volume, excluding spreads with '-' in symbol).

3. **5-minute bars**: Always rebuilt from 1-minute bars after backfill. Never manually edit bars_5m.

4. **Weekend/holiday handling**: Missing ORBs are stored as NULL. Scripts will not crash on days without data.

5. **Timezone awareness**: All timestamps in database are UTC (`TIMESTAMPTZ`). Session windows are defined in local time (Australia/Brisbane) then converted to UTC for queries.

6. **RSI calculation**: Uses Wilder's smoothing method with 14-period lookback. Calculated on 5-minute closes at 00:30 ORB.

7. **Data sources**:
   - Databento: Used for all historical backfill (recommended)
   - ProjectX: Optional, not used for deep history (limited historical range)
   - Raw DBN files stored in `dbn/` folder

8. **Schema migration**: The database uses `daily_features` (canonical table). If you have old data, you should wipe and rebuild:
   ```bash
   python pipeline/wipe_mgc.py
   python pipeline/backfill_databento_continuous.py 2020-12-20 2026-01-10
   ```

9. **Trading day change**: All backfill scripts now use 09:00→09:00 trading days (previously 00:00→00:00). This aligns with ORB strategy and session analysis. Old data will be incorrect.

10. **Project structure**: See `PROJECT_STRUCTURE.md` for current file organization. Test/experiment files are in `_archive/`.

## ⚠️ CRITICAL: Database and Config Synchronization (NEVER VIOLATE THIS)

**MANDATORY RULE: NEVER update validated_setups database without IMMEDIATELY updating config.py in the same operation.**

### ⚠️ ALWAYS RUN THIS TEST AFTER ANY CHANGES:

```bash
python test_app_sync.py
```

**Run this test EVERY TIME after:**
- Updating `validated_setups` database
- Modifying `trading_app/config.py`
- Adding new MGC/NQ/MPL setups
- Changing ORB filters
- Running `populate_validated_setups.py`
- Updating RR values or SL modes

**If you forget to run this test, the apps will use WRONG values and cause REAL MONEY LOSSES in live trading.**

### Why This Is Critical

Mismatches between database and config.py cause:
- Apps use WRONG filters
- Accept trades that should be rejected
- Reject trades that should be accepted
- **REAL MONEY LOSSES in live trading**
- Dangerous and unacceptable

### Synchronization Protocol

When updating MGC setups in validated_setups table:

1. **FIRST**: Update `gold.db` → `validated_setups` table
2. **IMMEDIATELY AFTER**: Update `trading_app/config.py` → `MGC_ORB_SIZE_FILTERS` dictionary
3. **VERIFY**: Run `python test_app_sync.py` to confirm synchronization
4. **ONLY PROCEED**: If ALL TESTS PASS

**NEVER skip step 2. NEVER skip step 3. NEVER proceed if tests fail.**

### Files That Must Always Match Exactly

- `gold.db` → `validated_setups.orb_size_filter` (for MGC rows)
- `trading_app/config.py` → `MGC_ORB_SIZE_FILTERS` dictionary values

For each ORB time (0900, 1000, 1100, 1800, 2300, 0030):
- Database filter value MUST equal config.py filter value (within 0.001 tolerance)
- If database has NULL filter, config.py must have None
- If database has 0.05, config.py must have 0.05
- Zero tolerance for mismatches

### Verification Command

```bash
python test_app_sync.py
```

Expected output:
```
ALL TESTS PASSED!

Your apps are now synchronized:
- config.py has optimized MGC filters
- validated_setups database is synchronized
- setup_detector.py works with all instruments
- data_loader.py filter checking works
- All components load without errors

Your apps are SAFE TO USE!
```

**If this fails: STOP ALL WORK and fix the mismatch before proceeding.**

**REMINDER: This test is your safety net. Always run it after changes to database or config.**

### Other Synchronized Components

These files depend on config.py and must be tested:
- `trading_app/setup_detector.py` - Reads validated_setups
- `trading_app/data_loader.py` - Uses config.py filters
- `trading_app/config.py` - Must match database

### When Updating Any Instrument (MGC, NQ, MPL)

Same rules apply:
- Update database first
- Update corresponding config section immediately (MGC_ORB_SIZE_FILTERS, NQ_ORB_SIZE_FILTERS, or MPL_ORB_SIZE_FILTERS)
- **Run `python test_app_sync.py`**
- Verify all tests pass
- **Do NOT skip this step**

### Experimental Strategies Validation (Added 2026-01-29)

The `experimental_strategies` table is a parallel strategy source that bypasses config.py entirely. To prevent bad data from entering production:

**MANDATORY: When updating experimental_strategies table:**

1. **Run validation script:**
   ```bash
   python scripts/check/check_experimental_strategies.py
   ```
   Checks for:
   - Expected R within bounds (-1.0 to +2.0R) - catches typos
   - Win rates within bounds (20% to 90%)
   - Minimum sample size (>= 15 trades)
   - Valid filter types and day_of_week values

2. **Test scanner:**
   ```bash
   python trading_app/experimental_scanner.py
   ```

3. **ONLY PROCEED if validation passes**

**DO NOT add experimental strategies without validation!** Bad data (e.g., expected_r = 2.5R instead of 0.25R typo) will mislead users in production.

