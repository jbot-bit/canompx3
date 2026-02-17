# Ralph Audit Report

Generated: 2026-02-18

## 1. Uncommitted Changes

### Files Changed (5 files, +442/-12 lines)

| File | Lines Changed | Review Status |
|------|--------------|---------------|
| `ROADMAP.md` | +39/-12 | PASS |
| `TRADING_RULES.md` | +33 | PASS |
| `docs/RESEARCH_ARCHIVE.md` | +272 | PASS |
| `trading_app/db_manager.py` | +33 | PASS |
| `trading_app/strategy_discovery.py` | +65 | PASS |

### Findings

**ROADMAP.md:**
- Updates P1 cross-instrument portfolio to DONE (NO-GO for 1000 LONG stacking) — consistent with new RESEARCH_ARCHIVE entry
- Adds DST contamination remediation status — matches CLAUDE.md documented status
- Updates P2 to reference DST winter seasonality signal — consistent with DST audit findings
- Updates P6 to reflect P1 findings (MNQ/MES correlation too high) — logical
- No stale references, no code violations

**TRADING_RULES.md:**
- Adds DST Audit summary after session table — consistent with CLAUDE.md DST contamination section
- Adds cross-instrument NO-GO to the table — consistent with P1 findings
- Adds DST CONTAMINATION WARNING section with table — matches CLAUDE.md exactly
- Adds DST annotations to 0900, 1800, 2300 session playbooks — compliant with "all future research must split by DST" rule
- No unqualified "edge" or "significant" language — RESEARCH_RULES.md compliant

**docs/RESEARCH_ARCHIVE.md:**
- 5 new research sections with full methodology disclosure — RESEARCH_RULES.md compliant
- Includes N trades, time periods, mechanism discussion, next steps — compliant
- Uses correct labels: "NO-GO", "COMPLETED", "STABLE/WINTER-DOMINANT/SUMMER-DOMINANT" — compliant
- Correlation numbers properly reported (+0.83 MNQ/MES, +0.40-0.44 MGC/equity)
- Red flags section for MES 0900 E1 — transparent reporting

**trading_app/db_manager.py:**
- Adds 5 DST columns to `experimental_strategies` DDL — matches documented DST remediation
- Adds 5 DST columns to `validated_setups` DDL — matches
- Migration code uses hardcoded string literals (not user input) — no SQL injection risk
- Migration uses try/except CatalogException pattern — idempotent (CLAUDE.md compliant)
- `verify_trading_app_schema()` updated with same 5 columns — schema verification stays in sync
- No import direction violations (stays in trading_app/)
- No hardcoded paths

**trading_app/strategy_discovery.py:**
- New import: `from pipeline.dst import (DST_AFFECTED_SESSIONS, is_winter_for_session, classify_dst_verdict)` — correct direction (trading_app imports from pipeline, allowed per CLAUDE.md one-way dependency rule)
- New function `compute_dst_split_from_outcomes()` — correctly returns CLEAN for non-affected sessions, uses proper DST resolvers
- INSERT SQL updated with 5 new DST columns — matches db_manager DDL
- `_INSERT_SQL` parameter count matches INSERT values — verified
- Trade day hash computation unchanged — FIX5 compliance maintained
- No hardcoded symbols, no hardcoded paths

### Compliance Checks

| Check | Result |
|-------|--------|
| Hardcoded symbols | PASS — no hardcoded instrument names |
| Import direction | PASS — one-way pipeline→trading_app maintained |
| Timezone hygiene | PASS — DST split uses date objects, UTC-aware |
| Schema consistency | PASS — DDL, migration, and verification all agree on DST columns |
| Security (no secrets) | PASS — no API keys, no credentials |
| SQL injection | PASS — all SQL uses parameterized queries or hardcoded literals |
| TRADING_RULES consistency | PASS — code changes match doc changes |
| RESEARCH_RULES compliance | PASS — all research sections follow mandatory disclosure rules |
| DST contamination rules | PASS — splits by DST regime for affected sessions |
| FIX5 trade day invariant | PASS — no changes to filter/outcome logic |

### Drift Check & Tests

**NOTE:** Python execution blocked by Windows sandbox permissions in this environment. Drift check and test suite could not be run. These must be verified manually or in a non-sandboxed session.

### Verdict: PASS (with caveat)

All uncommitted changes are consistent with CLAUDE.md, TRADING_RULES.md, and RESEARCH_RULES.md. No violations found. Code changes implement the documented DST remediation (Step 2: winter/summer split in strategy_validator.py and strategy_discovery.py). Documentation changes accurately reflect completed research. Drift check and tests could not be run due to sandbox permissions — recommend running before commit.

## 2. Pipeline Modules

### Files Reviewed

| File | Lines | Review Status |
|------|-------|---------------|
| `pipeline/ingest_dbn.py` | 523 | PASS |
| `pipeline/build_bars_5m.py` | 345 | PASS |
| `pipeline/build_daily_features.py` | 993 | PASS |
| `pipeline/paths.py` | 39 | PASS |
| `pipeline/init_db.py` | 247 | PASS |
| `pipeline/dst.py` | 386 | PASS |
| `pipeline/cost_model.py` | 279 | PASS |
| `pipeline/asset_configs.py` | 152 | PASS |

### Architecture Compliance

**Fail-Closed Principle:**
- `ingest_dbn.py`: All validation failures call `sys.exit(1)` — timestamp validation, OHLCV validation, PK safety, merge integrity, final honesty gates. PASS.
- `build_bars_5m.py`: Integrity failures raise exceptions inside transaction (auto-rollback via try/except). CLI exits via `sys.exit(1)`. PASS.
- `build_daily_features.py`: Build failures rollback transaction and re-raise. CLI exits on integrity failure. PASS.
- `asset_configs.py`: `get_asset_config()` calls `sys.exit(1)` for unknown instrument, missing DBN, missing start date. PASS.
- `cost_model.py`: `get_cost_spec()` raises `ValueError` for unknown instruments. PASS.

**Idempotent Operations:**
- `ingest_dbn.py`: Uses `INSERT OR REPLACE INTO bars_1m`. PASS.
- `build_bars_5m.py`: `DELETE` existing rows then `INSERT` new rows, wrapped in transaction. PASS.
- `build_daily_features.py`: `DELETE` existing rows (scoped to symbol + date range + orb_minutes) then `INSERT`, wrapped in transaction. PASS.

**One-Way Dependency (pipeline → trading_app):**
- Grep confirmed: NO pipeline module imports from `trading_app/`. Only `check_drift.py` references trading_app (for validation purposes), and it correctly self-excludes from the import direction check (line 374). PASS.

**Timezone Hygiene:**
- All DB columns use `TIMESTAMPTZ` (verified in `init_db.py` schema).
- `build_daily_features.py` properly converts between UTC and Brisbane using `zoneinfo.ZoneInfo`.
- Trading day boundary: 09:00 Brisbane = 23:00 UTC, correctly implemented.
- DST resolvers in `dst.py` use `zoneinfo` for all timezone math — no manual UTC offset arithmetic. PASS.

**GC → MGC Source Symbol Handling:**
- `ingest_dbn.py` line 348: INSERT stores `symbol` (from config, e.g., "MGC") as `symbol` column, and `r[1]` (the actual front contract name, e.g., "MGCG5") as `source_symbol`. The `asset_configs.py` outright_pattern for MGC matches `^MGC[FGHJKMNQUVXZ]\d{1,2}$`. Note: CLAUDE.md says "Raw data contains GC" but the pattern matches MGC contracts — this means the actual data file must contain MGC-prefixed symbols (not GC). The source_symbol column preserves the original contract identity. PASS.

**paths.py DUCKDB_PATH:**
- Resolution order: (1) `DUCKDB_PATH` env var → (2) `local_db/gold.db` if exists → (3) `gold.db` at project root. Matches CLAUDE.md documentation. PASS.

**Schema Matches Documentation:**
- `init_db.py` creates: `bars_1m` (8 cols, PK symbol+ts_utc), `bars_5m` (8 cols, PK symbol+ts_utc), `daily_features` (dynamic DDL from ORB_LABELS: 13 sessions × 9 cols + base cols, PK symbol+trading_day+orb_minutes). Matches CLAUDE.md data flow. PASS.

**Validation Gates:**
- Ingestion gates (7): DBN schema check (line 136), UTC timestamp validation (line 233), outright filter (line 243), OHLCV sanity (line 254), PK safety (line 295), merge integrity (line 354), final honesty gates (line 463). PASS.
- 5m aggregation gates (4): No duplicates (line 215), 5-minute alignment (line 230), OHLCV sanity (line 242), volume non-negative (line 257). PASS.

**DST Implementation:**
- `dst.py` implements 6 dynamic resolvers: `cme_open_brisbane`, `us_equity_open_brisbane`, `us_data_open_brisbane`, `london_open_brisbane`, `us_post_equity_brisbane`, `cme_close_brisbane`.
- All use proper timezone objects (`_US_EASTERN`, `_US_CHICAGO`, `_UK_LONDON`, `_BRISBANE`).
- `SESSION_CATALOG` is the master registry with break_groups preventing silent window shrinkage.
- `DST_AFFECTED_SESSIONS` correctly maps 0900/0030/2300 → "US", 1800 → "UK".
- `DST_CLEAN_SESSIONS` includes all Asia fixed + all dynamic sessions. PASS.

**Connection Handling:**
- `ingest_dbn.py`: Uses `atexit` handler for cleanup (line 171), explicit `con.close()` on normal exit (line 515). Adequate but not using `with` statement. Minor note — not a violation.
- `build_bars_5m.py`: Uses `with duckdb.connect()` context manager (line 303). PASS.
- `build_daily_features.py`: Uses `with duckdb.connect()` context manager (line 950). PASS.

### Minor Observations (Non-Violations)

1. **F-string SQL column names** (`build_daily_features.py:803`, `verify_daily_features` lines 863-868): Column names come from internally-controlled `ORB_LABELS` list or DataFrame columns — no user input, no injection risk.
2. **iterrows() in ingest_dbn.py:306**: Per-row iteration for front contract bars is bounded by per-day bar count (~1440 max per day), acceptable performance.
3. **ingest_dbn.py connection pattern**: Uses `atexit` + explicit close rather than `with` context manager. Functionally correct but slightly less Pythonic. Not a violation — the atexit handler covers crash paths.

### Drift Check & Tests

**NOTE:** Python execution blocked by Windows sandbox permissions. Drift check and test suite could not be run. These must be verified manually.

### Verdict: PASS

All pipeline modules comply with CLAUDE.md architecture rules. Fail-closed, idempotent, one-way dependency, timezone hygiene, schema consistency, and validation gates all verified. No violations found.

## 3. Trading App Modules

### Files Reviewed (Core Modules)

| File | Lines | Review Status |
|------|-------|---------------|
| `trading_app/outcome_builder.py` | 739 | PASS |
| `trading_app/strategy_discovery.py` | 696 | PASS |
| `trading_app/strategy_validator.py` | 638 | PASS |
| `trading_app/config.py` | 357 | PASS |
| `trading_app/db_manager.py` | 468 | PASS |

### Pre-Computed Outcomes Pattern (outcome_builder.py)

- **Pattern compliance:** Outcomes pre-computed for all (RR_TARGET × CONFIRM_BARS × ENTRY_MODEL) combinations per trading day. Grid: 6 RR targets × 5 CB options × 2 entry models (E3 always CB1). Matches CLAUDE.md data flow. PASS.
- **Cost model:** Uses `pipeline.cost_model.get_cost_spec()`, `pnl_points_to_r()`, `to_r_multiple()` for all R calculations. Costs always included. PASS.
- **Idempotent:** Uses `INSERT OR REPLACE INTO orb_outcomes`. Checkpoint/resume via `SELECT COUNT(*) FROM orb_outcomes WHERE trading_day = ?`. PASS.
- **Fail-closed:** Null results returned for no-signal or zero-risk cases. Ambiguous bars (both target and stop hit) default to loss. PASS.
- **Connection handling:** Uses `with duckdb.connect()` context manager. PASS.
- **Early exit:** Implements timed early exit from `config.EARLY_EXIT_MINUTES` (0900: 15min, 1000: 30min). Matches TRADING_RULES.md. PASS.
- **E3 CB optimization:** E3 always CB1 (higher CBs produce identical entry prices). Comment documents this. Matches config.py documentation. PASS.

### Grid Search (strategy_discovery.py)

- **Grid scope:** Iterates all (session × filter × entry_model × rr_target × confirm_bars) combos. Session-aware filters via `get_filters_for_grid()`. PASS.
- **Double-break exclusion:** `_build_filter_day_sets()` explicitly excludes `orb_{label}_double_break` days (line 357). Matches CLAUDE.md architecture. PASS.
- **DST regime split:** `compute_dst_split_from_outcomes()` computes winter/summer split for DST-affected sessions (0900/1800/0030/2300). Uses `DST_AFFECTED_SESSIONS` and `is_winter_for_session()` from `pipeline.dst`. Returns "CLEAN" for unaffected sessions. Fully compliant with CLAUDE.md DST contamination rules. PASS.
- **Canonical dedup:** `_mark_canonical()` groups by (instrument, orb_label, entry_model, rr_target, confirm_bars, trade_day_hash) and picks highest specificity filter as canonical. Trade day hash uses `compute_trade_day_hash()` from db_manager. PASS.
- **Metrics:** `compute_metrics()` correctly defines sample_size = wins + losses (scratches/early_exits excluded). Win rate denominator = wins + losses. Expectancy formula: `(WR × AvgWin_R) - (LR × AvgLoss_R)`. Sharpe uses sample variance (n-1). All consistent with CANONICAL_LOGIC.txt references. PASS.
- **Idempotent:** `INSERT OR REPLACE INTO experimental_strategies`. Preserves `created_at` timestamps on re-run. PASS.
- **Connection handling:** Uses `with duckdb.connect()` context manager. PASS.

### Validation Pipeline (strategy_validator.py)

- **7-phase validation:** Correctly implements all phases:
  1. Sample size gate (min_sample, default 30 = REGIME threshold per CLAUDE.md FIX5). PASS.
  2. Post-cost expectancy > 0. PASS.
  3. Yearly robustness with DORMANT regime waivers (ATR < 20, <= 5 trades). PASS.
  4. Stress test at 1.5× costs using `stress_test_costs()` from cost_model. PASS.
  4b. Walk-forward OOS (anchored expanding, 6m test windows). PASS.
  5. Sharpe ratio (optional, default None). PASS.
  6. Max drawdown (optional, default None). PASS.
- **DST passthrough:** Prefers discovery-computed DST values (correct for ALL filter types including VolumeFilter). Only falls back to `compute_dst_split()` if discovery left them NULL. This is the Feb 2026 bugfix (commit d1ab476). PASS.
- **Walk-forward:** Uses `run_walkforward()` from `trading_app.walkforward`. Result determines Phase 4b pass/fail. Non-rejecting if `--no-walkforward`. PASS.
- **Alias skip:** Non-canonical strategies get `validation_status='SKIPPED'`. Only canonical strategies undergo full validation. PASS.
- **Promotion:** Passing strategies INSERT OR REPLACE into `validated_setups` with all required columns including DST split. PASS.
- **Connection handling:** Uses `with duckdb.connect()` context manager. PASS.

### Classification Thresholds (config.py)

- **CORE_MIN_SAMPLES = 100, REGIME_MIN_SAMPLES = 30:** Matches CLAUDE.md FIX5 Classification Thresholds table exactly (CORE >= 100, REGIME 30-99, INVALID < 30). PASS.
- **classify_strategy():** Returns "CORE", "REGIME", or "INVALID" based on sample_size. Logic matches thresholds. PASS.
- **Portfolio overlay invariant:** FIX5 rules documented in comments (lines 348-357). "orb_outcomes contains ALL break-days regardless of filter. Overlay must ONLY write pnl_r on eligible days." PASS.
- **Entry models:** ENTRY_MODELS = ["E1", "E3"]. E2 removal documented. Matches TRADING_RULES.md. PASS.
- **ORB duration:** `ORB_DURATION_MINUTES` includes all 12 sessions (6 fixed + 6 dynamic). 1000 correctly uses 15m ORB. Matches TRADING_RULES.md variable aperture research. PASS.
- **Early exit:** `EARLY_EXIT_MINUTES` matches TRADING_RULES.md (0900: 15, 1000: 30, all others: None). PASS.
- **Grid filters:** L-filters, G2, G3 removed from grid. Only G4/G5/G6/G8 + NO_FILTER + VOL_RV12_N20 in grid. DIR_LONG added for session 1000 (H5 confirmed per TRADING_RULES.md). MES 1000 adds G4_L12/G5_L12 band filters (H2 confirmed). PASS.
- **TRADEABLE_INSTRUMENTS = ["MGC"]:** MCL permanently NO-GO, MNQ weak edge — documented. Matches TRADING_RULES.md. PASS.

### Schema Manager (db_manager.py)

- **6 tables defined:** orb_outcomes, experimental_strategies, validated_setups, validated_setups_archive, strategy_trade_days, edge_families. Matches CLAUDE.md data flow. PASS.
- **Idempotent:** All CREATE TABLE uses `IF NOT EXISTS`. Migrations use `try/except CatalogException` (column already exists = skip). PASS.
- **DST columns:** Both `experimental_strategies` and `validated_setups` have 5 DST columns (winter_n, winter_avg_r, summer_n, summer_avg_r, verdict). Migration applies to both tables. PASS.
- **Schema verification:** `verify_trading_app_schema()` checks all 6 tables and column sets including DST columns. Expected columns match DDL. PASS.
- **Connection handling:** `init_trading_app_schema()` uses `with duckdb.connect()`. `verify_trading_app_schema()` uses `with duckdb.connect(read_only=True)`. PASS.
- **F-string SQL in migrations:** Uses `f"ALTER TABLE {table} ADD COLUMN {col} {typedef}"` — but `table`, `col`, and `typedef` are all hardcoded internal strings (not user input). No injection risk. Minor note.
- **compute_trade_day_hash():** Uses MD5 of sorted comma-joined dates. Deterministic and consistent. PASS.

### FIX5 Trade Day Invariant Compliance

Verified across all modules:
1. **outcome_builder.py:** Computes outcomes for ALL break-days (no filter applied). Only checks `break_dir is None` to skip no-break days. PASS.
2. **strategy_discovery.py:** Applies filter via `_build_filter_day_sets()` → `matching_day_set`. Outcomes filtered by `o["trading_day"] in matching_day_set`. This correctly limits to eligible days while keeping outcomes universal. PASS.
3. **strategy_validator.py:** Validates from `experimental_strategies` which already have filter-scoped sample_size. Does not recompute trade days. PASS.
4. **config.py:** FIX5 invariant documented in comments (lines 348-357). PASS.

### DST Contamination Rules Compliance

- **strategy_discovery.py:** Computes DST split for every strategy at DST-affected sessions (0900/1800/0030/2300). Uses `is_winter_for_session()` per-day. Stores results in experimental_strategies. PASS.
- **strategy_validator.py:** Prefers discovery-computed DST values (passthrough fix). Falls back to SQL-based `compute_dst_split()` only if NULL. Logs DST info for affected sessions. Does NOT reject based on DST (info-only). PASS.
- **Both modules:** Use `DST_AFFECTED_SESSIONS` mapping from `pipeline.dst` as source of truth. PASS.

### Connection Leak Assessment

| Module | Pattern | Status |
|--------|---------|--------|
| outcome_builder.py | `with duckdb.connect()` | PASS |
| strategy_discovery.py | `with duckdb.connect()` | PASS |
| strategy_validator.py | `with duckdb.connect()` | PASS |
| db_manager.py | `with duckdb.connect()` | PASS |
| config.py | No DB access | N/A |

Secondary modules (cascade_table.py, live_config.py, validate_1800_*.py, market_state.py, ai/*.py, nested/*.py, regime/*.py) use `con = duckdb.connect()` with try/finally + `con.close()`. Functionally correct for exception safety but not using `with` context manager. All are read-only connections. **Minor observation, not a violation** — core write paths are properly managed.

### Minor Observations (Non-Violations)

1. **Secondary modules use try/finally pattern** instead of `with` for DB connections. Functionally safe but slightly less Pythonic. Not a CLAUDE.md violation.
2. **F-string SQL in db_manager.py migrations** uses internally-controlled strings only. No user input reaches SQL. Not an injection risk.
3. **walk_forward.py vs walkforward.py** — two similarly named files exist. `strategy_validator.py` imports from `walkforward.py` (the active one). `walk_forward.py` appears to be an older version. Potential dead code but not a violation.

### Drift Check & Tests

**NOTE:** Python execution blocked by Windows sandbox permissions. Drift check and test suite could not be run. Must be verified manually.

### Verdict: PASS

All core trading_app modules comply with CLAUDE.md, TRADING_RULES.md, and FIX5 rules. Pre-computed outcomes pattern correct. Grid search properly excludes double-breaks and applies session-aware filters. Validation implements all 7 phases including walk-forward. DST regime split computed and stored correctly. Classification thresholds match documentation. No violations found.

## 4. Test Coverage

### Test File Inventory

| Directory | Files | Description |
|-----------|-------|-------------|
| `tests/` | 4 | conftest.py, test_app_sync.py, test_integration_l1_l2.py, test_trader_logic.py |
| `tests/test_pipeline/` | 14 | Pipeline modules: schema, validation, DST, bars, features, drift, paths |
| `tests/test_trading_app/` | 22 | Trading app: config, discovery, validator, portfolio, entry rules, DB manager |
| `tests/test_trading_app/test_ai/` | 5 | AI query agent: cli, corpus, grounding, query agent, SQL adapter |
| `tests/test_trading_app/test_nested/` | 5 | Nested ORB: builder, discovery, schema, audit outcomes, resample |
| `tests/test_trading_app/test_regime/` | 3 | Regime analysis: discovery, schema, validator |
| `tests/test_research/` | 1 | Alt strategy utilities |
| **Total** | **66** | |

### CLAUDE.md Rule: No Unbuilt Feature References

- Searched for references to unbuilt features (live_trading, position_sizing, multi_instrument_portfolio, signal_aggregation): **NONE FOUND**. PASS.
- No tests reference ROADMAP planned features, placeholder functionality, or TODO-gated behavior. PASS.
- All tests exercise existing, documented behavior. PASS.

### DST Test Coverage (test_dst.py)

Comprehensive DST test coverage found in `tests/test_pipeline/test_dst.py` (448 lines):

| Area | Tests | Status |
|------|-------|--------|
| `is_us_dst()` transition detection | 11 tests across 2024-2026 | PASS |
| `is_uk_dst()` transition detection | 7 tests across 2024-2025 | PASS |
| CME_OPEN resolver (Brisbane times) | 5 tests (summer, winter, transitions) | PASS |
| US_EQUITY_OPEN resolver | 4 tests | PASS |
| US_DATA_OPEN resolver | 4 tests | PASS |
| LONDON_OPEN resolver | 5 tests | PASS |
| US_POST_EQUITY resolver | 6 tests | PASS |
| DYNAMIC_ORB_RESOLVERS registry | 2 tests (completeness + return types) | PASS |
| SESSION_CATALOG validation | 9 tests (keys, aliases, break groups, collisions) | PASS |
| Break window grouping | 2 tests (asia group, us group) | PASS |
| Seasonal shift verification | 4 tests (all resolvers shift between seasons) | PASS |

**DST contamination rules from CLAUDE.md are well-covered.** Tests verify all 6 dynamic resolvers return different values for summer vs winter, SESSION_CATALOG structure is valid, break groups prevent window shrinkage, and aliases map correctly. PASS.

### Validation Gate Test Coverage

CLAUDE.md documents 7 ingestion gates and 4 aggregation gates.

**Ingestion gates tested (test_validation.py):**
- OHLCV sanity (`validate_chunk`): 10 tests — NaN, infinite, negative, zero price, high<low, negative volume, zero volume, missing column, empty DF. PASS.
- UTC timestamp validation (`validate_timestamp_utc`): 4 tests — UTC passes, naive fails, wrong TZ fails, null timestamp fails. PASS.
- Other gates (DBN schema, outright filter, PK safety, merge integrity, honesty): Not unit-tested in isolation, but tested through integration in `test_full_pipeline.py` and `test_ingest_daily.py`. **PARTIAL** — individual gate functions aren't exported for direct testing; they run as part of the ingestion flow.

**Aggregation gates tested (test_idempotency.py, test_build_bars_5m.py):**
- No duplicates: Tested via `test_no_duplicate_rows`. PASS.
- Alignment: Tested via `test_row_count_matches_buckets`. PASS.
- OHLCV sanity: Tested via aggregation correctness tests. PASS.
- Volume non-negative: Covered by aggregation tests (input volume is non-negative). PASS.

**Coverage gap:** Ingestion gates 1 (DBN schema), 3 (outright filter), 5 (PK safety), 6 (merge integrity), and 7 (honesty) are not individually unit-tested. They are exercised through integration tests but would benefit from dedicated unit tests. **Minor gap — not a violation** (CLAUDE.md says "build guardrails for what exists" but doesn't mandate per-gate unit tests).

### Additional Test Coverage

| Feature | Test File | Status |
|---------|-----------|--------|
| Schema creation (init_db) | test_schema.py | 7 tests, ORB columns verified for all 13 labels. PASS. |
| Drift detection rules | test_check_drift.py | 22+ tests for all drift check functions. PASS. |
| Strategy validation (7 phases) | test_strategy_validator.py | 25+ tests including regime waivers, integration. PASS. |
| Config filters (ORB size, volume) | test_config.py | 38+ tests for NoFilter, OrbSizeFilter, VolumeFilter, ALL_FILTERS. PASS. |
| DB manager schema | test_db_manager.py | 11 tests for schema creation, idempotency, verification. PASS. |
| Cost model | test_cost_model.py | Tests R-multiple calculations. PASS. |
| Contract selection | test_contract_selection.py | Tests front contract logic. PASS. |
| GC→MGC mapping | test_gc_mgc_mapping.py | Tests source_symbol handling. PASS. |
| RSI edge cases | test_rsi_edge_cases.py | Tests Wilder's RSI. PASS. |
| Trading day assignment | test_trading_days.py | Tests 09:00 Brisbane boundary. PASS. |
| Path resolution | test_paths.py | Tests DUCKDB_PATH env var. PASS. |
| Timezone transitions | test_timezone_transitions.py | Tests DST boundary dates. PASS. |
| Idempotent 5m build | test_idempotency.py | 3 tests for double-run identity. PASS. |

### Hardcoded Path Check

- Searched for `OneDrive`, `C:\db`, `C:\canodrive`, `C:\Users` in tests/: **One hit** in `test_check_drift.py:190` — this is a **test case** that deliberately writes a hardcoded path to verify the drift check catches it. This is correct behavior (testing the detection rule). PASS.

### sys.path.insert Check

- Found one instance: `tests/test_trading_app/test_portfolio.py:12` uses `sys.path.insert(0, ...)`.
- **Assessment:** This is a legacy pattern — the project has `pip install -e .` available. However, the CLAUDE.md rule about sys.path hacks applies to production/research scripts, not test files. The test still works correctly. **Minor observation — not a violation.** Would be cleaner to remove it since pytest should find the package via the installed editable package.

### Test Suite Execution

**NOTE:** Python execution blocked by Windows sandbox permissions. Test suite could not be run in this iteration. Previous Ralph activity logs note test_app_sync.py passes in isolation; prior iterations confirm 1177 tests pass. Must be verified manually or in a non-sandboxed session.

### Verdict: PASS

Test suite provides comprehensive coverage of CLAUDE.md documented behavior:
- DST contamination rules thoroughly tested (52+ tests in test_dst.py)
- All drift detection rules have dedicated tests
- Schema, validation gates, idempotency all tested
- No references to unbuilt features
- No hardcoded paths (only test data for drift check tests)
- Classification thresholds and FIX5 rules verified through config and validator tests

**Minor gaps (non-blocking):**
1. Some ingestion gates lack individual unit tests (covered by integration tests)
2. One `sys.path.insert` in test_portfolio.py (legacy, not a violation)
3. Test execution could not be verified due to sandbox permissions

## 5. Scripts and Infrastructure

### File Inventory

| Directory | Files | Description |
|-----------|-------|-------------|
| `scripts/infra/` | 8 .py + 2 .sh + 1 .vbs + 2 .md | Infrastructure: backups, parallel rebuild, ingestion, ralph loop, telegram |
| `scripts/tools/` | 13 .py | Analysis tools: edge families, backtests, stress test, hypothesis tests |
| `scripts/reports/` | 2 .py | Reports: edge portfolio, walk-forward diagnostics |
| `scripts/migrations/` | 4 .py | Schema migrations: ATR20, Sharpe, trade days, dynamic columns |
| `scripts/ingestion/` | 4 .py | Per-instrument ingestion wrappers: MCL, MES, MNQ, MNQ-fast |
| `scripts/walkforward/` | 3 .py | Walk-forward scripts for 0900 session |
| `scripts/` (root) | 1 .py | operator_status.py |
| **Total** | **~40 files** | |

### Stale OneDrive/canodrive Paths

- Searched for `OneDrive` and `canodrive` across all scripts/: **NONE FOUND**. PASS.
- Project move from OneDrive path is clean — no stale references remain.

### Hardcoded DB Paths (C:\db\gold.db)

Many scripts use `C:/db/gold.db` as a default `--db-path` argument:

| Script | Pattern | Assessment |
|--------|---------|------------|
| `scripts/tools/build_edge_families.py` | `--db-path default="C:/db/gold.db"` | CLI arg, overrideable |
| `scripts/tools/profile_1000_runners.py` | `--db-path default=Path("C:/db/gold.db")` | CLI arg, overrideable |
| `scripts/tools/rolling_portfolio_assembly.py` | `--db-path default=Path("C:/db/gold.db")` | CLI arg, overrideable |
| `scripts/tools/audit_ib_single_break.py` | `--db-path default=Path("C:/db/gold.db")` | CLI arg, overrideable |
| `scripts/tools/smoke_test_new_filters.py` | `--db-path default=Path("C:/db/gold.db")` | CLI arg, overrideable |
| `scripts/migrations/backfill_strategy_trade_days.py` | `default="C:/db/gold.db"` | CLI arg, overrideable |
| `scripts/walkforward/wf_db_reversal_0900.py` | `args.db_path or Path("C:/db/gold.db")` | CLI arg, overrideable |
| `scripts/infra/parallel_rebuild.py` | Uses `DUCKDB_PATH` env var | Correct pattern |
| `scripts/infra/run_parallel_ingest.py` | `REBUILD_DIR` env + CLI arg | Correct pattern |

**Assessment:** All `C:/db/gold.db` references are CLI argument defaults, consistent with CLAUDE.md's documented scratch location pattern (`C:\db\gold.db` for long-running jobs). Every script allows override via `--db-path` or `DUCKDB_PATH` env var. **NOT a violation** — this is the documented workflow.

### CRITICAL FINDING: Hardcoded Telegram Bot Token

**File:** `scripts/infra/telegram_feed.py:19`
```python
BOT_TOKEN = "8572496011:AAFFDahKzbGbROndyPSFbH52VjoyCcmPWT0"
CHAT_ID = "6812728770"
```

**Issue:** Bot token and chat ID are hardcoded in source code. This file is tracked by git (untracked/new file in git status). If committed, the token would be exposed in the repository history.

**Risk:** Low-to-medium. This is a personal Telegram bot for process monitoring alerts — not a production API key. However, it violates the principle of not committing credentials. The bot token could be used to send messages to the user's chat.

**Recommendation:** Move `BOT_TOKEN` and `CHAT_ID` to `.env` file (which is gitignored) and read via `os.environ.get()`. Alternatively, add `scripts/infra/telegram_feed.py` to `.gitignore` if it should remain local-only.

**Note:** The file is currently untracked (`??` in git status), so it has NOT been committed yet. The `.gitignore` covers `.env` and `.env.*` but does not explicitly exclude `telegram_feed.py`.

### .env Handling

- `.env` and `.env.*` are in `.gitignore`. PASS.
- No `.env` file exists at project root (checked). PASS.
- `DATABENTO_API_KEY` is referenced in CLAUDE.md as a `.env` variable — consistent with `.env` being gitignored. PASS.
- The only `.env.example` files found are in gitignored external tools (`openclaw/`, `llm-code-scanner/`). PASS.

### Pre-Commit Hook (.githooks/pre-commit)

- **4-stage pipeline:** [0] Ruff lint → [1] Drift check → [2] Fast tests → [3] Syntax check. PASS.
- **Venv python resolution:** Uses `$PYTHON` variable pointing to `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (Linux). Avoids Windows Store redirect. PASS.
- **Fail-closed:** `set -e` at top, explicit `exit 1` on any stage failure. PASS.
- **PATH preservation:** Saves and restores `$PATH` after venv activate (prevents Git Bash utilities from disappearing). PASS.
- **Consistent with CLAUDE.md:** Documented as running "drift check + fast tests before every commit". Actually runs 4 stages (ruff, drift, tests, syntax). The extra ruff + syntax stages are bonus guardrails. PASS.

### CI Workflow (.github/workflows/ci.yml)

- **Triggers:** push to main, PR to main. PASS.
- **Steps:** checkout → Python 3.13 → install deps → ruff lint → drift check → UTF-8 encoding check → Tier 1 tests with coverage. PASS.
- **Consistent with CLAUDE.md:** "On push/PR: drift check + pure-function tests + schema tests." CI runs all tests (not just fast tests like pre-commit). PASS.
- **Note:** CI runs `--ignore=tests/test_trader_logic.py` (slow test). Pre-commit also ignores slow tests. Consistent. PASS.

### sys.path.insert Hacks

- **Found 1 instance:** `scripts/tools/hypothesis_test.py:23` — `sys.path.insert(0, str(PROJECT_ROOT))`
- **Assessment:** This is a standalone research tool script. CLAUDE.md's RESEARCH_RULES.md says "Research scripts: Always in research/, no sys.path hacks". However, this script is in `scripts/tools/` not `research/`. The sys.path hack is needed because the script isn't part of an installed package. **Minor observation** — would be cleaner to use relative imports or `pip install -e .`, but not a critical violation for a utility script.

### subprocess / os.system Usage

Multiple scripts use `subprocess.run()` for orchestrating pipeline steps:

| Script | Usage | Assessment |
|--------|-------|------------|
| `scripts/operator_status.py` | `subprocess.run(["git", ...])` | Safe — hardcoded args |
| `scripts/infra/parallel_rebuild.py` | `subprocess.Popen([sys.executable, ...])` | Safe — controlled args |
| `scripts/infra/run_parallel_ingest.py` | `subprocess.run([sys.executable, ...])` | Safe — controlled args |
| `scripts/infra/run_backfill_overnight.py` | `subprocess.run([sys.executable, ...])` | Safe — controlled args |
| `scripts/ingestion/ingest_mcl.py` | `subprocess.run(...)` | Safe — controlled args |
| `scripts/ingestion/ingest_mnq.py` | `subprocess.run(...)` | Safe — controlled args |
| `scripts/ingestion/ingest_mes.py` | `subprocess.run(...)` | Safe — controlled args |
| `scripts/tools/explore.py` | `os.system('cls' ...)` | Safe — hardcoded string |

**All subprocess calls use list-form arguments (no shell=True injection risk).** The one `os.system()` call uses a hardcoded string (`'cls'` or `'clear'`). PASS.

### Infrastructure Scripts Check

**scripts/infra/backup_db.py:**
- Uses `PROJECT_ROOT / "gold.db"` — correct, no hardcoded paths. PASS.
- Prunes old backups with configurable `--keep`. PASS.

**scripts/infra/ralph.sh:**
- Hardcoded PATH: `/c/Users/joshd/.local/bin` — user-specific but appropriate for a local-only script. PASS.
- Uses `--dangerously-skip-permissions` for automation (required for unattended Ralph loop). Acceptable for local dev tooling. PASS.

**scripts/infra/start_telegram_feed.vbs:**
- Hardcoded path: `C:\Users\joshd\canompx3\scripts\infra\telegram_feed.py` — user-specific, local-only VBS launcher. PASS.

**scripts/infra/check_root_hygiene.py:**
- Validates root directory contents against allowlists. Well-structured. PASS.

### Stale Script References

- `run_backfill_overnight.py` references `pipeline/ingest_dbn_daily.py` — **VERIFIED EXISTS**. PASS.
- No references to deleted or renamed scripts found. PASS.

### Compliance Checks

| Check | Result |
|-------|--------|
| Stale OneDrive/canodrive paths | PASS — none found |
| Hardcoded DB paths | PASS — all are CLI defaults matching documented scratch pattern |
| API keys / secrets in source | **FINDING** — Telegram bot token hardcoded in telegram_feed.py (untracked file) |
| .env handling | PASS — .env gitignored, no committed secrets |
| Pre-commit hook | PASS — 4-stage fail-closed pipeline |
| CI workflow | PASS — drift check + full tests on push/PR |
| sys.path.insert hacks | MINOR — 1 instance in hypothesis_test.py (scripts/tools/, not research/) |
| subprocess safety | PASS — all use list-form args, no shell injection |
| Stale script references | PASS — all referenced scripts exist |

### Verdict: PASS (with one finding)

All scripts and infrastructure are consistent with CLAUDE.md standards. No stale paths from the OneDrive migration. `C:/db/gold.db` defaults are the documented scratch pattern. Pre-commit hook and CI are properly configured with fail-closed behavior.

**One finding:** `telegram_feed.py` has a hardcoded Telegram bot token. The file is currently untracked, so no credentials have been committed. Recommend moving the token to `.env` before any future commit of this file.

## 6. Documentation Consistency

### Documents Cross-Referenced

| Document | Last Updated | Status |
|----------|-------------|--------|
| `CLAUDE.md` | Feb 2026 | 2 findings |
| `TRADING_RULES.md` | Feb 2026 | 1 finding |
| `ROADMAP.md` | Feb 2026 | 2 stale counts (cosmetic) |
| `REPO_MAP.md` | Pre-Feb 2026 | STALE — needs regeneration |

### Finding 1: CLAUDE.md Missing US_POST_EQUITY from Clean Sessions List

**Location:** CLAUDE.md line 93 — "What's NOT contaminated" section.

**Issue:** The clean sessions list says:
> All dynamic sessions (CME_OPEN, LONDON_OPEN, US_EQUITY_OPEN, US_DATA_OPEN, CME_CLOSE) — resolvers adjust per-day.

This lists **5** dynamic sessions but the code (`pipeline/dst.py:82`) defines **6** dynamic sessions in `DST_CLEAN_SESSIONS`:
```python
DST_CLEAN_SESSIONS = {"1000", "1100", "1130",
                       "CME_OPEN", "LONDON_OPEN", "US_EQUITY_OPEN", "US_DATA_OPEN",
                       "US_POST_EQUITY", "CME_CLOSE"}
```

**Missing:** `US_POST_EQUITY` (10:00 AM ET, ~30min after NYSE cash open). This session was added in commit `29fc88e` (Feb 18, 2026) but CLAUDE.md was not updated to include it.

**Impact:** Low — US_POST_EQUITY IS in dst.py's `DST_CLEAN_SESSIONS`, so the code is correct. Only the documentation list is incomplete.

### Finding 2: TRADING_RULES.md Says "4 Dynamic Sessions" — Code Has 6

**Location:** TRADING_RULES.md lines 10, 23-29 — Sessions glossary.

**Issue:** The glossary header says "Sessions (ORB Labels — 11 total)" with "Dynamic sessions (4, DST-aware)". The table lists only CME_OPEN, US_EQUITY_OPEN, US_DATA_OPEN, LONDON_OPEN.

**Missing:** `US_POST_EQUITY` (10:00 AM ET) and `CME_CLOSE` (2:45 PM CT). Both are defined in `pipeline/dst.py` SESSION_CATALOG and `pipeline/init_db.py` ORB_LABELS_DYNAMIC.

**Correct count:** 7 fixed + 6 dynamic = **13 non-alias sessions** (plus 2 aliases: TOKYO_OPEN, HK_SG_OPEN).

**Impact:** Medium — anyone reading TRADING_RULES.md for session definitions will not see US_POST_EQUITY or CME_CLOSE. However, both are relatively new and not yet part of validated strategies, so operational impact is low.

### Finding 3: ROADMAP.md Phase 3 — Stale Counts

**Location:** ROADMAP.md lines 24, 28.

**Issues:**
- Line 24: `trading_app/db_manager.py — 4 trading_app tables` → now has **6 tables** (orb_outcomes, experimental_strategies, validated_setups, validated_setups_archive, strategy_trade_days, edge_families). The extra 2 tables (strategy_trade_days, edge_families) were added in later phases.
- Line 28: `trading_app/strategy_validator.py — 6-phase validation + risk floor (17 tests)` → now has **7-phase validation** (walk-forward added as phase 4b). The source docstring already says "7-phase".

**Impact:** Cosmetic — these are historical phase descriptions documenting what was "DONE" at that time. The validator docstring is already updated. The current REPO_MAP (when regenerated) would pick up the "7-phase" from the docstring.

### Finding 4: REPO_MAP.md is STALE

**Location:** REPO_MAP.md (entire file).

**Issues:**
1. **Missing `pipeline/dst.py`** — a critical module (386 lines, 6 dynamic resolvers, SESSION_CATALOG) that is not listed in REPO_MAP at all.
2. **Stale generator path:** Header says "Auto-generated by `scripts/gen_repo_map.py`" but the actual script is at `scripts/tools/gen_repo_map.py` (confirmed by CLAUDE.md document authority table and filesystem).
3. **Stale LOC counts and descriptions:** `strategy_validator.py` listed as "232 LOC, 6-phase" but source says "7-phase" and LOC has changed.
4. **Missing recently added files:** `pipeline/log.py`, new scripts in `scripts/tools/` and `scripts/infra/`, new test files.
5. **Missing `scripts/tools/` subdirectory entirely** from Directory Tree (only lists `scripts/gen_repo_map.py` at scripts root level).

**Impact:** High — REPO_MAP.md is designated as the authoritative module index per CLAUDE.md document authority table. Anyone relying on it for file discovery would miss critical modules.

**Remediation:** Run `python scripts/tools/gen_repo_map.py` to regenerate. Blocked by sandbox in this session — must be done manually.

### Finding 5: CLAUDE.md Data Flow Diagram — Accurate

The data flow diagram in CLAUDE.md correctly maps to the actual code:
- `pipeline/ingest_dbn.py` → `gold.db:bars_1m` ✅
- `pipeline/build_bars_5m.py` → `gold.db:bars_5m` ✅
- `pipeline/build_daily_features.py` → `gold.db:daily_features` ✅
- `trading_app/outcome_builder.py` → `gold.db:orb_outcomes` ✅
- `trading_app/strategy_discovery.py` → `gold.db:experimental_strategies` ✅
- `trading_app/strategy_validator.py` → `gold.db:validated_setups` ✅
- `scripts/tools/build_edge_families.py` → `gold.db:edge_families` ✅

All modules exist at the stated paths and produce the documented outputs. PASS.

### Finding 6: Document Authority Table — Consistent

CLAUDE.md's document authority table correctly specifies:
- CLAUDE.md wins for code decisions ✅
- TRADING_RULES.md wins for trading logic ✅
- RESEARCH_RULES.md wins for research/analysis ✅
- REPO_MAP.md auto-generated from `scripts/tools/gen_repo_map.py` ✅ (correct path in CLAUDE.md, though REPO_MAP's own header has the wrong path)

No conflicts found between documents on their respective domains. PASS.

### Finding 7: ROADMAP Phase Status vs Actual Code — Mostly Accurate

| Phase | ROADMAP Status | Code Reality | Match |
|-------|---------------|-------------|-------|
| P1: Daily Features | DONE | `build_daily_features.py` exists, tested | ✅ |
| P2: Cost Model | DONE | `cost_model.py` exists, 20 tests | ✅ |
| P3: Trading App | DONE | All modules exist (stale counts noted above) | ✅ (cosmetic) |
| P4: DB/Config Sync | DONE | `test_app_sync.py` exists, drift checks active | ✅ |
| P5: Expanded Scan | DONE | Grid operational, 689K outcomes | ✅ |
| P5b: Entry Fix | DONE | E2 removed, risk floor, win PnL fix | ✅ |
| P6a-6d: Live Trading | DONE | portfolio.py, execution_engine.py, risk_manager.py, paper_trader.py | ✅ |
| P6e: Monitoring | TODO | Not built | ✅ |
| P7: Audit | DONE | audit findings documented | ✅ |
| P8a: Fill-Bar | DONE | `_check_fill_bar_exit()` exists | ✅ |
| P8b: Multi-Instrument Grid | TODO | DirectionFilter + MES band filters added to config.py | ⚠️ Partially done |
| P8e: Backfill 2016-2020 | IN PROGRESS | As documented | ✅ |

**Note on P8b:** The ROADMAP lists implementation steps 1-9. Steps 1-3 are partially done (DirectionFilter class exists, band filters exist, `get_filters_for_grid()` dispatch exists). However, the ROADMAP still says "TODO" for the whole item. This is a **conservative understatement** — some steps are already implemented. Not a violation.

### Finding 8: DST Contamination Section — CLAUDE.md vs dst.py

CLAUDE.md's DST contamination section correctly matches `pipeline/dst.py`:

| CLAUDE.md Claim | dst.py Reality | Match |
|----------------|---------------|-------|
| 0900 → US DST | `DST_AFFECTED_SESSIONS["0900"] = "US"` | ✅ |
| 1800 → UK DST | `DST_AFFECTED_SESSIONS["1800"] = "UK"` | ✅ |
| 0030 → US DST | `DST_AFFECTED_SESSIONS["0030"] = "US"` | ✅ |
| 2300 → US DST (never aligned) | `DST_AFFECTED_SESSIONS["2300"] = "US"` + comment | ✅ |
| 1000/1100/1130 clean | `DST_CLEAN_SESSIONS` includes these | ✅ |
| Dynamic sessions clean | `DST_CLEAN_SESSIONS` includes 6 dynamic | ✅ (but CLAUDE.md lists only 5 — see Finding 1) |
| Remediation status | DST columns, migration, verdict all in code | ✅ |

### Key Commands Section — Cannot Verify Execution

CLAUDE.md documents key commands. Python execution is blocked by sandbox, so commands could not be tested. However, all referenced scripts exist on disk:

| Command | Script Exists | Verified |
|---------|--------------|----------|
| `python pipeline/init_db.py` | ✅ | Path correct |
| `python pipeline/ingest_dbn.py` | ✅ | Path correct |
| `python pipeline/build_bars_5m.py` | ✅ | Path correct |
| `python pipeline/build_daily_features.py` | ✅ | Path correct |
| `python trading_app/outcome_builder.py` | ✅ | Path correct |
| `python trading_app/strategy_discovery.py` | ✅ | Path correct |
| `python trading_app/strategy_validator.py` | ✅ | Path correct |
| `python trading_app/paper_trader.py` | ✅ | Path correct |
| `python -m trading_app.live_config` | ✅ | Module exists |
| `python pipeline/check_drift.py` | ✅ | Path correct |
| `python -m pytest tests/ -v` | ✅ | Test dir exists |
| `python pipeline/health_check.py` | ✅ | Path correct |
| `python pipeline/dashboard.py` | ✅ | Path correct |
| `python scripts/reports/report_edge_portfolio.py` | ✅ | Path correct |

All 14 documented commands point to existing scripts. PASS.

### Compliance Summary

| Check | Result | Details |
|-------|--------|---------|
| ROADMAP phase status vs code | PASS (cosmetic stale counts) | P3 table/phase counts outdated but harmless |
| CLAUDE.md data flow vs imports | PASS | All module paths and outputs correct |
| Document authority — no conflicts | PASS | Each doc rules its domain |
| DST section vs dst.py | PASS (1 omission) | US_POST_EQUITY missing from CLAUDE.md list |
| Key commands — scripts exist | PASS | All 14 commands point to real files |
| REPO_MAP freshness | **FAIL** | Missing pipeline/dst.py, stale counts, wrong generator path |
| TRADING_RULES session count | STALE | Says 11 sessions (4 dynamic), actual is 13 (6 dynamic) |

### Verdict: PASS (with findings)

Cross-document consistency is generally good. The data flow diagram, DST contamination rules, document authority table, and phase statuses are accurate. Three documentation gaps need attention:

**Should fix (before next commit):**
1. REPO_MAP.md regeneration (run `python scripts/tools/gen_repo_map.py`)
2. TRADING_RULES.md session count: update to 13 sessions (6 dynamic), add US_POST_EQUITY and CME_CLOSE to dynamic sessions table
3. CLAUDE.md clean sessions list: add US_POST_EQUITY

**Cosmetic (low priority):**
4. ROADMAP.md Phase 3 stale counts (4 tables → 6, 6-phase → 7-phase)

## 7. Security and Guardrails

### SQL Injection Assessment

Searched all `.py` files for f-string SQL patterns (`f"...SELECT/INSERT/DELETE/ALTER..."`). Categorized each by injection risk:

**No risk — internally-controlled values only:**

| Location | Pattern | Why Safe |
|----------|---------|----------|
| `pipeline/init_db.py:187` | `f"DROP TABLE IF EXISTS {t}"` | `t` from hardcoded `TABLES` list |
| `pipeline/init_db.py:224` | `f"SELECT ... WHERE table_name = '{table_name}'"` | `table_name` from hardcoded list |
| `pipeline/build_daily_features.py` | Column names from `ORB_LABELS` | Internal constants, not user input |
| `trading_app/db_manager.py:271,286` | `f"ALTER TABLE ... ADD COLUMN {col} {typedef}"` | `col`/`typedef` from hardcoded DST columns dict |
| `scripts/migrations/migrate_add_dynamic_columns.py:72` | `f"ALTER TABLE daily_features ADD COLUMN {col_name} {dtype}"` | Hardcoded column defs |
| `scripts/tools/build_edge_families.py:134` | `f"ALTER TABLE edge_families ADD COLUMN {col} {typ}"` | Hardcoded column defs |
| `scripts/infra/run_parallel_ingest.py:106-107` | `f"INSERT ... FROM {alias}.bars_1m"` | `alias` from internal temp DB name |
| `trading_app/strategy_discovery.py:337` | `f"SELECT * FROM daily_features WHERE {where}"` | `where` from hardcoded column names with `?` params |
| `trading_app/strategy_fitness.py:211,254` | f-string SQL with hardcoded column names | All identifiers internal |
| `trading_app/nested/compare.py:30` | f-string SQL with hardcoded columns | Internal |
| `trading_app/setup_detector.py:60` | `f"SELECT * FROM daily_features WHERE {where_sql}"` | `where_sql` uses `?` parameterized values |

**Low risk — UI-facing but mitigated:**

| Location | Pattern | Mitigation |
|----------|---------|------------|
| `ui/db_reader.py:46` | `f"SELECT COUNT(*) FROM {t}"` | `t` from hardcoded `tables` list |
| `ui/db_reader.py:94` | `f"SELECT * FROM validated_setups WHERE expectancy_r >= {min_expectancy_r}"` | Float argument; read-only connection; DuckDB doesn't support stacked queries |
| `ui/db_reader.py:107` | `f"SELECT * ... WHERE trading_day = '{trading_day}'"` | String from Streamlit UI; read-only connection; local-only tool |
| `ui/db_reader.py:162` | `f"SELECT ... WHERE table_name='{t}'"` | `t` from DB query result (information_schema) |

**Assessment:** No SQL injection vulnerabilities found in production code paths. The `ui/db_reader.py` f-string patterns accept UI input without parameterization, but all connections are `read_only=True` and this is a local Streamlit dashboard. **Minor recommendation:** Convert `get_daily_features()` and `get_validated_strategies()` to use parameterized queries for defense-in-depth. Not a critical finding.

### Command Injection Assessment

Searched for `subprocess`, `os.system`, `eval()`, `exec()` across all `.py` files.

**subprocess.run() — all safe:**

| Module | Pattern | Assessment |
|--------|---------|------------|
| `pipeline/run_pipeline.py` | `subprocess.run(cmd, ...)` | `cmd` is list-form from hardcoded args |
| `pipeline/run_full_pipeline.py` | `subprocess.run(cmd, ...)` | `cmd` is list-form from hardcoded args |
| `pipeline/health_check.py` | `subprocess.run([sys.executable, ...])` | List-form, controlled args |
| `pipeline/dashboard.py` | `subprocess.run([sys.executable, ...])` | List-form, controlled args |
| `scripts/operator_status.py` | `subprocess.run(["git", ...])` | List-form, hardcoded |
| `scripts/infra/parallel_rebuild.py` | `subprocess.Popen([...])` | List-form, controlled |
| `scripts/infra/run_parallel_ingest.py` | `subprocess.run([...])` | List-form, controlled |
| `scripts/infra/run_backfill_overnight.py` | `subprocess.run([...])` | List-form, controlled |
| `scripts/ingestion/ingest_*.py` | `subprocess.run([...])` | List-form, controlled |

**shell=True — mitigated:**

| Module | Pattern | Mitigation |
|--------|---------|------------|
| `ui/sandbox_runner.py:151` | `subprocess.run(run_cmd, shell=True, ...)` | Has `_validate_command()` executable allowlist + `_has_shell_injection()` metacharacter blocker. Blocks pipes, chaining, eval, exec, backticks, `$(`. Input comes from Streamlit chat UI. |

**os.system() — safe:**

| Module | Pattern | Assessment |
|--------|---------|------------|
| `scripts/tools/explore.py:33` | `os.system('cls' if os.name == 'nt' else 'clear')` | Hardcoded string literal. No injection. |

**eval()/exec() — none found** in production or script code.

**Assessment:** No command injection vulnerabilities. All `subprocess` calls use list-form arguments except `sandbox_runner.py` which has explicit metacharacter filtering. PASS.

### Hardcoded Credentials Search

Searched for patterns: `password=`, `secret=`, `token=`, `api_key=`, `apikey=` (case-insensitive).

| File | Finding | Risk |
|------|---------|------|
| `scripts/infra/telegram_feed.py:19` | `BOT_TOKEN = "8572496011:AAFF..."` | **FINDING** — Telegram bot token hardcoded |
| `tests/test_trading_app/test_ai/test_query_agent.py:86` | `agent.api_key = "test-key"` | Test fixture — not a real key |

**Telegram token status:** The file is currently **untracked** (`??` in git status). It has NOT been committed to the repository. The `.gitignore` covers `.env` files but does not explicitly exclude `telegram_feed.py`.

**Recommendation:** Move `BOT_TOKEN` and `CHAT_ID` to `.env` and read via `os.environ.get()` before this file is ever committed. Already documented in Task 5 findings.

### .gitignore Coverage

Verified `.gitignore` covers required patterns:

| Pattern | Covered | Status |
|---------|---------|--------|
| `.env`, `.env.*` | ✅ Lines 34-35 | PASS |
| `gold.db`, `gold.db.wal` | ✅ Lines 2-3 | PASS |
| `__pycache__/`, `*.pyc` | ✅ Lines 24-25 | PASS |
| `*.egg-info/` | ✅ Line 26 | PASS |
| `.venv/`, `venv/`, `env/` | ✅ Lines 27-29 | PASS |
| `.claude/` | ✅ Line 44 | PASS |
| `.vscode/`, `.idea/` | ✅ Lines 38-39 | PASS |
| `dbn/`, `*.dbn`, `*.dbn.zst` | ✅ Lines 8-10 | PASS |
| `backups/` | ✅ Line 14 | PASS |
| `pipeline/checkpoints/` | ✅ Line 32 | PASS |
| `reports/`, `outputs/` | ✅ Lines 68-69 | PASS |

**No gaps found.** All sensitive files, databases, caches, and generated outputs are properly gitignored. PASS.

### DuckDB Connection Leak Prevention

Audited all `duckdb.connect()` calls across the codebase for proper cleanup:

**Core pipeline modules — context managers (`with`):**

| Module | Pattern | Status |
|--------|---------|--------|
| `pipeline/build_bars_5m.py:303` | `with duckdb.connect()` | PASS |
| `pipeline/build_daily_features.py:950` | `with duckdb.connect()` | PASS |
| `pipeline/init_db.py:180` | `with duckdb.connect()` | PASS |
| `pipeline/check_db.py:34` | `with duckdb.connect()` | PASS |

**Core trading_app modules — context managers (`with`):**

| Module | Pattern | Status |
|--------|---------|--------|
| `trading_app/db_manager.py:44,303` | `with duckdb.connect()` | PASS |
| `trading_app/outcome_builder.py:526` | `with duckdb.connect()` | PASS |
| `trading_app/strategy_discovery.py:519` | `with duckdb.connect()` | PASS |
| `trading_app/strategy_validator.py:375` | `with duckdb.connect()` | PASS |
| `trading_app/portfolio.py:258,526` | `with duckdb.connect()` | PASS |
| `trading_app/strategy_fitness.py:374,390` | `with duckdb.connect()` | PASS |
| `trading_app/walk_forward.py:139` | `with duckdb.connect()` | PASS |
| `trading_app/view_strategies.py` (5 calls) | `with duckdb.connect()` | PASS |
| `trading_app/paper_trader.py:186` | `with duckdb.connect()` | PASS |
| `trading_app/rolling_portfolio.py` (5 calls) | `with duckdb.connect()` | PASS |
| `trading_app/rolling_correlation.py` (4 calls) | `with duckdb.connect()` | PASS |

**Pipeline modules — try/finally + .close():**

| Module | Pattern | Status |
|--------|---------|--------|
| `pipeline/ingest_dbn.py:157` | `atexit.register(_close_con)` + explicit `con.close()` | PASS (adequate) |
| `pipeline/ingest_dbn_daily.py:262` | `atexit.register(_close_con)` + explicit `con.close()` | PASS (adequate) |
| `pipeline/ingest_dbn_mgc.py:522` | `atexit.register(_close_con)` + explicit `con.close()` | PASS (adequate) |
| `pipeline/audit_bars_coverage.py` (2 calls) | `try/finally + con.close()` | PASS |
| `pipeline/dashboard.py` (4 calls) | `try/finally + con.close()` | PASS |
| `pipeline/health_check.py:38` | `try/finally + con.close()` | PASS |

**Secondary trading_app modules — try/finally + .close():**

| Module | Pattern | Status |
|--------|---------|--------|
| `trading_app/ai/sql_adapter.py` (7 calls) | `try/finally + con.close()` | PASS |
| `trading_app/ai/corpus.py` (2 calls) | `try/finally + con.close()` | PASS |
| `trading_app/ai/strategy_matcher.py` | `try/finally + con.close()` | PASS |
| `trading_app/nested/builder.py` | `try/finally + con.close()` | PASS |
| `trading_app/nested/discovery.py` | `try/finally + con.close()` | PASS |
| `trading_app/nested/validator.py` | `try/finally + con.close()` | PASS |
| `trading_app/nested/compare.py` | `try/finally + con.close()` | PASS |
| `trading_app/nested/audit_outcomes.py` | `try/finally + con.close()` | PASS |
| `trading_app/regime/discovery.py` | `try/finally + con.close()` | PASS |
| `trading_app/regime/validator.py` | `try/finally + con.close()` | PASS |
| `trading_app/regime/compare.py` | `try/finally + con.close()` | PASS |

**UI modules — try/finally + .close():**

| Module | Pattern | Status |
|--------|---------|--------|
| `ui/db_reader.py` (all functions) | `try/finally + conn.close()` or `get_connection()` with try/finally | PASS |

**Remaining modules without explicit cleanup (bare connect):**

| Module | Pattern | Risk |
|--------|---------|------|
| `trading_app/cascade_table.py:50` | `con = duckdb.connect(read_only=True)` | Short-lived function, read-only |
| `trading_app/market_state.py:124` | `con = duckdb.connect(read_only=True)` | Short-lived function, read-only |
| `trading_app/live_config.py:106,146` | `con = duckdb.connect(read_only=True)` | Short-lived functions, read-only |
| `trading_app/validate_1800_composite.py:37` | `con = duckdb.connect(read_only=True)` | Script entry point |
| `trading_app/validate_1800_workhorse.py:50` | `con = duckdb.connect(read_only=True)` | Script entry point |

**Assessment:** All core write paths use `with` context managers. All secondary modules use either `try/finally + con.close()` or `atexit` handlers. A handful of utility/script modules have bare `connect()` calls without explicit cleanup, but these are all read-only connections in short-lived functions where Python's garbage collector will handle cleanup. The `check_drift.py` drift detection module specifically checks for connection leak patterns (has `_connect_leak_pipeline` and `_connect_leak_trading_app` checks). PASS.

### Drift Check Status

**NOTE:** Python execution blocked by Windows sandbox permissions. `python pipeline/check_drift.py` could not be run.

**Static verification of drift check implementation** (`pipeline/check_drift.py`):
- Checks for hardcoded symbols, `iterrows()` performance, import direction violations, connection leaks (pipeline + trading_app), schema mismatches, timezone hygiene, analytical honesty, and CLAUDE.md size cap.
- Each check function returns (pass/fail, issues list).
- All checks are exercised by 22+ dedicated tests in `test_check_drift.py`.
- Pre-commit hook runs drift check as stage [1] before allowing commits.
- CI runs drift check on every push/PR.

**Prior execution results:** Drift check passes as of latest committed code (confirmed by successful CI runs and prior test suite execution showing 1177 tests passing).

### Compliance Summary

| Check | Result | Details |
|-------|--------|---------|
| SQL injection | PASS | No user-input-to-SQL in production paths; UI read-only with minor f-string patterns |
| Command injection | PASS | All subprocess calls use list-form args; sandbox_runner has metachar filter |
| eval()/exec() | PASS | None found in production code |
| Hardcoded credentials | **FINDING** | Telegram bot token in untracked file (NOT committed) |
| .gitignore coverage | PASS | All sensitive files covered (.env, gold.db, credentials, caches) |
| Connection leak prevention | PASS | Core paths use `with`, secondary use try/finally, drift check enforces |
| Drift check implementation | PASS (static) | 23 checks, tested, enforced by pre-commit + CI |
| DuckDB parameterized queries | PASS | Production code uses `?` params; identifiers from internal constants |

### Verdict: PASS (with one prior finding)

Security posture is solid across the codebase:
- **No SQL injection** in production paths. All user-facing SQL in pipeline/trading_app uses parameterized queries or hardcoded identifiers. The UI module has minor f-string SQL patterns but all connections are read-only.
- **No command injection.** All subprocess calls use list-form arguments. The one `shell=True` usage has explicit metacharacter filtering.
- **No eval/exec** in production code.
- **No committed credentials.** The Telegram bot token is in an untracked file (finding carried from Task 5).
- **Comprehensive connection management.** Core write paths use context managers. Secondary modules use try/finally. Drift check enforces this statically.
- **Defense in depth.** Drift check (23 rules) + pre-commit hook (4 stages) + CI (lint + drift + tests) + validation gates (11 total).

**Minor recommendations (non-blocking):**
1. Convert `ui/db_reader.py` f-string SQL to parameterized queries (defense-in-depth for local UI)
2. Move Telegram credentials to `.env` before committing `telegram_feed.py`
