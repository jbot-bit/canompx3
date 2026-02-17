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
