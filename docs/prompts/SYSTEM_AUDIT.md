# Full System Audit — Self-Identifying Comprehensive Integrity Check

**Posture:** Nothing is in sync until proven in sync. Every document claim is stale until verified against code and data. Every "DONE" is incomplete until its artifacts exist and pass. Every drift check is toothless until you confirm it actually catches violations.

**When to run:** Before any major release, after any multi-day refactor, monthly maintenance, or whenever something feels "off."

**Activation:** This prompt is referenced in `.claude/rules/audit-prompts.md`. Claude Code should suggest running it when:
- The user asks for a "health check," "system audit," or "full check"
- More than 2 weeks have passed since the last audit
- A multi-file refactor or schema change just completed
- The user says something feels "off" or "out of sync"

---

## PHASE 0 — Self-Identification (What Already Exists)

Before doing ANYTHING, inventory the existing audit infrastructure so you don't duplicate it. The system already has automated checks — your job is to RUN them, interpret their results, then manually cover the gaps they can't.

### 0A. Inventory Existing Automated Checks
Read each of these files. Build the coverage map below.
- [ ] `pipeline/check_drift.py` — static code/config checks (note actual check count)
- [ ] `pipeline/health_check.py` — infra health (7 checks: deps, DB, DBN files, drift, integrity, tests, git hooks)
- [ ] `scripts/tools/audit_integrity.py` — data integrity (17 numbered checks, but **only 10 enforce** — checks 7-10, 13-14, 17 are informational-only with `return []`)
- [ ] `docs/prompts/ENTRY_MODEL_GUARDIAN.md` — entry model deep-dive (2-pass)
- [ ] `docs/prompts/PIPELINE_DATA_GUARDIAN.md` — pipeline data deep-dive (2-pass)
- [ ] ALL files in `.claude/rules/` — operational guardrails for Claude Code

```
AUTOMATED (just RUN these, record results):
  check_drift.py         → [N] static code/config checks
  health_check.py        → 7 infra checks (runs drift + integrity + tests internally)
  audit_integrity.py     → 10 enforcing checks + 7 informational displays
  pytest tests/          → [N] unit/integration tests

MANUAL DEEP-DIVES (exist, require execution):
  ENTRY_MODEL_GUARDIAN   → entry model code vs docs vs data
  PIPELINE_DATA_GUARDIAN → pipeline lineage, aggregation, integrity

THIS AUDIT FILLS (gaps none of the above cover):
  → Documentation vs reality sync (all docs)
  → Config registry completeness beyond drift check 12
  → Build chain staleness per instrument
  → Live trading config coherence
  → Research script hygiene (120 scripts, 197 outputs)
  → Asset config / cost model / environment correctness
  → .claude/rules/ staleness
  → docs/specs/ compliance
  → Smoke tests (can the system actually run end-to-end?)
  → Pending/inconclusive research item staleness
```

### 0B. Triage — What Changed Since Last Audit
- [ ] `git log --oneline --since="30 days ago" -- pipeline/ trading_app/ config.py` → changed production files
- [ ] `git log --oneline --since="30 days ago" -- CLAUDE.md TRADING_RULES.md RESEARCH_RULES.md ROADMAP.md` → changed docs
- [ ] `git log --oneline --since="30 days ago" -- tests/` → changed tests
- [ ] `git log --oneline --since="30 days ago" -- .claude/rules/ docs/prompts/` → changed rules/prompts
- [ ] `git log --oneline --since="30 days ago" -- pipeline/asset_configs.py pipeline/cost_model.py pipeline/dst.py` → changed infrastructure configs

**Focus 80% of manual effort on files that CHANGED.** Unchanged code that passed last audit only needs re-checking if a dependency changed under it.

### 0C. Locate Last Audit
- [ ] Check `research/output/` for previous audit reports (e.g., `HIGH_LEVEL_AUDIT_*.md`)
- [ ] If a prior audit exists, note its date and any open issues that may still be unresolved

---

## PHASE 1 — Run All Automated Checks (MECHANICAL EXECUTION — MANDATORY)

Run everything that's already built. Don't think — just execute and record.

**ANTI-HALLUCINATION RULE:** You MUST actually execute every command in this phase using bash/shell. Do NOT "read the code and infer" that checks pass. Do NOT say "based on the code, this likely passes." Reading a drift check's source code is NOT the same as running it — the check might pass in code but fail against current data. Copy-paste each command, capture real stdout/stderr, and record the actual output. If you cannot execute a command (e.g., no DB access), mark it `SKIPPED — NO EXECUTION ENVIRONMENT` in Section 7, not `PASS`.

This is the formal verification layer. Code reading is discovery (Phase 0). Mechanical execution is proof (Phase 1).

### 1A. Test Suite
```bash
pytest tests/ -x -q 2>&1 | tail -5
```
- [ ] Record: collected, passed, failed, errors
- [ ] **If ANY test fails → STOP. Fix first. Nothing else is trustworthy.**
- [ ] Compare count against CLAUDE.md and ROADMAP.md claims

### 1B. Drift Detection
```bash
python pipeline/check_drift.py 2>&1
```
- [ ] Record: total checks, passed, failed
- [ ] For each failure: `REAL_VIOLATION` / `STALE_CHECK` / `FALSE_POSITIVE`
- [ ] Compare check count against CLAUDE.md claim

### 1C. Health Check (runs drift + integrity + tests internally)
```bash
python pipeline/health_check.py 2>&1
```
- [ ] Record 7 check results

### 1D. Data Integrity Audit
```bash
python scripts/tools/audit_integrity.py 2>&1
```
- [ ] Record: enforcing checks (1-6, 11-12, 15-16) pass/fail
- [ ] Note informational output (7-10, 13-14, 17) for use in later phases
- [ ] **Capture the informational stats** — you'll need row counts, date ranges, FDR breakdown in Phase 4

### 1E. MCP Smoke Test
- [ ] `list_available_queries()` → returns without error, note template count
- [ ] `query_trading_db(template="table_counts")` → returns plausible numbers
- [ ] `get_strategy_fitness(summary_only=True)` → returns without error, counts plausible
- [ ] `get_canonical_context()` → returns without error

### 1F. End-to-End Smoke Test
```bash
# Can the paper trader actually run for 1 trading day without error?
python trading_app/paper_trader.py --instrument MGC --start 2025-01-02 --end 2025-01-03 2>&1 | tail -10
```
- [ ] Exits cleanly (no Python errors)
- [ ] If errors → `SMOKE_TEST_FAILURE` (investigate before proceeding)

**After Phase 1:** You have a baseline. Everything PASSED is provisionally clean. Everything FAILED is a known issue. Phases 2+ focus on what automated checks CAN'T cover.

---

## PHASE 2 — Infrastructure Config Audit (Manual)

These are foundational configs that no automated check fully validates.

### 2A. Asset Configs (`pipeline/asset_configs.py`)
For EACH instrument in ASSET_CONFIGS:
- [ ] `dbn_path` → does the file/directory actually exist on disk?
- [ ] `symbol` → matches what's stored in bars_1m? (query a sample row)
- [ ] `outright_pattern` → regex makes sense for the instrument's contract format?
- [ ] `minimum_start_date` → plausible? (MGC should be 2019+, MNQ 2024+, etc.)
- [ ] `enabled_sessions` → compare against `pipeline/dst.py:SESSION_CATALOG` keys. Any session enabled here but not in SESSION_CATALOG → `SESSION_CONFIG_DRIFT`
- [ ] Dead instruments (MCL, SIL, M6E) → still in config but should NOT have active validated_setups. Verify.

### 2B. Cost Model (`pipeline/cost_model.py`)
- [ ] TRADING_RULES.md claims MGC = $5.74/RT. Verify in code.
- [ ] Per-instrument friction: MGC $5.74, MNQ $2.74, MES $3.74, M2K $3.24. All in code?
- [ ] `friction_points` computation → uses the correct tick_value per instrument?
- [ ] Are there any instruments with cost specs that don't match TRADING_RULES.md? → `COST_MODEL_DRIFT`

### 2C. DST Session Resolution (`pipeline/dst.py`)
- [ ] SESSION_CATALOG has entries for all 10 documented sessions
- [ ] Spot-check: resolve session times for a known date (e.g., 2025-07-15 summer, 2025-01-15 winter). Do CME_REOPEN, LONDON_METALS, NYSE_OPEN resolve to correct UTC times?
- [ ] `DOW_ALIGNED_SESSIONS` and `DOW_MISALIGNED_SESSIONS` — do they match the table in TRADING_RULES.md?
- [ ] `validate_dow_filter_alignment()` exists and is called where needed?
- [ ] No old fixed-clock session names (`0900`, `1800`, `0030`, `2300`) anywhere in dst.py → drift check 33 should catch this but verify

### 2D. Environment / Paths
- [ ] `.env` file exists with `DUCKDB_PATH`, `DATABENTO_API_KEY`, `SYMBOL`, `TZ_LOCAL`
- [ ] `pipeline/paths.py` correctly reads `DUCKDB_PATH` from `.env` / environment
- [ ] No scratch DB lingering at `C:/db/gold.db` → drift check 37 should catch but verify
- [ ] `GOLD_DB_PATH` resolves to a file that exists and is readable

---

## PHASE 3 — Documentation vs Reality (Manual)

Documentation rots faster than code. No automated check verifies docs.

### 3A. CLAUDE.md Audit
- [ ] **Data flow diagram:** Trace each step by finding the writer module in code. Does each arrow exist?
- [ ] **Key commands:** Run each with `--help`. Exist? Flags match?
- [ ] **MCP tool table:** Compare against `list_available_queries()` from Phase 1. Phantom / undocumented templates?
- [ ] **Hard numbers** (drift check count, test count, strategy count, edge family count): Compare Phase 1 actuals. **If CLAUDE.md says 37 and you counted 35, that's a finding.**
- [ ] **Database location claim:** Matches `pipeline/paths.py` default?
- [ ] **Instrument status** (active vs dead): Matches `asset_configs.py`?
- [ ] **Strategy classification thresholds** (CORE >= 100, REGIME 30-99): Match `config.py`?
- [ ] **Source contract mapping table** (GC→MGC, ES→MES, RTY→M2K, etc.): Match `asset_configs.py`?

### 3B. TRADING_RULES.md Audit
- [ ] **Session table (10 sessions):** Compare every row against `dst.py:SESSION_CATALOG`. Flag mismatches.
- [ ] **Entry model table (E1/E2/E3):** Compare trigger logic against `entry_rules.py`. Line by line.
- [ ] **Cost model ($5.74/RT):** Match `cost_model.py`?
- [ ] **Live portfolio tiers 1/2/3:** Match `live_config.py`? Every family, every gate.
- [ ] **Confirmed edges table:** For each "DEPLOYED" edge, grep execution_engine.py + portfolio.py. Code path exists?
- [ ] **NO-GO table:** For each NO-GO, grep `pipeline/` and `trading_app/` for the feature name. Found in production → `NO_GO_ZOMBIE`.
- [ ] **Calendar filters status:** Doc says "not yet wired into portfolio.py/paper_trader.py." Still true?
- [ ] **ATR velocity filter:** `ATRVelocityFilter` in config.py matches documented 2-condition rule? Columns exist in daily_features?
- [ ] **Pending/Inconclusive table:** For EACH item, has it been resolved since the doc was last updated? If the research was completed but the table still says "NEXT STEP" or "WATCH" → `DOC_STALE`. Cross-reference against `research/output/` for completed work.

### 3C. RESEARCH_RULES.md Audit
- [ ] Sample size tiers match `config.py` CORE_MIN / REGIME_MIN?
- [ ] NO-GO rules cross-reference with TRADING_RULES.md NO-GO table. Both agree?
- [ ] "Current Research Status" section — is it current or stale?

### 3D. ROADMAP.md Audit
- [ ] **Every "DONE" phase:** Artifact exists? File/module at stated path? Test count matches?
- [ ] **Every "TODO" item:** NOT accidentally built? If code exists for a TODO → update ROADMAP or flag orphan.
- [ ] **Hard numbers** in phases: Cross-check Phase 1 actuals.
- [ ] **"Rules to Enforce" section:** Still accurate?

### 3E. REPO_MAP.md Audit
- [ ] Run `python scripts/tools/gen_repo_map.py` and diff. Any diff → `REPO_MAP_STALE`.

### 3F. .claude/rules/ Audit
These guide Claude Code behavior. If stale, Claude gets wrong instructions.
- [ ] `audit-prompts.md` — references all prompts in `docs/prompts/`? (Including this one!)
- [ ] `daily-features-joins.md` — triple-join rule still correct? `double_break` look-ahead warning still accurate?
- [ ] `mcp-usage.md` — decision framework matches current MCP templates from Phase 1E?
- [ ] `pipeline-patterns.md` — DST "fully resolved" still holds? Session names current?
- [ ] `validation-workflow.md` — validator flags match `strategy_validator.py --help`? MNQ `--no-walkforward` still required?

### 3G. docs/specs/ Compliance
- [ ] List all files in `docs/specs/`. For each:
  - Implemented? → Was spec followed? Deviations?
  - Not implemented? → Still in ROADMAP TODO?
  - Implemented but spec ignored → `SPEC_VIOLATION`

### 3H. Document Cross-Reference
- [ ] Trading logic in CLAUDE.md contradicts TRADING_RULES.md?
- [ ] Code structure in TRADING_RULES.md that should only be in CLAUDE.md?
- [ ] Research methodology in TRADING_RULES.md contradicts RESEARCH_RULES.md?
- [ ] `CANONICAL_*.txt` files exist? Treated as frozen per CLAUDE.md?

---

## PHASE 4 — Configuration Sync (Manual)

Automated drift checks cover SOME of this (checks 12-14, 26-30), but not all.

### 4A. Filter Registry (beyond drift check 12)
- [ ] ALL keys from `config.py:ALL_FILTERS`
- [ ] DISTINCT `filter_type` from validated_setups + experimental_strategies (via MCP)
- [ ] Every DB value in ALL_FILTERS → `PHANTOM_FILTER_IN_DB` if not
- [ ] Every ALL_FILTERS key has rows in experimental_strategies → `DEAD_FILTER` if not
- [ ] `_FILTER_SPECIFICITY` complete — every filter ranked
- [ ] `get_filters_for_grid()` returns correct session-aware filters

### 4B. Entry Model Sync (beyond drift check 13)
- [ ] `config.py:ENTRY_MODELS` = `["E1", "E2", "E3"]`. No E0.
- [ ] DB has zero E0 rows in production tables (Phase 1D should confirm — cross-check)
- [ ] `entry_rules.py` handles exactly E1, E2, E3
- [ ] `outcome_builder.py` iterates same three

### 4C. Session / ORB Label Sync (beyond drift check 32)
- [ ] `dst.py:SESSION_CATALOG` keys vs `config.py:ORB_LABELS` vs `asset_configs.py` enabled_sessions
- [ ] Any three-way mismatch → `SESSION_LABEL_DRIFT`
- [ ] SINGAPORE_OPEN excluded: execution_engine.py, portfolio.py, strategy_fitness.py

### 4D. Grid Sync
- [ ] `config.py:RR_TARGETS` + `CB_LEVELS` match `outcome_builder.py` iteration
- [ ] Match `strategy_discovery.py` grid
- [ ] E2/E3 restricted to CB1 only (drift check 30) — verify independently

### 4E. Threshold Sync
- [ ] `CORE_MIN`, `REGIME_MIN` in config.py used by `strategy_validator.py`
- [ ] `strategy_fitness.py` FIT/WATCH/DECAY/STALE thresholds documented and matching code
- [ ] `rolling_portfolio.py` stability thresholds match docs

### 4F. MCP Parameter Allowlists
- [ ] `sql_adapter.py:VALID_*` sets match `config.py` values
- [ ] Any config value missing from MCP allowlist → `MCP_ALLOWLIST_STALE`

---

## PHASE 5 — Database Integrity (Manual — beyond audit_integrity.py)

### 5A. Schema Alignment
- [ ] `init_db.py` CREATE TABLE vs actual DB schema (MCP `template="schema_info"`)
- [ ] Column in code but not DB → `SCHEMA_BEHIND`
- [ ] Column in DB but not code → `ORPHAN_COLUMN`
- [ ] All indexes exist as defined

### 5B. Row Count Ratios
From Phase 1D informational output + MCP `template="table_counts"`:

| Check | Expected | Tolerance |
|-------|----------|-----------|
| bars_5m ≈ bars_1m / 5 | Per instrument | ±5% |
| daily_features count divisible by 3 per (symbol) | Exact | 0 tolerance |
| validated_setups < experimental_strategies | Always | — |
| edge_families < validated_setups | Always (many:1) | — |

### 5C. Temporal Coverage Per Instrument
For each ACTIVE instrument (MGC, MNQ, MES, M2K):
- [ ] Earliest/latest `trading_day` in bars_1m, bars_5m, daily_features, orb_outcomes
- [ ] Date ranges consistent? (bars_1m >= bars_5m >= daily_features)
- [ ] **orb_outcomes latest == daily_features latest?** If not → `STALE_OUTCOMES`
- [ ] Any trading day gaps (missing weekdays that aren't holidays)?

### 5D. Orphan Detection
- [ ] validated_setups with `filter_type` NOT in `config.py:ALL_FILTERS` → `ORPHAN_STRATEGY`
- [ ] validated_setups with `entry_model` NOT in `config.py:ENTRY_MODELS` → `ORPHAN_STRATEGY`
- [ ] edge_families referencing nonexistent validated_setups → `ORPHAN_FAMILY`

---

## PHASE 6 — Build Chain Staleness

### 6A. Per-Instrument Build Chain Status
Full chain: `ingest → 5m bars → daily_features → outcomes → discovery → validation → edge_families`.

Build the table:

| Instrument | bars_1m latest | bars_5m latest | daily_features latest | orb_outcomes latest | experimental latest | validated latest | edge_families latest | Status |
|-----------|---------------|---------------|---------------------|--------------------|--------------------|-----------------|---------------------|--------|

- [ ] All same date → `UP_TO_DATE`
- [ ] Any behind → `BUILD_CHAIN_GAP` (name the stale step)

### 6B. Code-Change-Triggered Rebuilds
- [ ] `git log --oneline --since="60 days ago" -- trading_app/outcome_builder.py` → changes since last outcome build?
  - outcome_builder changed but outcomes not rebuilt → `REBUILD_NEEDED`
- [ ] Same for `strategy_discovery.py`, `strategy_validator.py`, `build_edge_families.py`
- [ ] Same for `pipeline/build_daily_features.py` (rebuilds cascade to everything downstream)

### 6C. Validator Flag Verification
- [ ] Each instrument's last run used correct flags?
  - MNQ: `--no-walkforward` (only 2 years of data)
  - All: `--no-regime-waivers --min-years-positive-pct 0.75`
- [ ] Edge families built AFTER last validation?

---

## PHASE 7 — Live Trading Readiness

### 7A. Live Config Coherence
- [ ] Every family in `live_config.py` Tier 1/2/3 exists in validated_setups AND edge_families
- [ ] Tier 2: families with `rolling_stability >= 0.6` — currently above? (Use `get_strategy_fitness`)
- [ ] Tier 3: regime-gated families currently FIT?
- [ ] SINGAPORE_OPEN excluded from all tiers

### 7B. Execution Engine Wiring
- [ ] Early exit rules coded: 15-min CME_REOPEN kill, 30-min TOKYO_OPEN kill
- [ ] IB-conditional exit: 120-min IB + 7-hour hold + opposed-kill in execution_engine.py
- [ ] TOKYO_OPEN LONG-ONLY: direction filter in execution path
- [ ] SINGAPORE_OPEN hard exclusion in execution_engine.py

### 7C. Risk Manager
- [ ] max concurrent, daily loss, max per ORB match `live_config.py` parameters
- [ ] Circuit breaker prevents new entries after limit hit (trace code path)

### 7D. Feature Parity — Backtest vs Live Entry Mechanics

> **Cross-reference:** ENTRY_MODEL_GUARDIAN.md Pass 2A-PARITY performs the same check at entry-model granularity. If this phase finds a violation, escalate to ENTRY_MODEL_GUARDIAN for per-model deep dive.

**This catches train/inference skew: where the backtest path and the live path make different entry decisions for the same bar data.**

The system has TWO entry paths:
- **Backtest path:** `outcome_builder.py` → calls `entry_rules.py:detect_break_touch()` and `detect_entry_with_confirm_bars()` directly
- **Live path:** `paper_trader.py` → `execution_engine.py` → (its own entry logic)

These MUST produce identical entry decisions for the same input data. Any divergence means backtest results don't predict live performance.

For EACH entry model (E1, E2, E3):
- [ ] **Trace the backtest code path:** `outcome_builder.py` → `entry_rules.py` function called, parameters passed, return values used
- [ ] **Trace the live code path:** `execution_engine.py` → how does it detect a break? What function/logic determines entry?
- [ ] **Compare entry trigger conditions:** Same ORB level comparison? Same bar field used (high/low/close)? Same direction logic?
- [ ] **Compare CB confirmation logic:** Same number of bars checked? Same close-above/below logic?
- [ ] **Compare early exit rules:** Do backtest outcomes use the same kill timers (15-min CME_REOPEN, 30-min TOKYO_OPEN) as execution_engine?
- [ ] **Compare cost model application:** friction_points applied the same way in outcome_builder as in paper_trader PnL calculation?
- [ ] Any divergence → `FEATURE_PARITY_VIOLATION` (CRITICAL severity — invalidates backtested edge)

**Spot-check method:** Pick 3 recent trades from paper_trader output. For each, find the same trading_day + session + direction in orb_outcomes. Do entry price, target, stop, and result agree? If not, trace the divergence.

---

## PHASE 8 — Test Suite Deep Check

### 8A. Coverage Gaps
- [ ] All `.py` in `pipeline/` and `trading_app/` → corresponding test file?
- [ ] Untested modules → `UNTESTED_MODULE`
- [ ] Critical modules that MUST have tests:
  - `pipeline/dst.py`, `pipeline/calendar_filters.py`, `pipeline/cost_model.py`
  - `trading_app/config.py`, `trading_app/strategy_fitness.py`
  - `trading_app/live_config.py`, `trading_app/mcp_server.py`

### 8B. Test Staleness
- [ ] Grep tests for `E0` → `STALE_TEST`
- [ ] Grep tests for old session names (`0900`, `1800`, `0030`, `2300`) → `STALE_TEST`
- [ ] Grep tests for hardcoded counts that may have changed → `BRITTLE_TEST`

---

## PHASE 9 — Research & Script Hygiene

### 9A. Research Script Inventory
There are ~120 research scripts and ~197 output files. Full inventory is impractical. Instead:
- [ ] For each **TRADING_RULES.md reference** to a script (e.g., `research/research_day_of_week.py`): does the file exist?
- [ ] For each **TRADING_RULES.md reference** to output (e.g., `research/output/day_of_week_breakdown.csv`): does the file exist?
- [ ] Spot-check 5 random research scripts: do they import cleanly? (`python -c "import research.script_name"`)
- [ ] Any scripts with names referencing dead features (E0, old sessions, NODBL filter) → `STALE_RESEARCH_SCRIPT`

### 9B. Research Findings vs Production Code
For each "Confirmed Edges" entry in TRADING_RULES.md:
- [ ] "DEPLOYED" → deployment traceable in code?
- [ ] "OPTIONAL" / "VALIDATED" → clear separation from mandatory logic?

### 9C. NO-GO Enforcement
For each NO-GO in TRADING_RULES.md:
- [ ] Grep `pipeline/` and `trading_app/` for feature name
- [ ] Found in production → `NO_GO_ZOMBIE`
- [ ] Found in `research/` only → fine

### 9D. Data Snooping Quarantine Check
- [ ] `strategy_discovery.py` does NOT access walk-forward window boundaries from `strategy_validator.py` → `DATA_LEAK_RISK` if it does
- [ ] `strategy_validator.py` walk-forward splits are deterministic and not tunable by discovery results
- [ ] No code path allows backtest-phase logic to peek at holdout-phase data
- [ ] Any research script that reads both discovery and validation data in the same run → flag for review
- [ ] Cross-reference RESEARCH_RULES.md "Bailey Rule" and sample-size requirements — still enforced in validator code?

### 9E. Pending/Inconclusive Staleness
For each item in TRADING_RULES.md "Pending / Inconclusive" table:
- [ ] Still pending? Or has research completed (output exists in `research/output/`) but table wasn't updated?
- [ ] Any item marked "NEXT STEP" that's been sitting for 30+ days without progress → flag

---

## PHASE 10 — Git & CI Hygiene

### 10A. Git Hooks
- [ ] `.githooks/pre-commit` exists and is executable
- [ ] `git config core.hooksPath` = `.githooks`
- [ ] Pre-commit hook runs drift + tests

### 10B. CI Pipeline
- [ ] `.github/workflows/` contains CI config
- [ ] CI triggers on push/PR
- [ ] CI runs same checks as pre-commit

### 10C. Repo Cleanliness
- [ ] `git status` — uncommitted production file changes?
- [ ] `.py` files in project root that should be in subdirectory?
- [ ] Scratch/temp files that shouldn't be tracked?
- [ ] Scratch DB at `C:/db/gold.db` lingering?

---

## Output Format

### Section 1: Audit Header
```
SYSTEM AUDIT — [DATE] — git commit [SHORT_HASH]
Last audit: [DATE or "unknown"]
Files changed since last audit: [N] production, [N] docs, [N] tests
Automated: [N] tests passed, [N]/[N] drift checks, [N]/10 integrity enforcing, MCP [OK/ERR]
Smoke test: [PASS/FAIL]
```

### Section 2: System Scorecard
```
SYSTEM HEALTH: X/10

Phase 1  (Automated):      [PASS/FAIL] — tests: N, drift: N/N, integrity: N/10, MCP: OK, smoke: OK
Phase 2  (Infra Config):   [PASS/FAIL] — N issues
Phase 3  (Documentation):  [PASS/FAIL] — N issues (N CRITICAL, N HIGH, N MEDIUM, N LOW)
Phase 4  (Config Sync):    [PASS/FAIL] — N issues
Phase 5  (Database):       [PASS/FAIL] — N issues
Phase 6  (Build Chain):    [PASS/FAIL] — instruments current: [list]
Phase 7  (Live Trading):   [PASS/FAIL] — N issues, feature parity: [IDENTICAL/DIVERGENT]
Phase 8  (Test Suite):     [PASS/FAIL] — N gaps, N stale
Phase 9  (Research):       [PASS/FAIL] — N orphans, N zombies, N stale pending, data quarantine: [CLEAN/LEAK]
Phase 10 (Git/CI):         [PASS/FAIL] — N issues
```

### Section 3: Issue Register
| ID | Phase | Severity | Tag | What's Claimed | What's Real | Evidence | Fix Type |
|----|-------|----------|-----|---------------|-------------|----------|----------|

Severity:
- **CRITICAL**: Data corruption, broken safety gate, live trading risk, filter leak, NO-GO zombie, feature parity violation (backtest≠live), data snooping leak, hallucinated pass (check not actually executed)
- **HIGH**: Doc materially wrong, config sync broken, build chain stale, MCP allowlist wrong, cost model wrong
- **MEDIUM**: Doc stale but not misleading, orphan artifacts, coverage gap, stale tests
- **LOW**: Cosmetic, count off slightly, repo cleanliness

**Consolidated Tag Vocabulary** (canonical reference — primary tags used across all 3 audit docs; individual Charges may define additional per-charge verdicts):

| Category | Tags | Source |
|----------|------|--------|
| **Severity** | `CRITICAL`, `HIGH`, `MEDIUM`, `LOW` | All docs |
| **Doc drift** | `DOC_STALE`, `DOC_WRONG`, `BOTH_WRONG` | SYSTEM_AUDIT, ENTRY_MODEL_GUARDIAN |
| **Config drift** | `CONFIG_DRIFT`, `COST_MODEL_DRIFT`, `THRESHOLD_DRIFT`, `SESSION_CONFIG_DRIFT`, `SESSION_LABEL_DRIFT`, `MCP_ALLOWLIST_STALE` | SYSTEM_AUDIT |
| **Schema** | `SCHEMA_DRIFT`, `SCHEMA_BEHIND`, `ORPHAN_COLUMN` | SYSTEM_AUDIT, PIPELINE_DATA_GUARDIAN |
| **Data integrity** | `DATA_INTEGRITY_VIOLATION`, `AGGREGATION_ERROR`, `FEATURE_ERROR`, `OUTCOME_ERROR`, `COUNT_ANOMALY`, `DATA_LOSS_RISK` | PIPELINE_DATA_GUARDIAN |
| **Write scope** | `SCOPED`, `OVER_DELETE`, `UNDER_DELETE` | PIPELINE_DATA_GUARDIAN |
| **Join safety** | `TRIPLE_JOIN_VIOLATION`, `TRIPLE_JOIN_TRAP_RISK`, `FILTER_LEAK` | PIPELINE_DATA_GUARDIAN |
| **Ghosts** | `PHANTOM_*`, `ZOMBIE_*`, `ORPHAN_*`, `NO_GO_ZOMBIE`, `DEAD_FILTER` | SYSTEM_AUDIT, ENTRY_MODEL_GUARDIAN |
| **Staleness** | `STALE_*`, `STALE_CHECK`, `STALE_TEST`, `STALE_OUTCOMES`, `STALE_RESEARCH_SCRIPT`, `REPO_MAP_STALE`, `BUILD_GAP`, `BUILD_CHAIN_GAP`, `REBUILD_NEEDED` | SYSTEM_AUDIT |
| **Gates** | `FAIL_CLOSED`, `FAIL_OPEN`, `GATE_MISSING`, `TOOTHLESS`, `TOOTHLESS_GATE` | PIPELINE_DATA_GUARDIAN, ENTRY_MODEL_GUARDIAN |
| **Parity** | `FEATURE_PARITY_VIOLATION`, `PARITY_VIOLATION`, `IDENTICAL_PATH`, `EQUIVALENT_LOGIC`, `DIVERGENT` | SYSTEM_AUDIT, ENTRY_MODEL_GUARDIAN |
| **Testing** | `UNTESTED`, `UNTESTED_MODULE`, `BRITTLE_TEST`, `SMOKE_TEST_FAILURE` | SYSTEM_AUDIT |
| **Research** | `DATA_LEAK_RISK`, `SPEC_VIOLATION`, `UNDOCUMENTED` | All docs |
| **Verdicts** | `MATCH`, `CLEAN`, `VERIFIED`, `ENFORCED`, `CONTAINED`, `ALL_SAFE`, `FULL_COVERAGE`, `ACCURATE`, `UP_TO_DATE`, `NO_DUPLICATES` | All docs |
| **Process** | `HALLUCINATED_PASS`, `REQUIRES_HUMAN_DECISION`, `SKIPPED` | All docs |
| **Fix types** | `DOC_FIX`, `CODE_FIX`, `CONFIG_FIX`, `DATA_FIX`, `REBUILD_NEEDED`, `DELETE` | SYSTEM_AUDIT |

### Section 4: Fix Plan (ordered by severity → dependency)
For each issue:
1. **File:line** or DB table + query
2. **Current state** (quoted)
3. **Required fix**
4. **Fix type:** `DOC_FIX` / `CODE_FIX` / `CONFIG_FIX` / `DATA_FIX` / `REBUILD_NEEDED` / `DELETE`
5. **Verification command** to confirm fix worked
6. **Depends on:** (other fixes that must happen first)

### Section 5: Build Chain Recommendations
```
MGC: [UP TO DATE / NEEDS REBUILD from {step}]
MNQ: [UP TO DATE / NEEDS REBUILD from {step}]
MES: [UP TO DATE / NEEDS REBUILD from {step}]
M2K: [UP TO DATE / NEEDS REBUILD from {step}]

Rebuild commands (if needed):
  {exact commands in correct order, per validation-workflow.md}
```

### Section 6: Stale Artifact Cleanup
- Orphan scripts, dead config entries, zombie E0/old-session references, stale test fixtures, orphan DB rows
- For each: what, where, delete or update

### Section 7: Self-Audit — What This Audit Didn't Cover
**Be honest about blind spots:**
- [ ] Any Phase 0 gaps you couldn't check? List them.
- [ ] Checks skipped due to DB access or tool limitations?
- [ ] Areas of uncertainty → recommend deep-dive guardian?
  - Entry model uncertainty → ENTRY_MODEL_GUARDIAN.md
  - Pipeline data uncertainty → PIPELINE_DATA_GUARDIAN.md
- [ ] audit_integrity.py informational checks (7-10, 13-14, 17) — were the displayed stats reviewed for anomalies?

---

## Execution Order & Run Modes

Phase numbers ARE the execution order. Earlier phases inform later ones.

**Quick mode (~30 min, weekly):**
Phase 0B (triage) → Phase 1 (all automated) → Phase 6 (build chain) → Phase 3A numbers check only

**Standard mode (~2-3 hours, monthly):**
All phases. Focus manual effort on files changed per Phase 0B.

**Deep mode (~half day, pre-release or "something is wrong"):**
All phases, no triage shortcuts, plus run ENTRY_MODEL_GUARDIAN + PIPELINE_DATA_GUARDIAN.

---

## Relationship to Other Audit Prompts

| Prompt | Scope | When |
|--------|-------|------|
| **SYSTEM_AUDIT.md (this)** | Full system — 11 phases, all layers | Monthly, pre-release, "something's off" |
| **ENTRY_MODEL_GUARDIAN.md** | Entry model code/docs/data/drift deep-dive | Before changing entry model logic |
| **PIPELINE_DATA_GUARDIAN.md** | Pipeline data lineage/aggregation/integrity deep-dive | Before changing pipeline/schema/sessions |

**Escalation mapping** — when this audit finds a failure, escalate to the matching guardian:

| Failure Type | Escalate To | Example |
|-------------|-------------|---------|
| Entry model enum mismatch | ENTRY_MODEL_GUARDIAN | E0 in data but not in config |
| Outcome computation error | ENTRY_MODEL_GUARDIAN | Wrong CB logic, stop/target calc |
| Strategy discovery/validation logic | BOTH | Validator writes data AND uses entry models |
| Schema drift (tables/columns) | PIPELINE_DATA_GUARDIAN | init_db vs actual DB mismatch |
| Ingestion gate failure | PIPELINE_DATA_GUARDIAN | Toothless gate, missing validation |
| Aggregation error (1m→5m, features) | PIPELINE_DATA_GUARDIAN | Wrong OHLCV rollup |
| Write scope mismatch (DELETE>INSERT) | PIPELINE_DATA_GUARDIAN | Broad DELETE wiping unprocessed rows |
| Join violation (missing orb_minutes) | PIPELINE_DATA_GUARDIAN | Triple-join trap |
| Filter leakage (orb_outcomes unfiltered) | PIPELINE_DATA_GUARDIAN | Direct query without daily_features |
| DST / session timing | PIPELINE_DATA_GUARDIAN | Fixed-clock remnants |
| Drift check count/definition wrong | BOTH | Stale count, missing check |
| Data leak (in-sample/OOS boundary) | BOTH | Discovery peeking at validation data |

---

## Rules of Engagement

1. **Run automated checks FIRST** — don't manually verify what a script already checks.
2. **Use MCP tools for ALL DB queries** — never raw SQL when a template exists.
3. **Verify against CODE and DATA, not other documents.** Docs can agree and both be wrong.
4. **Every finding cites evidence:** file:line, query result, or command output. No "it appears that."
5. **Run actual commands** — don't assume check_drift.py passes because it passed last week. Reading source code is NOT running it. If you cannot execute, mark `SKIPPED` — never mark `PASS` without stdout proof.
6. **Flag `REQUIRES_HUMAN_DECISION`** when intent is ambiguous.
7. **Timestamp the audit.** Date + git commit hash at the top.
8. **Be paranoid about counts.** 37 claimed, 35 counted = finding.
9. **Check for ABSENCE** — missing tests, missing drift checks, missing docs.
10. **Triage by recency** — 80% effort on files changed in last 30 days.
11. **Self-report blind spots** — Section 7 exists for a reason.
12. **Fix-then-verify** — every fix includes verification command.
13. **Know what audit_integrity.py actually enforces** — only 10 of 17 checks produce violations. The other 7 just print stats. Don't claim "17 integrity checks passed" when 7 can't fail.
14. **Mechanical execution only** — "I read the code and it looks correct" is NOT a passing result. Run the command, capture the output, cite the output. See Phase 1 header.

---

## Failure Remediation Protocol — Iterative Refinement Loop

When the audit discovers a failure, do NOT just log it and move on. Failures in early phases can invalidate later phases. Follow this protocol:

### Severity-Based Response

**CRITICAL failures (Phase 1 test/drift failure, data corruption, feature parity violation, data leak):**
1. **STOP the audit.** Do not proceed to later phases — their results are unreliable.
2. **Capture the exact error** — full stdout/stderr, not a summary.
3. **Escalate to the appropriate Guardian** if the failure is in their domain:
   - Entry model issue → read and execute `ENTRY_MODEL_GUARDIAN.md`
   - Pipeline data issue → read and execute `PIPELINE_DATA_GUARDIAN.md`
4. **Propose a fix** with the exact file:line and corrected code/config.
5. **Apply the fix** (if authorized by the user) or document it in the Fix Plan.
6. **Re-run the exact failing check** to prove the fix worked. Capture the new output.
7. **If the fix passes → resume the audit from the phase where you stopped.**
8. **If the fix fails → escalate to `REQUIRES_HUMAN_DECISION`.** Do NOT iterate more than twice on the same failure without human input.

**HIGH failures (doc materially wrong, config sync broken, build chain stale):**
1. **Log in the Issue Register** with full evidence.
2. **Continue the audit** — HIGH failures don't invalidate later phases.
3. **After the audit completes,** apply fixes in dependency order (Section 4).
4. **Re-run the specific checks** that failed to verify each fix.

**MEDIUM/LOW failures:**
1. Log in the Issue Register.
2. Continue the audit.
3. Batch-fix after audit completes.

### The Re-Run Rule
Every fix in Section 4 (Fix Plan) MUST include a `Verification command`. After applying the fix:
```
1. Run the verification command
2. Capture actual output
3. PASS → mark as RESOLVED in Issue Register
4. FAIL → the fix was wrong. Do NOT mark as resolved. Investigate further or escalate.
```

**Never claim a fix worked without re-running the check.** "I changed the code so it should pass now" is NOT verification.
