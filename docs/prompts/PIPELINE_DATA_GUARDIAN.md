# Pipeline Data Code Guardian — 2-Pass Self-Audit

**Posture:** Guilty until proven innocent. Every row in the database is SUSPECT until its provenance is traced from raw source through every transformation. Every aggregation is WRONG until independently verified. Every join is LEAKING until proven clean.

---

## PASS 1 — Discovery (Map the Data Surface)

You don't know what's in the database. You don't know if the pipeline produced it correctly. You don't know if counts make sense. Discover everything from first principles.

### 1A. Schema Excavation
Using MCP and code inspection, build a complete map of the data pipeline.

```
Files to inspect:
- pipeline/init_db.py → schema definitions, constraints, indexes
- pipeline/ingest_dbn.py → raw data ingestion, validation gates
- pipeline/build_bars_5m.py → 1m → 5m aggregation logic
- pipeline/build_daily_features.py → feature computation, ORB calc, session windows
- pipeline/dst.py → SESSION_CATALOG, dynamic session resolution
- pipeline/paths.py → DB path resolution
- pipeline/check_drift.py → drift checks (self-reports count at runtime)
- pipeline/health_check.py → health check logic
- trading_app/outcome_builder.py → outcome computation from features
- config.py → all thresholds, enums, valid values
```

For EACH table in the database:
- [ ] Record schema from `init_db.py` (CREATE TABLE statement)
- [ ] Record actual schema from DB (use MCP `template="schema_info"`)
- [ ] Compare — flag any mismatch → `SCHEMA_DRIFT`
- [ ] Record row count (use MCP `template="table_counts"`)
- [ ] Record which pipeline step writes to this table
- [ ] Record all indexes and constraints
- [ ] Record the DELETE+INSERT pattern used (idempotency check)

### 1B. Data Lineage Trace
For EACH table, trace the full lineage:

**bars_1m:**
- [ ] Source: Databento .dbn.zst files
- [ ] Writer: `ingest_dbn.py`
- [ ] 7 ingestion validation gates — list each one, verify it exists in code
- [ ] Symbol mapping: which symbols map to which source contracts? (GC→MGC, ES→MES, RTY→M2K, etc.)
- [ ] Does `source_symbol` column correctly record the actual contract?
- [ ] Date range coverage per instrument — any gaps?

**bars_5m:**
- [ ] Source: bars_1m
- [ ] Writer: `build_bars_5m.py`
- [ ] 4 aggregation validation gates — list each one, verify in code
- [ ] Aggregation logic: OHLCV rules (first open, max high, min low, last close, sum volume)
- [ ] Verify: `COUNT(bars_5m) * 5 ≈ COUNT(bars_1m)` per instrument per day (allowing for remainder bars)
- [ ] Any 5m bars with zero volume? Any with high < low? → `DATA_INTEGRITY_VIOLATION`

**daily_features:**
- [ ] Source: bars_1m + bars_5m
- [ ] Writer: `build_daily_features.py`
- [ ] Triple-row structure: 3 rows per (trading_day, symbol) — one per orb_minutes (5, 15, 30)
- [ ] Verify exact row count: `COUNT(daily_features) = COUNT(DISTINCT trading_day, symbol) * 3`
- [ ] If not exactly 3x → `TRIPLE_JOIN_TRAP_RISK`
- [ ] Session window logic — verify against `dst.py` SESSION_CATALOG
- [ ] ORB computation — verify high/low calculation matches code
- [ ] RSI/ATR computation — verify formula matches code
- [ ] Look-ahead columns: `double_break` is look-ahead — verify it's not used as entry filter anywhere

**orb_outcomes:**
- [ ] Source: daily_features + bars_1m
- [ ] Writer: `outcome_builder.py`
- [ ] Pre-computed outcomes table (count changes with rebuilds)
- [ ] Contains ALL break-days regardless of filter
- [ ] Verify: outcomes exist only for days where a break occurred
- [ ] Row count sanity: outcomes per instrument per orb_minutes — are counts plausible?

**validated_setups / experimental_strategies / edge_families:**
- [ ] Source: orb_outcomes + daily_features
- [ ] Writers: strategy_discovery.py, strategy_validator.py, build_edge_families.py
- [ ] Every validated strategy must have a traceable path from outcomes

### 1C. Instrument Coverage Audit
For EACH instrument (MGC, MNQ, MES, M2K, MCL, SIL, M6E):
- [ ] Date range in bars_1m
- [ ] Date range in bars_5m (should match)
- [ ] Date range in daily_features (should match)
- [ ] Date range in orb_outcomes
- [ ] Any gaps in trading day coverage? (missing days that should exist)
- [ ] Any extra days? (weekends, holidays that shouldn't have data)
- [ ] Active vs dead classification — does code treat dead instruments correctly?
- [ ] Source contract mapping — verify stored symbol vs source_symbol

### 1D. Time & Calendar Audit
- [ ] All DB timestamps are UTC — verify by sampling bars and checking against known market hours
- [ ] Trading day boundary: 09:00 Brisbane (23:00 UTC previous day) — verify assignment logic
- [ ] Bars before 09:00 local assigned to PREVIOUS trading day — verify in code AND data
- [ ] DST handling: session times resolved dynamically from `dst.py` — verify no fixed-clock remnants
- [ ] DOW alignment: Brisbane DOW = exchange DOW for all sessions except NYSE_OPEN — verify guard exists

---

## PASS 2 — Prosecution (Prove or Convict)

### 2A. Ingestion Integrity Trial

**Charge 1: Gate Failure**
For each of the 7 ingestion validation gates:
- Does the gate actually abort on failure? (trace the code — `raise` / `sys.exit` / `return`)
- Or does it just log a warning and continue? → `TOOTHLESS_GATE`
- Verdict per gate: `FAIL_CLOSED` / `FAIL_OPEN` / `GATE_MISSING`

**Charge 2: Symbol Contamination**
- Are there any rows in bars_1m where `symbol` doesn't match expected instruments?
- Are there rows where `source_symbol` is NULL or doesn't match the mapping table in CLAUDE.md?
- Verdict: `CLEAN` / `CONTAMINATED` (list offending rows)

**Charge 3: Duplicate Injection**
- Are there duplicate (timestamp, symbol) pairs in bars_1m?
- Does the idempotent DELETE+INSERT pattern actually prevent this?
- Verify: `SELECT timestamp, symbol, COUNT(*) FROM bars_1m GROUP BY 1,2 HAVING COUNT(*) > 1`
- Verdict: `NO_DUPLICATES` / `DUPLICATES_FOUND` (count and sample)

**Charge 4: Write Scope Integrity (Downstream Tables)**
For each table written by the trading app layer (validated_setups, edge_families, experimental_strategies):
- Trace the DELETE scope in the writer function — what WHERE clause does it use?
- Trace the INSERT scope — what rows are actually being written back?
- If DELETE is broader than INSERT → `DATA_LOSS_RISK` (data silently lost)
- Specific known vector: strategy_validator.py batch write DELETEs validated_setups.
  Must scope to `processed_orb_minutes` only, not entire instrument.
  (Feb 2026: instrument-wide DELETE with partial-aperture INSERT wiped 627 5m strategies)
- Also check: build_edge_families.py, outcome_builder.py --force, parallel_rebuild.py
- Verdict per table: `SCOPED` / `OVER_DELETE` / `UNDER_DELETE`

### 2B. Aggregation Integrity Trial

**Charge 1: 5m Bar Correctness**
Sample 10 random trading days across different instruments. For each:
- Manually aggregate bars_1m → 5m using the documented rules
- Compare against actual bars_5m rows
- Any mismatch → `AGGREGATION_ERROR`
- Verdict: `VERIFIED` / `ERRORS_FOUND`

**Charge 2: Daily Feature Correctness**
Sample 5 random trading days. For each:
- Recompute ORB high/low from bars_1m for each orb_minutes window
- Compare against daily_features
- Check session assignment against dst.py SESSION_CATALOG for that date
- Any mismatch → `FEATURE_ERROR`
- Verdict: `VERIFIED` / `ERRORS_FOUND`

**Charge 3: Outcome Correctness**
Sample 5 random outcomes. For each:
- Trace back to the bar data that produced the outcome
- Verify entry price, target, stop, result
- Check CB confirmation logic against actual bar closes
- Any mismatch → `OUTCOME_ERROR`
- Verdict: `VERIFIED` / `ERRORS_FOUND`

### 2C. Referential Integrity Trial

**Charge 1: Orphan Detection**
- Outcomes without matching daily_features rows
- Validated strategies without matching outcomes
- Edge families without matching validated strategies
- Daily features without matching bars
- Verdict per relationship: `CLEAN` / `ORPHANS_FOUND` (count)

**Charge 2: Join Safety**
- Is the triple-join (trading_day + symbol + orb_minutes) enforced everywhere daily_features is joined?
- Grep the entire codebase for joins on daily_features
- Flag any join missing orb_minutes → `TRIPLE_JOIN_VIOLATION`
- Verdict: `ALL_SAFE` / `VIOLATIONS_FOUND` (list file:line)

**Charge 3: Filter Leakage**
- `orb_outcomes` contains ALL break-days regardless of filter
- Verify: no code path queries orb_outcomes without applying filter_type from daily_features
- Check strategy_discovery.py, strategy_validator.py, outcome_builder.py
- Any direct orb_outcomes query without filter join → `FILTER_LEAK`
- Verdict: `CONTAINED` / `LEAKING` (list locations)

### 2D. Drift Check Completeness Trial

**Charge 1: Coverage Gaps**
- List all drift checks from check_drift.py (run it to get current count)
- For each data integrity issue type (duplicates, gaps, orphans, schema drift, count anomalies):
  - Is there a drift check that catches it?
  - If not → `UNCOVERED_RISK`
- Verdict: `FULL_COVERAGE` / `GAPS_FOUND` (list uncovered risks)

**Charge 2: Drift Check Accuracy**
- Run `python pipeline/check_drift.py` and capture output
- For each check that PASSES: independently verify the claim with a direct query
- For each check that FAILS: is the failure real or is the check buggy?
- Verdict per check: `ACCURATE` / `FALSE_POSITIVE` / `FALSE_NEGATIVE`

### 2E. Count Sanity Trial

**Charge: Implausible Numbers**
For each instrument, verify these ratios make sense:
- bars_1m count vs expected (trading days × ~1440 minutes, minus closed hours)
- bars_5m count vs bars_1m count (should be ~1/5)
- daily_features count vs trading days × 3
- orb_outcomes count vs daily_features break-days
- validated_setups count vs experimental_strategies count (validation is reductive)
- Any ratio wildly off → `COUNT_ANOMALY`

---

## Output Format

### Section 1: Data Surface Map

| Table | Schema Match | Row Count | Writer | Lineage Verified | Issues |
|-------|-------------|-----------|--------|-----------------|--------|

### Section 2: Instrument Coverage

| Instrument | bars_1m Range | bars_5m Range | daily_features Range | Gaps Found | Source Mapping Verified |
|-----------|--------------|--------------|---------------------|------------|----------------------|

### Section 3: Conviction Report

| ID | Category | Severity | Expected | Actual | Evidence | File:Line | Fix Type |
|----|----------|----------|----------|--------|----------|-----------|----------|

Severity scale:
- **CRITICAL**: Data corruption, missing validation gates, filter leakage, join violations, **write scope mismatch (DATA_LOSS_RISK)**
- **HIGH**: Orphaned records, count anomalies, schema drift
- **MEDIUM**: Undocumented behavior, missing drift checks
- **LOW**: Cosmetic issues, non-blocking warnings

> **Tag reference:** See SYSTEM_AUDIT.md "Consolidated Tag Vocabulary" for the canonical list of all verdict/severity tags used across all 3 audit docs.

### Section 4: Corrective Actions
For each conviction:
- Exact file path + line numbers
- Current code/config (quoted)
- Corrected version
- Whether CODE fix, DATA fix, or SCHEMA fix
- Verification query to confirm fix

### Section 5: Verification Queries
Independent queries to reproduce every finding. Another operator must be able to run these and reach the same conclusions.

### Section 6: Health Score
```
Pipeline Health: X/10
  Ingestion Gates:    [PASS/FAIL] (N of 7 verified)
  Aggregation:        [PASS/FAIL] (spot-check results)
  Feature Computation: [PASS/FAIL] (spot-check results)
  Referential Integrity: [PASS/FAIL] (orphan check)
  Join Safety:        [PASS/FAIL] (triple-join audit)
  Filter Containment: [PASS/FAIL] (filter leak check)
  Drift Coverage:     [PASS/FAIL] (N of total verified — run check_drift.py for count)
  Count Sanity:       [PASS/FAIL] (ratio checks)
  Time/Calendar:      [PASS/FAIL] (UTC + trading day)
  Schema Alignment:   [PASS/FAIL] (init_db vs actual)
```

---

## Rules of Engagement
1. Use MCP tools (`query_trading_db`, `get_canonical_context`) for all DB queries — never raw SQL when a template exists
2. For queries no template covers, write DuckDB-compatible SQL and document it for future template creation
3. When code and data disagree, DATA reveals what actually happened — the code may have been correct at write time but broken by a later change
4. Spot-checks must use RANDOM sampling (not cherry-picked examples). **Seed enforcement:** use `ORDER BY hash(trading_day || symbol) LIMIT N` or `USING SAMPLE N ROWS (REPEATABLE (42))` in DuckDB — never `ORDER BY RANDOM()` which is non-reproducible. Record the seed/method used so another auditor can reproduce the same sample.
5. Every claim must cite a file:line or query result — no "it appears that" or "it seems like"
6. Flag `IMPLEMENTATION UNCLEAR — REQUIRES HUMAN DECISION` when intent is genuinely ambiguous
7. NEVER trust row counts from documentation — query the actual database
8. NEVER trust schema descriptions from documentation — query the actual schema
9. NEVER assume idempotency works — verify with duplicate checks
10. Treat every pipeline step as potentially broken until you've traced input → transformation → output
11. **Mechanical execution only** — "I read the code and it looks correct" is NOT a passing verdict. Run the command, capture actual stdout, cite the output. If you cannot execute, mark `SKIPPED — NO EXECUTION ENVIRONMENT`, never `PASS`.
12. **NEVER mark a check as PASS without stdout proof.** Reading source code is discovery, not verification. A pipeline step might look correct in code but produce wrong data.
13. **Anti-hallucination:** If you find yourself writing "this likely passes" or "based on the code, this should work" — STOP. That is a `HALLUCINATED_PASS`. Execute or mark SKIPPED.

## Experimental Isolation — Data Snooping Quarantine

**This section prevents the most dangerous research sin: turning your test set into a training set.**

When auditing or analyzing pipeline data, the agent MUST respect the boundary between in-sample (training/discovery) and out-of-sample (validation/walk-forward) data. Violating this boundary silently destroys the statistical validity of every strategy in the system.

**Hard rules:**
1. **NEVER summarize, extract features from, or optimize against data designated as out-of-sample or walk-forward holdout.** If `strategy_validator.py` splits data into windows for walk-forward testing, those holdout windows are sacred — do not read their contents to inform upstream decisions.
2. **NEVER use validation results to retroactively tune discovery parameters.** If a strategy fails walk-forward, the correct response is to investigate the mechanism — NOT to adjust `config.py` thresholds until it passes.
3. **When spot-checking outcomes (Pass 2A/2B/2C), sample from the FULL date range.** But if findings suggest parameter changes, those changes must be re-validated on data the agent has NOT already examined during this audit.
4. **Flag any code path that allows discovery-phase logic to peek at validation-phase data** → `DATA_LEAK_RISK`. Example: if `strategy_discovery.py` can access walk-forward window boundaries from `strategy_validator.py`, that's a structural leak even if currently unused.
5. **The agent performing this audit is itself a potential data snooping vector.** If you read outcome distributions, win rates, or P&L curves during the audit, you now carry implicit knowledge of the test set. Document in Section 6 (Health Score) what data ranges you inspected, so the human researcher knows which windows may need fresh out-of-sample validation.

**Escalation:** If this audit discovers a structural data leak, escalate to `CRITICAL` severity and recommend re-running walk-forward validation from scratch on untainted data.
