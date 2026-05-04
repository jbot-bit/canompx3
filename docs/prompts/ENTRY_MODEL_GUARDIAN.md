# Entry Model Code Guardian — 2-Pass Self-Audit

**Posture:** Guilty until proven innocent. Every claim in documentation is WRONG until the code proves otherwise. Every strategy in the database is INVALID until its existence is justified by traceable logic.

---

## PASS 1 — Discovery (Define the Audit Surface)

You don't know what entry models exist. You don't know what CB levels are real. You don't know what ORB configurations are valid. Discover everything from scratch.

### 1A. Code Excavation
Scan the entire codebase for entry model definitions. Do NOT trust documentation — read source files only.

```
Files to inspect (non-exhaustive — grep for more):
- trading_app/outcome_builder.py → how entries are triggered
- trading_app/strategy_discovery.py → what models get grid-searched
- trading_app/strategy_validator.py → what models get validated
- pipeline/check_drift.py → what drift checks reference which models
- config.py → enum definitions, thresholds, valid combinations
- trading_app/mcp_server.py → what the MCP layer exposes
- TRADING_RULES.md → what docs CLAIM
- CANONICAL_*.txt → what frozen specs CLAIM
```

For EACH entry model variant found (E0, E1, E2, E3, or anything else):
- [ ] Record exact file path + line number where it's defined
- [ ] Record exact file path + line number where it's consumed
- [ ] Record the trigger logic (what causes an entry)
- [ ] Record CB interaction (which close bar levels are valid for this model)
- [ ] Record ORB minute interaction (5, 15, 30 — which combos are valid)
- [ ] Record any gating logic (filters, conditions, guards)

### 1B. Database Excavation
Using MCP `query_trading_db`, enumerate what actually EXISTS in the database:

```
Queries to run:
1. template="schema_info" → get actual column definitions
2. template="table_counts" → row counts per table
3. Strategy counts grouped by entry_model + confirm_bars + orb_minutes
   (use template="validated_summary" or appropriate template)
4. template="outcomes_stats" → what entry models have computed outcomes
5. Orphan check: strategies in validated_setups with no matching outcomes
```

For EACH entry_model + CB + orb_minutes combination found in data:
- [ ] Record row count
- [ ] Record whether code can produce this combination (trace the code path)
- [ ] Record whether documentation acknowledges this combination
- [ ] Flag any combination that EXISTS in data but has NO code path → `ZOMBIE`
- [ ] Flag any combination that code CAN produce but docs DON'T mention → `UNDOCUMENTED`
- [ ] Flag any combination that docs MENTION but code CANNOT produce → `PHANTOM`

### 1C. Drift Check Inventory
Read `pipeline/check_drift.py` line by line. For EACH drift check:
- [ ] Record check number, description, and what it validates
- [ ] Record which entry models / CB levels / ORB configs it references
- [ ] Record the purge/remediation action it claims to take
- [ ] Record whether the purge logic is actually implemented (trace the code)
- [ ] Flag checks that CLAIM to purge but don't have executable purge code → `TOOTHLESS`
- [ ] Flag checks that reference entry models not found in 1A → `STALE_CHECK`

---

## PASS 2 — Prosecution (Prove or Convict)

For every item discovered in Pass 1, demand proof. No benefit of the doubt.

### 2A. Entry Model Trial
For EACH entry model:

**Charge 1: Identity Fraud**
- Does the code definition match the documentation definition EXACTLY?
- Compare: trigger conditions, valid CB levels, valid ORB minutes, filter interactions
- Verdict: `MATCH` / `DOC_WRONG` / `CODE_WRONG` / `BOTH_WRONG`

**Charge 2: Unauthorized Reproduction**
- Can this model produce strategies that should not exist?
- Cross-reference: code's valid combinations vs database contents
- Are there strategies in the DB that this model's code path cannot produce?
- Verdict: `CLEAN` / `ZOMBIE_STRATEGIES_FOUND` (list them)

**Charge 3: Dereliction of Duty**
- Are there drift checks that should catch invalid strategies for this model?
- Do those drift checks actually execute? (trace the code path)
- Run the drift check logic manually against current DB state
- Are there strategies that SHOULD have been caught but weren't?
- Verdict: `ENFORCED` / `DRIFT_CHECK_FAILED` / `NO_DRIFT_CHECK_EXISTS`

### 2A-PARITY. Feature Parity Trial — Backtest vs Live Path

> **Cross-reference:** SYSTEM_AUDIT.md Phase 7D performs the same check at system level. If running the full system audit, Phase 7D covers this — run this pass only for entry-model-specific deep dives or when escalated from Phase 7D.

**This is the train/inference skew check.** The system has two separate paths that must agree:
- Backtest: `outcome_builder.py` → `entry_rules.py` (detect_break_touch, detect_entry_with_confirm_bars)
- Live: `paper_trader.py` → `execution_engine.py`

For EACH entry model:

**Charge: Implementation Divergence**
- Trace the EXACT entry trigger in `outcome_builder.py` for this model (file:line, function call, parameters)
- Trace the EXACT entry trigger in `execution_engine.py` for this model (file:line, function call, parameters)
- Do both use the same function from `entry_rules.py`? If execution_engine has its OWN implementation → `PARITY_VIOLATION` (two implementations will inevitably diverge)
- Compare: ORB level comparison (>=, >), bar field selection (high/low/close), direction logic, CB confirmation count and logic
- Compare: early exit / kill timer application — are the same time limits applied to outcomes as to live trades?
- Verdict: `IDENTICAL_PATH` (both call same function) / `EQUIVALENT_LOGIC` (different code, verified identical behavior) / `DIVERGENT` (different behavior — CRITICAL)

If `DIVERGENT`: identify which path is authoritative (usually outcome_builder, since that's what the edge was measured on) and flag the other for correction.

### 2B. Close Bar Confirmation Trial
For EACH CB level (CB1, CB2, CB3, etc.):

**Charge 1: Definition Drift**
- What does the code say constitutes a valid close bar at this level?
- What does documentation say?
- Do they match?
- Verdict: `MATCH` / `DRIFT` (specify direction)

**Charge 2: Interaction Integrity**
- How does this CB level interact with each entry model?
- Is every interaction documented?
- Is every documented interaction actually implemented?
- Verdict per combination: `VERIFIED` / `UNDOCUMENTED` / `PHANTOM`

### 2C. Strategy Lifecycle Trial
For EACH entry_model + CB + orb_minutes combination:

**Charge 1: Birth Certificate**
- Can you trace the EXACT code path that creates this strategy?
- File, function, line number for creation
- Verdict: `TRACEABLE` / `UNTRACEABLE`

**Charge 2: Survival Legitimacy**
- Should this strategy still exist given current drift checks and purge rules?
- Run purge logic against it — does it survive or should it have been killed?
- Verdict: `LEGITIMATE` / `SHOULD_BE_PURGED` (cite which check should have caught it)

**Charge 3: Classification Accuracy**
- Is it classified correctly (CORE/REGIME/INVALID) per config.py thresholds?
- Sample count vs threshold
- Verdict: `CORRECT` / `MISCLASSIFIED` (state actual vs expected)

---

## Output Format

### Section 1: Audit Surface Map
Table of every entry_model + CB + orb_minutes combination found across code, docs, and data with presence flags.

| Combo | In Code | In Docs | In DB | Row Count | Status |
|-------|---------|---------|-------|-----------|--------|

### Section 2: Conviction Report
Every discrepancy found, ordered by severity.

| ID | Category | Severity | What Docs Claim | What Code Does | What Data Shows | File:Line | Corrective Action |
|----|----------|----------|-----------------|----------------|-----------------|-----------|-------------------|

Severity scale:
- **CRITICAL**: Active strategies exist that shouldn't, or purge logic is broken
- **HIGH**: Documentation materially misrepresents code behavior
- **MEDIUM**: Undocumented but harmless combinations exist
- **LOW**: Cosmetic doc/code mismatch

> **Tag reference:** See SYSTEM_AUDIT.md "Consolidated Tag Vocabulary" for the canonical list of all verdict/severity tags used across all 3 audit docs.

### Section 3: Corrective Actions
For each conviction, provide:
- Exact file path + line numbers
- Current text/code (quoted)
- Corrected version
- Whether this is CODE fix or DOC fix or PURGE needed
- Verification query to confirm fix worked

### Section 4: Verification Queries
MCP queries or DuckDB SQL to independently verify every finding. Another operator should be able to run these queries and reach the same conclusions.

---

## Rules of Engagement
1. Read `TRADING_RULES.md` and `config.py` FIRST — but treat them as CLAIMS, not truth
2. Use MCP tools (`query_trading_db`, `get_strategy_fitness`, `get_canonical_context`) — never raw SQL when a template exists
3. When code and docs disagree, CODE is the current truth (but may itself be buggy)
4. When code and data disagree, DATA reveals what actually happened — investigate why
5. Flag `IMPLEMENTATION UNCLEAR — REQUIRES HUMAN DECISION` when intent is ambiguous
6. NEVER assume a purge worked — verify by querying what survived
7. NEVER assume documentation was written from code inspection — assume it was hallucinated until you verify each claim against source
8. **Mechanical execution only** — "I read the code and it looks correct" is NOT a passing verdict. Run the command, capture actual stdout, cite the output. If you cannot execute, mark `SKIPPED — NO EXECUTION ENVIRONMENT`, never `PASS`.
9. **NEVER mark a check as PASS without stdout proof.** Reading source code is discovery, not verification. A drift check might pass in code logic but fail against current data.
10. **Anti-hallucination:** If you find yourself writing "this likely passes" or "based on the code, this should work" — STOP. That is a `HALLUCINATED_PASS`. Execute or mark SKIPPED.
