# Custom Skills Suite — Implementation Plan

> **STALE WARNING (2026-03-09):** SQL examples in this plan use wrong column/table names
> (`strategy_fitness`, `v.symbol`, `avg_r`, `v.sharpe`). The actual skills in `.claude/commands/`
> have been corrected. Do NOT copy SQL from this plan — use the live skill files as source of truth.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Replace generic superpowers skills with 9 quant-pipeline-aware custom skills + fix 3 stale commands. All skills auto-trigger semantically so the user never needs to remember slash commands.

**Architecture:** Skills live in `.claude/commands/*.md`. Each skill has a description line that Claude Code uses for semantic matching. Skills encode project-specific invariants (seven sins, triple-join trap, canonical sources, BH FDR, fail-closed patterns) so every workflow is pipeline-aware.

**Tech Stack:** Markdown skill files in `.claude/commands/`, no code dependencies.

---

### Task 0: Write 4t.md skill

**Files:**
- Create: `.claude/commands/4t.md`

**Step 1: Write the skill file**

Write `.claude/commands/4t.md` with this exact content:

```markdown
Orient, design, detail, and validate a feature or change: $ARGUMENTS

Use when: user says "4t", "plan this", "design this", "how should we build", "feature planning", "approach this", "think through"

## The 4-Turn Flow

You are running the 4T feature planning workflow. Execute all 4 turns in order. Do NOT skip turns.

### Turn 1: ORIENT

Understand the landscape before proposing anything.

1. Parse $ARGUMENTS for the topic/feature
2. Read ALL files that could be affected — trace imports, check callers, map blast radius
3. Check `docs/specs/` for an existing spec (if one exists, follow it — do NOT redesign)
4. Check authority docs: CLAUDE.md for code decisions, TRADING_RULES.md for trading logic, RESEARCH_RULES.md for research methodology
5. Check canonical sources — will this change touch any?
   - Active instruments: `pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS`
   - Sessions: `pipeline.dst.SESSION_CATALOG`
   - Entry models/filters: `trading_app.config`
   - Cost specs: `pipeline.cost_model.COST_SPECS`
   - DB path: `pipeline.paths.GOLD_DB_PATH`
6. Articulate PURPOSE: why this matters, what breaks without it

Output: "ORIENT COMPLETE" + summary of affected files, blast radius, and purpose.

### Turn 2: DESIGN

Propose the solution architecture.

1. Data model: what tables/columns/fields change?
2. Interfaces: what functions/classes/APIs are added or modified?
3. Data flow: how does data move through the change?
4. Where it fits: which layer? (pipeline/ vs trading_app/ vs scripts/ vs ui/)
5. One-way dependency check: does this violate pipeline -> trading_app direction?
6. Propose 2-3 approaches with trade-offs if the design isn't obvious
7. State your recommendation and why

Output: Architecture diagram or description + recommendation.

### Turn 3: DETAIL

Ordered implementation steps specific enough to execute blindly.

1. List every file to create, modify, or delete with exact paths
2. For each file: what changes, in what order
3. Test strategy: what tests to write, what they verify
4. Migration: any data rebuild needed? Which instruments? Which apertures?
5. Drift checks: any new checks needed? Any existing checks affected?

Output: Numbered step list a junior dev could follow.

### Turn 4: VALIDATE

Risk assessment and verification plan.

1. What could go wrong? List failure modes.
2. What tests prove correctness? (not just "it runs" — what BEHAVIOR is verified?)
3. Drift checks: run `python pipeline/check_drift.py` mentally — will it still pass?
4. Rebuild requirements: does this need outcome_builder -> discovery -> validation chain?
5. Rollback plan: how do we undo this if it's wrong?
6. Guardian prompts: does this touch entry models (ENTRY_MODEL_GUARDIAN) or pipeline data (PIPELINE_DATA_GUARDIAN)?

Output: Risk table + verification checklist.

## After All 4 Turns

Save the design to `docs/plans/YYYY-MM-DD-<topic>-design.md` and present it to the user for approval.

## Rules

- ONE topic at a time. Never batch.
- If a spec exists in `docs/specs/`, follow it — do not redesign.
- NEVER skip ORIENT. Reading code is not optional.
- NEVER propose changes to files you haven't read.
- Apply YAGNI ruthlessly — remove unnecessary features from designs.
```

**Step 2: Verify the file was created**

Run: `cat .claude/commands/4t.md | head -5`
Expected: First 5 lines of the skill file.

---

### Task 1: Write 4tp.md skill

**Files:**
- Create: `.claude/commands/4tp.md`

**Step 1: Write the skill file**

Write `.claude/commands/4tp.md` with this exact content:

```markdown
Plan, design, and proceed to implementation for: $ARGUMENTS

Use when: user says "4tp", "plan and build", "design and implement", "full pipeline", "just do it and plan"

## 4TP = 4T + Proceed

This is the full-pipeline version of /4t. Run all 4 turns (orient, design, detail, validate), then AUTOMATICALLY:

### Phase 1: Run /4t

Execute the full 4T flow for $ARGUMENTS. All 4 turns, no shortcuts.

### Phase 2: Auto-Proceed (NO PAUSE)

After Turn 4 completes:

1. **Write design doc** to `docs/plans/YYYY-MM-DD-<topic>-design.md`
2. **Commit it**: `git add docs/plans/<file> && git commit -m "docs: 4TP design — <topic>"`
3. **Invoke writing-plans skill** to create the implementation plan from the design

Do NOT pause for approval between Phase 1 and Phase 2. The whole point of 4TP is zero stops.

### Rules

- Same rules as /4t apply (ORIENT is mandatory, read before proposing, YAGNI)
- If the design reveals the task is trivial (< 3 steps), skip the writing-plans skill and just do it
- If the design reveals the task is dangerous (schema change, entry model change), STOP and ask — override the "no stops" rule for safety
```

---

### Task 2: Write quant-debug.md skill

**Files:**
- Create: `.claude/commands/quant-debug.md`

**Step 1: Write the skill file**

Write `.claude/commands/quant-debug.md` with this exact content:

```markdown
Diagnose and fix a bug or unexpected behavior in the quant pipeline: $ARGUMENTS

Use when: ANY bug, test failure, unexpected behavior, "why is this broken", "wrong numbers", "row count off", "test failing", "unexpected results", "data looks wrong", error in pipeline/ or trading_app/

## Quant-Aware Debugging Protocol

You are debugging a quantitative trading pipeline. Follow this protocol EXACTLY. Do NOT guess at fixes.

### Step 0: Reproduce

Before anything else, reproduce the problem:
1. Run the failing command/test and capture the FULL output
2. Note the exact error message, line number, and file
3. If "wrong numbers" — get the ACTUAL numbers and the EXPECTED numbers

"Reading code is NOT verifying code." You must EXECUTE and read output.

### Step 1: Classify the Bug

Which category? This determines your checklist:

**A) Row Count / Data Inflation**
- [ ] Triple-join trap? `daily_features` has 3 rows per (trading_day, symbol) — one per orb_minutes. Missing `AND o.orb_minutes = d.orb_minutes` triples rows.
- [ ] Aperture mismatch? Comparing O5 data to O15/O30 without filtering.
- [ ] Filter not applied? Querying `orb_outcomes` without joining `daily_features` for filter eligibility.

**B) Wrong Values / Shifted Data**
- [ ] Timezone double-conversion? DuckDB `fetchdf()` returns `datetime64[us, Australia/Brisbane]` (already local). Adding `pd.Timedelta(hours=10)` double-converts. Use `.dt.tz_localize(None)` instead.
- [ ] DST contamination? Using hardcoded session times instead of `SESSION_CATALOG` from `pipeline/dst.py`.
- [ ] Trading day assignment? Bars before 09:00 Brisbane should be assigned to PREVIOUS trading day.
- [ ] Look-ahead leak? `double_break` is look-ahead. LAG() without `WHERE orb_minutes = 5` causes cross-aperture contamination.

**C) Missing Data / Empty Results**
- [ ] DuckDB replacement scan? DuckDB can reference in-scope pandas DataFrames in SQL — this is NOT a bug.
- [ ] Wrong instrument symbol? Check `ASSET_CONFIGS` — some instruments use full-size source contracts (GC for MGC, ES for MES, RTY for M2K).
- [ ] Filter too strict? G6/G8 filters producing low N is EXPECTED, not a bug.
- [ ] Date range gap? Check if daily_features and outcomes exist for the queried date range.

**D) Test Failure**
- [ ] Check `TEST_MAP` in `.claude/hooks/post-edit-pipeline.py` for the companion test file
- [ ] Is the test testing current behavior or stale behavior?
- [ ] Did a canonical source change? (session catalog, cost specs, asset configs)
- [ ] Is the test doing timezone math? Check for the `pd.Timedelta(hours=10)` anti-pattern.

**E) Pipeline / Build Failure**
- [ ] One-way dependency violated? pipeline/ -> trading_app/ only. Never reversed.
- [ ] Concurrent DuckDB writes? NEVER run two write processes against the same DB file.
- [ ] Fail-closed check? Is an exception being caught and returning success?
- [ ] Exit code checked? Zero is the only success.

### Step 2: Trace the Execution Path

1. Read the code at the actual error location (not from memory)
2. Trace imports — follow the call chain to find the root cause
3. Check for existing guards (try/except, if-checks, assertions) that should have caught this
4. Cross-reference canonical sources — is something hardcoded that should be imported?

### Step 3: Verify the Fix

After implementing a fix:
1. Run the original failing command — it must now succeed
2. Run targeted tests: `python -m pytest <test_file> -x -v`
3. Run drift check: `python pipeline/check_drift.py`
4. If the fix touches pipeline/ or trading_app/: run full test suite `python -m pytest tests/ -x -q`

### Step 4: Check Blast Radius

- Did the fix change any canonical source? If yes, check all importers.
- Did the fix change a JOIN? Verify row counts before and after.
- Did the fix change a calculation? Run `python scripts/tools/audit_integrity.py`.

### NEVER Do This

- Guess at a fix without reproducing the bug first
- "Fix" low trade counts by loosening filters (low N under strict filters is expected)
- Assume `double_break` can be used as a filter (it's look-ahead)
- Trust line numbers from error messages without reading the actual code
- Claim "fixed" without running the test and showing the output
```

---

### Task 3: Write quant-tdd.md skill

**Files:**
- Create: `.claude/commands/quant-tdd.md`

**Step 1: Write the skill file**

Write `.claude/commands/quant-tdd.md` with this exact content:

```markdown
Write tests first, then implement for: $ARGUMENTS

Use when: "write tests", "test first", "add tests for", "TDD", implementing any feature or bugfix in pipeline/ or trading_app/

## Quant TDD Protocol

Test-driven development for a quantitative trading pipeline. Tests come FIRST. Implementation follows.

### Step 1: Find the Right Test File

Check `TEST_MAP` in `.claude/hooks/post-edit-pipeline.py`:
```
pipeline/build_daily_features.py  -> tests/test_pipeline/test_build_daily_features.py
pipeline/build_bars_5m.py         -> tests/test_pipeline/test_build_bars_5m.py
pipeline/ingest_dbn.py            -> tests/test_pipeline/test_ingest.py
pipeline/check_drift.py           -> tests/test_pipeline/test_check_drift.py
pipeline/dst.py                   -> tests/test_pipeline/test_dst.py
pipeline/init_db.py               -> tests/test_pipeline/test_schema.py
pipeline/asset_configs.py         -> tests/test_pipeline/test_asset_configs.py
trading_app/outcome_builder.py    -> tests/test_trading_app/test_outcome_builder.py
trading_app/strategy_discovery.py -> tests/test_trading_app/test_strategy_discovery.py
trading_app/strategy_validator.py -> tests/test_trading_app/test_strategy_validator.py
trading_app/entry_rules.py        -> tests/test_trading_app/test_entry_rules.py
trading_app/paper_trader.py       -> tests/test_trading_app/test_paper_trader.py
trading_app/config.py             -> tests/test_trading_app/test_config.py
```

If no mapping exists, create the test file following the pattern: `tests/test_<module>/test_<filename>.py`

### Step 2: Read Existing Tests

Before writing new tests, read the existing test file to understand:
- Test style and patterns used
- Fixtures available (especially DuckDB in-memory fixtures)
- What's already tested (don't duplicate)

### Step 3: Write the Failing Test

Quant pipeline test patterns:

**Tripwire tests** (verify invariants hold):
```python
def test_cost_model_covers_all_active_instruments():
    """Every active instrument must have a cost spec."""
    from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
    from pipeline.cost_model import COST_SPECS
    for inst in ACTIVE_ORB_INSTRUMENTS:
        assert inst in COST_SPECS, f"Missing cost spec for {inst}"
```

**JOIN correctness tests** (verify no row inflation):
```python
def test_join_does_not_inflate_rows(con):
    """Join on all 3 keys: trading_day, symbol, orb_minutes."""
    before = con.execute("SELECT COUNT(*) FROM orb_outcomes WHERE symbol='MGC'").fetchone()[0]
    after = con.execute("""
        SELECT COUNT(*) FROM orb_outcomes o
        JOIN daily_features d ON o.trading_day = d.trading_day
          AND o.symbol = d.symbol AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol='MGC'
    """).fetchone()[0]
    assert after == before, f"JOIN inflated rows: {before} -> {after}"
```

**Idempotent tests** (verify DELETE+INSERT round-trip):
```python
def test_rebuild_is_idempotent(con):
    """Running twice produces identical results."""
    run_build(con, instrument='MGC', start='2024-01-01', end='2024-01-31')
    count_1 = con.execute("SELECT COUNT(*) FROM target_table WHERE symbol='MGC'").fetchone()[0]
    run_build(con, instrument='MGC', start='2024-01-01', end='2024-01-31')
    count_2 = con.execute("SELECT COUNT(*) FROM target_table WHERE symbol='MGC'").fetchone()[0]
    assert count_1 == count_2, f"Not idempotent: {count_1} vs {count_2}"
```

**Pipeline gate tests** (verify reject paths):
```python
def test_rejects_future_dates():
    """Pipeline must reject dates in the future."""
    with pytest.raises(ValueError, match="future"):
        ingest(instrument='MGC', start='2099-01-01', end='2099-12-31')
```

### Step 4: Run the Test — Verify It FAILS

```bash
python -m pytest tests/<test_file>::<test_name> -v
```

If it passes, your test doesn't test what you think. Fix the test.

### Step 5: Write Minimal Implementation

Write the smallest code that makes the test pass. No extras.

### Step 6: Run the Test — Verify It PASSES

```bash
python -m pytest tests/<test_file>::<test_name> -v
```

### Step 7: Run Full Suite

```bash
python -m pytest tests/ -x -q
```

No regressions allowed.

### Step 8: Commit

```bash
git add <test_file> <impl_file>
git commit -m "feat/fix: <description>"
```

### Rules

- NEVER write implementation before the test
- NEVER test against stale outcomes — rebuild first if schema changed
- NEVER hardcode expected values that come from canonical sources (import them)
- Tests must verify BEHAVIOR, not implementation details
- If a test needs gold.db data, use an in-memory DuckDB fixture, not the real DB
```

---

### Task 4: Write bloomey-review.md skill

**Files:**
- Create: `.claude/commands/bloomey-review.md`

**Step 1: Write the skill file**

Write `.claude/commands/bloomey-review.md` with this exact content:

```markdown
Institutional code review with seven sins analysis: $ARGUMENTS

Use when: "review this", "check my work", "is this good", "code review", "before I commit", "bloomey", "seven sins", "review my changes", "anything wrong with this"

## Mr. Bloomey — Head of Quant Review

You are the Bloomberg head-of-quant reviewer. Your job is to find real problems, not nitpick style. You grade ruthlessly but fairly. False positives damage your credibility — only flag what you can prove.

### Step 0: Identify What to Review

If $ARGUMENTS specifies files, review those.
If blank, review all uncommitted changes:
```bash
git diff --name-only HEAD
git diff --cached --name-only
```

Read EVERY changed file before reviewing. Never review code you haven't read.

### Section A: SEVEN SINS SCAN (Weight: 40%)

For each changed file, scan for the seven sins of quantitative investing:

| Sin | What to Look For | Severity |
|-----|------------------|----------|
| **Look-ahead bias** | `double_break` used as filter, future data in predictor, LAG() without `WHERE orb_minutes = 5` | CRITICAL |
| **Data snooping** | Claiming significance after grid search without BH FDR, cherry-picking strategies by OOS peek | CRITICAL |
| **Overfitting** | High Sharpe + N<30, passing only one year, too many parameters for sample size | HIGH |
| **Survivorship bias** | Ignoring dead instruments (MCL/SIL/M6E/MBT), ignoring purged E0/E3 when drawing conclusions | HIGH |
| **Storytelling bias** | Narrative around noise, p>0.05 dressed as "edge", "significant" without p-value | MEDIUM |
| **Outlier distortion** | Single extreme day driving aggregates, no year-by-year breakdown | MEDIUM |
| **Transaction cost illusion** | Missing COST_SPECS, ignoring spread+slippage+commission | HIGH |

### Section B: CANONICAL INTEGRITY (Weight: 20%)

- [ ] Any hardcoded instrument lists? (must import from `ACTIVE_ORB_INSTRUMENTS`)
- [ ] Any hardcoded session times? (must use `SESSION_CATALOG`)
- [ ] Any hardcoded cost numbers? (must use `COST_SPECS`)
- [ ] Any magic numbers without `@research-source` annotation?
- [ ] Authority hierarchy respected? (CLAUDE.md > TRADING_RULES.md > code)
- [ ] One-way dependency maintained? (pipeline/ -> trading_app/, never reversed)

### Section C: STATISTICAL RIGOR (Weight: 25%)

- [ ] Every quantitative claim has a p-value from an actual test?
- [ ] BH FDR applied after testing 50+ hypotheses?
- [ ] Sample size labels correct? (<30 INVALID, 30-99 REGIME, 100+ CORE)
- [ ] Year-by-year breakdown for any finding?
- [ ] Correct statistical test used? (Jobson-Korkie for Sharpe, t-test for means, Fisher for proportions)
- [ ] N computed correctly? (not inflated by bad JOINs)

### Section D: PRODUCTION READINESS (Weight: 15%)

- [ ] Fail-closed? (exceptions abort, never return success in health/audit paths)
- [ ] Idempotent? (safe to re-run with DELETE+INSERT pattern)
- [ ] Subprocess return codes checked? (zero is the only success)
- [ ] No `except Exception: pass` outside atexit handlers?
- [ ] DB writes are single-process? (no concurrent DuckDB writes)
- [ ] Test coverage? (check TEST_MAP for companion test file)

### Grading

| Grade | Criteria |
|-------|----------|
| **A** | Zero sins, canonical compliance, statistically sound, production-ready |
| **A-** | Minor style issues, no sins, all checks pass |
| **B+** | One MEDIUM sin or 1-2 canonical violations, otherwise solid |
| **B** | Multiple MEDIUM sins or one HIGH sin with mitigation |
| **C** | One CRITICAL sin or multiple HIGH sins |
| **D** | Multiple CRITICAL sins or fundamental design flaw |
| **F** | Look-ahead bias in production code, or data snooping without FDR |

### Output Format

```
=== BLOOMEY REVIEW ===
Files reviewed: [list]
Grade: [A/B/C/D/F]

Section A — Seven Sins: [score]
  [findings with line citations]

Section B — Canonical Integrity: [score]
  [findings with line citations]

Section C — Statistical Rigor: [score]
  [findings with line citations]

Section D — Production Readiness: [score]
  [findings with line citations]

Verdict: [1-2 sentence summary]
Action items: [numbered list of required changes]
========================
```

### Optional: M2.5 Second Opinion

If the review is for a significant change (schema, entry model, pipeline logic), offer to run `/m25-audit` for a second opinion. Triage any M2.5 findings per `.claude/rules/m25-audit.md`.

### Rules

- NEVER flag something you can't prove with a line citation
- NEVER override CLAUDE.md rules — if code follows CLAUDE.md, it's correct
- Check cross-file context before flagging (the guard may exist in another file)
- DuckDB replacement scans are NOT bugs (DataFrame in scope = valid SQL reference)
- `fillna(-999.0)` is an intentional domain sentinel, not a bug
- `except Exception: pass` in atexit handlers is correct shutdown cleanup
```

---

### Task 5: Write quant-verify.md skill

**Files:**
- Create: `.claude/commands/quant-verify.md`

**Step 1: Write the skill file**

Write `.claude/commands/quant-verify.md` with this exact content:

```markdown
Run pre-commit quality gates and verify everything passes: $ARGUMENTS

Use when: about to claim work is done, before commits, before PRs, "verify", "are we good", "run the gates", "check everything", "pre-commit"

## Quick Verification Gates

Lightweight pre-commit check. For full impact mapping, use /integrity-guardian instead.

### Run ALL 4 gates. ANY failure = STOP.

**Gate 1: Drift Detection**
```bash
python pipeline/check_drift.py
```
Check count is self-reported at runtime. NEVER hardcode "all N checks passed."

**Gate 2: Data Integrity**
```bash
python scripts/tools/audit_integrity.py
```

**Gate 3: Behavioral Audit**
```bash
python scripts/tools/audit_behavioral.py
```

**Gate 4: Test Suite**
```bash
python -m pytest tests/ -x -q
```

### Evidence Block

After all 4 gates, emit:

```
=== VERIFY GATES ===
Drift:      PASS/FAIL
Integrity:  PASS/FAIL
Behavioral: PASS/FAIL
Tests:      PASS/FAIL (N passed)
===================
```

### Rules

- ALL 4 must pass. No exceptions. No "it's just a minor fail."
- If ANY gate fails: stop, investigate, fix, re-run ALL gates.
- Never claim "done" without this evidence block.
- Never hardcode check counts — they are self-reported at runtime.
- This is the LIGHTWEIGHT check. For full impact mapping + evidence, use /integrity-guardian.
```

---

### Task 6: Write trade-book.md skill

**Files:**
- Create: `.claude/commands/trade-book.md`

**Step 1: Write the skill file**

Write `.claude/commands/trade-book.md` with this exact content:

```markdown
Show current trading book with full strategy details: $ARGUMENTS

Use when: "what do I trade", "what's live", "show strategies", "what's at [session]", "trading book", "portfolio", "show me what's validated", "what's FIT", "live strategies"

## Trade Book Query

Fast direct query against gold.db. Skips MCP for speed.

### Step 1: Parse Arguments

- If $ARGUMENTS contains a session name (e.g., "CME_REOPEN", "TOKYO_OPEN"): filter by that session
- If $ARGUMENTS contains an instrument (e.g., "MGC", "MNQ"): filter by that instrument
- If $ARGUMENTS contains "all" or is empty: show everything FIT

### Step 2: Query gold.db Directly

```bash
python -c "
import duckdb
from pipeline.paths import GOLD_DB_PATH

con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
df = con.execute('''
    SELECT
        v.symbol,
        v.orb_label,
        v.orb_minutes,
        v.entry_model,
        v.confirm_bars,
        v.filter_type,
        v.rr_target,
        v.direction,
        v.sample_size,
        v.win_rate,
        v.avg_r AS ExpR,
        v.sharpe,
        v.all_years_positive,
        v.years_tested,
        COALESCE(f.fitness_regime, 'UNKNOWN') as fitness
    FROM validated_setups v
    LEFT JOIN (
        SELECT strategy_id, fitness_regime
        FROM strategy_fitness
        WHERE as_of_date = (SELECT MAX(as_of_date) FROM strategy_fitness)
    ) f ON v.strategy_id = f.strategy_id
    WHERE 1=1
    -- Add filters based on $ARGUMENTS here
    ORDER BY v.symbol, v.orb_label, v.orb_minutes, v.rr_target
''').fetchdf()
con.close()

print(df.to_string(index=False))
print(f'\nTotal: {len(df)} strategies')
"
```

Modify the WHERE clause based on parsed arguments:
- Session filter: `AND v.orb_label = '{session}'`
- Instrument filter: `AND v.symbol = '{instrument}'`
- FIT only (default): `AND COALESCE(f.fitness_regime, 'UNKNOWN') = 'FIT'`

### Step 3: Present Results

Format as a clean table grouped by session (Brisbane time order):

**For each strategy show ALL of these (MANDATORY — never omit any):**
- Symbol, orb_label, orb_minutes (5/15/30)
- entry_model, confirm_bars
- filter_type, rr_target
- direction
- sample_size, win_rate, ExpR, Sharpe
- fitness_regime
- all_years_positive, years_tested

### Step 4: Summary

- Count by instrument
- Count by fitness regime (FIT/WATCH/DECAY/UNFIT)
- Flag any strategies with `all_years_positive = False`
- Note data freshness (latest as_of_date from strategy_fitness)

### Rules

- ALWAYS include rr_target — user explicitly demanded this
- NEVER use MCP for this query — too slow and may be stale
- NEVER cite strategy counts from memory — always query fresh
- Show WATCH strategies dimmed (mention but flag as "monitor only")
- Hide UNFIT/DECAY unless user asks for them
- If query returns 0 rows, check if gold.db exists and has data
```

---

### Task 7: Write regime-check.md skill

**Files:**
- Create: `.claude/commands/regime-check.md`

**Step 1: Write the skill file**

Write `.claude/commands/regime-check.md` with this exact content:

```markdown
Check portfolio fitness and regime health across all instruments: $ARGUMENTS

Use when: "fitness", "regime", "any decay", "how's the portfolio", "health of strategies", "regime check", "strategy health", "are strategies still working"

## Regime Health Check

Quick portfolio fitness snapshot. Shows regime distribution and flags transitions.

### Step 1: Query Current Fitness

```bash
python -c "
import duckdb
from pipeline.paths import GOLD_DB_PATH

con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

# Summary by instrument and regime
summary = con.execute('''
    SELECT
        v.symbol,
        COALESCE(f.fitness_regime, 'NO_FITNESS') as regime,
        COUNT(*) as count
    FROM validated_setups v
    LEFT JOIN (
        SELECT strategy_id, fitness_regime
        FROM strategy_fitness
        WHERE as_of_date = (SELECT MAX(as_of_date) FROM strategy_fitness)
    ) f ON v.strategy_id = f.strategy_id
    GROUP BY v.symbol, regime
    ORDER BY v.symbol, regime
''').fetchdf()

print('=== REGIME SUMMARY ===')
print(summary.to_string(index=False))

# Total counts
totals = con.execute('''
    SELECT
        COALESCE(f.fitness_regime, 'NO_FITNESS') as regime,
        COUNT(*) as count
    FROM validated_setups v
    LEFT JOIN (
        SELECT strategy_id, fitness_regime
        FROM strategy_fitness
        WHERE as_of_date = (SELECT MAX(as_of_date) FROM strategy_fitness)
    ) f ON v.strategy_id = f.strategy_id
    GROUP BY regime
    ORDER BY regime
''').fetchdf()

print('\n=== PORTFOLIO TOTALS ===')
print(totals.to_string(index=False))

# Data freshness
freshness = con.execute('''
    SELECT MAX(as_of_date) as latest, MIN(as_of_date) as earliest
    FROM strategy_fitness
''').fetchdf()
print(f'\nFitness data: {freshness.iloc[0][\"earliest\"]} to {freshness.iloc[0][\"latest\"]}')

con.close()
"
```

### Step 2: Flag Concerns

- Any instrument with 0 FIT strategies? -> RED FLAG
- Any instrument with > 50% DECAY/UNFIT? -> YELLOW FLAG
- Data freshness > 30 days old? -> STALE WARNING
- Significant shift from previous check? -> Note the transition

### Step 3: Present

One-liner per instrument, not a wall of text:

```
=== REGIME CHECK ===
MGC:  X FIT, Y WATCH, Z DECAY  [HEALTHY/CONCERN/CRITICAL]
MNQ:  X FIT, Y WATCH, Z DECAY  [HEALTHY/CONCERN/CRITICAL]
MES:  X FIT, Y WATCH, Z DECAY  [HEALTHY/CONCERN/CRITICAL]
M2K:  X FIT, Y WATCH, Z DECAY  [HEALTHY/CONCERN/CRITICAL]
Portfolio: N total, X% FIT
Data as of: YYYY-MM-DD
====================
```

### Rules

- NEVER cite counts from memory — always query fresh
- One query, one table, one summary. Keep it tight.
- If strategy_fitness table is empty, say so clearly — don't show zeros as if they're real.
```

---

### Task 8: Write post-rebuild.md skill

**Files:**
- Create: `.claude/commands/post-rebuild.md`

**Step 1: Write the skill file**

Write `.claude/commands/post-rebuild.md` with this exact content:

```markdown
Run post-rebuild audit chain after outcome/discovery/validation completes: $ARGUMENTS

Use when: after any rebuild chain completes, "post-rebuild", "rebuild done", "sync up", "finish the rebuild", "audit after rebuild"

## Post-Rebuild Chain

Run AFTER outcome_builder + strategy_discovery + strategy_validator have completed successfully.
This skill handles everything that comes AFTER the core rebuild.

Parse $ARGUMENTS for instrument (default: all active instruments).

### Step 1: Retire E3 Strategies

Validator promotes E3 strategies — this script retires them:
```bash
python scripts/migrations/retire_e3_strategies.py
```

### Step 2: Build Edge Families

```bash
python scripts/tools/build_edge_families.py --instrument $INSTRUMENT
```

If no instrument specified, run for all active:
```bash
for INST in MGC MNQ MES M2K; do
    python scripts/tools/build_edge_families.py --instrument $INST
done
```

### Step 3: Regenerate Repo Map

```bash
python scripts/tools/gen_repo_map.py
```

### Step 4: Audit Gates (ALL must pass)

```bash
python scripts/tools/audit_integrity.py
python pipeline/check_drift.py
python scripts/tools/audit_behavioral.py
python -m pytest tests/ -x -q
```

**ANY failure = STOP.** Fix the issue before proceeding.

### Step 5: Sync Pinecone Knowledge Base

```bash
python scripts/tools/sync_pinecone.py
```

### Step 6: Report

```
=== POST-REBUILD COMPLETE ===
Instrument:     $INSTRUMENT (or ALL)
E3 retire:      PASS/FAIL
Edge families:  PASS/FAIL
Repo map:       PASS/FAIL
Integrity:      PASS/FAIL
Drift:          PASS/FAIL
Behavioral:     PASS/FAIL
Tests:          PASS/FAIL (N passed)
Pinecone sync:  PASS/FAIL
=============================
```

### Rules

- Each step depends on the previous one succeeding
- NEVER skip the audit gates — they catch regressions from the rebuild
- NEVER skip Pinecone sync — generated snapshots go stale without it
- Stop on first failure, report which step failed
```

---

### Task 9: Fix discover.md

**Files:**
- Modify: `.claude/commands/discover.md`

**Step 1: Read the current file**

Read `.claude/commands/discover.md` to confirm current content.

**Step 2: Rewrite with fixes**

Replace the entire file. Key changes:
- Remove all `claude-mem:mem-search` references (dead tool)
- Update session names: use `CME_REOPEN` not `0900`, `TOKYO_OPEN` not `1000`, etc.
- E2 is now the default entry model (E0 is dead, E1 is conservative baseline)
- Save findings to memory files (`.claude/projects/.../memory/`) not claude-mem
- Remove DST split note (DST is fully resolved — all sessions are dynamic)
- Add semantic trigger description
- Fix the "Default entry model" line

New content for discover.md:

```markdown
Research edge discovery for instrument and session: $ARGUMENTS

Use when: "discover", "scan for edges", "research [instrument]", "find strategies", "edge discovery", "what works for [instrument]", "test [session]"

## Instructions

You are running an AI-assisted strategy research workflow. Follow these steps exactly.

### Step 1: Parse Arguments

Parse $ARGUMENTS for instrument (required) and session (optional).
Examples: "MGC CME_REOPEN", "MES NYSE_OPEN", "MNQ", "MGC all"

Default entry model: E2 (stop-market, industry standard). Use E1 for conservative baseline comparison.
E0 is DEAD — never use E0.

### Step 2: Check Research Memory

Check memory files in `.claude/projects/` memory directory for previous findings on this (instrument, session).
Search for files like `regime_findings.md`, `m2k_findings.md`, etc.
If previous findings exist, summarize them before running new scans.

### Step 3: Check Current Validated State

```bash
python -c "
import duckdb
from pipeline.paths import GOLD_DB_PATH
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
r = con.execute(\"SELECT COUNT(*) FROM validated_setups WHERE symbol='{INSTRUMENT}'\").fetchone()
print(f'Validated strategies for {INSTRUMENT}: {r[0]}')
con.close()
"
```

### Step 4: Run the Discovery Scanner

```bash
python research/discover.py --instrument {INSTRUMENT} --session {SESSION} --entry-model E2 --json
```
If no session specified, add `--all-sessions`.

### Step 5: Interpret Results

For each scan result, apply RESEARCH_RULES.md labels:
- BH-significant with p<0.005: "validated finding"
- BH-significant with p<0.05: "promising hypothesis"
- Not BH-significant: "statistical observation" (mention but don't recommend action)
- Baseline ExpR <= 0: "NO-GO -- negative baseline"

### Step 6: Report

**{INSTRUMENT} {SESSION} Discovery Report**

| Predictor | Delta | p-value | BH-sig? | N | Label |
|-----------|-------|---------|---------|---|-------|
| ... | ... | ... | ... | ... | ... |

Key findings: [2-3 bullet summary]
Recommended actions: [specific next steps, if any survive FDR]

### Step 7: Save Findings

Save findings to the appropriate memory file in `.claude/projects/.../memory/`:
- MGC findings -> `regime_findings.md` or `mgc_regime_analysis.md`
- MNQ findings -> update existing topic file or create new one
- New instrument -> create `{instrument}_findings.md`

Include: n_tests, n_significant, top findings, recommended actions, date.

### Rules (from RESEARCH_RULES.md)

- NEVER say "significant" without p-value
- NEVER say "edge" without BH FDR confirmation
- Sample size labels: <30 INVALID, 30-99 REGIME, 100+ CORE
- RSI/MACD/Bollinger are "guilty until proven" -- flag if they appear significant
- Always include year-by-year breakdown for any BH-significant finding
- All sessions are dynamic/event-based from SESSION_CATALOG -- DST is fully resolved
```

---

### Task 10: Fix validate-instrument.md DB path

**Files:**
- Modify: `.claude/commands/validate-instrument.md`

**Step 1: Read the current file**

Read `.claude/commands/validate-instrument.md`.

**Step 2: Fix the DB path note**

Replace the "Important" section at the bottom. Change:
```
- Canonical DB is `C:/db/gold.db` (set via `DUCKDB_PATH` in `.env`)
```
To:
```
- DB path resolved via `pipeline.paths.GOLD_DB_PATH` (reads `DUCKDB_PATH` from `.env`). C:/db/gold.db is the SCRATCH copy, not canonical.
```

Also add semantic trigger description to line 1:
```
Validate strategies for instrument $ARGUMENTS (default MGC). Uses the correct flag combination.

Use when: "validate", "run validation", "validate [instrument]", "re-validate", "strategy validation"
```

---

### Task 11: Update integrity-guardian.md scope

**Files:**
- Modify: `.claude/commands/integrity-guardian.md`

**Step 1: Read the current file**

Read `.claude/commands/integrity-guardian.md`.

**Step 2: Add scope clarification and semantic trigger**

Add to line 1 after the description:
```
Use when: task is COMPLETE and needs full impact verification, "integrity check", "full audit", "impact check", "guardian". For lightweight pre-commit gates only, use /quant-verify instead.
```

Add at the end:
```
## Scope Note

This is the COMPREHENSIVE post-task review with full impact mapping.
For lightweight "just run the 4 gates" pre-commit check, use /quant-verify.

| Situation | Use |
|-----------|-----|
| Quick pre-commit check | /quant-verify |
| Full post-task audit with impact map | /integrity-guardian |
| Institutional code review with grading | /bloomey-review |
```

---

### Task 12: Commit All Skills + Update Memory

**Files:**
- Modify: `.claude/commands/*.md` (all files from tasks 0-11)

**Step 1: Stage all changes**

```bash
git add .claude/commands/4t.md
git add .claude/commands/4tp.md
git add .claude/commands/quant-debug.md
git add .claude/commands/quant-tdd.md
git add .claude/commands/bloomey-review.md
git add .claude/commands/quant-verify.md
git add .claude/commands/trade-book.md
git add .claude/commands/regime-check.md
git add .claude/commands/post-rebuild.md
git add .claude/commands/discover.md
git add .claude/commands/validate-instrument.md
git add .claude/commands/integrity-guardian.md
```

**Step 2: Commit**

```bash
git commit -m "feat: 9 custom quant skills + 3 command fixes

New skills:
- /4t: orient-design-detail-validate workflow
- /4tp: 4T then auto-proceed to plan
- /quant-debug: pipeline-aware debugging (JOIN traps, DST, DuckDB)
- /quant-tdd: TDD with tripwire tests, idempotent patterns
- /bloomey-review: institutional code review with seven sins
- /quant-verify: lightweight pre-commit quality gates
- /trade-book: fast strategy lookup with full details
- /regime-check: portfolio fitness snapshot
- /post-rebuild: post-rebuild audit chain + pinecone sync

Fixes:
- discover.md: kill dead E0/claude-mem refs, update session names
- validate-instrument.md: fix DB path (was pointing to scratch copy)
- integrity-guardian.md: add scope clarification vs quant-verify"
```

**Step 3: Verify**

```bash
ls .claude/commands/
git log --oneline -1
```

Expected: 15 command files listed, latest commit shows the skills message.
