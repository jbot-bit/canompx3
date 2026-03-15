---
name: quant-debug
description: Diagnose and fix a bug or unexpected behavior in the quant pipeline
---
Diagnose and fix a bug or unexpected behavior in the quant pipeline: $ARGUMENTS

Use when: ANY bug, test failure, unexpected behavior, "why is this broken", "wrong numbers", "row count off", "test failing", "unexpected results", "data looks wrong", "something is off", "this doesn't look right", "what's wrong", error in pipeline/ or trading_app/

## Quant-Aware Debugging Protocol

You are debugging a quantitative trading pipeline. Follow this protocol EXACTLY. Do NOT guess at fixes.

### Step 0: Reproduce

Before anything else, reproduce the problem:
1. Run the failing command/test and capture the FULL output
2. Note the exact error message, line number, and file
3. If "wrong numbers" -- get the ACTUAL numbers and the EXPECTED numbers

"Reading code is NOT verifying code." You must EXECUTE and read output.

### Step 1: Classify the Bug

Which category? This determines your checklist:

**A) Row Count / Data Inflation**
- [ ] Triple-join trap? `daily_features` has 3 rows per (trading_day, symbol) -- one per orb_minutes. Missing `AND o.orb_minutes = d.orb_minutes` triples rows.
- [ ] Aperture mismatch? Comparing O5 data to O15/O30 without filtering.
- [ ] Filter not applied? Querying `orb_outcomes` without joining `daily_features` for filter eligibility.

**B) Wrong Values / Shifted Data**
- [ ] Timezone double-conversion? DuckDB `fetchdf()` returns `datetime64[us, Australia/Brisbane]` (already local). Adding `pd.Timedelta(hours=10)` double-converts. Use `.dt.tz_localize(None)` instead.
- [ ] DST contamination? Using hardcoded session times instead of `SESSION_CATALOG` from `pipeline/dst.py`.
- [ ] Trading day assignment? Bars before 09:00 Brisbane should be assigned to PREVIOUS trading day.
- [ ] Look-ahead leak? `double_break` is look-ahead. LAG() without `WHERE orb_minutes = 5` causes cross-aperture contamination.

**C) Missing Data / Empty Results**
- [ ] DuckDB replacement scan? DuckDB can reference in-scope pandas DataFrames in SQL -- this is NOT a bug.
- [ ] Wrong instrument symbol? Check `ASSET_CONFIGS` -- some instruments use full-size source contracts (GC for MGC, ES for MES, RTY for M2K).
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
2. Trace imports -- follow the call chain to find the root cause
3. Check for existing guards (try/except, if-checks, assertions) that should have caught this
4. Cross-reference canonical sources -- is something hardcoded that should be imported?

### Step 3: Verify the Fix

After implementing a fix:
1. Run the original failing command -- it must now succeed
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
