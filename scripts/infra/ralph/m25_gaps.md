<think>
The user wants me to review SQL queries in a trading pipeline, specifically focusing on:

1. JOIN correctness - daily_features has 3 rows per (trading_day, symbol) - one per orb_minutes. Any JOIN MUST include orb_minutes
2. Missing WHERE clauses that could cause row explosion
3. LAG()/window functions without proper PARTITION BY or WHERE orb_minutes = 5
4. Aggregations that could double-count due to JOIN fan-out
5. DELETE+INSERT idempotency - verify DELETE range matches INSERT range

The user provided two files:
1. `pipeline/build_daily_features.py` - builds daily_features from bars_1m and bars_5m
2. `trading_app/outcome_builder.py` - pre-computes outcomes for all RR targets

Let me analyze these files for SQL issues.

Looking at the code more carefully:

## build_daily_features.py

This file builds daily_features - it queries bars_1m and bars_5m but does NOT join with any existing table (it inserts new data). Let me check for issues:

1. **DuckDB replacement scans**: The file uses `con.execute(f"INSERT INTO daily_features ({col_names}) SELECT {col_names} FROM features_df")` - this is DuckDB's replacement scan feature where it can reference pandas DataFrames directly. This is correct as per the system prompt.

2. **DELETE+INSERT idempotency**: Lines 684-697 show:
```python
con.execute("""
    DELETE FROM daily_features
    WHERE symbol = ?
    AND trading_day >= ?
    AND trading_day <= ?
    AND orb_minutes = ?
""", [symbol, start_date, end_date, orb_minutes])
```
This correctly deletes based on symbol, trading_day, AND orb_minutes. The INSERT then inserts with the same orb_minutes. This looks correct.
 
## outcome_builder.py

Now examining the outcome_builder.py file. The key query that could present issues fetches daily_features rows with selected columns for each session label. This query joins multiple sessions across a single row, which might create problems if daily_features actually contains multiple rows per trading_day (one for each orb_minutes value). However, the WHERE clause filters by both symbol and orb_minutes, which prevents the row explosion issue from occurring.

The INSERT OR REPLACE statement pulls directly from batch_df into orb_outcomes, and DuckDB handles the DataFrame reference correctly. But I notice something important: the file retrieves daily_features with specific columns like orb_{lbl}_high and orb_{lbl}_break_dir, but it never actually applies an orb_minutes filter to this query.

Wait - looking at the parameter list, orb_minutes is already included in the WHERE clause: `WHERE symbol = ? AND orb_minutes = ?`. That's actually correct.

Now checking the checkpoint query that fetches already-computed days - it uses `SELECT DISTINCT trading_day FROM orb_outcomes WHERE symbol = ? AND orb_minutes = ?` and compares against trading_day values. The real issue is that the code treats trading_day as a `date` object but the checkpoint set was constructed from tuples returned by the query, so there's a type mismatch to investigate there.

Looking at how computed_days gets populated from the DISTINCT query, it creates a set of date objects since the column is already a DATE type in the schema. The later comparison against trading_day (also a date) should work correctly, so the checkpoint logic appears sound.

Now I need to check whether outcome_builder has any JOIN operations that could cause row explosion. The main data retrieval doesn't appear to join daily_features with anything else - it's a direct select that might return multiple rows per (symbol, trading_day) pair depending on different ORB parameters.

Wait, I see the issue now. The query filters by symbol and orb_minutes, but it's selecting across all sessions for each trading day. Since daily_features has one row per (trading_day, symbol, orb_minutes), filtering by a specific orb_minutes should guarantee a single row per day. That part checks out.

Now I need to look at whether any JOINs with daily_features happen elsewhere in the codebase, particularly in strategy_validator.py or other files that might pull from this table.
</think>

Looking at these files, I need to analyze the SQL patterns for JOIN correctness, WHERE clauses, and idempotency. Let me examine the code systematically.

## Analysis of `pipeline/build_daily_features.py`

This file **builds** daily_features from scratch - it does NOT join with daily_features, so there's no risk of JOIN fan-out here. Let me verify the key patterns:

### 1. SAFE PATTERNS

**DuckDB Replacement Scan (lines 691-695)** - CORRECT:
```python
col_names = ", ".join(features_df.columns)
con.execute(f"""
    INSERT INTO daily_features ({col_names})
    SELECT {col_names} FROM features_df
""")
```
This is documented DuckDB behavior for zero-copy DataFrame reference. Not a bug.

**DELETE+INSERT Idempotency (lines 678-686)** - CORRECT:
```python
con.execute("""
    DELETE FROM daily_features
    WHERE symbol = ?
    AND trading_day >= ?
    AND trading_day <= ?
    AND orb_minutes = ?
""", [symbol, start_date, end_date, orb_minutes])
```
The DELETE filters by `symbol`, `trading_day`, **AND** `orb_minutes`. The INSERT uses the same `orb_minutes` value. This is correct idempotency.

**Bulk load queries** - CORRECT:
Lines 558-565 and 577-587 use proper date filtering with `ts_utc >= ? AND ts_utc < ?` - no fan-out risk.

### 2. FINDINGS

**No SQL issues found in this file.** The build process is clean.

---

## Analysis of `trading_app/outcome_builder.py`

This file reads from `daily_features` and writes to `orb_outcomes`.

### 1. SAFE PATTERNS

**Checkpoint query (lines 330-337)** - CORRECT:
```python
computed_days = {
    r[0] for r in con.execute(
        "SELECT DISTINCT trading_day FROM orb_outcomes "
        "WHERE symbol = ? AND orb_minutes = ?",
        [instrument, orb_minutes],
    ).fetchall()
}
```
Filters by both `symbol` and `orb_minutes`. Correct.

**Daily features query (lines 320-328)** - CORRECT:
```python
query = f"""
    SELECT trading_day, symbol, orb_minutes,
           {', '.join(
               f'orb_{lbl}_high, orb_{lbl}_low, orb_{lbl}_break_dir, orb_{lbl}_break_ts'
               for lbl in sessions
           )}
    FROM daily_features
    WHERE symbol = ? AND orb_minutes = ?
    {date_filter}
    ORDER BY trading_day
"""
```
Uses parameterized `orb_minutes` from function argument. Correctly scoped.

### 2. FINDINGS

**FINDING 1: Minor - Type handling in checkpoint comparison**
- **Severity**: LOW
- **Location**: Lines 330-337, 397-399
- **Issue**: The checkpoint set uses `{r[0] for r in ...}` which creates a set of whatever type DuckDB returns. Later comparison at line 397 uses `if trading_day in computed_days:`. If DuckDB returns `datetime.date` objects (which it should for DATE columns), this works. But if there's any type coercion difference, it could silently skip or re-compute days.
- **Current code**:
```python
computed_days = {
    r[0] for r in con.execute(...).fetchall()
}
# ...
if trading_day in computed_days:
    continue
```
- **Fix**: Add explicit type conversion:
```python
computed_days = {
    r[0] for r in con.execute(...).fetchall()
} if not dry_run else set()
# Or ensure date comparison works:
if not dry_run and trading_day in computed_days:
```

This is LOW because it's protected by the `--force` flag which allows recovery.

---

## Cross-File Analysis

The user mentioned checking for LAG()/window functions without proper PARTITION BY. Let me check if there are any such patterns in the broader codebase context. Since I only have these two files, I'll note:

### Observations on Window Functions

Looking at both files, I don't see direct SQL `LAG()`, `ROW_NUMBER()`, or similar window functions in the queries. The computation of:
- RSI (build_daily_features.py) - done in Python using numpy
- GARCH forecasts - done in Python using arch package  
- Relative volume (rel_vol) - computed in Python loops
- ATR/ATR velocity - computed in Python loops

The window functions are **implemented in Python**, not SQL. This is actually safer because it avoids accidental cross-aperture leakage in SQL.

---

## Summary

| File | Pattern | Status |
|------|---------|--------|
| build_daily_features.py | DuckDB replacement scan | ✅ SAFE |
| build_daily_features.py | DELETE+INSERT idempotency | ✅ SAFE (includes orb_minutes) |
| build_daily_features.py | Bulk load WHERE clauses | ✅ SAFE |
| outcome_builder.py | Daily features query | ✅ SAFE (filters by orb_minutes) |
| outcome_builder.py | Checkpoint query | ✅ SAFE |
| outcome_builder.py | Checkpoint type handling | ⚠️ LOW (minor type concern) |

### Recommendations

**For build_daily_features.py**: No changes needed. The code is correct.

**For outcome_builder.py**: Consider adding explicit type safety for the checkpoint:

```python
# Line 330-337, change to:
computed_days: set[date] = set()
if not dry_run and not force:
    computed_days = {
        date.fromisoformat(str(r[0]))  # Explicit conversion
        for r in con.execute(
            "SELECT DISTINCT trading_day FROM orb_outcomes "
            "WHERE symbol = ? AND orb_minutes = ?",
            [instrument, orb_minutes],
        ).fetchall()
    }
```

However, this is a **very minor hardening** - the current code works correctly in practice because DuckDB's DATE type maps to Python's `date` natively.

---

### Note on Missing Cross-File Analysis

The system prompt mentions checking for JOINs in files like `strategy_validator.py` that might join `daily_features` with `orb_outcomes`. Those files were **not provided** in this review request. The two files analyzed do NOT have cross-table JOIN issues between daily_features and orb_outcomes - they each operate on one table at a time.