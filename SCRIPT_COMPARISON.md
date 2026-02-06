# HONEST COMPARISON: Original vs New Script

## FEATURES COMPARISON

| Feature | Original | New | Status |
|---------|----------|-----|--------|
| **Data Loading** | `store.to_df()` - ALL into RAM | `store.to_df(count=N)` - chunked | FIXED |
| **Trading Day Calc** | `df.apply()` over millions | Mostly vectorized | FIXED |
| **Outright Filter** | `is_outright_contract()` | Same pattern | SAME |
| **Contract Selection** | `max()` - non-deterministic | Expiry + lexicographic tiebreak | FIXED |
| **Validation Checks** | NaN, h>=max(o,c), l<=min(o,c), h>=l, vol>=0 | Same + prices>0, finite, h>=max(o,c,**l**) | STRICTER |
| **Fail Mode** | Skip bad rows, continue | ABORT on any bad row | CHANGED |
| **Insert Method** | INSERT OR REPLACE | INSERT OR REPLACE | SAME |
| **Transaction** | No explicit BEGIN/COMMIT | BEGIN/COMMIT/ROLLBACK | ADDED |
| **Chunking Unit** | Row count (chunk_size * 1000) | Trading days (chunk_days) | CHANGED |
| **Resume** | DB max date lookup | Checkpoint file (JSONL) | CHANGED |
| **bars_5m Build** | YES (lines 420-450) | NO | REMOVED (per rules) |
| **Schema Gate** | None | Verify ohlcv-1m | ADDED |
| **UTC Gate** | Assumes UTC | Assert UTC timezone | ADDED |
| **PK Safety** | None | Assert unique ts_utc before merge | ADDED |
| **Integrity Gate** | None | Check no dupes/NULLs after merge | ADDED |
| **Final Gates** | None | Verify type/dupes/NULLs at end | ADDED |
| **Progress Output** | Days processed, % | Batch count, rows written | CHANGED |
| **Summary** | days/bars/rejected/contracts | chunks/days/rows/contracts/time | CHANGED |

## CLI ARGUMENTS

| Argument | Original | New |
|----------|----------|-----|
| --dry-run | YES | YES |
| --start | YES | YES |
| --end | YES | YES |
| --resume | YES (DB max date) | YES (checkpoint) |
| --chunk-size | YES (rows) | NO (removed) |
| --chunk-days | NO | YES (trading days) |
| --batch-size | NO | YES (DBN read batch) |
| --retry-failed | NO | YES |

## WHAT'S MISSING FROM NEW SCRIPT

### 1. bars_5m Generation (INTENTIONALLY REMOVED)
**Original (lines 420-450):**
```python
# Rebuild bars_5m
con.execute("DELETE FROM bars_5m WHERE symbol = ?", [SYMBOL])
con.execute("INSERT INTO bars_5m ...")
```

**New:** Does not touch bars_5m at all.

**Why:** CANONICAL_backfill_dbn_mgc_rules.txt line 29 explicitly forbids:
> "Touch or generate ANY non-target tables (explicitly forbidden: bars_5m, daily_features, any derived tables)"

**Impact:** You need a SEPARATE script to build bars_5m after ingestion.

### 2. Rejection Reasons Summary
**Original (lines 408-412):**
```python
if stats['rejection_reasons']:
    print("Rejection reasons:")
    for reason, count in sorted(...):
        print(f"  {reason}: {count}")
```

**New:** ABORTS immediately on first invalid row, so no accumulation of rejection reasons.

**Why:** Fail-closed mode per rules - any invalid row aborts entire backfill.

**Impact:** You won't see a breakdown of "10 NaN, 5 negative volume, etc." - you'll just see the first error and abort.

### 3. Total Records Count at Start
**Original (lines 253-256):**
```python
df = store.to_df()
total_records = len(df)
print(f"Total records in file: {total_records:,}")
```

**New:** Doesn't count total records (would require loading everything to count).

**Impact:** You won't see "Total records in file: 6,401,788" at the start.

### 4. Unique Symbols Summary at Start
**Original (lines 262-272):**
```python
all_symbols = df['symbol'].unique()
print(f"Unique symbols in file: {len(all_symbols)}")
print(f"  Outright contracts: {len(outright_symbols)}")
print(f"  Spreads/other (will be ignored): {len(spread_symbols)}")
```

**New:** Doesn't show this summary (would require loading everything).

**Impact:** You won't see the symbol count breakdown at start.

## WHAT'S ADDED IN NEW SCRIPT

1. **CheckpointManager class** - Full JSONL checkpoint system
2. **validate_chunk()** - Vectorized validation
3. **validate_timestamp_utc()** - UTC assertion
4. **parse_expiry()** - For deterministic tiebreak
5. **compute_trading_days()** - Vectorized trading day calc
6. **check_pk_safety()** - PK assertion before merge
7. **check_merge_integrity()** - Integrity check after merge
8. **run_final_gates()** - Final honesty verification
9. **Proper transaction handling** - BEGIN/COMMIT/ROLLBACK
10. **Schema verification** - Assert ohlcv-1m before proceeding

## BEHAVIORAL DIFFERENCES

| Scenario | Original | New |
|----------|----------|-----|
| Bad row found | Skip it, continue | ABORT entire backfill |
| Equal volume tie | Random (Python dict order) | Deterministic (expiry/lex) |
| Interrupted run | Re-process from DB max date | Resume from checkpoint |
| Same range re-run | Overwrites (idempotent) | Skips done chunks (faster) |
| Missing schema | Proceeds anyway | ABORTS |

## VERDICT

**The new script has ALL core functionality of the original, with these changes:**

1. **bars_5m removed** - Per rules, need separate script
2. **Fail mode changed** - Abort vs skip (stricter)
3. **Resume mechanism different** - Checkpoint vs DB lookup (more robust)
4. **Some startup info missing** - No total count / symbol breakdown (can't count without loading)

**HONESTY CHECK:** The new script is MORE strict and MORE correct, but it does NOT build bars_5m. If you need bars_5m, that's a separate step now.
