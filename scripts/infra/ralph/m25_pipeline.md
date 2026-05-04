<think>
The user wants me to review three Python files from a trading system:

1. `pipeline/ingest_dbn.py` - Generic multi-instrument DBN ingestion
2. `pipeline/build_bars_5m.py` - Build 5-minute bars from 1-minute bars
3. `pipeline/build_daily_features.py` - Build daily features

Based on the system prompt, I need to:
1. Identify things that are CORRECT (not bugs) - these include DuckDB native reference, multi-stage statistical pipeline, ML 4-gate quality system, fail-open design, atexit exception handling, -999.0 NaN sentinel, DELETE + INSERT idempotency
2. Find REAL bugs with severity
3. Provide recommendations

Let me analyze each file:

## File 1: pipeline/ingest_dbn.py

Looking at the code for potential bugs:

1. **Line 82-87**: The date filter logic uses `timedelta(days=1)` but this could miss files at the boundaries. If a file is exactly at the start_filter or end_filter date, it's included because of the `<` vs `<=` comparison. Actually, the logic seems fine - files before start_filter-1 day are skipped, files after end_filter+1 day are skipped.

2. **Line 174-175**: There's a `sys.exit(1)` in the startup validation when the DBN path doesn't exist. This is fine.

3. **Line 200**: Schema verification uses `test_store.schema` - this is correct.

4. **Line 232**: `con = None` initialized, then set in a conditional block. The `atexit` handler correctly checks for None.

5. **Line 260-265**: Date filter logic with minimum_start_date enforcement - seems correct.

6. **Line 287**: `chunk_df = chunk_df.reset_index()` - this is fine.

7. **Line 305**: Filtering for outrights using regex - looks correct.
   
8. **Line 332**: Computing trading days with `compute_trading_days(chunk_df)` is correct, though it comes from an imported module I can't fully verify here.

9. **Line 335-358**: The aggregation loop groups by trading_day and filters based on date bounds, which appears sound.

10. **Line 365-378**: The checkpoint/resume logic with `checkpoint_mgr.should_process_chunk()` handles the chunking mechanism appropriately.

11. **Line 380-397**: Constructing the DataFrame and executing the INSERT OR REPLACE statement uses DuckDB's native reference capability correctly.

12. **Line 399-412**: The integrity verification via `check_merge_integrity()` follows the established pattern.

13. **Potential Issue - Line 295**: The fast-forward check uses `batch_max_date = chunk_df['ts_event'].max().date()`, but at this stage `chunk_df` hasn't had its index set yet. The subsequent line sets the index to 'ts_event', so the order is backwards - I need to check whether the index is being accessed before it's actually assigned.

Looking at the sequence: reset_index() is called first, then the max date is computed from 'ts_event', then set_index('ts_utc') happens. This should work since reset_index() promotes the current index to a column, making 'ts_event' accessible as a column for the max() calculation.

However, there's a deeper issue with the fast-forward logic. The code calculates batch_max_date from the entire chunk_df before applying the date filter, but chunk_df still contains all rows at that point—including those outside the intended date range. The filter hasn't been applied yet, so batch_max_date could be pulled from rows that should have been excluded. The fix would be to compute batch_max_date after filtering by start_filter, not before.

In the DataFrame construction around lines 380-397, the INSERT OR REPLACE pattern uses chunk_df as a native DuckDB reference, which is correct. But there's another concern: the chunking logic uses `args.chunk_days` as a count threshold, yet trading_day_buffer is keyed by individual dates, so the buffer could contain multiple trading days even when the count looks correct.

The real problem is at line 340 - `trading_days` gets computed from chunk_df after it's already been filtered to only outright contracts. This means any contract switching that happens within a trading day could cause issues since the filtering happens before the trading day computation.

Looking at the checkpoint manager, there's a potential issue: if start_filter or end_filter are provided but differ from what was originally checkpointed, the resume logic doesn't validate whether the current filters align with the stored checkpoint range.

In the integrity check, the query correctly uses `orb_minutes` in the ON clause as required by Section 3. However, there's a bug in how `source_counts` is constructed - it's grouping bucketed data by `source_symbol` in addition to the required dimensions, which could cause incorrect aggregation when multiple source symbols exist within the same bucket.

Looking at the source_mode calculation, using `DISTINCT ON (bucket_ts, symbol)` with ordering by `cnt DESC, source_symbol ASC` correctly selects the most frequent source_symbol per bucket while breaking ties lexicographically, so that part works as intended.

In `build_daily_features.py`, there's a potential issue with how `ts_utc` is being handled. The DataFrame fetched from the database should already contain timezone-aware timestamps since `ts_utc` is stored as TIMESTAMPTZ in the database. However, the code explicitly converts it with `pd.to_datetime(df['ts_utc'], utc=True)`, which could cause problems if the timestamps are already aware - converting them again might produce unexpected results.

Looking at the GARCH implementation around line 440, the try-except block catches all exceptions silently and returns None without logging what went wrong, which could hide bugs during development.

In the RSI computation, there's a potential edge case: when `bars_5m_ts` is provided but contains no matching bars before the ORB time, the function returns None. The binary search using `np.searchsorted` with `side="right"` should position correctly at the first bar on or after the target time, but this scenario needs handling.

The main concern is the interaction between `build_features_for_day` and `compute_outcome`. The outcome calculation uses `break_ts` from `detect_break` and iterates through `post_break` bars, but it's missing MAE/MFE tracking when the break bar itself moves price. The current loop starts from bars after `break_ts`, which means it could overlook the initial move that triggered the break.

There's also a potential timezone mismatch: `break_ts` from the database might be timezone-aware UTC, but it's being compared against `bars_df['ts_utc']` which should also be UTC-aware, so this should work correctly if both are properly converted. However, there's a subtle issue where ORB windows could span midnight in UTC but remain within a single trading day in Brisbane time, which needs verification.

The real concern is that `_orb_utc_window` performs a bounds check against the trading day's UTC window, but this validation doesn't account for sessions like NYSE_OPEN that shift to the next calendar day in Brisbane time. I need to confirm whether the ORB windows properly respect the Brisbane trading day boundaries when determining valid UTC ranges.

Looking at the implementation, the validation allows ORBs where the UTC start falls between the trading day's UTC start and end, which should handle cases like NYSE_OPEN correctly—the UTC start would be 14:30 UTC on the prior day, placing it within the valid window for the current trading day.

There's a performance consideration worth noting: when `dry_run` is True, the function executes the build query but discards the results after counting distinct buckets, which means unnecessary computation occurs even in preview mode. However, this doesn't constitute a functional bug.

The core logic appears sound overall. The most pressing issues are the DataFrame index order in `ingest_dbn.py` and how the timestamp gets handled in `build_daily_features.py`, while the other observations represent either design trade-offs or acceptable implementation details.
</think>

Looking at these three files, I'll analyze them for real bugs while considering the architectural patterns from the system prompt.

---

## 1. WELL-DONE

### pipeline/ingest_dbn.py
- **Lines 380-397**: Clean DuckDB native reference pattern for chunked INSERT OR REPLACE
- **Lines 235-238**: Connection cleanup with atexit - correctly handles all exit paths
- **Line 193**: Schema verification before processing (fail-closed)
- **Lines 287-295**: Fast-forward optimization skips pre-filter batches efficiently

### pipeline/build_bars_5m.py
- **Lines 60-62**: Proper idempotent DELETE + INSERT pattern
- **Lines 84-107**: Deterministic aggregation with proper tiebreak (cnt DESC, source_symbol ASC)
- **Lines 114-146**: Comprehensive integrity verification

### pipeline/build_daily_features.py
- **Lines 620-665**: Careful no-look-ahead post-pass for ATR, gap, compression, GARCH
- **Line 77-84**: DST classification validation at module load time (fail-closed)
- **Lines 350-361**: Proper Wilder's RSI implementation

---

## 2. FINDINGS

### CRITICAL

#### 1. Ingest: Batch filtering order causes ineffective fast-forward
**File:** `pipeline/ingest_dbn.py`  
**Lines:** 287-294

```python
for chunk_df in iter_dbn_chunks(dbn_path, args.batch_size, start_filter, end_filter):
    batch_num += 1

    chunk_df = chunk_df.reset_index()

    # FAST-FORWARD: Skip entire batches before start_filter
    batch_max_date = chunk_df['ts_event'].max().date()
    if batch_max_date < start_filter:
        skipped_batches += 1
        ...
        continue
```

**Problem:** The fast-forward check computes `batch_max_date` from the **entire** chunk before filtering. If `start_filter` is 2024-06-01 but a batch contains rows from 2024-05-30-2024-06-02, `batch_max_date` will be 2024-06-02 (which passes the filter), but the batch still contains 90% pre-filter data. The subsequent `trading_day` filtering (lines 344-347) handles this, but the fast-forward optimization is ineffective.

**Impact:** Wasted processing of irrelevant rows; performance degradation on large backfills.

**Fix:** Filter `chunk_df` by `start_filter` before computing `batch_max_date`:
```python
# Filter to date range first
chunk_df = chunk_df[chunk_df['ts_event'] >= pd.Timestamp(start_filter)]
if len(chunk_df) == 0:
    continue

batch_max_date = chunk_df['ts_event'].max().date()
```

---

### HIGH

#### 2. Daily Features: Relative volume warm-up bias
**File:** `pipeline/build_daily_features.py`  
**Lines:** 708-716

```python
for i in range(len(rows)):
    bvol = rows[i].get(bvol_col)
    if bvol is not None:
        # Compute rel_vol BEFORE appending today (no look-ahead)
        if len(prior_break_vols) >= 5:
            window = prior_break_vols[-20:]
            median_vol = float(sorted(window)[len(window) // 2])
            if median_vol > 0:
                rows[i][rel_col] = round(bvol / median_vol, 4)
        prior_break_vols.append(bvol)
```

**Problem:** On the **first** day with a break (i.e., `i` is the first break occurrence), `prior_break_vols` is empty, so `rel_vol` is never computed. This creates a systematic gap in the first N days where breaks occurred, but `gap_open_points`, `atr_20`, etc. are computed (with partial data) from day 1. The relative volume feature has a longer warm-up than other features.

**Impact:** Feature inconsistency; backtesting will have fewer valid samples for `rel_vol_*` vs other features.

**Fix:** Add comment documenting the warm-up difference, or compute a simpler initial value (e.g., 1.0) for days 2-5:
```python
if len(prior_break_vols) >= 5:
    # ... existing median logic
elif len(prior_break_vols) > 0 and bvol > 0:
    # Initial approximation during warm-up
    rows[i][rel_col] = 1.0
prior_break_vols.append(bvol)
```

---

#### 3. 5m Bars: Inconsistent index handling between paths
**File:** `pipeline/build_bars_5m.py`  
**Lines:** 74-76 vs 124-132

```python
# In build_5m_bars (line 74-76):
chunk_df = pd.DataFrame(
    [(r[0], symbol, r[1], r[2], r[3], r[4], r[5], r[6]) for r in chunk_rows],
    columns=['ts_utc', 'symbol', 'source_symbol', 'open', 'high', 'low', 'close', 'volume'],
)

# In dry-run (line 124-132):
count_result = con.execute("""
    SELECT COUNT(DISTINCT (EXTRACT(EPOCH FROM ts_utc)::BIGINT / 300 * 300))
    ...
""", [symbol, start_ts, end_ts]).fetchone()[0]
```

**Problem:** The dry-run count uses a manual epoch calculation while the actual build uses `time_bucket(INTERVAL '5 minutes', ts_utc)`. If DuckDB's `time_bucket` behavior differs from manual epoch integer division (e.g., edge cases around DST transitions), the dry-run count could mismatch the actual insert count.

**Impact:** Dry-run could report incorrect row counts; user may think something is wrong when it isn't (or vice versa).

**Fix:** Use consistent bucket logic:
```python
# In dry-run count query, use time_bucket for consistency
count_result = con.execute("""
    SELECT COUNT(DISTINCT time_bucket(INTERVAL '5 minutes', ts_utc))
    FROM bars_1m
    ...
""", [symbol, start_ts, end_ts]).fetchone()[0]
```

---

### MEDIUM

#### 4. Daily Features: Silent GARCH failure
**File:** `pipeline/build_daily_features.py`  
**Lines:** 639-644

```python
garch_vol = compute_garch_forecast(prior_closes)
if garch_vol is not None:
    rows[i]["garch_forecast_vol"] = garch_vol
    ...
```

**Problem:** `compute_garch_forecast` catches all exceptions silently and returns `None`. If the GARCH model fails repeatedly (e.g., numerical instability), there's no visibility into why.

**Impact:** Silent feature degradation; difficult to diagnose during research.

**Fix:** Add logging for GARCH failures:
```python
garch_vol = compute_garch_forecast(prior_closes)
if garch_vol is None and len(prior_closes) >= 252:
    logger.debug(f"GARCH failed to converge on {rows[i]['trading_day']}")
```

---

#### 5. Ingest: Potential timezone confusion in filename parsing
**File:** `pipeline/ingest_dbn.py`  
**Lines:** 47-53

```python
def _parse_dbn_filename_date(filepath: Path) -> date | None:
    """Extract date from Databento filename like glbx-mdp3-20210204.ohlcv-1m.dbn.zst."""
    m = re.search(r'(\d{8})', filepath.name)
    if m:
        try:
            return date(int(m.group(1)[:4]), int(m.group(1)[4:6]), int(m.group(1)[6:8]))
```

**Problem:** The docstring shows `20210204` format, which is YYYYMMDD. This is a calendar date in **UTC** terms from the DBN file metadata, but subsequent date filtering (lines 78-87) compares against `start_filter`/`end_filter` which are in **Brisbane local time** (from CLI args). The timezone is implicitly UTC for DBN data but the user's CLI dates may be interpreted as Brisbane dates.

**Impact:** If a user runs `--start 2024-01-01` meaning Brisbane calendar date, but the DBN file dates are UTC, there's a potential 1-day off-by-one depending on when the UTC-to-Brisbane boundary falls.

**Fix:** Document timezone expectation clearly in CLI help, or convert CLI dates to UTC for filtering:
```python
# In main():
start_filter = date.fromisoformat(args.start) if args.start else minimum_start_date
# Document: CLI dates are interpreted as Brisbane local time
```

---

#### 6. Daily Features: RSI lookback may be insufficient for first trading day
**File:** `pipeline/build_daily_features.py`  
**Lines:** 506-518

```python
# RSI needs up to 200 bars BEFORE the first trading day's 09:00, so extend lookback
rsi_lookback_start = range_start_utc - timedelta(days=10)  # ~200 5m bars ≈ ~3.5 days
```

**Problem:** The comment approximates ~200 bars with 10 days, but market hours vary. On low-liquidity days (weekends, holidays) there could be fewer than 200 5m bars in 10 days. If the first trading day is a Monday after a holiday, there may be insufficient 5m bars.

**Impact:** RSI may be NULL on first trading day even when data exists.

**Fix:** Use explicit bar count rather than day count, or query with LIMIT:
```python
# Query last N bars before range_start instead of date-based lookback
all_bars_5m_df = con.execute("""
    SELECT ts_utc, close
    FROM bars_5m
    WHERE symbol = ?
    AND ts_utc < ?::TIMESTAMPTZ
    ORDER BY ts_utc DESC
    LIMIT 300  -- Buffer for first-day RSI
""", [symbol, range_end_utc.isoformat()]).fetchdf()
```

---

### LOW

#### 7. Daily Features: Double-break computation iterates full window for each ORB
**File:** `pipeline/build_daily_features.py`  
**Lines:** 561-577

```python
for label in ORB_LABELS:
    # ... ORB range and break detection ...
    
    # Double-break detection (regime signal)
    row[f"orb_{label}_double_break"] = detect_double_break(
        bars_df, trading_day, label, orb_minutes,
        orb["high"], orb["low"],
    )
```

**Problem:** `detect_double_break` re-filters `bars_df` for the break detection window, which overlaps with the window already used in `detect_break`. This causes duplicate filtering work for each ORB label.

**Impact:** Minor performance impact; O(n * m) where n=ORB labels, m=bars in trading day.

**Fix:** Compute break detection window once and pass to both functions, or cache filtered window.

---

#### 8. 5m Bars: Missing source_symbol validation
**File:** `pipeline/build_bars_5m.py`  
**Lines:** 114-146 (verify_5m_integrity)

**Problem:** The integrity check verifies duplicates, alignment, OHLCV sanity, and volume, but doesn't verify that `source_symbol` is not NULL. The build query uses a JOIN with `source_mode`, which could theoretically produce NULLs if something goes wrong.

**Impact:** Potential silent data quality issue.

**Fix:** Add NULL check for source_symbol in verification:
```python
null_source = con.execute("""
    SELECT COUNT(*) FROM bars_5m
    WHERE symbol = ? AND ts_utc >= ? AND ts_utc < ?
    AND source_symbol IS NULL
""", [symbol, start_ts, end_ts]).fetchone()[0]
if null_source > 0:
    failures.append(f"NULL source_symbol: {null_source}")
```

---

## 3. RECOMMENDATIONS

| Priority | Issue | Fix Complexity |
|----------|-------|-----------------|
| 1 | Batch fast-forward ineffective | Low |
| 2 | 5m bars dry-run vs actual mismatch | Low |
| 3 | Relative volume warm-up bias | Medium |
| 4 | RSI insufficient lookback | Medium |
| 5 | GARCH silent failures | Low |
| 6 | DBN filename date timezone | Medium |
| 7 | Double-break redundant filtering | Low |
| 8 | Missing source_symbol validation | Low |

**Recommended immediate fixes (highest impact):**
1. Fix the batch fast-forward in ingest_dbn.py (issue #1) - significant performance gain for large backfills
2. Align dry-run vs actual bucket logic in build_bars_5m.py (issue #3) - correctness concern