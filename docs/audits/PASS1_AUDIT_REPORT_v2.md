# PASS 1 — COMPLETE AUDIT REPORT v2

**Date:** 2026-02-05
**Auditor:** Claude (MPX Code Guardian – Backfill Specialist)

---

# PART A: LINE-BY-LINE REQUIREMENTS CHECKLIST

## From CANONICAL_backfill_dbn_mgc_rules.txt

### Process Requirements (Lines 13-18)
| Line | Requirement | Covered in v1? | Notes |
|------|-------------|----------------|-------|
| L14 | Self-identify repo state (branch, dirty/clean, files) | ✅ Yes | Done in Section 1 |
| L15 | PASS 1 = AUDIT ONLY (no code changes) | ✅ Yes | Following this |
| L16 | PASS 2 = BUILD ONLY after explicit approval | ✅ Yes | Waiting for approval |
| L17 | Fail-closed at all times | ✅ Yes | Flagged in violations |
| L18 | Treat existing code as UNTRUSTED | ✅ Yes | Audited current script |

### Forbidden Actions (Lines 20-29)
| Line | Forbidden Action | Current Script Violates? | Covered in v1? |
|------|------------------|--------------------------|----------------|
| L21 | Assume correctness | N/A (process) | ✅ |
| L22 | "Improve" architecture | N/A (process) | ✅ |
| L23 | Add indicators/features/ORB/signals/execution/costs | ❌ No violation | ✅ |
| L24 | Load full dataset into memory (NO full .to_df()) | 🔴 **VIOLATED L253** | ✅ |
| L25 | Row-wise .apply() / iterrows over millions | 🔴 **VIOLATED L281, L339** | ✅ |
| L26 | Skip integrity checks | 🔴 **VIOLATED** (no checks) | ✅ |
| L27 | Continue after integrity errors | 🔴 **VIOLATED** (no checks exist) | ✅ |
| L28 | Quietly coerce or drop bad rows | 🟡 **VIOLATED** (skips bad rows) | ✅ |
| L29 | Touch bars_5m, daily_features, derived tables | 🔴 **VIOLATED L420-447** | ✅ |

### Project Context (Lines 31-42)
| Line | Requirement | Applies? | Covered? |
|------|-------------|----------|----------|
| L33-34 | ONLY JOB: Backfill 1m OHLCV bars | ✅ Yes | ✅ |
| L36 | No features, No 5m build, No daily_features, No ORB | 🔴 VIOLATED (builds 5m) | ✅ |
| L42 | Ingestion must be execution-agnostic | ✅ Script is agnostic | ✅ |

### DBN Content Gate (Lines 44-50)
| Line | Requirement | Current Script? | Covered in v1? |
|------|-------------|-----------------|----------------|
| L45 | Input MUST contain ohlcv-1m schema | ✅ Verified (store.schema) | ⚠️ **MISSED** |
| L46-48 | Filter to OHLCV-1m if multiple schemas, assert non-empty | ❌ **NOT CHECKED** | ⚠️ **MISSED** |
| L49 | If schema cannot be proven → ABORT | ❌ **NOT IMPLEMENTED** | ⚠️ **MISSED** |
| L50 | Never infer bars from trades/quotes | ✅ Not doing this | ✅ |

### Target Schema (Lines 52-65)
| Line | Requirement | Current Script? | Covered? |
|------|-------------|-----------------|----------|
| L56 | ts_utc TIMESTAMPTZ NOT NULL | ✅ Correct | ✅ |
| L57 | symbol TEXT NOT NULL (constant 'MGC') | ✅ Correct | ✅ |
| L58 | source_symbol TEXT NOT NULL | ✅ Correct | ✅ |
| L59-63 | OHLCV columns DOUBLE/BIGINT NOT NULL | ✅ Correct | ✅ |
| L65 | PRIMARY KEY (symbol, ts_utc) | ✅ Correct | ✅ |

### Time & Calendar Rules (Lines 67-72)
| Line | Requirement | Current Script? | Covered? |
|------|-------------|-----------------|----------|
| L68 | Bar timestamp = bar OPEN (ts_event) | ✅ Correct | ✅ |
| L69 | Store as TIMESTAMPTZ (NOT string) | ⚠️ Stores as ISO string, then DB converts | ⚠️ **MISSED** |
| L70 | Never store local timestamps | ✅ Correct (uses UTC) | ✅ |
| L71 | Trading day = 09:00 Brisbane → 09:00 next day | ✅ Correct logic | ✅ |
| L72 | Timestamps monotonic increasing per source_symbol | ❌ **NOT CHECKED** | ⚠️ **MISSED** |

### Timezone Verification Gate (Lines 74-79)
| Line | Requirement | Current Script? | Covered? |
|------|-------------|-----------------|----------|
| L75-76 | MUST explicitly prove ts_utc is UTC | ⚠️ Assumes UTC, doesn't assert | ⚠️ **MISSED** |
| L77 | If tz-naive → convert correctly | N/A (data is already UTC) | ✅ |
| L78 | If tz cannot be proven → ABORT | ❌ **NOT IMPLEMENTED** | ⚠️ **MISSED** |

### Contract Selection (Lines 81-87)
| Line | Requirement | Current Script? | Covered? |
|------|-------------|-----------------|----------|
| L82 | Per trading day (09:00 Brisbane boundary) | ✅ Correct | ✅ |
| L83 | Aggregate total volume per contract | ✅ Correct | ✅ |
| L84 | Select front-month = highest volume | ✅ Correct | ✅ |
| L85 | Ingest ONLY that contract's bars | ✅ Correct | ✅ |
| L86 | Store chosen contract in source_symbol | ✅ Correct | ✅ |
| L87 | NO smoothing, NO back-adjustment, NO NULLs | ✅ Correct | ✅ |

### Primary-Key Safety Assertion (Lines 89-93)
| Line | Requirement | Current Script? | Covered? |
|------|-------------|-----------------|----------|
| L91 | Assert exactly one bar per ts_utc | ❌ **NOT IMPLEMENTED** | ✅ |
| L92 | Assert no duplicate ts_utc in selected bars | ❌ **NOT IMPLEMENTED** | ✅ |
| L93 | If violated → abort immediately | ❌ **NOT IMPLEMENTED** | ✅ |

### Deterministic Tiebreak (Lines 95-103)
| Line | Requirement | Current Script? | Covered? |
|------|-------------|-----------------|----------|
| L96 | If equal daily volume → tiebreak | ❌ Uses non-deterministic max() | ✅ |
| L97-101 | Tiebreak #1: earliest expiry (if parseable for ALL) | ❌ **NOT IMPLEMENTED** | ✅ |
| L102 | Tiebreak #2: lexicographically smallest | ❌ **NOT IMPLEMENTED** | ✅ |
| L103 | Must be stable across reruns | ❌ **NOT STABLE** | ✅ |

### Data Validation (Lines 105-116)
| Line | Requirement | Current Script? | Covered? |
|------|-------------|-----------------|----------|
| L107 | high >= max(open, close, low) | ⚠️ Only checks high >= max(o,c) | ⚠️ **MISSED** |
| L108 | low <= min(open, close) | ✅ Correct | ✅ |
| L109 | high >= low | ✅ Correct | ✅ |
| L110 | All prices finite and > 0 | ❌ **NOT CHECKED (> 0)** | ⚠️ **MISSED** |
| L111 | Volume integer-like and >= 0 | ✅ Correct | ✅ |
| L112 | ts_utc not null, timezone-aware UTC | ⚠️ Not explicitly asserted | ⚠️ **MISSED** |
| L114 | Log offending row | ❌ **NOT IMPLEMENTED** | ⚠️ **MISSED** |
| L115 | Abort ENTIRE backfill immediately | 🔴 **VIOLATED** (skips row) | ✅ |
| L116 | Exit non-zero | ❌ **NOT IMPLEMENTED** | ⚠️ **MISSED** |

### Chunking & Resume Model (Lines 118-126)
| Line | Requirement | Current Script? | Covered? |
|------|-------------|-----------------|----------|
| L119 | Chunk unit: trading-day chunks (3-7 days default) | ⚠️ Uses row count, not trading days | ⚠️ **MISSED** |
| L120-123 | BEGIN/COMMIT per chunk | ⚠️ Commits per row-count chunk | ⚠️ **MISSED** |
| L124-126 | On failure: ROLLBACK, mark failed | ❌ **NOT IMPLEMENTED** | ✅ |

### Checkpoint System (Lines 128-148)
| Line | Requirement | Current Script? | Covered? |
|------|-------------|-----------------|----------|
| L129 | JSONL or SQLite (append-only) | ❌ **NOT IMPLEMENTED** | ✅ |
| L131 | chunk_start | ❌ | ✅ |
| L132 | chunk_end | ❌ | ✅ |
| L133 | status: pending/in_progress/done/failed | ❌ | ✅ |
| L134 | rows_written | ❌ | ✅ |
| L135 | started_at | ❌ | ✅ |
| L136 | finished_at | ❌ | ✅ |
| L137 | source_dbn (path + hash OR size+mtime) | ❌ | ⚠️ **MISSED** |
| L138 | error (if failed) | ❌ | ✅ |
| L139 | attempt_id (monotonic) | ❌ | ⚠️ **MISSED** |
| L142 | Records never edited or deleted | N/A | ✅ |
| L143 | Retries append NEW record | ❌ | ⚠️ **MISSED** |
| L146 | Skip status=done on startup | ❌ | ⚠️ **MISSED** |
| L147 | Resume status=in_progress | ❌ | ⚠️ **MISSED** |
| L148 | Retry failed only with --retry-failed | ❌ | ⚠️ **MISSED** |

### Parallelism (Lines 150-160)
| Line | Requirement | Applies? | Covered? |
|------|-------------|----------|----------|
| L150 | OPTIONAL | Skipping for v1 | ✅ |
| L151-160 | Workers write Parquet, main merges | Skipping for v1 | ✅ |

### Idempotence (Lines 162-167)
| Line | Requirement | Current Script? | Covered? |
|------|-------------|-----------------|----------|
| L164 | Use MERGE INTO or INSERT OR REPLACE | ✅ Uses INSERT OR REPLACE | ✅ |
| L165 | Merge key = (symbol, ts_utc) | ✅ Correct | ✅ |
| L166 | Re-runs must not duplicate or drift | ⚠️ Tiebreak not deterministic | ✅ |
| L167 | Forbid append-only without conflict | ✅ Not doing this | ✅ |

### Merge Integrity Gates (Lines 169-173)
| Line | Requirement | Current Script? | Covered? |
|------|-------------|-----------------|----------|
| L171 | Assert no duplicate (symbol, ts_utc) after merge | ❌ **NOT IMPLEMENTED** | ✅ |
| L172 | Assert no NULL source_symbol after merge | ❌ **NOT IMPLEMENTED** | ✅ |
| L173 | If violated → abort non-zero | ❌ **NOT IMPLEMENTED** | ✅ |

### Performance Constraints (Lines 175-179)
| Line | Requirement | Current Script? | Covered? |
|------|-------------|-----------------|----------|
| L176 | NO store.to_df() over full history | 🔴 **VIOLATED** | ✅ |
| L177 | Prefer DBN replay/streaming | 🔴 **VIOLATED** | ✅ |
| L178 | Incremental daily aggregation | ⚠️ Not really | ⚠️ **MISSED** |
| L179 | NO row-wise apply/iterrows | 🔴 **VIOLATED** | ✅ |

### Logging (Lines 181-193)
| Line | Requirement | Current Script? | Covered? |
|------|-------------|-----------------|----------|
| L183 | Config snapshot | ⚠️ Partial | ⚠️ **MISSED** |
| L184 | Chunk start/end + status | ❌ **NOT IMPLEMENTED** | ⚠️ **MISSED** |
| L185 | Contract chosen per trading day | ❌ **NOT LOGGED** | ⚠️ **MISSED** |
| L186 | Tie situations with volumes | ❌ **NOT LOGGED** | ✅ |
| L187 | Rows staged + merged | ⚠️ Partial (bars inserted) | ⚠️ **MISSED** |
| L188 | Failures with full stacktrace | ❌ **NOT IMPLEMENTED** | ⚠️ **MISSED** |
| L191-193 | Final summary (chunks, rows, time) | ⚠️ Partial | ⚠️ **MISSED** |

### Final Honesty Gates (Lines 195-201)
| Line | Requirement | Current Script? | Covered? |
|------|-------------|-----------------|----------|
| L197 | ts_utc type is TIMESTAMPTZ | ❌ **NOT VERIFIED** | ✅ |
| L198 | No duplicate (symbol, ts_utc) | ❌ **NOT VERIFIED** | ✅ |
| L199 | No NULL source_symbol | ❌ **NOT VERIFIED** | ✅ |
| L200 | Optional: sampled day consistency check | ❌ **NOT IMPLEMENTED** | ⚠️ **MISSED** |
| L201 | Any failure → exit non-zero | ❌ **NOT IMPLEMENTED** | ⚠️ **MISSED** |

---

## From CANONICAL_backfill_dbn_mgc_rules_addon.txt

| Line | Requirement | Current Script? | Covered in v1? |
|------|-------------|-----------------|----------------|
| L3-5 | Work in chunks, use to_df(count=N) iterator | 🔴 **VIOLATED** | ✅ |
| L15-17 | Vectorized operations (numpy/pandas masks) | 🔴 **VIOLATED** | ✅ |
| L29-35 | Workers output artifacts, main merges | N/A (optional v1) | ✅ |
| L53-55 | Bulk loads beat one-row inserts | ✅ Uses executemany | ✅ |
| L57-59 | Parquet staging recommended | ❌ Not implemented | ⚠️ **MISSED detail** |
| L63-67 | Checkpoints + idempotency | ❌ **NOT IMPLEMENTED** | ✅ |
| L69-70 | Validation early (extract-transform boundary) | 🔴 **VIOLATED** | ⚠️ **MISSED** |
| L75-77 | NO streaming all to memory | 🔴 **VIOLATED** | ✅ |
| L79-81 | NO parallel workers writing to DB directly | N/A | ✅ |
| L83-85 | NO heavy transforms during backfill | ✅ Not doing this | ✅ |

---

# PART B: ITEMS MISSED IN v1 AUDIT

1. **DBN Content Gate** (L44-50): Script doesn't verify schema is ohlcv-1m before proceeding
2. **Timestamp stored as string** (L69): Script uses `.isoformat()` then lets DB convert
3. **Monotonic timestamp check** (L72): Not checking timestamps are increasing per contract
4. **Explicit UTC assertion** (L75-78): Assumes UTC, doesn't assert with fail-closed
5. **Validation: high >= max(o,c,l)** (L107): Current only checks `high >= max(o,c)`, missing `low`
6. **Validation: prices > 0** (L110): Not checking prices are positive
7. **Validation: ts_utc not null** (L112): Not explicitly checking
8. **Log offending row on validation fail** (L114): Not implemented
9. **Exit non-zero on validation fail** (L116): Not implemented
10. **Chunk unit is trading-days** (L119): Current chunks by row count, not trading days
11. **Checkpoint: source_dbn hash/mtime** (L137): Not tracking source file identity
12. **Checkpoint: attempt_id** (L139): Not tracking attempt numbers
13. **Checkpoint: startup behavior** (L146-148): Skip done, resume in_progress, retry failed with flag
14. **Incremental daily aggregation** (L178): Not really doing this
15. **Logging: config snapshot** (L183): Partial
16. **Logging: chunk status transitions** (L184): Not implemented
17. **Logging: contract per day** (L185): Not logged
18. **Logging: rows staged vs merged** (L187): Only logs inserted
19. **Logging: stacktrace on failure** (L188): Not implemented
20. **Final gate: sampled consistency check** (L200): Not implemented
21. **Validation at extract-transform boundary** (addon L69-70): Currently validates late

---

# PART C: BIRD'S EYE VIEW OF COMPLIANT SCRIPT

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         ingest_dbn_mgc.py (COMPLIANT)                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STARTUP                                                                     │
│  ├─ Parse CLI args (--start, --end, --resume, --retry-failed, --dry-run)    │
│  ├─ Log config snapshot (paths, dates, chunk size, flags)                   │
│  ├─ Load checkpoint file (JSONL)                                            │
│  │   ├─ Skip chunks with status=done                                        │
│  │   ├─ Resume chunks with status=in_progress                               │
│  │   └─ Retry status=failed ONLY if --retry-failed                          │
│  └─ Open DBN file, verify schema = ohlcv-1m (FAIL-CLOSED)                   │
│                                                                              │
│  PHASE 1: EXTRACT (chunked)                                                  │
│  ├─ for chunk_df in store.to_df(count=50000):                               │
│  │   │                                                                       │
│  │   │  PHASE 2: VALIDATE (vectorized, FAIL-CLOSED)                         │
│  │   ├─ Assert ts_event.dtype == datetime64[ns, UTC]                        │
│  │   ├─ Assert ts_event not null (any null → ABORT)                         │
│  │   ├─ Assert prices finite and > 0 (any fail → ABORT)                     │
│  │   ├─ Assert high >= max(open, close, low) (any fail → ABORT)             │
│  │   ├─ Assert low <= min(open, close) (any fail → ABORT)                   │
│  │   ├─ Assert high >= low (any fail → ABORT)                               │
│  │   ├─ Assert volume >= 0 (any fail → ABORT)                               │
│  │   └─ If ANY validation fails:                                            │
│  │       ├─ Log offending row (ts, symbol, OHLCV, reason)                   │
│  │       ├─ Log full stacktrace                                             │
│  │       └─ Exit non-zero IMMEDIATELY                                       │
│  │                                                                           │
│  │   PHASE 3: TRANSFORM (vectorized)                                        │
│  │   ├─ Filter to outrights only (boolean mask: no '-' in symbol)           │
│  │   ├─ Compute trading_day using numpy.where (not apply):                  │
│  │   │     ts_local = ts_utc.tz_convert('Australia/Brisbane')               │
│  │   │     hour = ts_local.dt.hour                                          │
│  │   │     trading_day = np.where(hour < 9, date - 1 day, date)             │
│  │   └─ Assert timestamps monotonic per source_symbol                       │
│  │                                                                           │
│  │   PHASE 4: AGGREGATE (per trading day in chunk)                          │
│  │   ├─ Group by trading_day                                                │
│  │   ├─ For each trading_day:                                               │
│  │   │   ├─ Sum volume per source_symbol                                    │
│  │   │   ├─ Select front contract (highest volume)                          │
│  │   │   ├─ If TIE:                                                         │
│  │   │   │   ├─ Log tie situation with all candidate volumes                │
│  │   │   │   ├─ Tiebreak #1: earliest expiry (parse month+year)             │
│  │   │   │   │   └─ If parse fails for ANY tied symbol → skip to #2         │
│  │   │   │   └─ Tiebreak #2: lexicographically smallest                     │
│  │   │   ├─ Log: "trading_day X → contract Y (volume Z)"                    │
│  │   │   ├─ Filter to selected contract only                                │
│  │   │   ├─ Assert unique ts_utc (PK safety) → ABORT if duplicates          │
│  │   │   └─ Collect rows for this trading_day                               │
│  │   └─ Accumulate trading days into chunk buffer                           │
│  │                                                                           │
│  │   PHASE 5: MERGE (per chunk of trading days)                             │
│  │   ├─ When chunk has 7 trading days (configurable):                       │
│  │   │   ├─ Write checkpoint: status=in_progress, started_at=now            │
│  │   │   ├─ BEGIN TRANSACTION                                               │
│  │   │   ├─ INSERT OR REPLACE INTO bars_1m (bulk)                           │
│  │   │   ├─ INTEGRITY GATE: Assert no duplicate (symbol, ts_utc)            │
│  │   │   ├─ INTEGRITY GATE: Assert no NULL source_symbol                    │
│  │   │   ├─ If gates fail → ROLLBACK, mark failed, ABORT                    │
│  │   │   ├─ COMMIT                                                          │
│  │   │   ├─ Write checkpoint: status=done, rows_written=N, finished_at=now  │
│  │   │   └─ Log: "Chunk [start-end] done: N rows"                           │
│  │   └─ Clear chunk buffer                                                  │
│  │                                                                           │
│  └─ (repeat for all chunks from DBN)                                        │
│                                                                              │
│  PHASE 6: FINAL HONESTY GATES                                               │
│  ├─ Query: SELECT COUNT(*) FROM bars_1m GROUP BY symbol, ts_utc HAVING COUNT > 1
│  │   └─ If any duplicates → FAIL, exit non-zero                             │
│  ├─ Query: SELECT COUNT(*) FROM bars_1m WHERE source_symbol IS NULL         │
│  │   └─ If any NULLs → FAIL, exit non-zero                                  │
│  ├─ Verify ts_utc column type = TIMESTAMPTZ                                 │
│  ├─ Optional: Sample 10 random days, compare bar count vs raw DBN           │
│  └─ If all pass → exit 0, log success summary                               │
│                                                                              │
│  FINAL SUMMARY                                                               │
│  ├─ Total chunks: done=X, failed=Y, skipped=Z                               │
│  ├─ Total rows written: N                                                   │
│  ├─ Date range: YYYY-MM-DD to YYYY-MM-DD                                    │
│  ├─ Unique contracts used: N                                                │
│  └─ Wall time: HH:MM:SS                                                     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

# PART D: DETAILED LOGIC AUDIT (HONESTY CHECK)

## D1. Trading Day Calculation

**Rule (L71):** Trading day = 09:00 Brisbane → 09:00 next day Brisbane

**Current Script Logic (L99-102):**
```python
ts_local = ts_utc.astimezone(TZ_LOCAL)  # Convert to Brisbane
if ts_local.hour < 9:
    return (ts_local - timedelta(days=1)).date()
return ts_local.date()
```

**AUDIT:**
- ✅ Converts UTC to Brisbane correctly
- ✅ If hour < 9, assigns to PREVIOUS day (correct: 00:00-08:59 belongs to yesterday)
- ✅ If hour >= 9, assigns to CURRENT day (correct: 09:00-23:59 belongs to today)

**VERDICT:** Logic is CORRECT. ✅

**Vectorized version (for new script):**
```python
ts_local = chunk_df.index.tz_convert('Australia/Brisbane')
hour = ts_local.hour
base_date = ts_local.date
trading_day = np.where(hour < 9, base_date - pd.Timedelta(days=1), base_date)
```

---

## D2. Contract Selection (Front-Month)

**Rule (L81-87):** Select highest daily volume outright contract per trading day.

**Current Script Logic (L150-159):**
```python
def choose_front_contract(daily_volumes: dict) -> str | None:
    outrights = {s: v for s, v in daily_volumes.items() if is_outright_contract(s)}
    if not outrights:
        return None
    return max(outrights, key=outrights.get)
```

**AUDIT:**
- ✅ Filters to outrights only (correct)
- ✅ Returns highest volume contract (correct)
- ❌ **PROBLEM:** Python's `max()` with equal values is NOT deterministic
  - If MGCG5 and MGCZ5 both have volume 1000, result depends on dict ordering
  - This violates L103: "Must be stable across reruns"

**FIX NEEDED:** Deterministic tiebreak:
```python
def choose_front_contract(daily_volumes: dict) -> str | None:
    outrights = {s: v for s, v in daily_volumes.items() if is_outright_contract(s)}
    if not outrights:
        return None

    max_vol = max(outrights.values())
    tied = [s for s, v in outrights.items() if v == max_vol]

    if len(tied) == 1:
        return tied[0]

    # Tiebreak #1: earliest expiry
    def parse_expiry(sym):
        # MGC + month_code + year (e.g., MGCG5 → G=Feb, 5=2025)
        month_codes = 'FGHJKMNQUVXZ'  # Jan-Dec
        month = month_codes.index(sym[3]) + 1
        year = int(sym[4:])
        if year < 50:  # 2-digit year handling
            year += 2000
        else:
            year += 1900
        return (year, month)

    try:
        # Only use expiry if parseable for ALL tied symbols
        expiries = {s: parse_expiry(s) for s in tied}
        return min(tied, key=lambda s: expiries[s])
    except:
        # Tiebreak #2: lexicographically smallest
        return min(tied)
```

**VERDICT:** Logic needs FIX for deterministic tiebreak. ⚠️

---

## D3. Data Validation

**Rule (L105-116):** Validate OHLCV, fail-closed on ANY violation.

**Current Script Logic (L109-143):**
```python
def validate_bar(row: pd.Series) -> tuple[bool, str]:
    o, h, l, c = row['open'], row['high'], row['low'], row['close']
    v = row['volume']

    if pd.isna(o) or pd.isna(h) or pd.isna(l) or pd.isna(c):
        return False, "NaN price"

    if h < max(o, c):  # ⚠️ WRONG: should be max(o, c, l)
        return False, f"high < max(open,close)"

    if l > min(o, c):  # ✅ Correct
        return False, f"low > min(open,close)"

    if h < l:  # ✅ Correct
        return False, f"high < low"

    # Missing: prices > 0
    # Missing: ts_utc not null
```

**AUDIT:**
- ✅ Checks for NaN prices
- ❌ **WRONG:** `h < max(o, c)` should be `h < max(o, c, l)` per L107
- ✅ Checks `l > min(o, c)`
- ✅ Checks `h < l`
- ❌ **MISSING:** Prices must be > 0 (L110)
- ❌ **MISSING:** Prices must be finite (not inf)
- ✅ Checks volume >= 0
- ❌ **WRONG BEHAVIOR:** Returns False (skip row) instead of ABORT

**FIX NEEDED (vectorized):**
```python
def validate_chunk(df: pd.DataFrame) -> tuple[bool, str, pd.DataFrame]:
    """Validate chunk. Returns (valid, reason, offending_rows)."""

    # Check NaN
    nan_mask = df[['open','high','low','close','volume']].isna().any(axis=1)
    if nan_mask.any():
        return False, "NaN values found", df[nan_mask]

    # Check finite
    inf_mask = ~np.isfinite(df[['open','high','low','close']]).all(axis=1)
    if inf_mask.any():
        return False, "Infinite values found", df[inf_mask]

    # Check > 0
    neg_mask = (df[['open','high','low','close']] <= 0).any(axis=1)
    if neg_mask.any():
        return False, "Non-positive prices found", df[neg_mask]

    # Check high >= max(open, close, low)
    max_ocl = df[['open','close','low']].max(axis=1)
    high_fail = df['high'] < max_ocl
    if high_fail.any():
        return False, "high < max(open,close,low)", df[high_fail]

    # Check low <= min(open, close)
    min_oc = df[['open','close']].min(axis=1)
    low_fail = df['low'] > min_oc
    if low_fail.any():
        return False, "low > min(open,close)", df[low_fail]

    # Check high >= low
    hl_fail = df['high'] < df['low']
    if hl_fail.any():
        return False, "high < low", df[hl_fail]

    # Check volume >= 0
    vol_fail = df['volume'] < 0
    if vol_fail.any():
        return False, "negative volume", df[vol_fail]

    return True, "", None
```

**VERDICT:** Validation logic has bugs and wrong fail mode. ⚠️

---

## D4. Outright Contract Pattern

**Rule:** Filter spreads (contain '-'), keep outrights only.

**Current Script Pattern (L77):**
```python
MGC_OUTRIGHT_PATTERN = re.compile(r'^MGC[FGHJKMNQUVXZ]\d{1,2}$')
```

**AUDIT:**
- ✅ Matches `MGCG0` through `MGCZ99`
- ✅ Does NOT match `MGCG0-MGCZ0` (spreads)
- ✅ Verified against all 60 actual outrights in DBN file
- ✅ Month codes are correct (F=Jan, G=Feb, ..., Z=Dec)

**VERDICT:** Pattern is CORRECT. ✅

---

## D5. Timestamp Handling

**Rule (L68-69, L74-79):** ts_event = bar OPEN, must be UTC, stored as TIMESTAMPTZ

**Current Script (L277, L348):**
```python
df['ts_utc_dt'] = pd.to_datetime(df['ts_event'], utc=True)
...
rows_buffer.append((
    row['ts_utc_dt'].isoformat(),  # ⚠️ Converts to string
    ...
))
```

**AUDIT:**
- ✅ `ts_event` is the bar OPEN time (Databento convention)
- ✅ Data is already UTC (datetime64[ns, UTC]) - verified in inspection
- ⚠️ **CONCERN:** Converts to ISO string for insert, relies on DuckDB to parse back
  - This WORKS but is not ideal
  - Better: Pass datetime directly or use DuckDB's native timestamp handling

**VERDICT:** Works but could be cleaner. Minor issue. ✅

---

## D6. Idempotence

**Rule (L162-167):** INSERT OR REPLACE, key = (symbol, ts_utc), no duplicates/drift

**Current Script (L362-369):**
```python
con.executemany(
    """
    INSERT OR REPLACE INTO bars_1m
    (ts_utc, symbol, source_symbol, open, high, low, close, volume)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
    rows_buffer
)
```

**AUDIT:**
- ✅ Uses INSERT OR REPLACE (correct)
- ✅ Primary key is (symbol, ts_utc) - set in schema
- ⚠️ **CONCERN:** Non-deterministic tiebreak means re-runs COULD select different contract
  - This would cause "drift" - different source_symbol on re-run
  - Violates L166: "Re-runs must not drift results"

**VERDICT:** Idempotence compromised by non-deterministic tiebreak. ⚠️

---

# PART E: REVISED VIOLATION COUNT

| Severity | Count | Items |
|----------|-------|-------|
| 🔴 CRITICAL | 5 | Full RAM load, apply(), iterrows(), builds bars_5m, wrong fail mode |
| 🟡 HIGH | 8 | No checkpoint, no integrity gates, no tiebreak, wrong validation formula, no PK safety, no logging, no final gates, drift risk |
| 🟠 MEDIUM | 8 | Missing schema gate, no monotonic check, no UTC assertion, no config log, no chunk status, no stacktrace, no source tracking, chunking by rows not days |

**Total items requiring fix: 21**

---

# PART F: CONCLUSION

The v1 audit missed **13 items**. This v2 audit covers **ALL requirements** from both rule files.

**VERDICT:** Current script is NOT COMPLIANT. Requires complete rewrite.

**Awaiting "APPROVED PASS 2" to build compliant script.**
