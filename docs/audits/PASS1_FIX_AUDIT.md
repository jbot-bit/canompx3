# PASS 1 — FIX AUDIT REPORT

**Date:** 2026-02-06
**Scope:** Verify audit claims against repo. NO CODE CHANGES.
**Mode:** NO DRIFT — all changes must pass `check_drift.py`

---

## VERIFIED BUGS (must fix)

### V1. Hardcoded Windows absolute path
**File:** `pipeline/ingest_dbn_mgc.py:53`
**Snippet:**
```python
DBN_PATH = Path(r"C:\Users\sydne\OneDrive\Desktop\CANONICAL TRADING\OHLCV_MGC_FULL\glbx-mdp3-20100912-20260203.ohlcv-1m.dbn.zst")
```
**Evidence:** Only occurrence. `ingest_dbn.py` correctly uses `asset_configs.py` via `get_asset_config()`. The MGC-specific script still has the hardcoded fallback.
**Fix:** Replace with `PROJECT_ROOT / "OHLCV_MGC_FULL" / ...` (pattern already used at `ingest_dbn_mgc.py:46` where `PROJECT_ROOT = Path(__file__).parent.parent`). Keep as default, allow CLI `--dbn-path` override.

---

### V2. No requirements.txt or pyproject.toml
**Evidence:** Confirmed. Neither file exists. `pip freeze` shows 26 packages installed.
**Top-level imports actually used in pipeline/:**
- `duckdb==1.4.4`
- `databento==0.70.0`
- `pandas==2.3.3`
- `numpy==2.2.6`
- `pyarrow==23.0.0` (transitive dep of databento/pandas, but imported)
- `zstandard==0.25.0` (needed for .dbn.zst decompression)
**Fix:** Create minimal `requirements.txt` pinning these 6.

---

### V3. Connection leak on error paths (ingest_dbn.py)
**File:** `pipeline/ingest_dbn.py`
**Evidence:** `con` opened at line 159. `con.close()` only at lines 463 and 507. Seven `sys.exit(1)` calls between 159-464 that exit WITHOUT closing:
- Line 230 (timestamp validation)
- Line 254 (OHLCV validation)
- Line 291 (PK safety)
- Line 352 (integrity gate)
- Line 369 (merge exception)
- Line 420 (final chunk integrity)
- Line 436 (final chunk exception)

**Same pattern in** `pipeline/ingest_dbn_mgc.py`: `con` opened at line 539, multiple `sys.exit(1)` between 539-864 without `con.close()`:
- Line 615 (timestamp validation)
- Line 638 (OHLCV validation)
- Line 675 (PK safety)
- Line 746 (integrity gate)
- Line 765 (merge exception)
- Line 820 (final chunk integrity)
- Line 836 (final chunk exception)

**Impact on Windows:** DuckDB file lock not released. Next run may get "database locked".
**Fix:** Wrap main body in `try/finally` with `con.close()` in the `finally` block. Surgical — add one try/finally around the processing section, change `sys.exit(1)` to `return 1` within the try block, then `sys.exit(returncode)` after finally.

---

### V4. Connection leak on error path (build_bars_5m.py)
**File:** `pipeline/build_bars_5m.py`
**Evidence:** `con` opened at line 301. `con.close()` at lines 317 (integrity failure) and 337 (normal). But if `build_5m_bars()` raises an unexpected exception at line 305, `con` is NOT closed — the `except` at line 189 does `raise` which propagates up to `main()` where there's no catch.
**Fix:** Wrap `main()` body in `try/finally` with `con.close()`.

---

## FALSE / NOT-AN-ISSUE

### F1. "UTC vs local trading-day mismatch in build_bars_5m.py" — FALSE
**Claim:** Lines 52-53 use `00:00 UTC` boundaries, misses bars from Brisbane trading day.
**Evidence against:** `build_bars_5m.py` is a **UTC-to-UTC aggregation layer**. It takes 1m bars (stored in UTC) and buckets them into 5m bars (stored in UTC). It does NOT need to know about Brisbane trading days. The `--start/--end` args are **UTC calendar dates**, not trading dates.
- The DELETE window and INSERT window use identical `[start_date 00:00Z, end_date+1 00:00Z)` range — consistent.
- No data is "missed" — bars_1m rows outside the range simply aren't included in this build, and will be included when a wider range is specified.
- Brisbane trading day logic belongs in `daily_features` (not yet built), NOT in 5m aggregation.
**Verdict:** The code is correct. The audit claim applied daily_features logic to the wrong layer.

---

### F2. "Transaction atomicity gap in build_bars_5m.py" — FALSE
**Claim:** DELETE + INSERT not atomic, data lost if INSERT fails.
**Evidence against:** `build_bars_5m.py:149-192`:
```python
con.execute("BEGIN TRANSACTION")
try:
    con.execute("DELETE FROM bars_5m WHERE ...")
    con.execute("INSERT INTO bars_5m ...")
    con.execute("COMMIT")
except Exception as e:
    con.execute("ROLLBACK")    # <-- restores deleted rows
    raise
```
The ROLLBACK on exception restores all deleted rows. DuckDB transactions are ACID. This is textbook correct.
**Verdict:** No atomicity gap. The audit claim ignored the ROLLBACK.

---

### F3. ".iterrows() is 1000x slower bottleneck" — EXAGGERATED
**Claim:** `front_df.iterrows()` at `ingest_dbn.py:298` and `ingest_dbn_mgc.py:684` is a critical bottleneck.
**Evidence against:**
- `front_df` = bars for ONE trading day, ONE contract = ~1440 rows max (24h * 60min)
- `check_drift.py:104-109` **explicitly allows** `front_df.iterrows()` as an exception, with comment: "This is on already-filtered single-contract single-day data, not bulk"
- Over a full 5-year backfill (~1250 trading days): 1250 * 1440 = 1.8M total iterations, but in 1440-row chunks. Each chunk takes <10ms.
- The **real** bottleneck is I/O (DBN parsing, DB writes), not this loop.

**The `.apply()` calls are also acceptable:**
- `ingest_dbn.py:235` / `ingest_dbn_mgc.py:621`: regex filter on symbol column (~50K rows per batch). `check_drift.py:89` explicitly allows this pattern.
- `ingest_dbn_mgc.py:360`: lambda on `trading_days[mask]` (subset where hour < 9). `check_drift.py:119` explicitly allows this.

**Verdict:** Not a bottleneck. The drift checker already reviewed and approved these patterns. Replacing them risks introducing bugs for negligible gain.

---

## RISKS (not fixing now — rationale)

### R1. No CHECK constraints on schema
**Evidence:** `init_db.py:31-57` — no CHECK constraints. `high < low`, negative prices accepted at DB level.
**Rationale for deferral:** Python validation in `validate_chunk()` catches all OHLCV violations before INSERT. Adding CHECK constraints requires schema change. fix.txt says "No schema migrations unless explicitly allowed." Existing behavior is safe due to fail-closed validation gate.

### R2. No additional indexes beyond PK
**Evidence:** Only `PRIMARY KEY (symbol, ts_utc)` on both tables.
**Rationale for deferral:** Current data is 30K rows. PK already covers the most common query pattern `WHERE symbol = ? AND ts_utc >= ? AND ts_utc < ?`. DuckDB's columnar storage handles full scans efficiently at this scale. Indexes become necessary at 1M+ rows.

### R3. "0 rows = success" in build_bars_5m.py
**Evidence:** `build_bars_5m.py:65-67` returns 0 rows with exit code 0 when no source data.
**Rationale for deferral:** This is arguably correct behavior — "nothing to build" is not an error. The runner (`run_pipeline.py`) doesn't exist for bars_5m standalone use. Adding a warning is low-risk but out of scope for fix.txt.

### R4. Checkpoint file has no locking
**Evidence:** `ingest_dbn_mgc.py:176` — `open(file, 'a')` with no flock.
**Rationale for deferral:** Concurrent runs are not a supported use case. The checkpoint system is single-process by design. Documented risk, not a bug.

### R5. bars_5m.source_symbol allows NULL
**Evidence:** `init_db.py:49` — `source_symbol TEXT` (no NOT NULL) vs bars_1m which has `source_symbol TEXT NOT NULL`.
**Rationale for deferral:** Schema change. fix.txt says no schema migrations. The build_5m query always produces non-NULL source_symbol from bars_1m data.

---

## PROPOSED MINIMAL PATCH LIST

| Priority | File | Change | Risk |
|----------|------|--------|------|
| P1 | `pipeline/ingest_dbn_mgc.py:53` | Replace hardcoded path with `PROJECT_ROOT / ...` | Minimal — same path, portable form |
| P2 | `requirements.txt` (new file) | Pin 6 top-level deps | Zero — additive only |
| P3 | `pipeline/ingest_dbn.py` | Wrap con usage in try/finally for con.close() | Minimal — no logic change, only cleanup |
| P3 | `pipeline/ingest_dbn_mgc.py` | Same try/finally pattern | Minimal |
| P3 | `pipeline/build_bars_5m.py` | Same try/finally pattern | Minimal |

**NOT changing (per evidence):**
- build_bars_5m.py date window logic (F1 — already correct)
- Transaction handling in build_bars_5m.py (F2 — already correct)
- .iterrows() / .apply() patterns (F3 — already approved by check_drift.py)
- Schema constraints (R1, R5 — out of scope per fix.txt)
- Indexes (R2 — premature at current scale)

---

## GATES TO RUN AFTER PASS 2

```bash
python pipeline/check_drift.py
python pipeline/ingest_dbn_mgc.py --dry-run --start 2024-01-01 --end 2024-01-07
python pipeline/ingest_dbn.py --instrument MGC --dry-run --start 2024-01-01 --end 2024-01-07
python pipeline/build_bars_5m.py --instrument MGC --start 2024-01-01 --end 2024-01-31
python pipeline/check_db.py
```

Plus correctness query:
```sql
SELECT COUNT(*) FROM bars_5m WHERE EXTRACT(MINUTE FROM ts_utc)::INT % 5 != 0;
-- Expected: 0
```

---

**PASS 1 COMPLETE. Awaiting "APPROVE PASS 2" to implement patches.**
