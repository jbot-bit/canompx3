# PASS 1 — AUDIT REPORT

**Date:** 2026-02-05
**Auditor:** Claude (MPX Code Guardian – Backfill Specialist)
**Status:** AUDIT COMPLETE — WAITING FOR PASS 2 APPROVAL

---

## 1. REPO STATE IDENTIFICATION

| Item | Status |
|------|--------|
| Git repo | ❌ Not a git repo |
| Branch | N/A |
| Dirty/Clean | N/A |
| DATABASE_SCHEMA_SOURCE_OF_TRUTH.md | ❌ Not present |
| CLAUDE.md | ✅ Present (31KB) |
| CANONICAL_backfill_dbn_mgc_rules.txt | ✅ Present |
| CANONICAL_backfill_dbn_mgc_rules_addon.txt | ✅ Present |

**Relevant Files Found:**
- `OHLCV_MGC_FULL/ingest_dbn_mgc.py` — Current ingest script (UNTRUSTED)
- `pipeline/init_db.py` — Database initialization (created this session)
- `pipeline/paths.py` — Path constants (created this session)
- `pipeline/check_db.py` — DB inspection tool (created this session)
- `gold.db` — Empty DuckDB with bars_1m/bars_5m tables (created this session)
- `OHLCV_MGC_FULL/glbx-mdp3-20100912-20260203.ohlcv-1m.dbn.zst` — Source data (73MB compressed)

---

## 2. SOURCE DATA VERIFIED

| Property | Value |
|----------|-------|
| File | glbx-mdp3-20100912-20260203.ohlcv-1m.dbn.zst |
| Size | 73 MB compressed |
| Total records | 6,401,788 |
| Date range | 2010-09-12 to 2026-02-03 |
| Schema | ohlcv-1m ✅ |
| Timestamp format | datetime64[ns, UTC] ✅ Already UTC |
| Unique symbols | 543 (60 outrights, 483 spreads) |
| Outright pattern | `^MGC[FGHJKMNQUVXZ]\d{1,2}$` ✅ Matches all 60 |

**Chunked Iterator Verified:**
- `store.to_df(count=50000)` returns iterator ✅
- Can process in 50K row chunks without RAM explosion ✅

---

## 3. CURRENT SCRIPT VIOLATIONS

### Auditing: `OHLCV_MGC_FULL/ingest_dbn_mgc.py`

| Line | Violation | Rule Reference | Severity |
|------|-----------|----------------|----------|
| 253 | `df = store.to_df()` — Loads full 6.4M rows into RAM | CANONICAL_rules L176: "NO store.to_df() over full history" | 🔴 CRITICAL |
| 281-284 | `df.apply(lambda...)` — Row-wise apply over millions | CANONICAL_rules L179: "NO row-wise pandas apply" | 🔴 CRITICAL |
| 339 | `for _, row in front_df.iterrows()` — Row-wise iteration | CANONICAL_rules L179: "NO iterrows over full dataset" | 🔴 CRITICAL |
| 420-447 | Builds bars_5m after ingestion | CANONICAL_rules L29: "explicitly forbidden: bars_5m" | 🔴 CRITICAL |
| 159 | `max(outrights, key=...)` — Non-deterministic tiebreak | CANONICAL_rules L95-103: Deterministic tiebreak required | 🟡 HIGH |
| N/A | No checkpoint system | CANONICAL_rules L128-148: Checkpoint MANDATORY | 🟡 HIGH |
| N/A | No integrity gates after merge | CANONICAL_rules L169-173: Assert no duplicates | 🟡 HIGH |
| N/A | No PK safety assertion before merge | CANONICAL_rules L89-93: Assert unique ts_utc | 🟡 HIGH |
| N/A | No staging artifacts (Parquet) | ADDON L57-59: Parquet staging recommended | 🟠 MEDIUM |
| N/A | Validation aborts single row, not entire backfill | CANONICAL_rules L113-116: Abort ENTIRE backfill | 🟡 HIGH |

### Summary of Violations:
- **CRITICAL:** 4
- **HIGH:** 5
- **MEDIUM:** 1

---

## 4. COMPLIANCE CHECKLIST

### From CANONICAL_backfill_dbn_mgc_rules.txt:

| Requirement | Current Status | Action Needed |
|-------------|----------------|---------------|
| Chunked reads (no full .to_df()) | ❌ VIOLATED | Use `store.to_df(count=N)` iterator |
| Vectorized trading day calc | ❌ VIOLATED | Replace apply() with numpy.where |
| Vectorized validation | ❌ VIOLATED | Replace iterrows with boolean masks |
| No bars_5m generation | ❌ VIOLATED | Remove lines 420-447 |
| Deterministic tiebreak | ❌ VIOLATED | Add expiry parsing + lexicographic fallback |
| Checkpoint system (JSONL/SQLite) | ❌ MISSING | Implement append-only checkpoint |
| Integrity gates after merge | ❌ MISSING | Add duplicate/NULL checks |
| PK safety assertion | ❌ MISSING | Assert unique ts_utc per trading day |
| Fail-closed on invalid row | ❌ PARTIAL | Currently skips bad rows, should ABORT |
| Logging (contract per day, ties) | ❌ PARTIAL | Needs tie logging |
| Final honesty gates | ❌ MISSING | Add post-backfill verification |

### From CANONICAL_backfill_dbn_mgc_rules_addon.txt:

| Requirement | Current Status | Action Needed |
|-------------|----------------|---------------|
| Chunked iterator | ❌ VIOLATED | Already flagged above |
| Vectorized operations | ❌ VIOLATED | Already flagged above |
| Parquet staging artifacts | ❌ MISSING | Optional but recommended |
| Parallel workers → artifacts | ❌ MISSING | Optional for v1 |
| Main merge step only | ❌ VIOLATED | Current script writes directly |
| Bulk loads (not row inserts) | ✅ COMPLIANT | Uses executemany |
| Idempotence via INSERT OR REPLACE | ✅ COMPLIANT | Already uses this |

---

## 5. PROPOSED ARCHITECTURE (NEW SCRIPT)

Based on both rule files, the new script should follow this pattern:

```
┌─────────────────────────────────────────────────────────────┐
│                    ingest_dbn_mgc.py (v2)                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. EXTRACT (chunked)                                       │
│     └─ store.to_df(count=50000) → iterator                  │
│                                                             │
│  2. TRANSFORM (vectorized, per chunk)                       │
│     ├─ Filter outrights (boolean mask)                      │
│     ├─ Calculate trading_day (numpy.where)                  │
│     ├─ Validate OHLCV (vectorized boolean masks)            │
│     └─ If ANY invalid → ABORT ENTIRE BACKFILL               │
│                                                             │
│  3. AGGREGATE (per trading day)                             │
│     ├─ Sum volume per contract                              │
│     ├─ Select front-month (deterministic tiebreak)          │
│     └─ Assert unique ts_utc (PK safety)                     │
│                                                             │
│  4. STAGE (optional but recommended)                        │
│     └─ Write chunk to temp Parquet artifact                 │
│                                                             │
│  5. MERGE (per chunk, transactional)                        │
│     ├─ BEGIN                                                │
│     ├─ INSERT OR REPLACE INTO bars_1m                       │
│     ├─ Integrity gates (no duplicates, no NULL)             │
│     ├─ COMMIT (or ROLLBACK on failure)                      │
│     └─ Update checkpoint (append-only JSONL)                │
│                                                             │
│  6. FINAL GATES                                             │
│     ├─ Verify ts_utc type = TIMESTAMPTZ                     │
│     ├─ Verify no duplicate (symbol, ts_utc)                 │
│     ├─ Verify no NULL source_symbol                         │
│     └─ Exit 0 (success) or non-zero (failure)               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. FILE CHANGES REQUIRED

### DELETE:
- None (keep old script for reference in oldscripts/)

### MOVE:
- `OHLCV_MGC_FULL/ingest_dbn_mgc.py` → `oldscripts/ARCHIVE_ingest_dbn_mgc_v1.py`

### CREATE:
| File | Purpose |
|------|---------|
| `pipeline/ingest_dbn_mgc.py` | New compliant ingest script |
| `pipeline/checkpoints/` | Directory for checkpoint JSONL files |
| `pipeline/staging/` | Directory for temp Parquet artifacts (optional) |

### MODIFY:
- `pipeline/init_db.py` — Ensure bars_1m schema matches exactly (already correct)

---

## 7. CLI INTERFACE (PROPOSED)

```bash
# Full backfill (will take time)
python pipeline/ingest_dbn_mgc.py

# Date range backfill
python pipeline/ingest_dbn_mgc.py --start 2020-01-01 --end 2025-12-31

# Resume from checkpoint
python pipeline/ingest_dbn_mgc.py --resume

# Retry failed chunks
python pipeline/ingest_dbn_mgc.py --retry-failed

# Dry run (validate only)
python pipeline/ingest_dbn_mgc.py --dry-run

# Configure chunk size (trading days per commit)
python pipeline/ingest_dbn_mgc.py --chunk-days 7

# Configure row batch size for DBN reading
python pipeline/ingest_dbn_mgc.py --batch-size 50000
```

---

## 8. RISK ASSESSMENT

| Risk | Mitigation |
|------|------------|
| RAM explosion | Chunked iterator (50K rows max in memory) |
| Data corruption | Transactional commits per chunk + rollback |
| Non-deterministic results | Deterministic tiebreak + stable sort |
| Silent bad data | Fail-closed validation → abort on ANY invalid row |
| Lost progress | Checkpoint system → resume from last done chunk |
| Duplicate rows | PK safety + integrity gates |

---

## 9. PASS 1 CONCLUSION

**Current script `OHLCV_MGC_FULL/ingest_dbn_mgc.py` is NOT COMPLIANT.**

It violates 4 CRITICAL and 5 HIGH severity rules from the canonical spec.

**Running this script will:**
- Potentially crash from RAM exhaustion (6.4M rows × pandas overhead)
- Take hours due to row-wise operations
- Generate forbidden bars_5m table
- Produce non-deterministic results on contract ties
- Have no checkpoint/resume capability
- Skip bad rows instead of failing closed

---

## 10. NEXT STEP

**Awaiting user approval to proceed to PASS 2 (BUILD).**

When approved, I will:
1. Archive the current script
2. Create a new compliant `pipeline/ingest_dbn_mgc.py`
3. Implement checkpoint system
4. Test with `--dry-run` first
5. Run full backfill

---

**END OF PASS 1 AUDIT REPORT**
