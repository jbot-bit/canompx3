# DB Performance Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement Items 1-5 from DB_PERFORMANCE_PLAN.md — PRAGMA tuning, DataFrame inserts, batched queries, Parquet export, scratch-run wrapper.

**Architecture:** Centralized `db_config.py` configures all DuckDB connections. Writers switch from `executemany()` to `INSERT FROM df`. Daily features bulk-loads bars instead of per-day queries. Parquet exports decouple reads from gold.db. Scratch-run script wraps long jobs.

**Tech Stack:** DuckDB 1.4.4, pandas, Python 3.12

---

### Task 1: Create `pipeline/db_config.py`

**Files:**
- Create: `pipeline/db_config.py`

**Step 1: Create the module**

```python
"""Standard DuckDB connection tuning. Call immediately after connect()."""

def configure_connection(con, *, writing: bool = False):
    con.execute("SET memory_limit = '8GB'")
    con.execute("SET temp_directory = 'C:/db/.tmp'")
    if writing:
        con.execute("SET preserve_insertion_order = false")
```

**Step 2: Verify import works**

Run: `python -c "from pipeline.db_config import configure_connection; print('OK')"`

---

### Task 2: Write test for `db_config`

**Files:**
- Create: `tests/test_pipeline/test_db_config.py`

Test that configure_connection sets PRAGMAs on a fresh in-memory DuckDB connection.

---

### Task 3: Wire `db_config` into pipeline files

**Files:**
- Modify: `pipeline/ingest_dbn.py:157` — after `duckdb.connect()`
- Modify: `pipeline/ingest_dbn_daily.py:262` — after `duckdb.connect()`
- Modify: `pipeline/build_bars_5m.py:303` — inside `with` block
- Modify: `pipeline/build_daily_features.py:969` — inside `with` block

Add `from pipeline.db_config import configure_connection` and call it after connect.

---

### Task 4: Wire `db_config` into trading_app files

**Files:**
- Modify: `trading_app/outcome_builder.py:526` — inside `with` block
- Modify: `trading_app/strategy_discovery.py:519` — inside `with` block
- Modify: `trading_app/strategy_validator.py:376` — inside `with` block

---

### Task 5: Add drift check 27 — unconfigured connections

**Files:**
- Modify: `pipeline/check_drift.py` — add `check_db_config_usage()` and wire into `main()`

Scan pipeline/ and trading_app/ for `duckdb.connect(` calls. Flag any file that connects but doesn't import/call `configure_connection`.

---

### Task 6: Run tests + commit Item 1

Run: `python -m pytest tests/ -x -q`
Commit: "perf: add PRAGMA tuning module (db_config.py) and wire into all DB consumers"

---

### Task 7: Replace `executemany()` in `ingest_dbn.py`

**Files:**
- Modify: `pipeline/ingest_dbn.py:345,414` — convert tuple list to DataFrame, use `INSERT FROM df`

---

### Task 8: Replace `executemany()` in `outcome_builder.py`

**Files:**
- Modify: `trading_app/outcome_builder.py:680` — convert day_batch list to DataFrame, use `INSERT FROM df`

---

### Task 9: Replace `executemany()` in `strategy_discovery.py`

**Files:**
- Modify: `trading_app/strategy_discovery.py:651-656` — convert insert_batch to DataFrame, use `INSERT FROM df`

---

### Task 10: Run tests + commit Item 2

Run: `python -m pytest tests/ -x -q`
Commit: "perf: replace executemany() with INSERT FROM DataFrame (5-10x faster inserts)"

---

### Task 11: Batch bars_1m queries in `build_daily_features.py`

**Files:**
- Modify: `pipeline/build_daily_features.py:156-180,635-679,751`

Bulk-load all bars_1m for the date range in one query, then slice per trading day in Python using sorted index.

---

### Task 12: Batch RSI queries in `build_daily_features.py`

**Files:**
- Modify: `pipeline/build_daily_features.py:453-490`

Bulk-load bars_5m for the date range, compute RSI per day from the in-memory array instead of one query per day.

---

### Task 13: Run tests + commit Item 3

Run: `python -m pytest tests/ -x -q`
Commit: "perf: batch daily_features queries (1500 queries -> ~2)"

---

### Task 14: Create `pipeline/export_parquet.py`

**Files:**
- Create: `pipeline/export_parquet.py`

Export tables to Parquet using DuckDB's `COPY ... TO ... (FORMAT PARQUET)`.

---

### Task 15: Wire Parquet export into pipeline + test

**Files:**
- Create: `tests/test_pipeline/test_export_parquet.py`

---

### Task 16: Commit Item 4

Commit: "feat: add Parquet export layer for read-independent analysis"

---

### Task 17: Create `scripts/infra/scratch_run.py`

**Files:**
- Create: `scripts/infra/scratch_run.py`

Copies gold.db to scratch, runs command, swaps back on success.

---

### Task 18: Commit Item 5

Commit: "feat: add scratch_run.py for safe long-running jobs"
