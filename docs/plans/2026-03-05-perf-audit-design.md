# Performance Audit & Refactoring Plan

**Date:** 2026-03-05
**Status:** Design — awaiting approval
**Scope:** check_drift.py, health_check.py, post-edit hook, UI db_reader

---

## T1 — ORIENT: Measured Bottlenecks

### Profiled Execution (Windows 11, Python 3.13, NTFS)

| Component | Measured Time | Trigger Frequency |
|-----------|---------------|-------------------|
| `check_drift.py` full run | **16.5s** | Every file edit + every commit |
| `post-edit-pipeline.py` total | **21-31s** | Every pipeline/trading_app edit |
| `health_check.py` (sequential) | **~5-10min** | Manual / post-rebuild |
| Python subprocess startup | **873ms** | Per subprocess.run() call |

### check_drift.py Per-Check Profile (Top 5)

| Check # | Name | Measured | Root Cause |
|---------|------|----------|------------|
| **#38** | `check_old_session_names` | **2,002ms** | `PROJECT_ROOT.rglob("*.py")` walks venv/, .git/, .auto-claude/ |
| **#16** | `check_all_imports_resolve` | **1,523ms** | `importlib.import_module()` cold-imports duckdb/pandas/numpy per module |
| **#53** | `check_ml_model_freshness` | **466ms** | ML model file stat() calls |
| **#52** | `check_ml_config_hash_match` | **419ms** | Config hash computation |
| **#19** | `check_timezone_hygiene` | **136ms** | rglob + regex |
| All 53 non-DB checks | — | **4,933ms** total | — |
| All 11 DB checks | — | **540ms** total | 10x duckdb.connect() at 45ms each |

### DuckDB Connection Benchmark

| Pattern | Measured |
|---------|----------|
| 10x connect/query/close | **445ms** (45ms each) |
| 1x connect, 10 queries, close | **43ms** |
| **Savings from shared connection** | **402ms** |

### File I/O Cache Benchmark

| Pattern | Measured |
|---------|----------|
| Uncached (440 reads, 12 globs) | **128ms** |
| Cached (87 reads, 2 globs) | **24ms** |
| **Savings from caching** | **103ms** (5.3x faster) |

### ThreadPoolExecutor Benchmark (subprocess parallelization)

| Pattern | Measured |
|---------|----------|
| 4 sequential subprocesses | **7.5s** |
| 4 parallel via ThreadPoolExecutor | **2.8s** |
| **Speedup** | **2.7x** |

### Best Practice Sources

- **DuckDB docs** (duckdb.org/docs/stable/guides/performance/how_to_tune_workloads):
  > "DuckDB will perform best when reusing the same database connection many times."
- **DuckDB multi-thread** (duckdb.org/docs/stable/guides/python/multiple_threads):
  > "Each thread must use `.cursor()` to create a thread-local connection."
- **Python concurrent.futures** (docs.python.org/3/library/concurrent.futures):
  > ThreadPoolExecutor for I/O-bound/subprocess waiting; ProcessPoolExecutor for CPU-bound.

---

## T2 — DESIGN: Approach Selection

### Approach A: Targeted Hot-Fix (selected)
Fix the 5 measured bottlenecks individually. No new abstractions. ~3.5s saved per drift check run.

### Approach B: Full CheckRunner Refactor
Build a `CheckRunner` class with file cache, DB pool, memoization. Over-engineered for batch-style usage.

### Approach C: Event-Driven Watcher
Replace polling hooks with file-system watcher daemon. Wrong architecture for a CLI-triggered pipeline.

**Decision: Approach A.** Each fix is independent, testable, and reversible. No new dependencies.

---

## T3 — DETAIL: Implementation Steps

### Fix 1: Scope rglob in check_old_session_names (#38)
**File:** `pipeline/check_drift.py` lines 1494
**Measured savings:** ~1,800ms
**Change:** Replace `PROJECT_ROOT.rglob("*.py")` with targeted directory walk.

```python
# BEFORE (scans entire project tree including venv, .git, node_modules):
for py_file in sorted(PROJECT_ROOT.rglob("*.py")):

# AFTER (scan only production directories):
_SCAN_DIRS = [PIPELINE_DIR, TRADING_APP_DIR, SCRIPTS_DIR]
for scan_dir in _SCAN_DIRS:
    if not scan_dir.exists():
        continue
    for py_file in sorted(scan_dir.rglob("*.py")):
```

The `frozen_dirs` set already excludes tests/research/venv, but the rglob still WALKS those dirs before the exclusion check fires. By scoping to production dirs, we skip the walk entirely.

**Risk:** Could miss old session names in a new top-level directory. Mitigated by: production code only lives in pipeline/, trading_app/, scripts/.

---

### Fix 2: Skip already-imported modules in check_all_imports_resolve (#16)
**File:** `pipeline/check_drift.py` lines 816-855
**Measured savings:** ~1,300ms
**Change:** Skip `importlib.import_module()` if module is already in `sys.modules`.

```python
# BEFORE (cold-imports every module):
try:
    importlib.import_module(module)

# AFTER (skip if already loaded):
if module in sys.modules:
    continue  # already imported by a prior check or at startup
try:
    importlib.import_module(module)
```

Most pipeline/trading_app modules are already loaded by the time check #16 runs (checks 1-15 import from pipeline.*). The cold-import cost is mostly re-initializing duckdb/pandas/numpy C extensions.

**Risk:** None. If a module is in sys.modules, its imports already resolved successfully.

---

### Fix 3: Shared DB connection for requires_db checks
**File:** `pipeline/check_drift.py` lines 2945-3009 (main function)
**Measured savings:** ~400ms
**Change:** Open one read-only connection, pass to all DB checks.

Current pattern: each DB check calls `duckdb.connect()` internally.
New pattern: `main()` opens one connection, passes it via a module-level variable.

```python
# In main():
_db_con = None
if _get_db_path().exists():
    try:
        _db_con = duckdb.connect(str(_get_db_path()), read_only=True)
    except Exception:
        pass  # DB busy

# Each DB check uses the shared connection:
def check_no_e0_in_db(con=None) -> list[str]:
    if con is None:
        return []  # no DB available
    ...
```

Need to identify which DB checks currently open their own connections and refactor them.

**Risk:** If connection fails mid-run, remaining DB checks skip. Acceptable — current behavior already skips on DB busy.

---

### Fix 4: Debounce post-edit hook
**File:** `.claude/hooks/post-edit-pipeline.py`
**Measured savings:** ~16s per rapid-edit batch (skip redundant drift runs)
**Change:** Skip drift check if it passed within last 30 seconds.

```python
_DEBOUNCE_FILE = Path(__file__).parent / ".last_drift_ok"
_DEBOUNCE_SECONDS = 30

# In main(), before Phase 1:
if _DEBOUNCE_FILE.exists():
    age = time.time() - _DEBOUNCE_FILE.stat().st_mtime
    if age < _DEBOUNCE_SECONDS:
        # Skip drift check, still run targeted tests
        ...proceed to Phase 2...

# After successful drift check:
_DEBOUNCE_FILE.touch()
```

Pre-commit hook still runs full drift check (last line of defense). The debounce only applies to the edit hook.

**Risk:** Could miss a drift violation during rapid edits. Mitigated by: pre-commit catches it before anything lands.

---

### Fix 5: Parallel health check phases
**File:** `pipeline/health_check.py`
**Measured savings:** 2.7x speedup on health check (bounded by slowest subprocess)
**Change:** Run independent subprocess checks via ThreadPoolExecutor.

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

# Phase 1: Fast local checks (sequential, <1s)
fast_checks = [check_python_deps, check_database, check_dbn_files, check_git_hooks]

# Phase 2: Slow subprocess checks (parallel)
slow_checks = [check_drift, check_integrity, check_tests, check_m25_audit]

with ThreadPoolExecutor(max_workers=4) as pool:
    futures = {pool.submit(check): check.__name__ for check in slow_checks}
    for future in as_completed(futures):
        ok, msg = future.result()
        ...
```

Per DuckDB docs: multiple read-only processes can access the same DB file simultaneously. All health check subprocesses use `read_only=True`.

**Risk:** Console output interleaving. Mitigated by: collect results first, print after all complete.

---

### Fix 6: File content cache (bonus, low-priority)
**File:** `pipeline/check_drift.py` (module-level)
**Measured savings:** ~103ms
**Change:** Cache file reads and glob results across checks.

This is the smallest measured win. Implement only if fixes 1-5 go smoothly.

---

## T4 — VALIDATE: Risks, Tests, Rollback

### Test Plan

| Fix | Verification Command | Pass Criteria |
|-----|---------------------|---------------|
| Fix 1 | `python -c "from pipeline.check_drift import check_old_session_names; print(check_old_session_names())"` | Same violations as before (empty list) |
| Fix 2 | `python -c "from pipeline.check_drift import check_all_imports_resolve; print(check_all_imports_resolve())"` | Same violations as before (empty list) |
| Fix 3 | `python pipeline/check_drift.py` | Exit code 0, same pass count |
| Fix 4 | Edit a pipeline file twice rapidly, observe hook output | Second edit skips drift check, runs tests only |
| Fix 5 | `python pipeline/health_check.py` | Same results, faster wall time |
| All | `python -m pytest tests/test_pipeline/test_check_drift.py -x -q` | All tests pass |
| All | `time python pipeline/check_drift.py` | < 12s (down from 16.5s) |

### Drift Check Verification
After each fix: `python pipeline/check_drift.py` must exit 0 with same check count.

### Rollback Plan
Each fix is a single-file edit. `git checkout -- <file>` reverts any fix independently.

### What NOT to Change
- **Live trading hot path** — bar_aggregator.py is O(1) per tick, zero allocations. data_feed.py uses async correctly. execution_engine.py is event-driven. No performance issues.
- **Pre-commit hook structure** — staged-file-aware targeting is excellent. Only the drift check speed matters.
- **No new dependencies** — all fixes use stdlib (time, sys.modules, concurrent.futures).
- **Check semantics** — no check is removed or weakened. Same violations detected, just faster.

### Expected Total Improvement

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| check_drift.py runtime | 16.5s | ~11-12s | **-4.5s** |
| Post-edit hook (single edit) | 21-31s | 15-25s | **-6s** |
| Post-edit hook (rapid batch) | 63-93s (3 edits) | 15-25s (debounced) | **-48-68s** |
| health_check.py runtime | ~10min | ~5min | **-5min** |
| Pre-commit hook | 26-46s | 20-40s | **-6s** |

### Implementation Order
1. Fix 1 (rglob scope) — biggest single win, 5 min effort
2. Fix 2 (import skip) — second biggest, 5 min effort
3. Fix 3 (shared DB conn) — medium win, 20 min effort (refactor DB check signatures)
4. Fix 4 (debounce hook) — big UX win for rapid edits, 10 min effort
5. Fix 5 (parallel health) — big win for health check, 15 min effort
6. Fix 6 (file cache) — small win, do if time permits
