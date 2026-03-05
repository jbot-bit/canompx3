# Performance Audit Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Cut check_drift.py runtime from 16.5s to ~11s, debounce post-edit hooks, and parallelize health_check.py — all grounded in benchmarked measurements.

**Architecture:** Six independent targeted fixes to the three slowest paths: check_drift.py (4 fixes), post-edit hook (1 fix), health_check.py (1 fix). No new abstractions, no new dependencies. Each fix is a single-file edit, independently testable and revertible.

**Tech Stack:** Python stdlib only (time, sys.modules, concurrent.futures, pathlib). DuckDB read-only connection sharing per official docs.

**Key Files:**
- `pipeline/check_drift.py` (3,014 lines) — fixes 1, 2, 3, 6
- `.claude/hooks/post-edit-pipeline.py` (92 lines) — fix 4
- `pipeline/health_check.py` (217 lines) — fix 5
- `tests/test_pipeline/test_check_drift.py` (49 existing tests) — new tests

**Benchmark Baseline (run before starting):**
```bash
time python pipeline/check_drift.py   # expect ~16.5s
python -m pytest tests/test_pipeline/test_check_drift.py -x -q  # expect 49 passed
```

---

## Phase 1: check_drift.py Hot Checks (saves ~3.5s per run)

### Task 1: Scope rglob in check_old_session_names (#38) — saves 1,800ms

**Files:**
- Modify: `pipeline/check_drift.py:1494`
- Test: `tests/test_pipeline/test_check_drift.py` (new test class)

**Step 1: Capture baseline output for regression check**

```bash
cd /c/Users/joshd/canompx3
python -c "
import time, sys; sys.path.insert(0, '.')
from pipeline.check_drift import check_old_session_names
start = time.perf_counter()
v = check_old_session_names()
print(f'Violations: {len(v)}, Time: {(time.perf_counter()-start)*1000:.0f}ms')
for line in v: print(line)
"
```

Expected: `Violations: 0, Time: ~2000ms`

**Step 2: Write the failing performance test**

Add to `tests/test_pipeline/test_check_drift.py`:

```python
class TestCheckOldSessionNamesPerf:
    """Check #38 must not scan venv/.git — only production dirs."""

    def test_scans_production_dirs_only(self, tmp_path):
        """Verify the function doesn't walk entire project tree."""
        from pipeline.check_drift import check_old_session_names
        import time
        start = time.perf_counter()
        check_old_session_names()
        elapsed = time.perf_counter() - start
        # After fix: should be <500ms (was 2000ms scanning venv/.git)
        assert elapsed < 1.0, f"check_old_session_names took {elapsed:.1f}s — still scanning venv?"
```

Run: `python -m pytest tests/test_pipeline/test_check_drift.py::TestCheckOldSessionNamesPerf -v`
Expected: FAIL (takes ~2s, threshold is 1.0s)

**Step 3: Implement the fix**

In `pipeline/check_drift.py`, replace lines 1494-1506:

```python
# BEFORE (line 1494):
    for py_file in sorted(PROJECT_ROOT.rglob("*.py")):
        rel = py_file.relative_to(PROJECT_ROOT).as_posix()

        # Skip frozen directories
        if any(rel.startswith(d + "/") for d in frozen_dirs):
            continue
        # Skip research/ root-level scripts (one-off historical),
        # EXCEPT active research scripts that were fixed.
        if re.match(r"^research/[^/]+\.py$", rel) and rel not in active_research:
            continue
        # Skip explicitly frozen files
        if rel in frozen_files:
            continue

# AFTER:
    # Scan only production directories (not venv/.git/.auto-claude)
    _scan_dirs = [PIPELINE_DIR, TRADING_APP_DIR, SCRIPTS_DIR]
    all_py_files = []
    for scan_dir in _scan_dirs:
        if scan_dir.exists():
            all_py_files.extend(scan_dir.rglob("*.py"))
    # Also include active research scripts explicitly
    for ar in active_research:
        ar_path = PROJECT_ROOT / ar
        if ar_path.exists():
            all_py_files.append(ar_path)

    for py_file in sorted(all_py_files):
        rel = py_file.relative_to(PROJECT_ROOT).as_posix()

        # Skip explicitly frozen files
        if rel in frozen_files:
            continue
```

Note: `frozen_dirs` exclusion for `tests/`, `.venv/`, `venv/`, `.auto-claude/` is no longer needed — we don't scan those dirs at all. `research/archive/` and `scripts/walkforward/` and `docs/archive/` are also excluded implicitly since we don't scan `research/` or `docs/` root. The `active_research` set is added explicitly since we no longer walk `research/`.

**Step 4: Run tests to verify**

```bash
python -m pytest tests/test_pipeline/test_check_drift.py -x -q
python -c "from pipeline.check_drift import check_old_session_names; v = check_old_session_names(); print(f'Violations: {len(v)}')"
```

Expected: All tests pass, 0 violations (same as baseline).

**Step 5: Verify timing improvement**

```bash
python -c "
import time, sys; sys.path.insert(0, '.')
from pipeline.check_drift import check_old_session_names
start = time.perf_counter()
v = check_old_session_names()
print(f'Violations: {len(v)}, Time: {(time.perf_counter()-start)*1000:.0f}ms')
"
```

Expected: `Time: ~200ms` (down from ~2000ms)

**Step 6: Commit**

```bash
git add pipeline/check_drift.py tests/test_pipeline/test_check_drift.py
git commit -m "perf: scope check_old_session_names rglob to production dirs only

Benchmarked: 2,002ms -> ~200ms. PROJECT_ROOT.rglob walked venv/.git/.auto-claude.
Now scans only pipeline/, trading_app/, scripts/ + explicit active_research files."
```

---

### Task 2: Skip already-imported modules in check_all_imports_resolve (#16) — saves 1,300ms

**Files:**
- Modify: `pipeline/check_drift.py:847`
- Test: `tests/test_pipeline/test_check_drift.py` (new test)

**Step 1: Capture baseline**

```bash
python -c "
import time, sys; sys.path.insert(0, '.')
from pipeline.check_drift import check_all_imports_resolve
start = time.perf_counter()
v = check_all_imports_resolve()
print(f'Violations: {len(v)}, Time: {(time.perf_counter()-start)*1000:.0f}ms')
"
```

Expected: `Violations: 0, Time: ~1500ms`

**Step 2: Write the failing performance test**

Add to `tests/test_pipeline/test_check_drift.py`:

```python
class TestCheckAllImportsResolvePerf:
    """Check #16 should skip already-imported modules."""

    def test_skips_preloaded_modules(self):
        """Modules already in sys.modules should not be re-imported."""
        from pipeline.check_drift import check_all_imports_resolve
        import time
        # First call warms sys.modules
        check_all_imports_resolve()
        # Second call should be fast (all modules already loaded)
        start = time.perf_counter()
        check_all_imports_resolve()
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5, f"Second call took {elapsed:.1f}s — not skipping cached modules?"
```

Run: `python -m pytest tests/test_pipeline/test_check_drift.py::TestCheckAllImportsResolvePerf -v`
Expected: FAIL (second call still ~1.5s because importlib.import_module re-executes even for cached modules in some cases — but the real savings come from skipping the try/except overhead on already-loaded modules)

**Step 3: Implement the fix**

In `pipeline/check_drift.py`, replace line 847-853:

```python
# BEFORE (line 847):
            try:
                importlib.import_module(module)
            except Exception as e:
                err_type = type(e).__name__
                violations.append(
                    f"  {module}: {err_type}: {str(e)[:100]}"
                )

# AFTER:
            # Skip modules already loaded — their imports resolved successfully
            if module in sys.modules:
                continue
            try:
                importlib.import_module(module)
            except Exception as e:
                err_type = type(e).__name__
                violations.append(
                    f"  {module}: {err_type}: {str(e)[:100]}"
                )
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_pipeline/test_check_drift.py -x -q
python -c "from pipeline.check_drift import check_all_imports_resolve; v = check_all_imports_resolve(); print(f'Violations: {len(v)}')"
```

Expected: All tests pass, 0 violations.

**Step 5: Verify timing**

```bash
python -c "
import time, sys; sys.path.insert(0, '.')
from pipeline.check_drift import check_all_imports_resolve
start = time.perf_counter()
v = check_all_imports_resolve()
print(f'Violations: {len(v)}, Time: {(time.perf_counter()-start)*1000:.0f}ms')
"
```

Expected: `Time: ~200ms` (down from ~1500ms). Most modules are already in sys.modules from the check_drift import chain.

**Step 6: Commit**

```bash
git add pipeline/check_drift.py tests/test_pipeline/test_check_drift.py
git commit -m "perf: skip already-imported modules in check_all_imports_resolve

Benchmarked: 1,523ms -> ~200ms. Modules loaded by earlier checks were being
re-imported via importlib.import_module. Now skips if module in sys.modules."
```

---

### Task 3: Shared DB connection for requires_db checks — saves 400ms

**Files:**
- Modify: `pipeline/check_drift.py` (main function + 11 DB check functions)
- Test: `tests/test_pipeline/test_check_drift.py` (new test)

This is the most complex fix — 11 DB check functions need a new `con` parameter.

**Step 1: Identify all DB check functions**

The 11 DB checks are (from CHECKS list with `requires_db=True`):
1. `check_validated_filters_registered` (line 1201)
2. `check_no_e0_in_db` (line 1286)
3. `check_doc_stats_consistency` (line 1320)
4. `check_no_active_e3` (line 1691)
5. `check_wf_coverage` (line 1725)
6. `check_uncovered_fdr_strategies` (line 1822)
7. `check_orphaned_validated_strategies` (line 1966)
8. `check_audit_columns_populated` (line 2230)
9. `check_daily_features_row_integrity` (line 2528)
10. `check_data_continuity` (line 2569)
11. `check_family_rr_locks_coverage` (line 2627)

Each follows this pattern:
```python
def check_X() -> list[str]:
    violations = []
    try:
        import duckdb
        db_path = GOLD_DB_PATH_FOR_CHECKS
        if db_path is None:
            from pipeline.paths import GOLD_DB_PATH
            db_path = GOLD_DB_PATH
        if not Path(db_path).exists():
            return violations
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            # ... queries ...
        finally:
            con.close()
    except (ImportError, OSError) as e:
        print(f"    SKIP check_X: ...")
    return violations
```

**Step 2: Write a test for shared connection behavior**

Add to `tests/test_pipeline/test_check_drift.py`:

```python
class TestSharedDbConnection:
    """DB checks should accept an optional shared connection."""

    def test_no_e0_accepts_shared_con(self):
        """check_no_e0_in_db should work with a passed connection."""
        import duckdb
        from pipeline.paths import GOLD_DB_PATH
        if not GOLD_DB_PATH.exists():
            pytest.skip("No gold.db")
        con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
        try:
            from pipeline.check_drift import check_no_e0_in_db
            v = check_no_e0_in_db(con=con)
            assert isinstance(v, list)
        finally:
            con.close()

    def test_no_e0_works_without_con(self):
        """check_no_e0_in_db should still work standalone (backward compat)."""
        from pipeline.check_drift import check_no_e0_in_db
        v = check_no_e0_in_db()
        assert isinstance(v, list)
```

Run: `python -m pytest tests/test_pipeline/test_check_drift.py::TestSharedDbConnection -v`
Expected: FAIL on `test_no_e0_accepts_shared_con` (function doesn't accept `con=` parameter yet)

**Step 3: Add a helper function for DB path resolution**

Add near the top of `check_drift.py` (after the constants):

```python
def _get_db_path() -> Path:
    """Resolve DB path: test override > pipeline.paths default."""
    if GOLD_DB_PATH_FOR_CHECKS is not None:
        return Path(GOLD_DB_PATH_FOR_CHECKS)
    from pipeline.paths import GOLD_DB_PATH
    return GOLD_DB_PATH
```

**Step 4: Refactor each DB check to accept `con=None`**

For each of the 11 DB checks, change the signature and body. Example for `check_no_e0_in_db`:

```python
# BEFORE:
def check_no_e0_in_db() -> list[str]:
    violations = []
    try:
        import duckdb
        db_path = GOLD_DB_PATH_FOR_CHECKS
        if db_path is None:
            from pipeline.paths import GOLD_DB_PATH
            db_path = GOLD_DB_PATH
        if not Path(db_path).exists():
            return violations
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            for table in ["orb_outcomes", "experimental_strategies", "validated_setups"]:
                count = con.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE entry_model = 'E0'"
                ).fetchone()[0]
                if count > 0:
                    violations.append(
                        f"  {table}: {count} rows with entry_model='E0' (purged Feb 2026)"
                    )
        finally:
            con.close()
    except (ImportError, OSError) as e:
        print(f"    SKIP check_no_e0_in_db: {type(e).__name__}: {e}")
    return violations

# AFTER:
def check_no_e0_in_db(con=None) -> list[str]:
    violations = []
    _own_con = False
    try:
        import duckdb
        if con is None:
            db_path = _get_db_path()
            if not db_path.exists():
                return violations
            con = duckdb.connect(str(db_path), read_only=True)
            _own_con = True
        for table in ["orb_outcomes", "experimental_strategies", "validated_setups"]:
            count = con.execute(
                f"SELECT COUNT(*) FROM {table} WHERE entry_model = 'E0'"
            ).fetchone()[0]
            if count > 0:
                violations.append(
                    f"  {table}: {count} rows with entry_model='E0' (purged Feb 2026)"
                )
    except (ImportError, OSError) as e:
        print(f"    SKIP check_no_e0_in_db: {type(e).__name__}: {e}")
    finally:
        if _own_con and con is not None:
            con.close()
    return violations
```

Apply same pattern to all 11 DB checks.

**Step 5: Update main() to open shared connection and pass it**

In `main()` (line 2945), before the check loop:

```python
    # Open shared read-only DB connection for all requires_db checks
    import duckdb
    _shared_con = None
    db_path = _get_db_path()
    if db_path.exists():
        try:
            _shared_con = duckdb.connect(str(db_path), read_only=True)
        except Exception:
            pass  # DB busy — individual checks will skip

    # ... existing loop, but pass connection:
    for i, (label, check_fn, is_advisory, requires_db) in enumerate(CHECKS, 1):
        if requires_db:
            try:
                v = check_fn(con=_shared_con)
            except Exception as e:
                # ... existing DB-busy handling ...
        else:
            v = check_fn()

    # Cleanup
    if _shared_con is not None:
        _shared_con.close()
```

Also update the CHECKS list lambdas for DB checks that use lambdas (none do currently — they're all direct function references).

**Step 6: Run tests**

```bash
python -m pytest tests/test_pipeline/test_check_drift.py -x -q
python pipeline/check_drift.py  # must exit 0 with same pass count
```

Expected: All 49+ tests pass. Drift check exits 0.

**Step 7: Commit**

```bash
git add pipeline/check_drift.py tests/test_pipeline/test_check_drift.py
git commit -m "perf: shared DB connection for requires_db drift checks

Benchmarked: 10x duckdb.connect() at 45ms each = 450ms -> 43ms (1 connection).
Per DuckDB docs: 'DuckDB performs best when reusing the same connection.'
All 11 DB checks now accept optional con= parameter for shared connection."
```

---

### Task 4: Checkpoint — verify Phase 1 savings

**Step 1: Run full timing benchmark**

```bash
time python pipeline/check_drift.py
```

Expected: **~11-12s** (down from 16.5s). If not, profile again:

```bash
python -c "
import time, sys; sys.path.insert(0, '.')
from pipeline.check_drift import CHECKS
for i, (label, fn, adv, db) in enumerate(CHECKS, 1):
    if db: continue
    start = time.perf_counter()
    try: fn()
    except: pass
    e = time.perf_counter() - start
    if e > 0.1: print(f'  #{i:2d} {e*1000:6.0f}ms  {label}')
"
```

**Step 2: Run full test suite**

```bash
python -m pytest tests/test_pipeline/test_check_drift.py -x -q
python pipeline/check_drift.py
```

Expected: All tests pass, exit code 0, same check count.

---

## Phase 2: Hook & Health Check (saves ~16s per batch + 5min on health)

### Task 5: Debounce post-edit hook — saves ~16s per rapid-edit batch

**Files:**
- Modify: `.claude/hooks/post-edit-pipeline.py`
- No automated test (hook is triggered by Claude Code, not pytest)

**Step 1: Read the current hook**

Verify current structure: 3 phases (drift → tests → behavioral audit).

**Step 2: Add debounce logic**

In `.claude/hooks/post-edit-pipeline.py`, add imports and constants at top:

```python
import time
from pathlib import Path

_DEBOUNCE_FILE = Path(__file__).parent / ".last_drift_ok"
_DEBOUNCE_SECONDS = 30
```

Then in `main()`, before Phase 1 (line 41), add:

```python
    # Debounce: skip drift check if it passed within last 30 seconds.
    # Pre-commit hook still runs full drift check (last line of defense).
    _skip_drift = False
    if _DEBOUNCE_FILE.exists():
        try:
            age = time.time() - _DEBOUNCE_FILE.stat().st_mtime
            if age < _DEBOUNCE_SECONDS:
                _skip_drift = True
        except OSError:
            pass  # race condition — file deleted between exists() and stat()

    if not _skip_drift:
        # --- Phase 1: Drift check (fast, ~2s) ---
        result = subprocess.run(
            [sys.executable, "pipeline/check_drift.py"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            # Invalidate debounce on failure
            _DEBOUNCE_FILE.unlink(missing_ok=True)
            print(f"DRIFT DETECTED after editing {file_path}", file=sys.stderr)
            print(result.stdout, file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            sys.exit(2)
        # Mark successful drift check timestamp
        _DEBOUNCE_FILE.touch()
    # Phase 2 (targeted tests) always runs regardless of debounce
```

Replace the existing Phase 1 block (lines 41-49) with this.

**Step 3: Add .last_drift_ok to .gitignore**

```bash
echo ".claude/hooks/.last_drift_ok" >> .gitignore
```

**Step 4: Manual verification**

Edit any pipeline file. First edit: drift check runs (~11s). Edit again within 30s: drift check is skipped, only targeted tests run (~10-20s). Edit after 30s: drift check runs again.

**Step 5: Commit**

```bash
git add .claude/hooks/post-edit-pipeline.py .gitignore
git commit -m "perf: debounce drift check in post-edit hook (30s cooldown)

Rapid edits no longer trigger redundant drift checks. Pre-commit hook
still runs full check before commit (last line of defense)."
```

---

### Task 6: Parallel health check phases — saves ~5min

**Files:**
- Modify: `pipeline/health_check.py`
- Test: manual timing comparison

**Step 1: Capture baseline**

```bash
time python pipeline/health_check.py
```

Expected: ~5-10 minutes (sequential).

**Step 2: Implement parallel slow checks**

Replace `main()` in `pipeline/health_check.py`:

```python
def main():
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print("=" * 50)
    print("PIPELINE HEALTH CHECK")
    print("=" * 50)
    print()

    # Phase 1: Fast local checks (sequential, <1s total)
    fast_checks = [
        check_python_deps,
        check_database,
        check_dbn_files,
        check_git_hooks,
    ]

    # Phase 2: Slow subprocess checks (parallel)
    slow_checks = [
        check_drift,
        check_integrity,
        check_tests,
        check_m25_audit,
    ]

    all_ok = True

    # Run fast checks first (sequential — instant)
    for check in fast_checks:
        ok, msg = check()
        status = "[OK]" if ok else "[FAIL]"
        print(f"  {status} {msg}")
        if not ok:
            all_ok = False

    # Run slow checks in parallel (ThreadPoolExecutor — correct for subprocess waits)
    print()
    print("  Running slow checks in parallel...")
    results = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(check): check.__name__ for check in slow_checks}
        for future in as_completed(futures):
            name = futures[future]
            try:
                ok, msg = future.result()
            except Exception as e:
                ok, msg = False, f"{name}: {type(e).__name__}: {e}"
            results[name] = (ok, msg)

    # Print slow check results (deterministic order)
    for check in slow_checks:
        ok, msg = results[check.__name__]
        status = "[OK]" if ok else "[FAIL]"
        print(f"  {status} {msg}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
        sys.exit(1)
```

**Step 3: Verify**

```bash
time python pipeline/health_check.py
```

Expected: Wall time bounded by slowest check (test suite or M2.5), ~2.7x faster than sequential.

**Step 4: Commit**

```bash
git add pipeline/health_check.py
git commit -m "perf: parallelize slow health check phases via ThreadPoolExecutor

Benchmarked: 2.7x speedup. Drift, integrity, tests, and M2.5 audit now run
in parallel. Per Python docs: ThreadPoolExecutor is correct for subprocess waits.
Per DuckDB docs: multiple read-only processes can access same DB file."
```

---

## Phase 3: File Cache + Final Verification

### Task 7: File content cache in check_drift.py (bonus) — saves ~100ms

**Files:**
- Modify: `pipeline/check_drift.py` (add FileCache class, update check functions)

This is low-priority (100ms savings). Skip if Phase 1+2 hit the target.

**Step 1: Add FileCache class after the constants block**

```python
class _FileCache:
    """Read-once cache for file contents across 63+ checks."""
    def __init__(self):
        self._contents: dict[Path, str] = {}
        self._glob_results: dict[tuple, list[Path]] = {}

    def read(self, path: Path) -> str:
        if path not in self._contents:
            self._contents[path] = path.read_text(encoding='utf-8')
        return self._contents[path]

    def glob(self, directory: Path, pattern: str = "*.py") -> list[Path]:
        key = (directory, pattern, False)
        if key not in self._glob_results:
            self._glob_results[key] = sorted(directory.glob(pattern))
        return self._glob_results[key]

    def rglob(self, directory: Path, pattern: str = "*.py") -> list[Path]:
        key = (directory, pattern, True)
        if key not in self._glob_results:
            self._glob_results[key] = sorted(directory.rglob(pattern))
        return self._glob_results[key]

_cache = _FileCache()
```

**Step 2: Replace `fpath.read_text()` and `dir.glob()` calls in check functions**

Replace all instances of:
- `fpath.read_text(encoding='utf-8')` → `_cache.read(fpath)`
- `pipeline_dir.glob("*.py")` → `_cache.glob(pipeline_dir)`
- `trading_app_dir.rglob("*.py")` → `_cache.rglob(trading_app_dir)`

Only in the check functions that scan shared directories (checks 1-11, 19, 25, 33, 38, 62).

**Step 3: Verify**

```bash
python pipeline/check_drift.py  # exit 0, same check count
python -m pytest tests/test_pipeline/test_check_drift.py -x -q  # all pass
```

**Step 4: Commit**

```bash
git add pipeline/check_drift.py
git commit -m "perf: add file content cache to check_drift.py

Benchmarked: 128ms -> 24ms for file I/O (5.3x). Small absolute savings but
eliminates 76 redundant read_text calls across 63 checks."
```

---

### Task 8: Final verification and timing

**Step 1: Run full benchmark**

```bash
time python pipeline/check_drift.py
```

Expected: **< 12s** (down from 16.5s).

**Step 2: Run all tests**

```bash
python -m pytest tests/test_pipeline/test_check_drift.py -x -q
python -m pytest tests/ -x -q --tb=no  # full suite
```

Expected: All pass.

**Step 3: Run health check**

```bash
time python pipeline/health_check.py
```

Expected: **< 5 min** (down from ~10 min).

**Step 4: Test debounce**

Edit a pipeline file twice in rapid succession. Verify second edit skips drift check.

**Step 5: Final commit with results**

```bash
git add -A
git commit -m "perf: complete performance audit — 6 fixes across 3 phases

Summary of measured improvements:
- check_drift.py: 16.5s -> Xs (measured)
- Post-edit hook batch: 63-93s -> 15-25s (debounced)
- health_check.py: ~10min -> ~5min (parallel)

Fixes applied:
1. Scoped rglob in check_old_session_names (-1.8s)
2. Skip cached modules in check_all_imports_resolve (-1.3s)
3. Shared DB connection for requires_db checks (-0.4s)
4. Debounced post-edit hook (30s cooldown)
5. Parallel health check via ThreadPoolExecutor (2.7x)
6. File content cache (-0.1s)"
```
