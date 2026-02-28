# Pipeline Ingest Audit Fixes

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Fix 6 self-identified issues in the pipeline ingestion layer — config mismatches, integrity check scoping, missing DB config, and dishonest docstrings.

**Architecture:** All fixes are in the `pipeline/` layer. No changes to trading_app/ or downstream tables. The key insight is that `asset_configs.py` has wrong patterns for MGC and MES when used by `ingest_dbn.py`, even though `ingest_dbn_daily.py` has a hardcoded workaround for MGC. We fix the source of truth (asset_configs.py) and remove the workaround.

**Tech Stack:** Python, DuckDB, regex, pytest

**Evidence from DB (verified 2026-02-28):**
- MGC source_symbols: ALL `GC*` (GCZ2, GCG5, etc.) — NOT MGC contracts
- MES source_symbols: MIX of `ES*` (pre-2024) and `MES*` (2024+)
- M2K source_symbols: ALL `RTY*` — correctly matched by config
- MNQ/NQ: correctly handled by separate NQ config entry

---

### Task 1: Fix MGC outright_pattern and prefix_len in asset_configs.py

**Why:** `asset_configs.py` says MGC pattern is `^MGC...` with `prefix_len=3`, but ALL MGC data in the DB uses GC contracts (GCZ2, GCM4, etc.). Running `python pipeline/ingest_dbn.py --instrument MGC` would match ZERO outrights and silently produce empty output. The correct pattern is `^GC...` with `prefix_len=2`, matching the legacy `ingest_dbn_mgc.py` and `ingest_dbn_daily.py` override.

**Files:**
- Modify: `pipeline/asset_configs.py:36-48` (MGC config block)
- Modify: `tests/test_pipeline/test_gc_mgc_mapping.py` (update docstring line 9)
- Test: `tests/test_pipeline/test_gc_mgc_mapping.py` (add asset_configs test)

**Step 1: Write the failing test**

Add to `tests/test_pipeline/test_gc_mgc_mapping.py`:

```python
from pipeline.asset_configs import ASSET_CONFIGS

class TestAssetConfigMgcPattern:
    """Verify asset_configs.py MGC entry matches GC source data."""

    def test_mgc_pattern_matches_gc_contracts(self):
        """MGC config must match GC outrights (data source is full-size Gold)."""
        pattern = ASSET_CONFIGS["MGC"]["outright_pattern"]
        for sym in ["GCM4", "GCZ4", "GCG25", "GCQ5"]:
            assert pattern.match(sym), f"MGC pattern should match {sym}"

    def test_mgc_pattern_rejects_mgc_contracts(self):
        """MGC config must NOT match MGC outrights (we use GC source data)."""
        pattern = ASSET_CONFIGS["MGC"]["outright_pattern"]
        for sym in ["MGCM4", "MGCZ4", "MGCG25"]:
            assert not pattern.match(sym), f"MGC pattern should not match {sym}"

    def test_mgc_prefix_len_is_2(self):
        """GC contracts have 2-char prefix before month code."""
        assert ASSET_CONFIGS["MGC"]["prefix_len"] == 2
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline/test_gc_mgc_mapping.py::TestAssetConfigMgcPattern -v`
Expected: FAIL — pattern is currently `^MGC...` and prefix_len is 3

**Step 3: Fix asset_configs.py MGC entry**

Change `pipeline/asset_configs.py:36-48`:

```python
"MGC": {
    # Source data is GC (full-size Gold, $100/pt) — same price, stored as symbol='MGC'.
    # Identical pattern to RTY→M2K, SI→SIL, 6E→M6E. Cost model uses MGC micro specs ($10/pt).
    # GC has better 1m bar coverage than native MGC contracts.
    "dbn_path": PROJECT_ROOT / "DB" / "GOLD_DB_FULLSIZE",
    "symbol": "MGC",
    "outright_pattern": re.compile(r'^GC[FGHJKMNQUVXZ]\d{1,2}$'),
    "prefix_len": 2,
    "minimum_start_date": date(2019, 1, 1),
    "schema_required": "ohlcv-1m",
    "enabled_sessions": [
        "CME_REOPEN", "TOKYO_OPEN", "SINGAPORE_OPEN", "LONDON_METALS",
        "US_DATA_830", "NYSE_OPEN", "US_DATA_1000",
        "COMEX_SETTLE",
    ],
},
```

**Step 4: Update test docstring**

Change `tests/test_pipeline/test_gc_mgc_mapping.py:9`:

From: `The multi-instrument asset_configs.py uses MGC pattern directly.`
To: `The multi-instrument asset_configs.py uses the GC pattern (matching source data).`

**Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_pipeline/test_gc_mgc_mapping.py -v`
Expected: ALL PASS

**Step 6: Run full pipeline tests to check for regressions**

Run: `python -m pytest tests/test_pipeline/ -x -q`
Expected: ALL PASS

**Step 7: Commit**

```bash
git add pipeline/asset_configs.py tests/test_pipeline/test_gc_mgc_mapping.py
git commit -m "fix: correct MGC outright_pattern and prefix_len in asset_configs (S1/S2 audit)"
```

---

### Task 2: Add ES config entry for MES pre-2024 data (S3 fix)

**Why:** MES DB has both ES contracts (2019-2024) and MES contracts (2024+). The current `^MES...` pattern would miss all ES contracts on re-ingest. The NQ→MNQ pattern (separate `NQ` config entry) shows the correct approach: add an `ES` entry that maps ES contracts to MES symbol.

**Files:**
- Modify: `pipeline/asset_configs.py` (add ES config entry)
- Test: `tests/test_pipeline/test_gc_mgc_mapping.py` (add ES/MES config tests)

**Step 1: Verify the MES data directory structure**

Run: `ls /c/Users/joshd/canompx3/DB/MES_DB/ | head -20`
Determine: Does MES_DB contain ES daily files, MES daily files, or both?

**Step 2: Write the failing test**

Add to `tests/test_pipeline/test_gc_mgc_mapping.py`:

```python
class TestAssetConfigMesPattern:
    """Verify MES config handles both ES (pre-2024) and MES (2024+) source data."""

    def test_mes_pattern_matches_mes_contracts(self):
        """MES config must match native MES outrights (2024+)."""
        pattern = ASSET_CONFIGS["MES"]["outright_pattern"]
        for sym in ["MESM4", "MESZ4", "MESH25"]:
            assert pattern.match(sym), f"MES pattern should match {sym}"

    def test_es_config_exists(self):
        """An ES config entry must exist for pre-2024 ES→MES mapping."""
        assert "ES" in ASSET_CONFIGS, "ES config needed for pre-2024 data"

    def test_es_pattern_matches_es_contracts(self):
        """ES config must match ES outrights."""
        pattern = ASSET_CONFIGS["ES"]["outright_pattern"]
        for sym in ["ESH5", "ESM9", "ESZ24"]:
            assert pattern.match(sym), f"ES pattern should match {sym}"

    def test_es_stores_as_mes_symbol(self):
        """ES config must store data under MES symbol."""
        assert ASSET_CONFIGS["ES"]["symbol"] == "MES"

    def test_es_prefix_len_is_2(self):
        """ES contracts have 2-char prefix (ES)."""
        assert ASSET_CONFIGS["ES"]["prefix_len"] == 2
```

**Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline/test_gc_mgc_mapping.py::TestAssetConfigMesPattern -v`
Expected: FAIL — no ES config entry exists

**Step 4: Add ES config entry to asset_configs.py**

Add after the MES entry (before M2K):

```python
"ES": {
    # Source data: ES (E-mini S&P 500, $50/pt) for 2019-2024 backfill.
    # Same price as MES on same exchange. Stored as symbol='MES', source_symbol='ESH22' etc.
    # Identical pattern to GC→MGC, NQ→MNQ, RTY→M2K.
    "dbn_path": PROJECT_ROOT / "DB" / "MES_DB",
    "symbol": "MES",
    "outright_pattern": re.compile(r'^ES[FGHJKMNQUVXZ]\d{1,2}$'),
    "prefix_len": 2,
    "minimum_start_date": date(2019, 2, 12),
    "schema_required": "ohlcv-1m",
    "enabled_sessions": [
        "CME_REOPEN", "TOKYO_OPEN", "SINGAPORE_OPEN", "LONDON_METALS",
        "US_DATA_830", "NYSE_OPEN", "US_DATA_1000",
        "COMEX_SETTLE", "CME_PRECLOSE", "NYSE_CLOSE",
    ],
},
```

**NOTE:** The `dbn_path` for ES and MES may or may not be the same directory. Step 1 determines this. If ES data lives in a separate directory (e.g., `DB/ES_DB`), adjust accordingly. If same directory, the outright_pattern will correctly filter ES-only vs MES-only contracts from the shared file set.

**Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_pipeline/test_gc_mgc_mapping.py -v`
Expected: ALL PASS

**Step 6: Run full test suite**

Run: `python -m pytest tests/test_pipeline/ -x -q`
Expected: ALL PASS

**Step 7: Commit**

```bash
git add pipeline/asset_configs.py tests/test_pipeline/test_gc_mgc_mapping.py
git commit -m "fix: add ES config entry for pre-2024 MES data (S3 audit)"
```

---

### Task 3: Remove MGC hardcoded override from ingest_dbn_daily.py

**Blocked by:** Task 1 (MGC pattern must be correct in asset_configs first)

**Why:** `ingest_dbn_daily.py:89-97` has a hardcoded `if instrument.upper() == "MGC"` override that was the workaround for the wrong asset_configs pattern. Now that asset_configs is correct, this divergence is unnecessary. All instruments should go through the same config path.

**Files:**
- Modify: `pipeline/ingest_dbn_daily.py:77-105` (simplify get_ingest_config)

**Step 1: Simplify get_ingest_config to remove MGC special case**

Replace `pipeline/ingest_dbn_daily.py:77-105`:

```python
def get_ingest_config(instrument: str) -> dict:
    """
    Return ingestion-specific config for an instrument.

    All instruments (including MGC) now use asset_configs.py as source of truth.
    The outright_pattern in asset_configs matches the actual source data contracts
    (e.g., GC for MGC, RTY for M2K, ES for the ES backfill entry).

    Returns dict with keys: symbol, outright_pattern, prefix_len,
    minimum_start_date, data_dir.
    """
    config = get_asset_config(instrument)

    return {
        "symbol": config["symbol"],
        "outright_pattern": config["outright_pattern"],
        "prefix_len": config["prefix_len"],
        "minimum_start_date": config["minimum_start_date"],
        "data_dir": config["dbn_path"],
    }
```

**Step 2: Remove unused imports**

Check if `GC_OUTRIGHT_PATTERN` and `MINIMUM_START_DATE` are still used elsewhere in `ingest_dbn_daily.py`. If only used in the removed MGC override, remove them from the import at line 53-55.

**Step 3: Run pipeline tests**

Run: `python -m pytest tests/test_pipeline/ -x -q`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add pipeline/ingest_dbn_daily.py
git commit -m "refactor: remove MGC hardcoded override from ingest_dbn_daily (S4 audit)"
```

---

### Task 4: Scope check_merge_integrity with symbol filter

**Why:** `check_merge_integrity()` checks `FROM bars_1m WHERE DATE(ts_utc) BETWEEN ? AND ?` without a symbol filter. This means it checks ALL instruments in that date range, not just the one being ingested. It also has a date scoping gap: chunks are defined by trading_day but the query uses DATE(ts_utc), which differs for overnight bars.

**Files:**
- Modify: `pipeline/ingest_dbn_mgc.py:376-405` (add symbol param)
- Modify: `pipeline/ingest_dbn.py:418` (pass symbol to call)
- Modify: `pipeline/ingest_dbn_daily.py:430` (pass symbol to call)
- Modify: `pipeline/ingest_dbn_mgc.py:732` (pass symbol in legacy main)
- Test: `tests/test_pipeline/test_validation.py` (add integrity check test)

**Step 1: Write the failing test**

Add to `tests/test_pipeline/test_validation.py`:

```python
class TestCheckMergeIntegrity:
    """Tests for post-merge integrity gate."""

    def test_accepts_symbol_parameter(self):
        """check_merge_integrity should accept optional symbol parameter."""
        import inspect
        sig = inspect.signature(check_merge_integrity)
        assert 'symbol' in sig.parameters, "check_merge_integrity must accept symbol param"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline/test_validation.py::TestCheckMergeIntegrity -v`
Expected: FAIL — no symbol parameter exists

**Step 3: Update check_merge_integrity signature and queries**

Change `pipeline/ingest_dbn_mgc.py:376-405`:

```python
def check_merge_integrity(con: duckdb.DuckDBPyConnection, chunk_start: str, chunk_end: str, symbol: str = None) -> tuple[bool, str]:
    """
    Assert no duplicates or NULL source_symbol after merge.

    Must run AFTER merge. If symbol is provided, scopes checks to that symbol only.
    Uses a 1-day buffer on the date range to account for overnight bars whose
    DATE(ts_utc) may precede the trading day.
    """
    symbol_clause = "AND symbol = ?" if symbol else ""
    params_base = [chunk_start, chunk_end]
    if symbol:
        params_base.append(symbol)

    # Check for duplicates (with 1-day buffer for overnight bars)
    dupe_check = con.execute(f"""
        SELECT symbol, ts_utc, COUNT(*) as cnt
        FROM bars_1m
        WHERE DATE(ts_utc) BETWEEN CAST(? AS DATE) - INTERVAL 1 DAY AND ?
        {symbol_clause}
        GROUP BY symbol, ts_utc
        HAVING COUNT(*) > 1
        LIMIT 5
    """, params_base).fetchall()

    if dupe_check:
        return False, f"Duplicate (symbol, ts_utc) found after merge: {dupe_check}"

    # Check for NULL source_symbol
    null_params = [chunk_start, chunk_end]
    if symbol:
        null_params.append(symbol)
    null_check = con.execute(f"""
        SELECT COUNT(*) FROM bars_1m
        WHERE DATE(ts_utc) BETWEEN CAST(? AS DATE) - INTERVAL 1 DAY AND ?
        {symbol_clause}
        AND source_symbol IS NULL
    """, null_params).fetchone()[0]

    if null_check > 0:
        return False, f"NULL source_symbol found after merge: {null_check} rows"

    return True, ""
```

**Step 4: Update all callers to pass symbol**

In `pipeline/ingest_dbn.py:418`:
```python
int_ok, int_reason = check_merge_integrity(con, chunk_start, chunk_end, symbol=symbol)
```

In `pipeline/ingest_dbn.py:488`:
```python
int_ok, int_reason = check_merge_integrity(con, chunk_start, chunk_end, symbol=symbol)
```

In `pipeline/ingest_dbn_daily.py:430`:
```python
int_ok, int_reason = check_merge_integrity(con, chunk_start, chunk_end, symbol=symbol)
```

In `pipeline/ingest_dbn_daily.py:497`:
```python
int_ok, int_reason = check_merge_integrity(con, chunk_start, chunk_end, symbol=symbol)
```

In `pipeline/ingest_dbn_mgc.py:732`:
```python
int_ok, int_reason = check_merge_integrity(con, chunk_start, chunk_end, symbol='MGC')
```

In `pipeline/ingest_dbn_mgc.py:807`:
```python
int_ok, int_reason = check_merge_integrity(con, chunk_start, chunk_end, symbol='MGC')
```

**Step 5: Run tests**

Run: `python -m pytest tests/test_pipeline/ -x -q`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add pipeline/ingest_dbn_mgc.py pipeline/ingest_dbn.py pipeline/ingest_dbn_daily.py tests/test_pipeline/test_validation.py
git commit -m "fix: scope check_merge_integrity with symbol filter and 1-day buffer (L1 audit)"
```

---

### Task 5: Add configure_connection to ingest_dbn_mgc.py

**Why:** `ingest_dbn_mgc.py` opens DuckDB without calling `configure_connection()`, missing memory_limit and temp_directory settings that `ingest_dbn.py` and `ingest_dbn_daily.py` both apply. Low priority since the file is deprecated, but keeps the library honest.

**Files:**
- Modify: `pipeline/ingest_dbn_mgc.py:527-528`

**Step 1: Add configure_connection call**

After `pipeline/ingest_dbn_mgc.py:527` (`con = duckdb.connect(str(DB_PATH))`), add:

```python
from pipeline.db_config import configure_connection
configure_connection(con, writing=True)
```

**Step 2: Run tests**

Run: `python -m pytest tests/test_pipeline/ -x -q`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add pipeline/ingest_dbn_mgc.py
git commit -m "fix: add configure_connection to deprecated ingest_dbn_mgc.py (S5 audit)"
```

---

### Task 6: Correct VECTORIZED claims in docstrings

**Why:** Several functions claim "VECTORIZED" in comments but use `.apply(lambda)` or `iterrows()`. Honesty over outcome — either vectorize the code or fix the comments.

**Files:**
- Modify: `pipeline/ingest_dbn_mgc.py` (lines 10, 611-613, 640, 676)
- Modify: `pipeline/ingest_dbn.py` (lines 10, 306-310)

**Step 1: Fix ingest_dbn_mgc.py docstring claim**

Line 10: Change `- VECTORIZED: No apply() or iterrows() over large data` to:
```
- MOSTLY VECTORIZED: Minimal apply() for outright filtering and trading day edge case
```

Line 611 comment: Change `# FILTER TO OUTRIGHTS FIRST (VECTORIZED)` to:
```
# FILTER TO OUTRIGHTS FIRST (apply — small N per chunk)
```

Line 640 comment is fine — `compute_trading_days` IS mostly vectorized (the apply is only on the masked subset).

Line 676: The `iterrows()` loop is in the deprecated `main()` function. Add a comment:
```python
# NOTE: iterrows() here — deprecated path. Multi-instrument version uses vectorized zip.
for ts_utc, row in front_df.iterrows():
```

**Step 2: Fix ingest_dbn.py docstring claim**

Line 10: Change `- VECTORIZED: No apply() or iterrows() over large data` to:
```
- MOSTLY VECTORIZED: Minimal apply() for outright filtering; core ops are vectorized
```

Line 306 comment: Change `# FILTER TO OUTRIGHTS (VECTORIZED, CONFIG-DRIVEN)` to:
```
# FILTER TO OUTRIGHTS (CONFIG-DRIVEN, apply for regex match)
```

**Step 3: Run tests**

Run: `python -m pytest tests/test_pipeline/ -x -q`
Expected: ALL PASS (docstring-only changes)

**Step 4: Commit**

```bash
git add pipeline/ingest_dbn_mgc.py pipeline/ingest_dbn.py
git commit -m "docs: correct VECTORIZED claims in ingest docstrings (S6 audit)"
```

---

## Execution Order

```
Task 1 (MGC pattern fix) ──→ Task 3 (remove hardcoded override)
Task 2 (ES config entry)
Task 4 (integrity check scoping)
Task 5 (configure_connection)
Task 6 (docstring corrections)
```

Tasks 1, 2, 4, 5, 6 are independent of each other.
Task 3 is blocked by Task 1.

## Verification After All Tasks

```bash
python -m pytest tests/ -x -q                    # Full test suite
python pipeline/check_drift.py                    # Drift detection
python pipeline/health_check.py                   # Health check
```
