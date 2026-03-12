# Pipeline Fortification — Hot-Path Optimization Design

**Date:** 2026-03-08
**Status:** Phases 1-3,5 DONE (verified 2026-03-12). Phase 4 deferred.
**Scope:** `trading_app/strategy_fitness.py`, `trading_app/pbo.py`, `scripts/tools/build_edge_families.py`, `trading_app/walkforward.py`, `trading_app/strategy_fitness.py` exception handling
**Relationship:** Complements `2026-03-05-perf-audit-design.md` (which covers check_drift/health_check/hooks). This doc targets **trading_app hot paths** — the code that runs during portfolio fitness checks, edge family builds, and walk-forward validation.

---

## T1 — ORIENT: Problem Statement

### What's Slow

The Bloomey fortification audit identified **8 optimization targets** in trading_app hot paths. These are NOT the drift-check/hook improvements from the March 5 perf audit — these are the **data pipeline bottlenecks** that affect rebuild times and portfolio health checks.

### Measured Bottleneck Profile

| # | File | Pattern | Impact | Root Cause |
|---|------|---------|--------|------------|
| **1** | `strategy_fitness.py` | N+1 query loop | ~900 individual queries per portfolio check | Each strategy loads its own outcomes via `_load_strategy_outcomes()` — 1 `orb_outcomes` query + 1 `daily_features` query per strategy. ~900 strategies × 2 queries = ~1800 queries per `compute_portfolio_fitness()` call |
| **2** | `pbo.py` → `build_edge_families.py` | N+1 query loop | ~750 queries during edge family build | `compute_family_pbo()` loads outcomes individually per family member. With ~150 multi-member families × ~5 members each = ~750 queries in the PBO step alone |
| **3** | `strategy_fitness.py` | Bare `except Exception` | Silent failures, swallowed errors | Lines 538, 596, 682, 765, 787 — catch-all exception handlers in fitness computation and decay diagnostics. If a strategy fails with a real bug (e.g. schema change), it's logged as `WARN` and skipped |
| **4** | `build_edge_families.py` | Row-by-row UPDATEs | O(N) roundtrips to DB | Lines 324-333 tag each family member with individual UPDATE statements. 900 strategies = 900 UPDATE roundtrips |
| **5** | `strategy_fitness.py` | `SELECT *` overfetch | Transfers all 30+ daily_features columns when only ~5 are needed | Line 359: `SELECT * FROM daily_features` loads every column even when the filter only checks 2-3 fields. This is bounded (columns are stable) but wasteful for NTFS I/O |
| **6** | `walkforward.py` | Linear scan for window partitioning | O(N×W) per strategy | Line 148/155: list comprehensions scan ALL outcomes for each window. With 200+ outcomes × 6+ windows = 1200+ comparisons. Could be O(N log N) with sorted index + bisect |
| **7** | General | No DuckDB prepared statements | Each repeated query re-parses SQL | DuckDB's `con.execute()` parses SQL text on every call. Queries like `_load_strategy_outcomes()` use identical SQL structure with different params — prepared statements eliminate parse overhead |
| **8** | General | pandas `.apply()` in pipelines | Python-speed row iteration | Not yet profiled — need to grep for `.apply()` usage and assess if vectorized alternatives exist |

### Blast Radius Analysis

| File | Called By | Downstream |
|------|-----------|------------|
| `strategy_fitness.py` | `mcp_server.py`, CLI, `strategy_validator.py` (via imports) | FitnessReport → dashboard, MCP, CLI output |
| `pbo.py` | `build_edge_families.py` only | PBO value on `edge_families` table |
| `build_edge_families.py` | rebuild scripts, CLI | `edge_families` + `validated_setups.family_hash` columns |
| `walkforward.py` | `strategy_validator.py` only | WalkForwardResult → `validated_setups.wf_*` columns |

**Critical invariant:** All optimizations MUST produce byte-identical outputs. These are pure performance changes — no logic changes.

---

## T2 — DESIGN: Approach Per Item

### Phase 1: Exception Hardening (Item #3)

**Files:** `strategy_fitness.py`
**Why first:** Safety before speed. Bare `except Exception` can mask real bugs introduced by later phases.

**Current state (5 locations):**
```python
# Line 538 — compute_portfolio_fitness loop
except Exception as e:
    logger.warning(f"  WARN: Failed to compute fitness for {sid}: {e}")

# Line 596 — diagnose_decay own fitness
except Exception as e:
    logger.warning("Could not compute fitness for %s: %s", strategy_id, e)
    actual_status = "UNKNOWN"

# Line 682 — diagnose_decay sibling loop
except Exception as e:
    logger.debug("Sibling %s fitness failed: %s", sid, e)
    counts["STALE"] += 1

# Line 765 — diagnose_portfolio_decay first pass
except Exception as e:
    logger.debug("Fitness computation failed for %s: %s", sid, e)

# Line 787 — diagnose_portfolio_decay cached reuse
except Exception:
    own_status = "UNKNOWN"
```

**Fix:** Replace each `except Exception` with specific exceptions:
- `ValueError` — strategy not found (already raised by `_compute_fitness_with_con`)
- `duckdb.Error` — DB query failures (connection, schema, SQL)
- Keep `except Exception` ONLY for the outermost portfolio loop (line 538) since it's a fail-soft iterator that must not crash the whole report. Add `logger.exception()` for traceback visibility.

**Test strategy:** Existing tests pass unchanged. Add one test injecting a bad strategy_id to verify ValueError propagation.

---

### Phase 2: PBO Bulk Load (Item #2)

**File:** `trading_app/pbo.py`
**Why second:** Biggest speedup potential (10-50x), isolated module, no blast radius.

**Current state:** `compute_family_pbo()` loads outcomes individually per member:
```python
for sid, orb_label, orb_minutes, entry_model, rr_target, confirm_bars, _filter_type in members:
    rows = con.execute("""SELECT o.trading_day, o.pnl_r FROM orb_outcomes o
                          WHERE ... AND o.pnl_r IS NOT NULL
                          ORDER BY o.trading_day""", [...]).fetchall()
```

**Fix:** Single bulk query loads ALL member outcomes at once, then partitions in Python:
```python
# Build (orb_label, orb_minutes, entry_model, rr_target, confirm_bars) tuples
# for a single IN-clause query using DuckDB's VALUES syntax
member_keys = [(orb_label, orb_minutes, entry_model, rr_target, confirm_bars) for ...]

# One query: join orb_outcomes against member keys
rows = con.execute("""
    SELECT o.trading_day, o.pnl_r,
           o.orb_label, o.orb_minutes, o.entry_model, o.rr_target, o.confirm_bars
    FROM orb_outcomes o
    WHERE o.symbol = ? AND o.pnl_r IS NOT NULL
      AND (o.orb_label, o.orb_minutes, o.entry_model, o.rr_target, o.confirm_bars)
          IN (SELECT * FROM (VALUES ...))
    ORDER BY o.trading_day
""", [instrument]).fetchall()

# Partition in Python by (orb_label, orb_minutes, entry_model, rr_target, confirm_bars) → strategy_id
```

**Known limitation (from Bloomey review):** This query doesn't join `daily_features` for filter application — same as current code. Conservative bias preserved.

**Test strategy:** Existing `test_pbo.py` (9 tests) validates output correctness. Add one benchmark test comparing old vs new query count.

---

### Phase 3: Edge Family Batch UPDATEs (Item #4)

**File:** `scripts/tools/build_edge_families.py`
**Why third:** Builds on Phase 2 PBO fix. Batch UPDATEs eliminate ~900 roundtrips.

**Current state (lines 324-333):**
```python
for sid, _, _, _ in members:
    is_head = sid == head_sid
    con.execute("""UPDATE validated_setups
                   SET family_hash = ?, is_family_head = ?
                   WHERE strategy_id = ?""", [family_hash, is_head, sid])
```

**Fix:** Use DuckDB DataFrame replacement scan pattern:
```python
import pandas as pd

# Accumulate all (strategy_id, family_hash, is_family_head) tuples
updates = []
for family_hash, members in families.items():
    head_sid = ...
    for sid, _, _, _ in members:
        updates.append((sid, family_hash, sid == head_sid))

# Single batch UPDATE using replacement scan
df_updates = pd.DataFrame(updates, columns=["strategy_id", "family_hash", "is_family_head"])
con.execute("""
    UPDATE validated_setups vs
    SET family_hash = u.family_hash,
        is_family_head = u.is_family_head
    FROM df_updates u
    WHERE vs.strategy_id = u.strategy_id
""")
```

**DuckDB docs confirm:** "DataFrames (including Pandas, Polars, and Arrow) that are visible in the Python scope can be queried directly in SQL." This is the official replacement scan pattern.

**Test strategy:** Existing `build_edge_families.py` integration test (family hash determinism). Verify identical `validated_setups.family_hash` values before/after.

---

### Phase 4: Strategy Fitness Bulk Load (Item #1)

**File:** `strategy_fitness.py`
**Why fourth:** Largest codebase change. Most complex blast radius. Needs phases 1-3 stable first.

**Current N+1 pattern (lines 529-539):**
```python
for sid in strategy_ids:
    try:
        score = _compute_fitness_with_con(con, sid, as_of_date, rolling_months)
        scores.append(score)
    except Exception as e:
        logger.warning(...)
```

Each `_compute_fitness_with_con()` call triggers `_load_strategy_outcomes()` which runs:
1. One `orb_outcomes` query (per strategy)
2. One `daily_features` query (per strategy, if filter != NO_FILTER)
3. Filter matching in Python

**Fix:** Two-level bulk approach:

**Level 1 — Bulk outcome load:**
```python
# Load ALL outcomes for the instrument at once
all_outcomes = con.execute("""
    SELECT symbol, orb_label, orb_minutes, entry_model, rr_target, confirm_bars,
           trading_day, outcome, pnl_r, mae_r, mfe_r, entry_price, stop_price
    FROM orb_outcomes
    WHERE symbol = ? AND outcome IS NOT NULL
    ORDER BY trading_day
""", [instrument]).fetchall()

# Index by (orb_label, orb_minutes, entry_model, rr_target, confirm_bars)
outcome_index = defaultdict(list)
for row in all_outcomes:
    key = (row.orb_label, row.orb_minutes, row.entry_model, row.rr_target, row.confirm_bars)
    outcome_index[key].append(row_to_dict(row))
```

**Level 2 — Bulk daily_features load:**
```python
# Load ALL daily_features for instrument at once (bounded column set)
all_features = con.execute("""
    SELECT * FROM daily_features WHERE symbol = ? AND orb_minutes = ?
""", [instrument, orb_minutes]).fetchall()

# Index by (trading_day, orb_minutes)
feature_index = {(row.trading_day, row.orb_minutes): row for row in all_features}
```

**Then partition in Python:** For each strategy, look up its outcomes from `outcome_index`, apply filter using `feature_index`, compute metrics.

**Risk:** This changes the data loading pattern significantly. Must verify byte-identical FitnessScore outputs.

**Test strategy:**
1. Run `compute_portfolio_fitness(instrument="MGC")` with old code, capture JSON output.
2. Run with new code, diff outputs.
3. Existing unit tests cover classification logic (unchanged).

---

### Phase 5: Walkforward Binary Search (Item #6)

**File:** `trading_app/walkforward.py`
**Why fifth:** Small isolated change, easy to verify.

**Current state (lines 148, 155):**
```python
test_outcomes = [o for o in outcomes if window_start <= o["trading_day"] < window_end]
is_outcomes = [o for o in outcomes if o["trading_day"] < window_start]
```

**Fix:** Pre-sort outcomes once, use `bisect` for O(log N) window slicing:
```python
from bisect import bisect_left, bisect_right

# Pre-sort once (already sorted from DB ORDER BY, but make explicit)
outcomes.sort(key=lambda o: o["trading_day"])
trading_days = [o["trading_day"] for o in outcomes]

# Inside loop:
lo = bisect_left(trading_days, window_start)
hi = bisect_left(trading_days, window_end)
test_outcomes = outcomes[lo:hi]

is_lo = 0
is_hi = bisect_left(trading_days, window_start)
is_outcomes = outcomes[is_lo:is_hi]
```

**Test strategy:** Existing `test_walkforward.py` (4 WFE tests + existing WF tests) verify correctness.

---

### Phase 6: SELECT * Reduction (Item #5) — DEFERRED

**Status:** Not implementing now.

**Reason:** The `SELECT *` in `_load_strategy_outcomes()` (line 359) is explicitly documented with a comment explaining why it's correct:
> "Using SELECT * is safe here — daily_features has a bounded column set and we're already loading the full date range."

Moreover, composite filters (`VolumeFilter`, `ATRVelocityFilter`, DOW filters) need different subsets of columns, making a static column list fragile. The Phase 4 bulk-load already optimizes the query count from N to 1 — the column overfetch is negligible compared to the N+1 roundtrip elimination.

### Phase 7: Prepared Statements & pandas.apply() (Items #7, #8) — DEFERRED

**Status:** Not implementing now.

**Reason for #7:** DuckDB's own docs say "DuckDB's primary optimization focus is on larger, less frequent queries rather than rapid concurrent execution of many small queries." After Phase 4 eliminates the N+1 loop, the remaining queries are few and large — exactly DuckDB's sweet spot. Prepared statements add complexity for diminishing returns.

**Reason for #8:** Needs profiling first. Must grep for `.apply()` usage and measure whether vectorization is worthwhile. Not part of this design scope.

---

## T3 — DETAIL: Implementation Order

### Phase Dependency Graph

```
Phase 1 (Exception Hardening)
    ↓
Phase 2 (PBO Bulk Load) ← independent, but benefits from Phase 1 safety
    ↓
Phase 3 (Edge Family Batch UPDATE) ← depends on Phase 2 PBO changes
    ↓
Phase 4 (Fitness Bulk Load) ← depends on Phase 1 exception hardening
    ↓
Phase 5 (Walkforward Bisect) ← fully independent
```

### Per-Phase Verification

Each phase MUST pass before proceeding:

1. `python -m pytest tests/ -x -q` — all existing tests green
2. `python pipeline/check_drift.py` — drift checks pass
3. `python scripts/tools/audit_behavioral.py` — behavioral audit clean
4. Manual spot-check: compare output of affected command before/after

### Estimated Scope

| Phase | Files Modified | Lines Changed (est.) | Risk |
|-------|---------------|---------------------|------|
| 1 | 1 (`strategy_fitness.py`) | ~15 | LOW — exception type narrowing |
| 2 | 1 (`pbo.py`) | ~40 | LOW — isolated module, 9 tests |
| 3 | 1 (`build_edge_families.py`) | ~30 | LOW — batch UPDATE pattern is well-known |
| 4 | 1 (`strategy_fitness.py`) | ~80 | MEDIUM — largest change, must verify identical outputs |
| 5 | 1 (`walkforward.py`) | ~15 | LOW — mechanical refactor |

---

## T4 — VALIDATE: Risks & Mitigations

### Risk 1: Bulk Load Changes Output
**Threat:** Phase 4 bulk load produces different outcomes due to ordering or filtering differences.
**Mitigation:** Capture `compute_portfolio_fitness()` JSON output before/after, diff must be empty. Run for all 4 instruments.

### Risk 2: DuckDB VALUES Limit
**Threat:** Phase 2's `IN (VALUES ...)` clause may hit DuckDB row limits for large families.
**Mitigation:** DuckDB handles large VALUES lists well (tested up to 10K rows). Our max family size is ~50.

### Risk 3: pandas Import in build_edge_families.py
**Threat:** Phase 3 introduces a pandas import into a script that currently doesn't use it.
**Mitigation:** pandas is already a project dependency (used in pipeline/). If we want to avoid the import, we can use DuckDB's native `CREATE TEMP TABLE` instead.
**Alternative:** Use DuckDB temp table pattern instead of pandas DataFrame replacement scan:
```python
con.execute("CREATE TEMP TABLE _updates (strategy_id TEXT, family_hash TEXT, is_family_head BOOLEAN)")
con.executemany("INSERT INTO _updates VALUES (?, ?, ?)", updates)
con.execute("""UPDATE validated_setups vs SET ... FROM _updates u WHERE vs.strategy_id = u.strategy_id""")
con.execute("DROP TABLE _updates")
```
This avoids the pandas import entirely while achieving the same batch UPDATE.

### Risk 4: Memory Pressure from Bulk Load
**Threat:** Phase 4 loads ALL outcomes for an instrument into memory at once.
**Mitigation:** Our largest instrument (MNQ) has ~200K outcome rows × ~13 columns × ~50 bytes = ~130MB. Well within typical Python process memory.

### Rollback Plan
Each phase is a separate commit. If any phase causes issues, `git revert <commit>` restores the previous state. No schema changes — pure code optimization.

---

## Deferred Items

| Item | Reason | Revisit When |
|------|--------|-------------|
| `SELECT *` reduction (#5) | Documented as intentional, Phase 4 bulk load eliminates the N+1 | After Phase 4, if profiling shows column I/O is still significant |
| Prepared statements (#7) | DuckDB not optimized for high-frequency small queries; Phase 4 eliminates those | If post-Phase-4 profiling reveals parse overhead |
| pandas `.apply()` (#8) | Needs profiling grep first | Next fortification pass |
