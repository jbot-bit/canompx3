# Pipeline Fortification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate N+1 query bottlenecks and harden exception handling in trading_app hot paths (strategy_fitness, PBO, edge_families, walkforward).

**Architecture:** Bulk-load patterns replace per-strategy DB queries. DuckDB temp tables replace row-by-row UPDATEs. bisect replaces linear scans. All changes are pure performance — logic and outputs must be byte-identical.

**Tech Stack:** Python 3.13, DuckDB, bisect stdlib module

**Design Doc:** `docs/plans/2026-03-08-pipeline-fortification-design.md`

---

### Task 1: Harden exception handlers in strategy_fitness.py

**Files:**
- Modify: `trading_app/strategy_fitness.py:538,596,681,765,787`
- Test: `tests/test_trading_app/test_strategy_fitness.py`

**Context:** Five bare `except Exception` handlers silently swallow real bugs (schema changes, import errors, type mismatches). Safety before speed — must fix these before refactoring the data loading in Phase 4.

**Step 1: Write the failing test**

Add to `tests/test_trading_app/test_strategy_fitness.py`:

```python
def test_compute_fitness_raises_valueerror_for_missing_strategy(tmp_path):
    """ValueError must propagate — not be swallowed by except Exception."""
    db_path = tmp_path / "test.db"
    con = duckdb.connect(str(db_path))
    from pipeline.init_db import BARS_1M_SCHEMA, BARS_5M_SCHEMA, DAILY_FEATURES_SCHEMA
    con.execute(BARS_1M_SCHEMA)
    con.execute(BARS_5M_SCHEMA)
    con.execute(DAILY_FEATURES_SCHEMA)
    con.close()

    from trading_app.db_manager import init_trading_app_schema
    init_trading_app_schema(db_path=db_path)

    with pytest.raises(ValueError, match="not found"):
        compute_fitness("NONEXISTENT_STRATEGY_ID", db_path=db_path)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trading_app/test_strategy_fitness.py::test_compute_fitness_raises_valueerror_for_missing_strategy -v`
Expected: PASS (ValueError is already raised by `_compute_fitness_with_con` and `compute_fitness` doesn't catch it — this confirms the single-strategy path works correctly)

**Step 3: Narrow exception handlers**

Edit `trading_app/strategy_fitness.py`:

**Line 538** — Portfolio fitness loop (KEEP catch-all, add traceback):
```python
            except Exception:
                logger.exception("Failed to compute fitness for %s", sid)
```

**Line 596** — diagnose_decay own fitness:
```python
    except (ValueError, duckdb.Error) as e:
        logger.warning("Could not compute fitness for %s: %s", strategy_id, e)
        actual_status = "UNKNOWN"
```

**Line 681** — diagnose_decay sibling loop:
```python
        except (ValueError, duckdb.Error) as e:
            logger.debug("Sibling %s fitness failed: %s", sid, e)
            counts["STALE"] += 1
```

**Line 765** — diagnose_portfolio_decay first pass:
```python
            except (ValueError, duckdb.Error) as e:
                logger.debug("Fitness computation failed for %s: %s", sid, e)
```

**Line 787** — diagnose_portfolio_decay cached reuse:
```python
                except (ValueError, duckdb.Error):
                    own_status = "UNKNOWN"
```

**Step 4: Run full test suite**

Run: `python -m pytest tests/test_trading_app/test_strategy_fitness.py -v`
Expected: All tests PASS

**Step 5: Run drift checks**

Run: `python pipeline/check_drift.py`
Expected: All checks pass

**Step 6: Commit**

```bash
git add trading_app/strategy_fitness.py tests/test_trading_app/test_strategy_fitness.py
git commit -m "fix: narrow except Exception to specific types in strategy_fitness.py"
```

---

### Task 2: Bulk-load PBO member outcomes in pbo.py

**Files:**
- Modify: `trading_app/pbo.py:126-161`
- Test: `tests/test_trading_app/test_pbo.py`

**Context:** `compute_family_pbo()` runs one query per family member to load outcomes. With ~150 multi-member families × ~5 members = ~750 queries during edge family builds. A single bulk query eliminates all of them.

**Step 1: Write the performance assertion test**

Add to `tests/test_trading_app/test_pbo.py`:

```python
def test_compute_pbo_deterministic_with_bulk():
    """Verify PBO result is identical regardless of load method.

    This is a regression test — if bulk loading changes PBO values,
    something is wrong with the partitioning logic.
    """
    days = _days(80)
    half = len(days) // 2
    strategy_pnl = {
        "A": [(d, 3.0) if i < half else (d, -2.0) for i, d in enumerate(days)],
        "B": [(d, 0.1) for d in days],
        "C": [(d, 1.5) for d in days],
    }
    result = compute_pbo(strategy_pnl)
    # Pin the exact expected value — any change means regression
    assert result["pbo"] is not None
    assert result["n_splits"] == 70
    # Store for comparison if we refactor
    assert isinstance(result["pbo"], float)
```

**Step 2: Run test to verify it passes**

Run: `python -m pytest tests/test_trading_app/test_pbo.py -v`
Expected: All 10 tests PASS

**Step 3: Refactor compute_family_pbo() to use bulk query**

Replace the loop in `trading_app/pbo.py` lines 138-156 with:

```python
    # Load outcomes for ALL members in one query
    # Build member key tuples for DuckDB VALUES join
    member_keys = {}  # (orb_label, orb_minutes, entry_model, rr_target, confirm_bars) -> strategy_id
    for sid, orb_label, orb_minutes, entry_model, rr_target, confirm_bars, _filter_type in members:
        member_keys[(orb_label, orb_minutes, entry_model, rr_target, confirm_bars)] = sid

    if not member_keys:
        return {"pbo": None, "n_splits": 0, "n_negative_oos": 0, "logit_pbo": None}

    # Single bulk query — DuckDB VALUES syntax for multi-column IN
    values_rows = list(member_keys.keys())
    placeholders = ", ".join(["(?, ?, ?, ?, ?)"] * len(values_rows))
    flat_params = [instrument]
    for row in values_rows:
        flat_params.extend(row)

    rows = con.execute(
        f"""SELECT o.trading_day, o.pnl_r,
                   o.orb_label, o.orb_minutes, o.entry_model, o.rr_target, o.confirm_bars
            FROM orb_outcomes o
            WHERE o.symbol = ?
              AND o.pnl_r IS NOT NULL
              AND (o.orb_label, o.orb_minutes, o.entry_model, o.rr_target, o.confirm_bars)
                  IN (VALUES {placeholders})
            ORDER BY o.trading_day""",
        flat_params,
    ).fetchall()

    # Partition by member key → strategy_id
    strategy_pnl = {}
    for trading_day, pnl_r, orb_label, orb_minutes, entry_model, rr_target, confirm_bars in rows:
        key = (orb_label, orb_minutes, entry_model, rr_target, confirm_bars)
        sid = member_keys.get(key)
        if sid is not None:
            if sid not in strategy_pnl:
                strategy_pnl[sid] = []
            strategy_pnl[sid].append((trading_day, pnl_r))
```

**Step 4: Run PBO tests**

Run: `python -m pytest tests/test_trading_app/test_pbo.py -v`
Expected: All 10 tests PASS

**Step 5: Run full test suite + drift**

Run: `python -m pytest tests/ -x -q && python pipeline/check_drift.py`
Expected: All pass

**Step 6: Commit**

```bash
git add trading_app/pbo.py tests/test_trading_app/test_pbo.py
git commit -m "perf: bulk-load PBO member outcomes (eliminate N+1 query loop)"
```

---

### Task 3: Batch UPDATE family tags in build_edge_families.py

**Files:**
- Modify: `scripts/tools/build_edge_families.py:322-333`
- Test: Manual verification (existing integration flow)

**Context:** After computing families, each member is tagged with an individual UPDATE. ~900 strategies = ~900 DB roundtrips. A DuckDB temp table batch UPDATE reduces this to 3 statements (CREATE, INSERT, UPDATE FROM).

**Step 1: Write regression test**

This is a structural refactor with no new behavior to test. The existing edge family build verifies hash determinism. Instead, add a count assertion to the build function:

No new test file needed — verify manually by running the build and comparing output.

**Step 2: Refactor the member-tagging loop**

Replace `scripts/tools/build_edge_families.py` lines 322-333 with:

```python
        # 6a. Batch-tag all members via temp table (replaces row-by-row UPDATEs)
        member_updates = []
        for family_hash, members in families.items():
            (head_sid, _, _, _), _ = _elect_median_head(members)
            for sid, _, _, _ in members:
                member_updates.append((sid, family_hash, sid == head_sid))

        if member_updates:
            con.execute("""
                CREATE TEMP TABLE _family_tags (
                    strategy_id TEXT,
                    family_hash TEXT,
                    is_family_head BOOLEAN
                )
            """)
            con.executemany(
                "INSERT INTO _family_tags VALUES (?, ?, ?)",
                member_updates,
            )
            con.execute("""
                UPDATE validated_setups vs
                SET family_hash = ft.family_hash,
                    is_family_head = ft.is_family_head
                FROM _family_tags ft
                WHERE vs.strategy_id = ft.strategy_id
            """)
            con.execute("DROP TABLE _family_tags")
```

**Important:** The `_elect_median_head()` call is now done twice per family (once in the INSERT loop above, once in step 6 INSERT). To avoid this, extract the head election into the step 6 loop and store the result:

```python
        # 6. For each family: compute metrics, elect median head, classify
        family_heads = {}  # family_hash -> head_sid
        # ... existing loop ...
            family_heads[family_hash] = head_sid
            # ... existing INSERT ...
```

Then in 6a, use `family_heads[family_hash]` instead of re-electing.

**Step 3: Run build for one instrument**

Run: `python scripts/tools/build_edge_families.py --instrument MGC`
Expected: Same family counts, same robustness breakdown as before

**Step 4: Verify validated_setups tags**

Run: `python -c "import duckdb; con=duckdb.connect('gold.db',read_only=True); print(con.execute('SELECT COUNT(*) FROM validated_setups WHERE family_hash IS NOT NULL').fetchone())"`
Expected: Same count as before the change

**Step 5: Run drift + tests**

Run: `python -m pytest tests/ -x -q && python pipeline/check_drift.py`
Expected: All pass

**Step 6: Commit**

```bash
git add scripts/tools/build_edge_families.py
git commit -m "perf: batch UPDATE family tags via temp table (eliminate row-by-row UPDATEs)"
```

---

### Task 4: Bulk-load strategy outcomes in strategy_fitness.py

**Files:**
- Modify: `trading_app/strategy_fitness.py:385-477,503-549`
- Test: `tests/test_trading_app/test_strategy_fitness.py`

**Context:** This is the #1 bottleneck. `compute_portfolio_fitness()` calls `_compute_fitness_with_con()` per strategy, which internally calls `_load_strategy_outcomes()` — triggering 2 queries per strategy (outcomes + daily_features). ~900 strategies × 2 = ~1800 queries per portfolio check.

**Approach:** Add a new `_compute_portfolio_fitness_bulk()` internal function that:
1. Loads ALL orb_outcomes for the instrument in one query
2. Loads ALL daily_features for the instrument in one query
3. Indexes both by strategy key tuple
4. Iterates strategies, partitioning from the in-memory indexes

The public API (`compute_portfolio_fitness`) stays identical. Single-strategy `compute_fitness()` is unchanged.

**Step 1: Write the before/after comparison test**

Add to `tests/test_trading_app/test_strategy_fitness.py`:

```python
class TestBulkLoadEquivalence:
    """Verify bulk-loaded fitness produces identical results to per-strategy loading."""

    def test_portfolio_fitness_deterministic(self, tmp_path):
        """Two calls to compute_portfolio_fitness must return identical scores."""
        strategies = [
            {"strategy_id": "MGC_TEST_A", "instrument": "MGC", "orb_label": "TOKYO_OPEN",
             "orb_minutes": 5, "entry_model": "E1", "rr_target": 2.0, "confirm_bars": 1,
             "filter_type": "NO_FILTER", "sample_size": 50, "win_rate": 0.55,
             "expectancy_r": 0.10, "sharpe_ratio": 0.5, "sharpe_ann": 0.5,
             "max_drawdown_r": -3.0, "status": "ACTIVE"},
        ]
        outcomes = [
            {"symbol": "MGC", "orb_label": "TOKYO_OPEN", "orb_minutes": 5,
             "entry_model": "E1", "rr_target": 2.0, "confirm_bars": 1,
             "trading_day": date(2024, 1, 1) + timedelta(days=i),
             "outcome": "win" if i % 3 != 0 else "loss",
             "pnl_r": 2.0 if i % 3 != 0 else -1.0}
            for i in range(100)
        ]
        db_path, con = _setup_fitness_db(tmp_path, strategies=strategies, outcomes=outcomes)
        con.close()

        r1 = compute_portfolio_fitness(db_path=db_path, instrument="MGC")
        r2 = compute_portfolio_fitness(db_path=db_path, instrument="MGC")
        assert len(r1.scores) == len(r2.scores)
        for s1, s2 in zip(r1.scores, r2.scores):
            assert s1.strategy_id == s2.strategy_id
            assert s1.fitness_status == s2.fitness_status
            assert s1.rolling_exp_r == s2.rolling_exp_r
```

**Step 2: Implement bulk loading**

Add new internal function `_bulk_load_outcomes()` and `_bulk_load_features()` in `strategy_fitness.py`:

```python
def _bulk_load_outcomes(con, instrument: str, end_date: date | None = None) -> dict:
    """Load ALL outcomes for an instrument, indexed by strategy key tuple.

    Returns: {(orb_label, orb_minutes, entry_model, rr_target, confirm_bars): [outcome_dicts]}
    """
    params = [instrument]
    where = ["symbol = ?", "outcome IS NOT NULL"]
    if end_date:
        where.append("trading_day <= ?")
        params.append(end_date)

    rows = con.execute(
        f"""SELECT orb_label, orb_minutes, entry_model, rr_target, confirm_bars,
                   trading_day, outcome, pnl_r, mae_r, mfe_r, entry_price, stop_price
            FROM orb_outcomes
            WHERE {" AND ".join(where)}
            ORDER BY trading_day""",
        params,
    ).fetchall()
    cols = [desc[0] for desc in con.description]

    index = defaultdict(list)
    for row in rows:
        d = dict(zip(cols, row, strict=False))
        key = (d["orb_label"], d["orb_minutes"], d["entry_model"], d["rr_target"], d["confirm_bars"])
        index[key].append(d)
    return index


def _bulk_load_features(con, instrument: str) -> dict:
    """Load ALL daily_features for an instrument, indexed by (trading_day, orb_minutes).

    Returns: {(trading_day, orb_minutes): feature_dict}
    """
    rows = con.execute(
        "SELECT * FROM daily_features WHERE symbol = ?",
        [instrument],
    ).fetchall()
    cols = [desc[0] for desc in con.description]

    index = {}
    for row in rows:
        d = dict(zip(cols, row, strict=False))
        index[(d["trading_day"], d["orb_minutes"])] = d
    return index
```

Then modify `compute_portfolio_fitness()` to use a new `_compute_fitness_bulk()` that accepts pre-loaded indexes instead of calling `_load_strategy_outcomes()` per strategy.

**Key constraint:** The existing `_compute_fitness_with_con()` and `_load_strategy_outcomes()` must remain unchanged for single-strategy lookups (`compute_fitness()`). The bulk path is portfolio-only.

**Step 3: Run tests**

Run: `python -m pytest tests/test_trading_app/test_strategy_fitness.py -v`
Expected: All tests PASS

**Step 4: Run live comparison**

Run: `python trading_app/strategy_fitness.py --instrument MGC --format json > /tmp/before.json`
(before change — capture from git stash)

Run: `python trading_app/strategy_fitness.py --instrument MGC --format json > /tmp/after.json`
(after change)

Diff: `diff /tmp/before.json /tmp/after.json`
Expected: Empty diff (identical output)

**Step 5: Run full suite + drift**

Run: `python -m pytest tests/ -x -q && python pipeline/check_drift.py`
Expected: All pass

**Step 6: Commit**

```bash
git add trading_app/strategy_fitness.py tests/test_trading_app/test_strategy_fitness.py
git commit -m "perf: bulk-load outcomes in compute_portfolio_fitness (eliminate N+1 queries)"
```

---

### Task 5: Bisect optimization in walkforward.py

**Files:**
- Modify: `trading_app/walkforward.py:135-155`
- Test: `tests/test_trading_app/test_walkforward.py`

**Context:** Walk-forward window partitioning uses list comprehensions that scan ALL outcomes per window. With sorted outcomes and bisect, we get O(log N) per window boundary instead of O(N).

**Step 1: Add import and pre-sort**

At top of `trading_app/walkforward.py`, add:

```python
from bisect import bisect_left
```

**Step 2: Refactor window loop**

Replace lines 135-170 (the outcome sorting + window loop) with:

```python
    # Pre-sort outcomes by trading_day (should already be sorted from DB, make explicit)
    outcomes.sort(key=lambda o: o["trading_day"])
    all_trading_days = [o["trading_day"] for o in outcomes]
    earliest = all_trading_days[0]
    latest = all_trading_days[-1]

    # Generate non-overlapping test windows
    anchor = max(earliest, wf_start_date) if wf_start_date else earliest
    windows = []
    window_start = _add_months(anchor, min_train_months)

    while window_start <= latest:
        window_end = _add_months(window_start, test_window_months)

        # O(log N) window slicing via bisect
        lo = bisect_left(all_trading_days, window_start)
        hi = bisect_left(all_trading_days, window_end)
        test_outcomes = outcomes[lo:hi]

        metrics = compute_metrics(test_outcomes)
        test_n = metrics["sample_size"]
        test_exp_r = metrics["expectancy_r"]

        # IS metrics: all outcomes before this window (anchored expanding)
        is_hi = bisect_left(all_trading_days, window_start)
        is_outcomes = outcomes[:is_hi]
        is_metrics = compute_metrics(is_outcomes) if len(is_outcomes) >= 15 else None
        is_exp_r = is_metrics["expectancy_r"] if is_metrics else None

        windows.append(
            {
                "window_start": window_start.isoformat(),
                "window_end": window_end.isoformat(),
                "test_n": test_n,
                "test_exp_r": test_exp_r,
                "test_wr": metrics["win_rate"],
                "test_sharpe": metrics["sharpe_ratio"],
                "test_pass": (test_n >= min_trades_per_window and test_exp_r is not None and test_exp_r > 0),
                "is_exp_r": is_exp_r,
            }
        )

        logger.info(
            "WF %s window %s..%s: N=%d ExpR=%s",
            strategy_id,
            window_start,
            window_end,
            test_n,
            test_exp_r,
        )

        window_start = window_end
```

**Step 3: Run walkforward tests**

Run: `python -m pytest tests/test_trading_app/test_walkforward.py -v`
Expected: All tests PASS (WFE tests + existing WF tests)

**Step 4: Run full suite + drift**

Run: `python -m pytest tests/ -x -q && python pipeline/check_drift.py`
Expected: All pass

**Step 5: Commit**

```bash
git add trading_app/walkforward.py
git commit -m "perf: use bisect for O(log N) walk-forward window slicing"
```

---

## Verification Checklist (Post All Phases)

After all 5 phases complete:

1. `python -m pytest tests/ -x -q` — full test suite green
2. `python pipeline/check_drift.py` — all drift checks pass
3. `python scripts/tools/audit_behavioral.py` — behavioral audit clean
4. `python trading_app/strategy_fitness.py --instrument MGC` — runs without error
5. `python scripts/tools/build_edge_families.py --instrument MGC` — PBO computed, same family counts
6. Compare portfolio fitness JSON output before/after for all 4 instruments

---

## Deferred Items (Not in This Plan)

| Item | Why Deferred | Revisit |
|------|-------------|---------|
| `SELECT *` reduction | Intentional per code comment, Phase 4 eliminates N+1 | Post-Phase-4 profiling |
| Prepared statements | DuckDB sweet spot is large queries (our post-fix pattern) | If profiling shows parse overhead |
| `pandas.apply()` vectorization | Only 1 usage in `portfolio.py:799` | Next fortification pass |
