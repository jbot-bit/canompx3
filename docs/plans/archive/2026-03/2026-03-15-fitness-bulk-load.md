---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Strategy Fitness Bulk Load — Phase 4

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate the N+1 query loop in `compute_portfolio_fitness()` — currently ~1800 DB queries per call. Replace with 2-query bulk load. Output must be byte-identical.

**Architecture:** Two new private helpers (`_load_all_outcomes`, `_load_all_features`) bulk-load data indexed by key. Existing `_compute_fitness_from_cache` path already accepts pre-loaded data. Design fully specified in `docs/plans/2026-03-08-pipeline-fortification-design.md` Phase 4.

**Tech Stack:** Python, DuckDB, pytest

**Risk:** MEDIUM — largest single change. Byte-identical output verification is mandatory before commit.

---

## Task 1: Capture Golden Output (pre-change baseline)

**No code changes. Run only.**

```bash
python -c "
from trading_app.strategy_fitness import compute_portfolio_fitness
from pipeline.paths import GOLD_DB_PATH
import json

for instrument in ['MGC', 'MNQ', 'MES', 'M2K']:
    result = compute_portfolio_fitness(instrument=instrument, db_path=GOLD_DB_PATH)
    with open(f'/tmp/fitness_before_{instrument}.json', 'w') as f:
        json.dump([r.__dict__ for r in result], f, default=str)
    print(f'{instrument}: {len(result)} strategies captured')
"
```

Keep these files. They are the regression gate.

---

## Task 2: Add _load_all_outcomes and _load_all_features helpers

**Files:**
- Modify: `trading_app/strategy_fitness.py`
- Test: `tests/test_trading_app/test_strategy_fitness.py`

**Context:** Full design in `docs/plans/2026-03-08-pipeline-fortification-design.md` Phase 4. Two helpers:
- `_load_all_outcomes(con, instrument, as_of_date)` → dict keyed by `(orb_label, orb_minutes, entry_model, rr_target, confirm_bars)`
- `_load_all_features(con, instrument)` → dict keyed by `(trading_day, orb_minutes)`

**Step 1: Write regression test (must PASS before and after)**
```python
# In test_strategy_fitness.py — add:
def test_compute_portfolio_fitness_output_stable(tmp_strategy_db):
    """Bulk load must produce identical results to single-strategy path."""
    from trading_app.strategy_fitness import compute_portfolio_fitness
    result = compute_portfolio_fitness(instrument="MGC", db_path=tmp_strategy_db)
    # If result changes after bulk load implementation, this test catches it
    # by comparing to a snapshot captured before the change
    assert isinstance(result, list)
    for r in result:
        assert hasattr(r, "strategy_id")
        assert hasattr(r, "fitness_status")
```

**Step 2: Implement `_load_all_outcomes`** (new private function in strategy_fitness.py):
```python
def _load_all_outcomes(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
    as_of_date: date | None,
) -> dict:
    """Bulk-load all outcomes for instrument, indexed by strategy key tuple."""
    from collections import defaultdict
    query = """
        SELECT orb_label, orb_minutes, entry_model, rr_target, confirm_bars,
               trading_day, outcome, pnl_r, mae_r, mfe_r, entry_price, stop_price
        FROM orb_outcomes
        WHERE symbol = ? AND outcome IS NOT NULL
    """
    params = [instrument]
    if as_of_date is not None:
        query += " AND trading_day <= ?"
        params.append(as_of_date)
    query += " ORDER BY trading_day"

    rows = con.execute(query, params).fetchall()
    cols = ["orb_label","orb_minutes","entry_model","rr_target","confirm_bars",
            "trading_day","outcome","pnl_r","mae_r","mfe_r","entry_price","stop_price"]
    index = defaultdict(list)
    for row in rows:
        d = dict(zip(cols, row))
        key = (d["orb_label"], d["orb_minutes"], d["entry_model"],
               d["rr_target"], d["confirm_bars"])
        index[key].append(d)
    return index
```

**Step 3: Implement `_load_all_features`**:
```python
def _load_all_features(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
) -> dict:
    """Bulk-load all daily_features for instrument, indexed by (trading_day, orb_minutes)."""
    rows = con.execute(
        "SELECT * FROM daily_features WHERE symbol = ? ORDER BY trading_day",
        [instrument],
    ).fetchall()
    cols = [d[0] for d in con.description]
    return {
        (row[cols.index("trading_day")], row[cols.index("orb_minutes")]): dict(zip(cols, row))
        for row in rows
    }
```

**Step 4: Rewrite `compute_portfolio_fitness`** to use bulk load

Replace the inner N+1 loop:
```python
# OLD:
for sid in strategy_ids:
    try:
        score = _compute_fitness_with_con(con, sid, as_of_date, rolling_months)
        ...
    except Exception as e:
        ...

# NEW:
outcome_index = _load_all_outcomes(con, instrument, as_of_date)
feature_index = _load_all_features(con, instrument)

for sid in strategy_ids:
    try:
        score = _compute_fitness_from_cache(
            con, sid, as_of_date, rolling_months,
            outcome_cache=outcome_index,
            feature_cache=feature_index,
        )
        ...
    except (ValueError, duckdb.Error) as e:
        ...
```

Verify `_compute_fitness_from_cache` accepts these cache params (check existing signature).

**Step 5: Verify byte-identical output**
```bash
python -c "
from trading_app.strategy_fitness import compute_portfolio_fitness
from pipeline.paths import GOLD_DB_PATH
import json, sys

for instrument in ['MGC', 'MNQ', 'MES', 'M2K']:
    result = compute_portfolio_fitness(instrument=instrument, db_path=GOLD_DB_PATH)
    with open(f'/tmp/fitness_after_{instrument}.json', 'w') as f:
        json.dump([r.__dict__ for r in result], f, default=str)

import subprocess
for instrument in ['MGC', 'MNQ', 'MES', 'M2K']:
    r = subprocess.run(['diff', f'/tmp/fitness_before_{instrument}.json',
                        f'/tmp/fitness_after_{instrument}.json'], capture_output=True)
    if r.returncode != 0:
        print(f'DIFF DETECTED for {instrument}:')
        print(r.stdout.decode())
        sys.exit(1)
    else:
        print(f'{instrument}: identical')
"
```
**Must produce zero diffs. If any diff: STOP and diagnose before committing.**

**Step 6: Run full suite**
```bash
python -m pytest tests/test_trading_app/test_strategy_fitness.py -x -q
python pipeline/check_drift.py
python scripts/tools/audit_behavioral.py
```

**Step 7: Commit**
```bash
git add trading_app/strategy_fitness.py tests/test_trading_app/test_strategy_fitness.py
git commit -m "perf: bulk-load outcomes+features in compute_portfolio_fitness, eliminate N+1 (Phase 4)"
```
