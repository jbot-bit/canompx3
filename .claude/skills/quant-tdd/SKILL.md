---
name: quant-tdd
description: Write tests first, then implement for
---
Write tests first, then implement for: $ARGUMENTS

Use when: "write tests", "test first", "TDD", implementing any feature/bugfix in pipeline/ or trading_app/

## Step 1: Find Test File

Check `TEST_MAP` in `.claude/hooks/post-edit-pipeline.py` for the companion test. If no mapping, create `tests/test_<module>/test_<filename>.py`.

## Step 2: Read Existing Tests

Understand test style, fixtures (especially DuckDB in-memory), what's already covered.

## Step 3: Write Failing Test

Key patterns for this pipeline:

**Tripwire** (invariants):
```python
def test_cost_model_covers_active():
    from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
    from pipeline.cost_model import COST_SPECS
    for inst in ACTIVE_ORB_INSTRUMENTS:
        assert inst in COST_SPECS
```

**JOIN correctness** (no row inflation):
```python
def test_join_no_inflate(con):
    before = con.execute("SELECT COUNT(*) FROM orb_outcomes WHERE symbol='MGC'").fetchone()[0]
    after = con.execute("SELECT COUNT(*) FROM orb_outcomes o JOIN daily_features d ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes WHERE o.symbol='MGC'").fetchone()[0]
    assert after == before
```

## Step 4: Run — Verify FAILS

```bash
python -m pytest tests/<file>::<test> -v
```

If it passes, your test doesn't test what you think. Fix the test.

## Step 5: Implement

Write smallest code that makes the test pass. Before implementing: if change touches `pipeline/` or `trading_app/`, run `/blast-radius [target file]`.

## Step 6: Run — Verify PASSES + Full Suite

```bash
python -m pytest tests/<file>::<test> -v
python -m pytest tests/ -x -q
```

## Step 7: Commit

## Rules

- NEVER write implementation before the test
- NEVER hardcode expected values from canonical sources — import them
- Tests verify BEHAVIOR, not implementation details
- In-memory DuckDB fixture for tests, never real gold.db
