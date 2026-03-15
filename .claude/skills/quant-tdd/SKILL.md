---
name: quant-tdd
description: Write tests first, then implement for
---
Write tests first, then implement for: $ARGUMENTS

Use when: "write tests", "test first", "add tests for", "TDD", implementing any feature or bugfix in pipeline/ or trading_app/

## Quant TDD Protocol

Test-driven development for a quantitative trading pipeline. Tests come FIRST. Implementation follows.

### Step 1: Find the Right Test File

Check `TEST_MAP` in `.claude/hooks/post-edit-pipeline.py`:

| Source File | Test File |
|-------------|-----------|
| `pipeline/build_daily_features.py` | `tests/test_pipeline/test_build_daily_features.py` |
| `pipeline/build_bars_5m.py` | `tests/test_pipeline/test_build_bars_5m.py` |
| `pipeline/ingest_dbn.py` | `tests/test_pipeline/test_ingest.py` |
| `pipeline/check_drift.py` | `tests/test_pipeline/test_check_drift.py` |
| `pipeline/dst.py` | `tests/test_pipeline/test_dst.py` |
| `pipeline/init_db.py` | `tests/test_pipeline/test_schema.py` |
| `pipeline/asset_configs.py` | `tests/test_pipeline/test_asset_configs.py` |
| `trading_app/outcome_builder.py` | `tests/test_trading_app/test_outcome_builder.py` |
| `trading_app/strategy_discovery.py` | `tests/test_trading_app/test_strategy_discovery.py` |
| `trading_app/strategy_validator.py` | `tests/test_trading_app/test_strategy_validator.py` |
| `trading_app/entry_rules.py` | `tests/test_trading_app/test_entry_rules.py` |
| `trading_app/paper_trader.py` | `tests/test_trading_app/test_paper_trader.py` |
| `trading_app/config.py` | `tests/test_trading_app/test_config.py` |

If no mapping exists, create the test file following the pattern: `tests/test_<module>/test_<filename>.py`

### Step 2: Read Existing Tests

Before writing new tests, read the existing test file to understand:
- Test style and patterns used
- Fixtures available (especially DuckDB in-memory fixtures)
- What's already tested (don't duplicate)

### Step 3: Write the Failing Test

Quant pipeline test patterns:

**Tripwire tests** (verify invariants hold):
```python
def test_cost_model_covers_all_active_instruments():
    """Every active instrument must have a cost spec."""
    from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
    from pipeline.cost_model import COST_SPECS
    for inst in ACTIVE_ORB_INSTRUMENTS:
        assert inst in COST_SPECS, f"Missing cost spec for {inst}"
```

**JOIN correctness tests** (verify no row inflation):
```python
def test_join_does_not_inflate_rows(con):
    """Join on all 3 keys: trading_day, symbol, orb_minutes."""
    before = con.execute("SELECT COUNT(*) FROM orb_outcomes WHERE symbol='MGC'").fetchone()[0]
    after = con.execute("""
        SELECT COUNT(*) FROM orb_outcomes o
        JOIN daily_features d ON o.trading_day = d.trading_day
          AND o.symbol = d.symbol AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol='MGC'
    """).fetchone()[0]
    assert after == before, f"JOIN inflated rows: {before} -> {after}"
```

**Idempotent tests** (verify DELETE+INSERT round-trip):
```python
def test_rebuild_is_idempotent(con):
    """Running twice produces identical results."""
    run_build(con, instrument='MGC', start='2024-01-01', end='2024-01-31')
    count_1 = get_count(con)
    run_build(con, instrument='MGC', start='2024-01-01', end='2024-01-31')
    count_2 = get_count(con)
    assert count_1 == count_2, f"Not idempotent: {count_1} vs {count_2}"
```

**Pipeline gate tests** (verify reject paths):
```python
def test_rejects_future_dates():
    """Pipeline must reject dates in the future."""
    with pytest.raises(ValueError, match="future"):
        ingest(instrument='MGC', start='2099-01-01', end='2099-12-31')
```

### Step 4: Run the Test -- Verify It FAILS

```bash
python -m pytest tests/<test_file>::<test_name> -v
```

If it passes, your test doesn't test what you think. Fix the test.

### Step 5: Write Minimal Implementation

Write the smallest code that makes the test pass. No extras.

### Step 6: Run the Test -- Verify It PASSES

```bash
python -m pytest tests/<test_file>::<test_name> -v
```

### Step 7: Run Full Suite

```bash
python -m pytest tests/ -x -q
```

No regressions allowed.

### Step 8: Commit

```bash
git add <test_file> <impl_file>
git commit -m "feat/fix: <description>"
```

### Rules

- NEVER write implementation before the test
- NEVER test against stale outcomes -- rebuild first if schema changed
- NEVER hardcode expected values that come from canonical sources (import them)
- Tests must verify BEHAVIOR, not implementation details
- If a test needs gold.db data, use an in-memory DuckDB fixture, not the real DB
