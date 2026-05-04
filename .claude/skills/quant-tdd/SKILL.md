---
name: quant-tdd
description: Use when implementing any feature or bugfix in pipeline/ or trading_app/, before writing implementation code
---

# Quant Pipeline TDD

No production code without a failing test first.

## Cycle

```
RED: write one failing test → verify it fails correctly
GREEN: write minimal code to pass → verify all tests pass
REFACTOR: clean up → verify still green
REPEAT
```

## RED — Write Failing Test

### 1. Find Test File

Check `TEST_MAP` in `.claude/hooks/post-edit-pipeline.py`. If no mapping: `tests/test_<module>/test_<filename>.py`.

### 2. Read Existing Tests

Understand fixtures, style, coverage. Shared fixture `tmp_db` in `tests/conftest.py` creates in-memory DuckDB with canonical schema via `pipeline.init_db`.

### 3. Write Test — One Behavior

**Tripwire** (canonical source invariant):
```python
def test_cost_model_covers_active():
    from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
    from pipeline.cost_model import COST_SPECS
    for inst in ACTIVE_ORB_INSTRUMENTS:
        assert inst in COST_SPECS
```

**JOIN correctness** (no row inflation):
```python
def test_join_no_inflate(tmp_db):
    before = tmp_db.execute(
        "SELECT COUNT(*) FROM orb_outcomes WHERE symbol='MGC'"
    ).fetchone()[0]
    after = tmp_db.execute(
        "SELECT COUNT(*) FROM orb_outcomes o "
        "JOIN daily_features d ON o.trading_day=d.trading_day "
        "AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes "
        "WHERE o.symbol='MGC'"
    ).fetchone()[0]
    assert after == before
```

**Idempotency** (DELETE+INSERT re-run safe):
```python
def test_rebuild_idempotent(tmp_db):
    build(tmp_db, instrument="MGC", start="2024-01-01", end="2024-01-31")
    count_1 = tmp_db.execute("SELECT COUNT(*) FROM target_table").fetchone()[0]
    build(tmp_db, instrument="MGC", start="2024-01-01", end="2024-01-31")
    count_2 = tmp_db.execute("SELECT COUNT(*) FROM target_table").fetchone()[0]
    assert count_1 == count_2
```

**Fail-closed** (validation aborts on bad input):
```python
def test_rejects_bad_input(tmp_db):
    with pytest.raises(ValueError, match="expected pattern"):
        process(tmp_db, bad_data)
```

### 4. Verify RED

```bash
python -m pytest tests/<file>::<test> -v
```

Must **fail** (not error). Failure message must match expected missing behavior. If it passes: wrong test. If it errors: fix import/fixture/typo, re-run.

---

## GREEN — Minimal Implementation

Smallest code to pass the test. If touching `pipeline/` or `trading_app/`, run `/blast-radius [target file]` first.

### Verify GREEN

```bash
python -m pytest tests/<file>::<test> -v
python -m pytest tests/ -x -q
```

Your test passes + no regressions. If your test fails: fix code, not test. If other tests fail: fix now.

---

## REFACTOR — Clean Up

Remove duplication, improve names, extract helpers. All tests stay green. No new behavior.

---

## Project Constraints

| Rule | Source |
|------|--------|
| In-memory DuckDB only — never real `gold.db` | `tests/conftest.py` |
| Import canonical sources — never hardcode lists/constants | `integrity-guardian.md` §2 |
| `tmp_db` fixture creates schema via `pipeline.init_db` | `tests/conftest.py:23` |
| Fail fast: `pytest -x` flag always | `CLAUDE.md` |
| Companion test file: check `TEST_MAP` | `.claude/hooks/post-edit-pipeline.py:18` |
| `/blast-radius` before touching production code | `auto-skill-routing.md` |
| Drift check after implementation | `python pipeline/check_drift.py` |

## Canonical Sources (import, never hardcode)

| Data | Import |
|------|--------|
| Active instruments | `pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS` |
| All instrument configs | `pipeline.asset_configs.ASSET_CONFIGS` |
| Session catalog | `pipeline.dst.SESSION_CATALOG` |
| Entry models / filters | `trading_app.config` |
| Cost specs | `pipeline.cost_model.COST_SPECS` |
| DB path | `pipeline.paths.GOLD_DB_PATH` |

## Completion Gate

- [ ] Every new function has a test
- [ ] Each test failed before implementation (RED verified)
- [ ] Each test passes after implementation (GREEN verified)
- [ ] Full suite passes (`python -m pytest tests/ -x -q`)
- [ ] No hardcoded values — canonical imports only
- [ ] Drift checks pass (`python pipeline/check_drift.py`)
