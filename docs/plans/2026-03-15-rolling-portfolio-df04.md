# Rolling Portfolio DF-04: orb_minutes Hardcode

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Fix the hardcoded `orb_minutes=5` in `compute_day_of_week_stats()` so DOW filter eligibility uses the correct aperture when called with non-5m data.

**Architecture:** One function, one caller. Add `orb_minutes: int = 5` param (backward-compat default), pass to SQL, remove TODO comment.

**Tech Stack:** Python, DuckDB, pytest

---

## Task 1: Fix compute_day_of_week_stats orb_minutes param

**Files:**
- Modify: `trading_app/rolling_portfolio.py`
- Test: `tests/test_trading_app/test_rolling_portfolio.py`

**Context:**
- `FamilyResult` has no `orb_minutes` field — can't read from families
- Only caller is `rolling_portfolio.py:591` (CLI `__main__` block) — no external callers confirmed by Ralph Loop iter history
- Default `orb_minutes=5` preserves current behavior exactly

**Step 1: Write failing test**
```python
# In test_rolling_portfolio.py — add:
def test_compute_dow_stats_respects_orb_minutes(tmp_path):
    """orb_minutes param must reach the SQL query (not hardcoded to 5)."""
    import duckdb
    from trading_app.rolling_portfolio import compute_day_of_week_stats, FamilyResult

    db = tmp_path / "test.db"
    con = duckdb.connect(str(db))
    # Create daily_features with ONLY orb_minutes=15 rows
    con.execute("""
        CREATE TABLE daily_features (
            symbol TEXT, trading_day DATE, orb_minutes INTEGER,
            orb_CME_REOPEN_size DOUBLE
        )
    """)
    con.execute("""
        INSERT INTO daily_features VALUES
        ('MGC', '2026-01-02', 15, 5.0),
        ('MGC', '2026-01-05', 15, 5.0)
    """)
    # orb_outcomes needed for the inner query
    con.execute("""
        CREATE TABLE orb_outcomes (
            symbol TEXT, orb_label TEXT, entry_model TEXT, orb_minutes INTEGER,
            trading_day DATE, pnl_r DOUBLE
        )
    """)
    con.execute("INSERT INTO orb_outcomes VALUES ('MGC','CME_REOPEN','E1',15,'2026-01-02',0.5)")
    con.close()

    fam = FamilyResult(
        family_id="CME_REOPEN_E1_NO_FILTER", orb_label="CME_REOPEN", entry_model="E1",
        filter_type="NO_FILTER", windows_total=1, windows_passed=1,
        weighted_stability=1.0, classification="STABLE", avg_expectancy_r=0.1,
        avg_sharpe=1.0, total_sample_size=2, oos_cumulative_r=0.5,
        double_break_degraded_windows=0,
    )
    # With orb_minutes=15, should find the rows; with default=5 would find nothing
    results = compute_day_of_week_stats(db, [fam], train_months=3,
                                        instrument="MGC", orb_minutes=15)
    assert results[0].day_of_week_stats is not None
```

**Step 2: Run to verify FAIL** (function has no `orb_minutes` param)

**Step 3: Update `compute_day_of_week_stats` in `rolling_portfolio.py`**

Change signature:
```python
def compute_day_of_week_stats(
    db_path: Path,
    family_results: list[FamilyResult],
    train_months: int,
    instrument: str = "MGC",
    orb_minutes: int = 5,   # ADD — was hardcoded to 5
) -> list[FamilyResult]:
```

Change SQL WHERE at line 323:
```python
            WHERE symbol = ? AND orb_minutes = ?
```
And update params from `[instrument]` to `[instrument, orb_minutes]`.

Remove the TODO comment block (lines 315-318).

**Step 4: Run to verify PASS**
```bash
python -m pytest tests/test_trading_app/test_rolling_portfolio.py -x -q
python pipeline/check_drift.py
```

**Step 5: Commit**
```bash
git add trading_app/rolling_portfolio.py tests/test_trading_app/test_rolling_portfolio.py
git commit -m "fix: pass orb_minutes through compute_day_of_week_stats, remove hardcoded 5 (DF-04)"
```
